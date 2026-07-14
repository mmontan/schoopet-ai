"""OAuth HTTP handler — IAM connector callback only."""
import asyncio
import copy
import html
import logging
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from ..auth.connector import finalize_iam_credentials
from ..email.handler import register_gmail_watch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth", tags=["oauth"])

_session_manager = None
_agent_client = None
_discord_sender = None

# Seconds to wait after credentials:finalize before resuming the agent.
# Gives the IAM connector backend time to transition from uri_consent_required
# to consent_pending so the ADK poll loop (not the immediate-raise path) runs.
_CREDENTIAL_PROPAGATION_DELAY_SECONDS: float = 5.0


async def _resume_agent_after_consent(
    user_id: str,
    session_id: str,
    credential_fc_id: str,
    auth_config_dict: dict,
) -> None:
    """Background task: wait for IAM connector propagation then resume the agent.

    The delay lets the IAM connector transition from uri_consent_required to
    consent_pending before the agent retries the blocked workspace tool, so
    the ADK's consent_pending poll loop runs instead of the immediate-raise path.
    """
    uid = user_id if len(user_id) > 4 else user_id
    credential_key = auth_config_dict.get("credentialKey", "unknown") if auth_config_dict else "unknown"
    logger.info(
        f"[oauth] resume_after_consent: sleeping {_CREDENTIAL_PROPAGATION_DELAY_SECONDS}s "
        f"for IAM propagation — user={uid} session={session_id} "
        f"fc_id={credential_fc_id!r} credentialKey={credential_key!r}"
    )
    await asyncio.sleep(_CREDENTIAL_PROPAGATION_DELAY_SECONDS)

    # Strip authUri before responding so ADK treats this as "consent complete" rather
    # than "still waiting for auth". Sending back the original authUri causes ADK to
    # keep the session in uri_consent_required state, making every workspace tool call
    # return empty even though credentials:finalize already stored the token.
    auth_config_complete = copy.deepcopy(auth_config_dict) if auth_config_dict else {}
    exchanged = auth_config_complete.get("exchangedAuthCredential") or {}
    oauth2 = dict(exchanged.get("oauth2") or {})
    oauth2.pop("authUri", None)
    oauth2.pop("auth_uri", None)
    if exchanged or oauth2:
        auth_config_complete.setdefault("exchangedAuthCredential", {})["oauth2"] = oauth2

    logger.info(f"[oauth] resume_after_consent: sending credential response for user={uid}")
    try:
        events = await _agent_client.send_credential_response(
            user_id=user_id,
            session_id=session_id,
            credential_function_call_id=credential_fc_id,
            auth_config_dict=auth_config_complete,
        )
        logger.info(
            f"[oauth] resume_after_consent: credential response sent, "
            f"{len(events)} events received for user={uid}"
        )
    except Exception as e:
        logger.error(
            f"[oauth] resume_after_consent: failed to resume agent for user={uid}: {e}",
            exc_info=True,
        )
        return

    # Resolve any auto-credential requests the resumed session emitted so that
    # workspace tools can actually fetch the stored token from the IAM connector.
    try:
        events, remaining_cred_req = await _agent_client.resolve_iam_credential_events(
            user_id=user_id,
            session_id=session_id,
            events=events,
            context="resume_after_consent",
        )
        if remaining_cred_req:
            logger.warning(
                f"[oauth] resume_after_consent: credential still interactive after resume "
                f"for user={uid} — nonce={remaining_cred_req.nonce[:8] if remaining_cred_req.nonce else 'n/a'}..."
            )
    except Exception as e:
        logger.warning(
            f"[oauth] resume_after_consent: resolve_iam_credential_events failed for user={uid}: {e}",
            exc_info=True,
        )

    logger.info(f"[oauth] resume_after_consent: clearing pending credential for user={uid}")
    await _session_manager.clear_pending_credential(user_id)

    if _discord_sender and events:
        from ..agent.client import AgentEngineClient
        response_text = AgentEngineClient.extract_text(events)
        logger.info(
            f"[oauth] resume_after_consent: agent response {len(response_text)} chars for user={uid}"
        )
        if response_text:
            try:
                await _discord_sender.send(user_id, response_text)
                logger.info(f"[oauth] Forwarded post-consent agent response to Discord user={uid}")
            except Exception as e:
                logger.warning(
                    f"[oauth] Failed to forward post-consent response to Discord for user={uid}: {e}",
                    exc_info=True,
                )

    logger.info(f"[oauth] resume_after_consent: requesting Gmail watch setup for user={uid}")
    try:
        await register_gmail_watch(user_id)
        logger.info(f"[oauth] Gmail watch setup requested for user={uid}")
    except Exception as e:
        logger.warning(f"[oauth] Gmail watch setup request failed for user={uid}: {e}", exc_info=True)


def init_oauth_services(session_manager=None, agent_client=None, discord_sender=None):
    """Initialize OAuth services for the handler."""
    global _session_manager, _agent_client, _discord_sender
    _session_manager = session_manager
    _agent_client = agent_client
    _discord_sender = discord_sender


def _success_html(email: str, service_name: str = "Google Calendar") -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authorization Successful</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 400px;
                margin: 1rem;
            }}
            .success-icon {{ font-size: 4rem; margin-bottom: 1rem; }}
            h1 {{ color: #22c55e; margin-bottom: 0.5rem; }}
            p {{ color: #666; line-height: 1.6; }}
            .email {{
                background: #f0f9ff;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-family: monospace;
                color: #0369a1;
                margin: 1rem 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="success-icon">&#10004;</div>
            <h1>Authorization Successful!</h1>
            <p>Your {html.escape(service_name)} has been connected.</p>
            <div class="email">{html.escape(email)}</div>
            <p>You can now close this window and use Schoopet.</p>
        </div>
    </body>
    </html>
    """


def _error_html(message: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authorization Failed</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }}
            .container {{
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 400px;
                margin: 1rem;
            }}
            .error-icon {{ font-size: 4rem; margin-bottom: 1rem; }}
            h1 {{ color: #ef4444; margin-bottom: 0.5rem; }}
            p {{ color: #666; line-height: 1.6; }}
            .message {{
                background: #fef2f2;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                color: #991b1b;
                margin: 1rem 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="error-icon">&#10006;</div>
            <h1>Authorization Failed</h1>
            <div class="message">{html.escape(message)}</div>
            <p>Please try again by requesting a new authorization link from Schoopet.</p>
        </div>
    </body>
    </html>
    """


@router.get("/authorize")
async def oauth_authorize(
    nonce: str = Query(..., description="Pending credential nonce"),
    uid: str = Query(..., description="URL-encoded user_id"),
):
    """Redirect to the Google OAuth consent page for a pending credential request."""
    if not _session_manager:
        return HTMLResponse(_error_html("Service not initialized."), status_code=503)

    user_id = unquote(uid)
    uid_tag = user_id if len(user_id) > 4 else user_id
    logger.info(
        f"[oauth] /authorize: user={uid_tag} nonce={nonce[:8] if nonce else 'n/a'}..."
    )

    pending = await _session_manager.get_pending_credential(user_id)
    if not pending or pending.get("nonce") != nonce:
        logger.warning(
            f"[oauth] /authorize: no matching pending credential for user={uid_tag} "
            f"nonce={nonce[:8] if nonce else 'n/a'}... "
            f"(found={bool(pending)} nonce_match={pending.get('nonce') == nonce if pending else False})"
        )
        return HTMLResponse(
            _error_html("Authorization link not found or already used."),
            status_code=404,
        )

    auth_uri = pending.get("auth_uri", "")
    if not auth_uri:
        logger.error(f"[oauth] /authorize: pending credential has no auth_uri for user={uid_tag}")
        return HTMLResponse(_error_html("Authorization URI missing."), status_code=404)

    logger.info(
        f"[oauth] /authorize: redirecting user={uid_tag} "
        f"to auth_uri={auth_uri[:100]}..."
    )
    return RedirectResponse(url=auth_uri)


@router.get("/connector/callback", response_class=HTMLResponse)
async def connector_callback(
    background_tasks: BackgroundTasks,
    uid: str = Query(None, description="URL-encoded user_id embedded in continue_uri"),
    user_id_validation_state: str = Query(None, description="Opaque state from IAM connector (no nonce echoed)"),
    connector_name: str = Query(None, description="Connector resource name from IAM connector"),
    state: str = Query(None, description="Nonce echoed by IAM connector (Path A, not used for routing)"),
    nonce: str = Query(None, description="Alternate nonce parameter (not used for routing)"),
    error: str = Query(None),
    error_description: str = Query(None),
):
    """Finalize IAM connector consent and resume the agent session."""
    logger.info(
        f"[oauth] /connector/callback received: uid={uid!r} "
        f"has_user_id_validation_state={bool(user_id_validation_state)} "
        f"has_connector_name={bool(connector_name)} connector_name={connector_name!r} "
        f"has_state={bool(state)} has_error={bool(error)}"
    )

    if error:
        logger.warning(
            f"[oauth] /connector/callback OAuth error: error={error!r} "
            f"description={error_description!r}"
        )
        return HTMLResponse(_error_html(error_description or error), status_code=400)

    if not _session_manager:
        logger.error("[oauth] /connector/callback: session manager not initialized")
        return HTMLResponse(_error_html("Service not initialized."), status_code=503)

    if not _agent_client:
        logger.error("[oauth] /connector/callback: agent client not initialized")
        return HTMLResponse(_error_html("Service not initialized."), status_code=503)

    routing_user_id = unquote(uid) if uid else None
    if not routing_user_id:
        logger.error("[oauth] /connector/callback: missing uid — cannot route to user")
        return HTMLResponse(_error_html("Missing user identity in callback."), status_code=400)

    uid_tag = routing_user_id if len(routing_user_id) > 4 else routing_user_id
    pending = await _session_manager.get_pending_credential(routing_user_id)
    if not pending:
        logger.warning(
            f"[oauth] /connector/callback: no pending credential for user={uid_tag}"
        )
        return HTMLResponse(
            _error_html("No pending authorization found. Please try again."),
            status_code=400,
        )

    consent_nonce = pending.get("nonce", "")
    needs_finalize = bool(user_id_validation_state)

    user_id = pending.get("user_id", "")
    session_id = pending.get("session_id", "")
    credential_fc_id = pending.get("credential_function_call_id", "")
    auth_config_dict = pending.get("auth_config_dict") or {}
    credential_key = auth_config_dict.get("credentialKey", "unknown")

    logger.info(
        f"[oauth] /connector/callback: consent complete for user={uid_tag} "
        f"session={session_id} nonce={consent_nonce[:8] if consent_nonce else 'n/a'}... "
        f"needs_finalize={needs_finalize} credentialKey={credential_key!r} "
        f"fc_id={credential_fc_id!r}"
    )

    # Call credentials:finalize to store the token in the IAM connector backend.
    # Required per the IAM connector 3LO spec — without this the credential is
    # not stored and subsequent get_auth_credential calls keep returning None.
    if needs_finalize:
        effective_connector = connector_name or pending.get("auth_config_dict", {}).get(
            "authScheme", {}
        ).get("name", "")
        logger.info(
            f"[oauth] /connector/callback: calling credentials:finalize "
            f"connector={effective_connector!r} user={uid_tag}"
        )
        try:
            await finalize_iam_credentials(
                connector_name=effective_connector,
                user_id=user_id,
                consent_nonce=consent_nonce,
                user_id_validation_state=user_id_validation_state,
            )
            logger.info(f"[oauth] /connector/callback: credentials:finalize succeeded for user={uid_tag}")
        except Exception as e:
            logger.error(
                f"[oauth] /connector/callback: credentials:finalize failed for user={uid_tag}: {e}",
                exc_info=True,
            )
            return HTMLResponse(_error_html("Failed to finalize authorization. Please try again."), status_code=500)
    else:
        logger.info(
            f"[oauth] /connector/callback: skipping credentials:finalize "
            f"(no user_id_validation_state) for user={uid_tag}"
        )

    logger.info(
        f"[oauth] /connector/callback: scheduling resume_after_consent "
        f"for user={uid_tag} session={session_id}"
    )
    background_tasks.add_task(
        _resume_agent_after_consent,
        user_id=user_id,
        session_id=session_id,
        credential_fc_id=credential_fc_id,
        auth_config_dict=auth_config_dict,
    )
    return HTMLResponse(_success_html("your Google account", "Google Workspace"))

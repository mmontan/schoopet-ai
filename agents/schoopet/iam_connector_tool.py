"""Diagnostic tool for querying the IAM connector directly.

Mirrors the approach in ADK's GcpAuthProvider: uses the sync
IAMConnectorCredentialsServiceClient with transport="rest" via
asyncio.to_thread, and manually deserializes operation.response/metadata
(the Pre-GA service does not support LRO polling via operation.result()).
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

from google.cloud.iamconnectorcredentials_v1alpha import (
    IAMConnectorCredentialsServiceClient,
    RetrieveCredentialsMetadata,
    RetrieveCredentialsRequest,
    RetrieveCredentialsResponse,
)
from google.api_core.client_options import ClientOptions

from .gcp_auth import GOOGLE_PERSONAL_SCOPES, build_auth_config

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 1.0
_POLL_TIMEOUT_SEC = 15.0


def _make_client() -> IAMConnectorCredentialsServiceClient:
    client_options = None
    if host := os.environ.get("IAM_CONNECTOR_CREDENTIALS_TARGET_HOST"):
        client_options = ClientOptions(api_endpoint=host)
    return IAMConnectorCredentialsServiceClient(
        client_options=client_options, transport="rest"
    )


async def check_connector_credential(tool_context=None, target_user_id: str = "") -> dict:
    """Query the IAM connector to inspect what credential it holds for a user.

    Mirrors ADK's GcpAuthProvider: uses the sync REST client via
    asyncio.to_thread and manually deserializes the operation proto, because
    the Pre-GA service does not support LRO polling via operation.result().

    Args:
        target_user_id: User ID to query. Defaults to the current session user.

    Returns:
        Dict with has_credential, token_prefix, header, expire_time, expired,
        valid_seconds, scopes, connector, user_id, consent_status, and error.
    """
    connector_name = os.getenv("IAM_CONNECTOR_GOOGLE_PERSONAL_NAME", "")
    if not connector_name:
        return {"error": "IAM_CONNECTOR_GOOGLE_PERSONAL_NAME env var not set"}

    user_id = target_user_id or (getattr(tool_context, "user_id", "") if tool_context else "")
    if not user_id:
        return {"error": "No user_id available — pass target_user_id or call from a user session"}

    continue_uri = build_auth_config(user_id).auth_scheme.continue_uri or ""

    logger.info(
        "[iam-connector-tool] starting retrieve_credentials: user=%s connector=%s "
        "scopes=%s continue_uri=%s",
        user_id, connector_name, GOOGLE_PERSONAL_SCOPES, continue_uri,
    )

    # NOTE: force_refresh was removed from the v1alpha IAM connector API. Sending it
    # (True or False) causes a 400 "Unknown name forceRefresh: Cannot find field".
    # The main ADK retrieve path is already patched in gcp_auth.py to omit it; this
    # diagnostic tool must do the same or every retrieve_credentials call 400s.
    request = RetrieveCredentialsRequest(
        connector=connector_name,
        user_id=user_id,
        scopes=GOOGLE_PERSONAL_SCOPES,
        continue_uri=continue_uri,
    )

    client = _make_client()

    def _call_retrieve() -> object:
        op = client.retrieve_credentials(request)
        return op.operation  # raw proto Operation

    try:
        operation = await asyncio.to_thread(_call_retrieve)
    except Exception as exc:
        logger.error("[iam-connector-tool] retrieve_credentials RPC failed for user=%s: %s", user_id, exc)
        return {"error": str(exc), "user_id": user_id, "connector": connector_name}

    logger.info(
        "[iam-connector-tool] initial operation: user=%s done=%s has_response=%s has_metadata=%s has_error=%s",
        user_id, operation.done, bool(operation.response.value), bool(operation.metadata.value),
        operation.HasField("error"),
    )

    # Unpack helper — mirrors ADK's _unpack_operation
    def _unpack(op):
        response = None
        metadata = None
        if op.response and op.response.value:
            response = RetrieveCredentialsResponse.deserialize(op.response.value)
            logger.info("[iam-connector-tool] raw response for user=%s: %s", user_id, response)
        if op.metadata and op.metadata.value:
            metadata = RetrieveCredentialsMetadata.deserialize(op.metadata.value)
            logger.info("[iam-connector-tool] raw metadata for user=%s: %s", user_id, metadata)
        return response, metadata

    # If done immediately, return result
    if operation.done:
        if operation.HasField("error"):
            logger.error("[iam-connector-tool] operation error for user=%s: %s", user_id, operation.error.message)
            return {"error": operation.error.message, "user_id": user_id, "connector": connector_name}
        response, _ = _unpack(operation)
        return _build_result(response, user_id, connector_name)

    # Not done — check metadata to decide whether to poll or return immediately
    _, metadata = _unpack(operation)

    if metadata and getattr(metadata, "consent_pending", False):
        # 2-legged OAuth: poll until token appears or timeout
        logger.info("[iam-connector-tool] consent_pending — polling for token: user=%s", user_id)
        import time as _time
        end = _time.time() + _POLL_TIMEOUT_SEC
        while _time.time() < end:
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            try:
                operation = await asyncio.to_thread(_call_retrieve)
            except Exception as exc:
                logger.error("[iam-connector-tool] poll RPC failed for user=%s: %s", user_id, exc)
                return {"error": str(exc), "user_id": user_id, "connector": connector_name}
            logger.info(
                "[iam-connector-tool] poll result: user=%s done=%s has_error=%s",
                user_id, operation.done, operation.HasField("error"),
            )
            if operation.done:
                if operation.HasField("error"):
                    return {"error": operation.error.message, "user_id": user_id, "connector": connector_name}
                response, _ = _unpack(operation)
                return _build_result(response, user_id, connector_name)
        logger.warning("[iam-connector-tool] poll timeout for user=%s", user_id)
        return {"has_credential": False, "consent_status": "consent_pending_timeout", "user_id": user_id, "connector": connector_name}

    if metadata and getattr(metadata, "uri_consent_required", None):
        uri_info = metadata.uri_consent_required
        auth_uri = getattr(uri_info, "authorization_uri", None) or str(uri_info)
        logger.warning("[iam-connector-tool] uri_consent_required for user=%s auth_uri=%s", user_id, auth_uri)
        return {
            "has_credential": False,
            "consent_status": "uri_consent_required",
            "authorization_uri": auth_uri,
            "user_id": user_id,
            "connector": connector_name,
        }

    if metadata and getattr(metadata, "consent_rejected", False):
        logger.warning("[iam-connector-tool] consent_rejected for user=%s", user_id)
        return {"has_credential": False, "consent_status": "consent_rejected", "user_id": user_id, "connector": connector_name}

    # Unknown pending state
    logger.warning("[iam-connector-tool] unknown pending state for user=%s metadata=%s", user_id, metadata)
    return {"has_credential": False, "consent_status": "unknown_pending", "user_id": user_id, "connector": connector_name}


def _build_result(response: RetrieveCredentialsResponse | None, user_id: str, connector_name: str) -> dict:
    if not response or not response.token:
        logger.warning("[iam-connector-tool] operation done but no token for user=%s", user_id)
        return {"has_credential": False, "consent_status": "done_no_token", "user_id": user_id, "connector": connector_name}

    token = response.token
    header = response.header or ""
    scopes = list(response.scopes) if response.scopes else []
    expire_time_pb = response.expire_time
    token_prefix = token[:8] + "..."

    result: dict = {
        "has_credential": True,
        "token_prefix": token_prefix,
        "header": header,
        "scopes": scopes,
        "user_id": user_id,
        "connector": connector_name,
    }

    # expire_time comes back as DatetimeWithNanoseconds (proto-plus datetime),
    # not a raw protobuf Timestamp, so we can't use .seconds.
    dt = None
    if isinstance(expire_time_pb, datetime):
        dt = expire_time_pb if expire_time_pb.tzinfo else expire_time_pb.replace(tzinfo=timezone.utc)
    elif expire_time_pb and getattr(expire_time_pb, "seconds", None):
        dt = datetime.fromtimestamp(expire_time_pb.seconds, tz=timezone.utc)

    if dt is not None:
        expire_iso = dt.isoformat()
        result["expire_time"] = expire_iso
        now = datetime.now(timezone.utc)
        if dt <= now:
            result["expired"] = True
            result["expired_seconds_ago"] = round((now - dt).total_seconds())
            logger.warning(
                "[iam-connector-tool] credential EXPIRED for user=%s expired_at=%s expired_seconds_ago=%s",
                user_id, expire_iso, result["expired_seconds_ago"],
            )
        else:
            result["expired"] = False
            result["valid_seconds"] = round((dt - now).total_seconds())
            logger.info(
                "[iam-connector-tool] credential valid for user=%s expires=%s valid_seconds=%s",
                user_id, expire_iso, result["valid_seconds"],
            )
    else:
        result["expire_time"] = None
        logger.info("[iam-connector-tool] no expire_time for user=%s", user_id)

    logger.info(
        "[iam-connector-tool] result: user=%s has_credential=True token_prefix=%s expire=%s expired=%s",
        user_id, token_prefix, result.get("expire_time"), result.get("expired"),
    )
    return result

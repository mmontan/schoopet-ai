"""Vertex AI Agent Engine client wrapper."""

import asyncio
from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Union

import vertexai
import vertexai.agent_engines
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)

_RAW_RESPONSE_PREFIX = "Failed to parse response as JSON. Raw response: "

# HTTP status codes that are worth retrying with backoff.
_RETRYABLE_CODES = {429, 503, 500}
# Maximum retries for transient stream errors (inside the client).
_STREAM_MAX_RETRIES = 3
# Base delay in seconds for exponential backoff between retries.
_STREAM_RETRY_BASE_DELAY = 2.0


def _split_json_objects(text: str) -> list[dict]:
    """Parse one or more concatenated JSON objects from text.

    Agent Engine occasionally sends two ADK events glued together in a single
    stream chunk (e.g. function_response + final model response).  The genai
    library's chunk parser calls json.loads() on the whole chunk and raises
    UnknownApiResponseError / JSONDecodeError("Extra data").  This helper
    splits the raw string into individual objects so callers can recover.
    """
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            break
    return objects


class _TokenExpiredError(Exception):
    """Internal signal that the Vertex AI access token expired mid-stream."""


@dataclass
class StreamResult:
    """Result of streaming agent events, including any errors encountered."""

    events: list = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def error_summary(self) -> str:
        """Human-readable summary of errors encountered during streaming."""
        if not self.errors:
            return ""
        parts = []
        for err in self.errors:
            code = err.get("code", "unknown")
            message = err.get("message", "no details")
            parts.append(f"code={code}: {message}")
        return "; ".join(parts)


@dataclass(frozen=True)
class AdkCredentialRequest:
    """An adk_request_credential function call extracted from an event stream."""

    function_call_id: str
    original_function_call_id: str
    auth_uri: str
    nonce: str
    auth_config_dict: dict

    def to_firestore(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "original_function_call_id": self.original_function_call_id,
            "auth_uri": self.auth_uri,
            "nonce": self.nonce,
            "auth_config_dict": self.auth_config_dict,
        }


@dataclass(frozen=True)
class AdkConfirmationRequest:
    """Native ADK confirmation request extracted from an event stream."""

    function_call_id: str
    original_function_call: dict[str, Any]
    tool_confirmation: dict[str, Any]

    @property
    def tool_name(self) -> str:
        return str(self.original_function_call.get("name") or "unknown_tool")

    @property
    def tool_args(self) -> dict[str, Any]:
        args = self.original_function_call.get("args")
        return args if isinstance(args, dict) else {}

    @property
    def original_function_call_id(self) -> str:
        return str(self.original_function_call.get("id") or "")

    @property
    def hint(self) -> str:
        return str(self.tool_confirmation.get("hint") or "")

    @property
    def payload(self) -> Any:
        return self.tool_confirmation.get("payload")

    def to_firestore(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "original_function_call": self.original_function_call,
            "tool_confirmation": self.tool_confirmation,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "original_function_call_id": self.original_function_call_id,
            "hint": self.hint,
            "payload": self.payload,
        }


def _log_tool_activity(event: Event, user_id: str) -> None:
    """Log function calls and responses from an ADK event."""
    uid = user_id if len(user_id) > 4 else user_id

    for fc in event.get_function_calls() or []:
        args = fc.args or {}
        # Truncate large args (e.g. email bodies) to keep logs readable
        summarized = {
            k: (str(v)[:120] if isinstance(v, str) else v) for k, v in args.items()
        }
        logger.info(f"Tool call [{uid}]: {fc.name}({summarized})")

    for fr in event.get_function_responses() or []:
        response = fr.response or {}
        output = response.get("output") or response.get("result") or response
        output_str = str(output)
        if len(output_str) > 200:
            output_str = output_str[:200] + "…"
        # Empty result on a workspace tool almost always means the IAM connector
        # credential was auto-resolved but the underlying API call still failed
        # (e.g. token not yet propagated, wrong scopes, or connector state mismatch).
        if "result" in response and response["result"] == "":
            logger.warning(
                f"Tool response [{uid}]: {fr.name} → EMPTY RESULT "
                f"(credential may not be providing a valid token) "
                f"full_response={response!r}"
            )
        else:
            logger.info(f"Tool response [{uid}]: {fr.name} → {output_str}")


class AgentEngineClient:
    """Client for interacting with Vertex AI Agent Engine.

    Wraps the Agent Engine SDK to provide session management and
    message streaming for the SMS gateway.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        agent_engine_id: str,
        timeout_seconds: int = 30,
    ):
        """Initialize the Agent Engine client.

        Args:
            project_id: Google Cloud project ID.
            location: Google Cloud region (e.g., us-central1).
            agent_engine_id: The Agent Engine resource ID.
            timeout_seconds: Timeout for agent queries (default: 30).
        """
        self._project_id = project_id
        self._location = location
        self._agent_engine_id = agent_engine_id
        self._timeout = timeout_seconds

        self._resource_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{agent_engine_id}"
        )
        self._engine = None
        self._init_client()
        logger.info(f"Initialized AgentEngineClient for {self._resource_name}")

    def _init_client(self):
        """(Re)initialize the Vertex AI client and agent engine reference.

        Called at startup and again whenever the token expires (401).
        vertexai.init() reinitializes the global config with fresh ADC credentials.
        """
        vertexai.init(project=self._project_id, location=self._location)
        self._engine = vertexai.agent_engines.AgentEngine(self._resource_name)

    async def create_session(self, user_id: str, state: dict | None = None) -> str:
        """Create a new Agent Engine session.

        Args:
            user_id: User identifier.
            state: Optional initial session state (e.g., {"channel": "discord"}).

        Returns:
            The session ID string.
        """
        result = await self._engine.async_create_session(user_id=user_id, state=state)
        session_id = result["id"]
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete an Agent Engine session.

        Best-effort — failures are logged and swallowed so they cannot
        block downstream session creation.
        """
        try:
            await self._engine.async_delete_session(
                user_id=user_id, session_id=session_id
            )
            logger.info(f"Deleted session {session_id} for user {user_id}")
        except Exception as e:
            logger.warning(
                f"Failed to delete session {session_id} for user {user_id}: {e}"
            )

    async def send_message(
        self,
        user_id: str,
        session_id: str,
        message: Union[str, types.Content],
    ) -> str:
        """Send a message to the agent and collect the full response.

        Streams the response from the agent and concatenates all text parts
        into a single response string.

        Args:
            user_id: User identifier.
            session_id: Agent Engine session ID.
            message: The user's message — either a plain string or a
                     types.Content with multimodal parts (text + inline_data).

        Returns:
            The complete agent response text.

        Raises:
            asyncio.TimeoutError: If the agent doesn't respond within timeout.
        """
        events = await self.send_message_events(user_id, session_id, message)
        return self.extract_text(events)

    async def send_message_events(
        self,
        user_id: str,
        session_id: str,
        message: Union[str, types.Content],
        on_tool_call: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[Event]:
        """Send a message to the agent and return native ADK events.

        Retries up to _STREAM_MAX_RETRIES times with exponential backoff
        when the stream completes with only retryable errors (e.g. 429)
        and no text content.
        """
        last_result: StreamResult | None = None
        for attempt in range(1 + _STREAM_MAX_RETRIES):
            result = await self._stream_message(
                user_id, session_id, message, on_tool_call
            )
            response_text = self.extract_text(result.events)

            if response_text or not result.has_errors:
                # Got a real response or a clean empty (no errors) — done.
                self._last_stream_errors = result.errors
                return result.events

            # Check if all errors are retryable.
            all_retryable = all(
                err.get("code") in _RETRYABLE_CODES for err in result.errors
            )
            if not all_retryable:
                # Non-retryable error — give up immediately.
                break

            last_result = result
            if attempt < _STREAM_MAX_RETRIES:
                delay = _STREAM_RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Retryable stream error(s) for user=%s session=%s "
                    "(attempt %d/%d): %s — retrying in %.1fs",
                    user_id,
                    session_id,
                    attempt + 1,
                    _STREAM_MAX_RETRIES + 1,
                    result.error_summary,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted or non-retryable error.
        final = last_result or result
        self._last_stream_errors = final.errors
        return final.events

    @property
    def last_stream_errors(self) -> list[dict[str, Any]]:
        """Errors encountered during the most recent send_message_events call."""
        return getattr(self, "_last_stream_errors", [])

    @property
    def last_stream_error_summary(self) -> str:
        """Human-readable summary of errors from the most recent stream."""
        errors = self.last_stream_errors
        if not errors:
            return ""
        parts = []
        for err in errors:
            code = err.get("code", "unknown")
            message = err.get("message", "no details")
            parts.append(f"code={code}: {message}")
        return "; ".join(parts)

    async def _stream_message(
        self,
        user_id: str,
        session_id: str,
        message: Union[str, types.Content],
        on_tool_call: Callable[[str], Awaitable[None]] | None = None,
    ) -> StreamResult:
        """Stream agent events, tracking any errors encountered."""
        logger.info(f"Sending message to agent: user={user_id}, session={session_id}")

        msg_payload: Union[str, dict] = (
            message
            if isinstance(message, str)
            else message.model_dump(exclude_none=True, mode="json")
        )

        result = StreamResult()
        t_start = time.monotonic()
        ttft_ms: float | None = None

        async def _emit(event: Event) -> None:
            nonlocal ttft_ms
            result.events.append(event)
            if ttft_ms is None and self.extract_text([event]):
                ttft_ms = (time.monotonic() - t_start) * 1000
            _log_tool_activity(event, user_id)
            if on_tool_call:
                for fc in event.get_function_calls() or []:
                    if fc.name and not fc.name.startswith("adk_"):
                        await on_tool_call(fc.name)

        def _record_stream_error(error_dict: dict) -> None:
            """Log and record an Agent Engine error from the stream."""
            code = error_dict.get("code", "unknown")
            message = error_dict.get("message", "no details")
            result.errors.append(error_dict)
            logger.error(
                "Agent Engine stream error: code=%s, message=%s (user=%s, session=%s)",
                code,
                message,
                user_id,
                session_id,
            )

        async def stream_response():
            try:
                async for event in self._engine.async_stream_query(
                    user_id=user_id,
                    session_id=session_id,
                    message=msg_payload,
                ):
                    # Reinitialize client on expired token and propagate so caller retries
                    if isinstance(event, dict) and event.get("code") == 401:
                        logger.warning(
                            "Access token expired — reinitializing Vertex AI client"
                        )
                        self._init_client()
                        raise _TokenExpiredError()
                    logger.debug(f"Event type: {type(event).__name__}, event: {event}")

                    if isinstance(event, dict):
                        if "code" in event and "message" in event:
                            _record_stream_error(event)
                            continue
                        await _emit(self._coerce_event(event))
                        continue

                    parsed_event = (
                        event
                        if isinstance(event, Event)
                        else Event.model_validate(event)
                    )
                    await _emit(parsed_event)
            except _TokenExpiredError:
                raise
            except Exception as exc:
                # Agent Engine occasionally concatenates two JSON objects in one
                # stream chunk, causing UnknownApiResponseError("Extra data").
                # Recover by splitting the raw payload into individual objects.
                exc_str = str(exc)
                if _RAW_RESPONSE_PREFIX not in exc_str:
                    raise
                raw = exc_str[
                    exc_str.index(_RAW_RESPONSE_PREFIX) + len(_RAW_RESPONSE_PREFIX) :
                ]
                recovered = _split_json_objects(raw)
                if not recovered:
                    raise
                logger.warning(
                    f"Recovered {len(recovered)} events from concatenated JSON stream chunk"
                )
                for event_dict in recovered:
                    if "code" in event_dict and "message" in event_dict:
                        _record_stream_error(event_dict)
                        continue
                    await _emit(self._coerce_event(event_dict))

        # Apply timeout to the streaming operation; retry once on token expiry
        try:
            await asyncio.wait_for(stream_response(), timeout=self._timeout)
        except _TokenExpiredError:
            logger.info("Retrying stream after token refresh")
            result.events.clear()
            result.errors.clear()
            ttft_ms = None
            t_start = time.monotonic()
            await asyncio.wait_for(stream_response(), timeout=self._timeout)

        total_ms = (time.monotonic() - t_start) * 1000
        response_text = self.extract_text(result.events)
        error_suffix = f", stream_errors={len(result.errors)}" if result.errors else ""
        logger.info(
            f"Agent response: {len(response_text)} chars "
            f"[ttft={ttft_ms:.0f}ms, total={total_ms:.0f}ms{error_suffix}]"
            if ttft_ms is not None
            else f"Agent response: {len(response_text)} chars "
            f"[no text received, total={total_ms:.0f}ms{error_suffix}]"
        )

        return result

    async def send_confirmation_response(
        self,
        user_id: str,
        session_id: str,
        confirmation_function_call_id: str,
        confirmed: bool,
    ) -> list[Event]:
        """Resolve an ADK confirmation request with a native function response."""
        return await self.send_confirmation_responses_batch(
            user_id=user_id,
            session_id=session_id,
            confirmations=[(confirmation_function_call_id, confirmed)],
        )

    async def send_confirmation_responses_batch(
        self,
        user_id: str,
        session_id: str,
        confirmations: list[tuple[str, bool]],
    ) -> list[Event]:
        """Resolve multiple ADK confirmations in a single agent turn.

        Sends all responses as one types.Content with multiple FunctionResponse
        parts so the agent runs exactly once regardless of batch size.
        """
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_confirmation",
                    id=fc_id,
                    response={"confirmed": confirmed},
                )
            )
            for fc_id, confirmed in confirmations
        ]
        content = types.Content(role="user", parts=parts)
        return await self.send_message_events(user_id, session_id, content)

    @staticmethod
    def extract_text(events: list[Event]) -> str:
        """Extract concatenated text parts from ADK events."""
        chunks: list[str] = []
        for event in events:
            content = getattr(event, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def extract_confirmation_requests(
        events: list[Event],
    ) -> list[AdkConfirmationRequest]:
        """Extract native ADK confirmation function calls from events."""
        confirmations: list[AdkConfirmationRequest] = []
        for event in events:
            for function_call in event.get_function_calls() or []:
                if function_call.name != "adk_request_confirmation":
                    continue
                args = function_call.args or {}
                original = (
                    args.get("originalFunctionCall")
                    or args.get("original_function_call")
                    or {}
                )
                tool_confirmation = (
                    args.get("toolConfirmation") or args.get("tool_confirmation") or {}
                )
                confirmations.append(
                    AdkConfirmationRequest(
                        function_call_id=str(function_call.id or ""),
                        original_function_call=original
                        if isinstance(original, dict)
                        else {},
                        tool_confirmation=(
                            tool_confirmation
                            if isinstance(tool_confirmation, dict)
                            else {}
                        ),
                    )
                )
        return confirmations

    @staticmethod
    def extract_gcp_auto_credential_requests(
        events: list[Event],
    ) -> list[tuple[str, dict]]:
        """Find gcpAuthProviderScheme adk_request_credential calls without an auth URI.

        These occur when the IAM connector credential is already stored in the backend
        (user already consented) but ADK's session state still has a pending credential
        request. The gateway should auto-respond so ADK can fetch the stored token
        without prompting the user again.

        Returns list of (function_call_id, auth_config_dict) tuples.
        """
        results = []
        for event in events:
            for fc in event.get_function_calls() or []:
                if fc.name != "adk_request_credential":
                    continue
                args = fc.args or {}
                auth_config = args.get("authConfig") or args.get("auth_config") or {}
                if not isinstance(auth_config, dict):
                    continue
                scheme_type = (auth_config.get("authScheme") or {}).get("type", "")
                if scheme_type != "gcpAuthProviderScheme":
                    continue
                exchanged = auth_config.get("exchangedAuthCredential") or {}
                oauth2 = exchanged.get("oauth2") or {}
                if oauth2.get("auth_uri") or oauth2.get("authUri"):
                    continue  # has auth URI — needs user action, handled by extract_credential_requests
                credential_key = auth_config.get("credentialKey", "unknown")
                logger.info(
                    f"[auto-cred] Stored IAM credential request: "
                    f"fc_id={fc.id!r} credentialKey={credential_key!r}"
                )
                results.append((str(fc.id or ""), auth_config))
        if results:
            logger.info(
                f"[auto-cred] {len(results)} auto-resolvable credential request(s) found"
            )
        return results

    @staticmethod
    def extract_credential_requests(events: list[Event]) -> list[AdkCredentialRequest]:
        """Extract adk_request_credential function calls from events."""
        requests: list[AdkCredentialRequest] = []
        for event in events:
            for function_call in event.get_function_calls() or []:
                if function_call.name != "adk_request_credential":
                    continue
                args = function_call.args or {}
                fc_id = str(function_call.id or "")
                original_fc_id = str(
                    args.get("functionCallId") or args.get("function_call_id") or ""
                )
                auth_config = args.get("authConfig") or args.get("auth_config") or {}
                if isinstance(auth_config, dict):
                    auth_config_dict = auth_config
                else:
                    try:
                        auth_config_dict = dict(auth_config)
                    except Exception:
                        auth_config_dict = {}
                scheme_type = (auth_config_dict.get("authScheme") or {}).get("type", "")
                credential_key = auth_config_dict.get("credentialKey", "unknown")
                exchanged = (
                    auth_config_dict.get("exchangedAuthCredential")
                    or auth_config_dict.get("exchanged_auth_credential")
                    or {}
                )
                oauth2 = exchanged.get("oauth2") or {}
                auth_uri = str(oauth2.get("auth_uri") or oauth2.get("authUri") or "")
                nonce = str(oauth2.get("nonce") or "")
                logger.info(
                    f"[cred-extract] adk_request_credential: fc_id={fc_id!r} "
                    f"scheme={scheme_type!r} credentialKey={credential_key!r} "
                    f"has_auth_uri={bool(auth_uri)}"
                )
                if not auth_uri:
                    continue
                logger.warning(
                    f"[cred-extract] Interactive auth URI required: fc_id={fc_id!r} "
                    f"credentialKey={credential_key!r} nonce={nonce!r} "
                    f"auth_uri={auth_uri[:80]}..."
                )
                requests.append(
                    AdkCredentialRequest(
                        function_call_id=fc_id,
                        original_function_call_id=original_fc_id,
                        auth_uri=auth_uri,
                        nonce=nonce,
                        auth_config_dict=auth_config_dict,
                    )
                )
        return requests

    # Max iterations for resolving stored (no-authUri) credential requests.
    _AUTO_RESOLVE_CAP = 3
    # Max "settle" retries when an interactive (authUri) request appears but the
    # stored credential may actually be valid — see resolve_iam_credential_events.
    _SETTLE_CAP = 2

    async def _auto_resolve_stored_credentials(
        self,
        user_id: str,
        session_id: str,
        events: list[Event],
        ctx: str,
        uid: str,
    ) -> list[Event]:
        """Respond to stored (no-authUri) credential requests so ADK fetches the token."""
        for _i in range(self._AUTO_RESOLVE_CAP):
            auto_creds = self.extract_gcp_auto_credential_requests(events)
            if not auto_creds:
                if _i > 0:
                    logger.info(
                        f"{ctx}IAM connector auto-resolve complete after {_i} iteration(s) "
                        f"for {uid}"
                    )
                return events
            fc_id, auth_config = auto_creds[0]
            credential_key = auth_config.get("credentialKey", "unknown")
            logger.info(
                f"{ctx}Auto-resolving stored IAM connector credential for {uid}: "
                f"fc_id={fc_id!r} credentialKey={credential_key!r} "
                f"(iteration {_i + 1}/{self._AUTO_RESOLVE_CAP}, {len(auto_creds)} pending)"
            )
            try:
                events = await self.send_credential_response(
                    user_id=user_id,
                    session_id=session_id,
                    credential_function_call_id=fc_id,
                    auth_config_dict=auth_config,
                )
            except asyncio.TimeoutError:
                logger.error(f"{ctx}Timeout during IAM credential auto-resolve for {uid}")
                return events
        remaining = self.extract_gcp_auto_credential_requests(events)
        logger.warning(
            f"{ctx}IAM connector auto-resolve cap ({self._AUTO_RESOLVE_CAP}) reached for {uid}: "
            f"{len(remaining)} credential request(s) still pending — "
            f"stored credential may be invalid or IAM connector state is inconsistent"
        )
        return events

    @staticmethod
    def _strip_auth_uri(auth_config_dict: dict) -> dict:
        """Return a copy of the auth config with authUri removed from oauth2.

        Echoing an adk_request_credential response that still contains authUri keeps
        ADK in the uri_consent_required state. Stripping it makes ADK treat the
        response as "consent complete" and re-retrieve the stored token — which
        succeeds when a valid credential already exists in the IAM connector backend.
        """
        stripped = copy.deepcopy(auth_config_dict) if auth_config_dict else {}
        exchanged = (
            stripped.get("exchangedAuthCredential")
            or stripped.get("exchanged_auth_credential")
            or {}
        )
        oauth2 = dict(exchanged.get("oauth2") or {})
        oauth2.pop("authUri", None)
        oauth2.pop("auth_uri", None)
        if exchanged or oauth2:
            stripped.setdefault("exchangedAuthCredential", {})["oauth2"] = oauth2
        return stripped

    async def resolve_iam_credential_events(
        self,
        user_id: str,
        session_id: str,
        events: list[Event],
        context: str = "",
    ) -> tuple[list[Event], "AdkCredentialRequest | None"]:
        """Auto-resolve stored IAM credentials; detect interactive auth requests.

        Runs the auto-resolve loop (credentials already consented, no auth URI),
        then checks whether an interactive credential request (with auth URI) remains.
        If one does, runs a bounded "settle" retry before declaring it genuine: the
        IAM connector intermittently returns uri_consent_required on the first
        retrieve of a fresh session even when a valid credential is already stored
        (a retry in the same turn succeeds). Echoing the request back with authUri
        stripped makes ADK re-retrieve the stored token; if it is genuinely missing
        or expired ADK returns another authUri and we fall through to prompting.

        Returns (final_events, credential_request_or_none). If credential_request_or_none
        is not None, the caller must store the pending credential and send an auth link.
        """
        uid = user_id if len(user_id) > 4 else user_id
        ctx = f"[{context}] " if context else ""

        events = await self._auto_resolve_stored_credentials(
            user_id, session_id, events, ctx, uid
        )

        credential_requests = self.extract_credential_requests(events)
        if not credential_requests:
            return events, None

        for _s in range(self._SETTLE_CAP):
            req = credential_requests[0]
            logger.info(
                f"{ctx}IAM connector interactive auth for {uid} — settle retry "
                f"{_s + 1}/{self._SETTLE_CAP}: fc_id={req.function_call_id!r} "
                f"nonce={(req.nonce[:8] + '...') if req.nonce else 'n/a'}"
            )
            try:
                events = await self.send_credential_response(
                    user_id=user_id,
                    session_id=session_id,
                    credential_function_call_id=req.function_call_id,
                    auth_config_dict=self._strip_auth_uri(req.auth_config_dict),
                )
            except asyncio.TimeoutError:
                logger.error(f"{ctx}Timeout during IAM credential settle retry for {uid}")
                break
            events = await self._auto_resolve_stored_credentials(
                user_id, session_id, events, ctx, uid
            )
            credential_requests = self.extract_credential_requests(events)
            if not credential_requests:
                logger.info(
                    f"{ctx}IAM connector credential settled after {_s + 1} settle "
                    f"retry(ies) for {uid} — no interactive auth needed"
                )
                return events, None

        req = credential_requests[0]
        logger.warning(
            f"{ctx}IAM connector interactive auth required for {uid}: "
            f"nonce={req.nonce[:8]}... fc_id={req.function_call_id!r} "
            f"credentialKey={req.auth_config_dict.get('credentialKey', 'unknown')!r} "
            f"auth_uri={req.auth_uri[:80]}..."
        )
        return events, req

    async def send_credential_response(
        self,
        user_id: str,
        session_id: str,
        credential_function_call_id: str,
        auth_config_dict: dict,
    ) -> list[Event]:
        """Resume an agent session after IAM connector consent is complete."""
        uid = user_id if len(user_id) > 4 else user_id
        credential_key = auth_config_dict.get("credentialKey", "unknown")
        exchanged = auth_config_dict.get("exchangedAuthCredential") or {}
        oauth2 = exchanged.get("oauth2") or {}
        token_val = (
            oauth2.get("accessToken")
            or oauth2.get("access_token")
            or oauth2.get("token")
            or ""
        )
        logger.info(
            f"Sending credential response: user={uid} "
            f"fc_id={credential_function_call_id!r} credentialKey={credential_key!r} "
            f"has_exchangedAuthCredential={bool(exchanged)} "
            f"has_token={bool(token_val)} "
            f"auth_config_keys={list(auth_config_dict.keys())}"
        )
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_credential",
                        id=credential_function_call_id,
                        response=auth_config_dict,
                    )
                )
            ],
        )
        return await self.send_message_events(user_id, session_id, content)

    @staticmethod
    def _coerce_event(event: dict) -> Event:
        """Validate a stream dict into an ADK Event, with a legacy text fallback."""
        try:
            return Event.model_validate(event)
        except Exception:
            content = event.get("content")
            if isinstance(content, dict):
                return Event(
                    author=str(event.get("author") or "agent"),
                    content=types.Content.model_validate(content),
                )
            raise

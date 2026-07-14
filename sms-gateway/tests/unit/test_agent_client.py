"""Unit tests for the Agent Engine client wrapper."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.events import Event
from google.genai import types

from src.agent.client import AgentEngineClient, AdkConfirmationRequest


class _FakeEngine:
    def __init__(self, events):
        self._events = events

    async def async_stream_query(self, **kwargs):
        for event in self._events:
            yield event

    async def async_create_session(self, **kwargs):
        return {"id": "test-session-id"}

    async def async_delete_session(self, **kwargs):
        pass


def _make_client(events):
    client = AgentEngineClient.__new__(AgentEngineClient)
    client._timeout = 1
    client._resource_name = "projects/test/locations/us-central1/reasoningEngines/123"
    client._engine = _FakeEngine(events)
    client._init_client = MagicMock()
    return client


@pytest.mark.asyncio
async def test_send_message_collects_text_from_dict_events():
    client = _make_client(
        [
            {"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}},
            {"content": {"parts": [{"text": "!"}]}},
        ]
    )

    response = await client.send_message(
        user_id="user-123",
        session_id="session-123",
        message="hi",
    )

    assert response == "Hello world!"
    client._init_client.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_events_validates_dict_events():
    event_dict = {
        "author": "agent",
        "content": {"role": "model", "parts": [{"text": "Hello"}]},
    }
    client = _make_client([event_dict])

    events = await client.send_message_events(
        user_id="user-123",
        session_id="session-123",
        message="hi",
    )

    assert len(events) == 1
    assert isinstance(events[0], Event)
    assert AgentEngineClient.extract_text(events) == "Hello"


def test_extract_confirmation_requests_finds_native_function_calls():
    event = Event(
        author="agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="adk_request_confirmation",
                        id="confirm-1",
                        args={
                            "originalFunctionCall": {
                                "id": "tool-1",
                                "name": "send_email",
                                "args": {"to": "person@example.com"},
                            },
                            "toolConfirmation": {
                                "hint": "Send this email?",
                                "payload": {"risk": "external_send"},
                            },
                        },
                    )
                )
            ],
        ),
    )

    confirmations = AgentEngineClient.extract_confirmation_requests([event])

    assert len(confirmations) == 1
    assert confirmations[0].function_call_id == "confirm-1"
    assert confirmations[0].tool_name == "send_email"
    assert confirmations[0].tool_args == {"to": "person@example.com"}
    assert confirmations[0].original_function_call_id == "tool-1"
    assert confirmations[0].hint == "Send this email?"


@pytest.mark.asyncio
async def test_send_confirmation_response_sends_native_function_response():
    client = AgentEngineClient.__new__(AgentEngineClient)
    client.send_message_events = AsyncMock(return_value=[])

    await client.send_confirmation_response(
        user_id="user-123",
        session_id="session-123",
        confirmation_function_call_id="confirm-1",
        confirmed=False,
    )

    message = client.send_message_events.call_args.args[2]
    function_response = message.parts[0].function_response
    assert function_response.name == "adk_request_confirmation"
    assert function_response.id == "confirm-1"
    assert function_response.response == {"confirmed": False}


@pytest.mark.asyncio
async def test_send_confirmation_responses_batch_sends_all_in_one_turn():
    client = AgentEngineClient.__new__(AgentEngineClient)
    client.send_message_events = AsyncMock(return_value=[])

    await client.send_confirmation_responses_batch(
        user_id="user-123",
        session_id="session-123",
        confirmations=[("confirm-1", True), ("confirm-2", False)],
    )

    client.send_message_events.assert_awaited_once()
    message = client.send_message_events.call_args.args[2]
    assert len(message.parts) == 2
    fr0 = message.parts[0].function_response
    assert fr0.name == "adk_request_confirmation"
    assert fr0.id == "confirm-1"
    assert fr0.response == {"confirmed": True}
    fr1 = message.parts[1].function_response
    assert fr1.id == "confirm-2"
    assert fr1.response == {"confirmed": False}


def test_confirmation_request_serializes_for_firestore():
    confirmation = AdkConfirmationRequest(
        function_call_id="confirm-1",
        original_function_call={"id": "tool-1", "name": "send_email", "args": {"to": "x"}},
        tool_confirmation={"hint": "approve?", "payload": {"kind": "email"}},
    )

    data = confirmation.to_firestore()

    assert data["function_call_id"] == "confirm-1"
    assert data["tool_name"] == "send_email"
    assert data["tool_args"] == {"to": "x"}


def _interactive_credential_event(fc_id: str = "cred-1", auth_uri: str = "https://accounts.google.com/o/oauth2/v2/auth?x=1"):
    """Build an event carrying an interactive (authUri) adk_request_credential call."""
    return Event(
        author="agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="adk_request_credential",
                        id=fc_id,
                        args={
                            "functionCallId": fc_id,
                            "authConfig": {
                                "authScheme": {"type": "gcpAuthProviderScheme"},
                                "exchangedAuthCredential": {
                                    "authType": "oauth2",
                                    "oauth2": {
                                        "authUri": auth_uri,
                                        "nonce": "d372753b-6e62-4f31-bc2d-6a86b92d8258",
                                    },
                                },
                                "credentialKey": "adk_gcpAuthProviderScheme_abc_",
                            },
                        },
                    )
                )
            ],
        ),
    )


def test_strip_auth_uri_removes_auth_uri_from_oauth2():
    auth_config = {
        "authScheme": {"type": "gcpAuthProviderScheme"},
        "exchangedAuthCredential": {
            "authType": "oauth2",
            "oauth2": {"authUri": "https://x", "auth_uri": "https://x", "nonce": "n"},
        },
        "credentialKey": "k",
    }

    stripped = AgentEngineClient._strip_auth_uri(auth_config)

    assert "authUri" not in stripped["exchangedAuthCredential"]["oauth2"]
    assert "auth_uri" not in stripped["exchangedAuthCredential"]["oauth2"]
    assert stripped["exchangedAuthCredential"]["oauth2"]["nonce"] == "n"
    # original left untouched
    assert auth_config["exchangedAuthCredential"]["oauth2"]["authUri"] == "https://x"


@pytest.mark.asyncio
async def test_resolve_iam_credential_settles_without_prompting_when_credential_valid():
    """First-call uri_consent_required that clears on the settle retry must not prompt."""
    client = AgentEngineClient.__new__(AgentEngineClient)
    # The settle retry returns a turn with no credential request → credential is valid.
    client.send_credential_response = AsyncMock(return_value=[])

    events, req = await client.resolve_iam_credential_events(
        user_id="592762352374644736",
        session_id="s1",
        events=[_interactive_credential_event()],
        context="async_task",
    )

    assert req is None
    client.send_credential_response.assert_awaited_once()
    # The credential response echoed back must NOT contain the authUri.
    sent_config = client.send_credential_response.await_args.kwargs["auth_config_dict"]
    assert "authUri" not in sent_config["exchangedAuthCredential"]["oauth2"]


@pytest.mark.asyncio
async def test_resolve_iam_credential_prompts_when_still_interactive_after_settle():
    """A genuinely missing credential keeps returning authUri → prompt after settle cap."""
    client = AgentEngineClient.__new__(AgentEngineClient)
    # Every settle retry still comes back interactive → genuine re-auth needed.
    client.send_credential_response = AsyncMock(
        return_value=[_interactive_credential_event()]
    )

    events, req = await client.resolve_iam_credential_events(
        user_id="592762352374644736",
        session_id="s1",
        events=[_interactive_credential_event()],
        context="async_task",
    )

    assert req is not None
    assert req.auth_uri.startswith("https://accounts.google.com")
    assert client.send_credential_response.await_count == AgentEngineClient._SETTLE_CAP

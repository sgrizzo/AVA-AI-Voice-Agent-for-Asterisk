"""
Azure OpenAI Realtime provider implementation.

This module integrates Azure OpenAI's GPT realtime models via WebSocket
into the Asterisk AI Voice Agent. It reuses the same Realtime API event
protocol as the upstream OpenAI provider but differs in:

- WebSocket URL construction (Azure resource hostname, /openai/v1/realtime or
  /openai/realtime path, model/deployment query parameters)
- Authentication (Bearer token from AZURE_OPENAI_API_KEY via ``api-key``
  header or query string, per Azure docs)
- GA vs preview API versioning (``model=`` vs ``deployment=`` + ``api-version=``)

Audio pipeline, event handling, tool calling, egress pacer, barge-in, and
greeting flow are identical to the OpenAI provider since Azure exposes the
same Realtime API surface.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
import uuid
import audioop
from typing import Any, Dict, Optional, List

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from structlog import get_logger
from prometheus_client import Gauge, Info

from .base import AIProviderInterface, ProviderCapabilities
from ..audio import (
    convert_pcm16le_to_target_format,
    mulaw_to_pcm16le,
    resample_audio,
)
from ..config import AzureOpenAIRealtimeProviderConfig

# Tool calling support
from src.tools.registry import tool_registry
from src.tools.adapters.openai import OpenAIToolAdapter

logger = get_logger(__name__)


def _log_provider_task_exception(task: asyncio.Task) -> None:
    """Done-callback: log exceptions from fire-and-forget provider tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Provider background task failed", task_name=task.get_name(), error=str(exc), exc_info=exc)


_COMMIT_INTERVAL_SEC = 0.2
_KEEPALIVE_INTERVAL_SEC = 15.0

_AZURE_ASSUMED_OUTPUT_RATE = Gauge(
    "ai_agent_azure_openai_assumed_output_sample_rate_hz",
    "Configured Azure OpenAI Realtime output sample rate per call",
)
_AZURE_PROVIDER_OUTPUT_RATE = Gauge(
    "ai_agent_azure_openai_provider_output_sample_rate_hz",
    "Provider-advertised Azure OpenAI Realtime output sample rate per call",
)
_AZURE_MEASURED_OUTPUT_RATE = Gauge(
    "ai_agent_azure_openai_measured_output_sample_rate_hz",
    "Measured Azure OpenAI Realtime output sample rate per call",
)
_AZURE_SESSION_AUDIO_INFO = Info(
    "ai_agent_azure_openai_session_audio",
    "Azure OpenAI Realtime session audio format assumptions and provider acknowledgements",
)


class AzureOpenAIRealtimeProvider(AIProviderInterface):
    """
    Azure OpenAI Realtime provider using server-side WebSocket transport.

    Lifecycle:
    1. start_session(call_id) -> establishes WebSocket session.
    2. send_audio(bytes) -> converts inbound AudioSocket frames to PCM16 24 kHz,
       base64-encodes, and streams via input_audio_buffer.
    3. Provider output deltas are decoded, resampled to AudioSocket format, and
       emitted as AgentAudio / AgentAudioDone events.
    4. stop_session() -> closes the WebSocket and cancels background tasks.
    """

    def __init__(
        self,
        config: AzureOpenAIRealtimeProviderConfig,
        on_event,
        gating_manager=None,
    ):
        super().__init__(on_event)
        self.config = config
        self.websocket: Optional[ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

        self._call_id: Optional[str] = None
        self._pending_response: bool = False
        self._current_response_id: Optional[str] = None
        self._greeting_response_id: Optional[str] = None
        self._greeting_completed: bool = False
        self._last_barge_in_emit_ts: float = 0.0
        self._farewell_response_id: Optional[str] = None
        self._hangup_after_response: bool = False
        self._farewell_timeout_task: Optional[asyncio.Task] = None
        self._greeting_vad_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._in_audio_burst: bool = False
        self._audio_seen_response_ids: set[str] = set()
        self._farewell_waiting_for_audio_done: bool = False
        self._response_audio_start_time: Optional[float] = None
        self._min_response_time_before_interrupt: float = 2.5
        self._first_output_chunk_logged: bool = False
        self._closing: bool = False
        self._closed: bool = False

        self._input_resample_state: Optional[tuple] = None
        self._output_resample_state: Optional[tuple] = None
        self._transcript_buffer: str = ""
        self._input_info_logged: bool = False
        self._allowed_tools: Optional[List[str]] = None

        self._turn_start_time: Optional[float] = None
        self._turn_first_audio_received: bool = False
        self._session_store = None
        self._pending_audio_provider_rate: bytearray = bytearray()

        self._gating_manager = gating_manager
        if self._gating_manager:
            logger.info("Audio gating enabled for Azure OpenAI Realtime (echo prevention)")
        self._last_commit_ts: float = 0.0
        self._audio_lock: asyncio.Lock = asyncio.Lock()
        self._provider_output_format: str = "pcm16"
        self._provider_reported_output_rate: Optional[int] = None
        self._output_meter_start_ts: float = 0.0
        self._output_meter_last_log_ts: float = 0.0
        self._output_meter_bytes: int = 0
        self._output_rate_warned: bool = False
        self._active_output_sample_rate_hz: Optional[float] = (
            float(self.config.output_sample_rate_hz) if getattr(self.config, "output_sample_rate_hz", None) else None
        )
        self._session_output_bytes_per_sample: int = 2
        self._session_output_encoding: str = "pcm16"
        self._outfmt_acknowledged: bool = False
        self._inferred_provider_encoding: Optional[str] = None
        self._inference_logged: bool = False
        self._egress_pacer_enabled: bool = bool(getattr(config, "egress_pacer_enabled", False))
        try:
            self._egress_pacer_warmup_ms: int = int(getattr(config, "egress_pacer_warmup_ms", 320))
        except Exception:
            self._egress_pacer_warmup_ms = 320
        self._outbuf: bytearray = bytearray()
        self._pacer_task: Optional[asyncio.Task] = None
        self._pacer_running: bool = False
        self._pacer_start_ts: float = 0.0
        self._pacer_underruns: int = 0
        self._pacer_lock: asyncio.Lock = asyncio.Lock()
        self._fallback_pcm24k_done: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None

        self.tool_adapter = OpenAIToolAdapter(tool_registry)
        logger.info("Azure OpenAI Realtime provider initialized with tool support")

        try:
            if self.config.input_encoding:
                self.config.input_encoding = self.config.input_encoding.strip()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Interface methods
    # ------------------------------------------------------------------ #

    def describe_alignment(
        self,
        *,
        audiosocket_format: str,
        streaming_encoding: str,
        streaming_sample_rate: int,
    ) -> List[str]:
        issues: List[str] = []
        inbound_enc = (self.config.input_encoding or "slin16").lower()
        inbound_rate = int(self.config.input_sample_rate_hz or 0)
        target_enc = (self.config.target_encoding or "ulaw").lower()
        target_rate = int(self.config.target_sample_rate_hz or 0)

        def _class(enc: str) -> str:
            e = (enc or "").lower()
            if e in ("ulaw", "mulaw", "g711_ulaw", "mu-law"):
                return "ulaw"
            if e in ("slin", "slin16", "linear16", "pcm16", "pcm"):
                return "pcm16"
            return e

        if inbound_enc in ("slin16", "linear16", "pcm16") and _class(audiosocket_format) == "ulaw":
            issues.append(
                "Azure OpenAI inbound encoding is PCM16 but AudioSocket format is μ-law; set audiosocket.format=slin16 "
                "or change azure_openai_realtime.input_encoding to ulaw."
            )
        if target_rate and target_rate != streaming_sample_rate:
            issues.append(
                f"Azure OpenAI target_sample_rate_hz={target_rate} but streaming sample rate is {streaming_sample_rate}."
            )

        provider_rate = int(self.config.provider_input_sample_rate_hz or 0)
        provider_enc = (getattr(self.config, "provider_input_encoding", None) or "linear16").lower()
        if provider_rate:
            if provider_enc in ("ulaw", "mulaw", "g711_ulaw", "mu-law", "alaw", "g711_alaw") and provider_rate != 8000:
                issues.append(
                    f"Azure OpenAI provider_input_sample_rate_hz={provider_rate}; G.711 should be 8000 Hz."
                )
            elif provider_enc in ("slin16", "linear16", "pcm16") and provider_rate not in (16000, 24000):
                issues.append(
                    f"Azure OpenAI provider_input_sample_rate_hz={provider_rate}; for PCM16 use 16000 or 24000 Hz."
                )

        return issues

    @property
    def supported_codecs(self):
        fmt = (self.config.target_encoding or "ulaw").lower()
        return [fmt]

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            input_encodings=["ulaw", "linear16"],
            input_sample_rates_hz=[8000, 16000],
            output_encodings=["mulaw", "pcm16"],
            output_sample_rates_hz=[8000, 24000],
            preferred_chunk_ms=20,
            can_negotiate=False,
            is_full_agent=True,
            has_native_vad=True,
            has_native_barge_in=True,
            requires_continuous_audio=True,
        )

    def parse_ack(self, event_data: Dict[str, Any]) -> Optional[ProviderCapabilities]:
        event_type = event_data.get('type')
        if event_type != 'session.updated':
            return None
        try:
            session = event_data.get('session', {})
            input_format = session.get('input_audio_format', 'pcm16')
            output_format = session.get('output_audio_format', 'pcm16')
            sample_rate = 24000
            format_map = {
                'pcm16': 'linear16',
                'g711_ulaw': 'mulaw',
                'g711_alaw': 'alaw',
            }
            input_enc = format_map.get(input_format, input_format)
            output_enc = format_map.get(output_format, output_format)
            logger.info(
                "Parsed Azure OpenAI session.updated ACK",
                call_id=self._call_id,
                input_format=input_format,
                output_format=output_format,
                sample_rate=sample_rate,
            )
            return ProviderCapabilities(
                input_encodings=[input_enc],
                input_sample_rates_hz=[sample_rate],
                output_encodings=[output_enc],
                output_sample_rates_hz=[sample_rate],
                preferred_chunk_ms=20,
                can_negotiate=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse Azure OpenAI session.updated event",
                call_id=self._call_id,
                error=str(exc),
            )
            return None

    async def start_session(self, call_id: str, context: Optional[Dict[str, Any]] = None):
        if not self.config.api_key:
            raise ValueError("Azure OpenAI Realtime provider requires AZURE_OPENAI_API_KEY")
        if not self.config.resource_name:
            raise ValueError("Azure OpenAI Realtime provider requires resource_name")

        await self.stop_session()
        self._call_id = call_id
        self._pending_response = False
        self._in_audio_burst = False
        self._first_output_chunk_logged = False
        self._input_resample_state = None
        self._output_resample_state = None
        self._transcript_buffer = ""
        self._closing = False
        self._closed = False

        self._session_ack_event = asyncio.Event()
        self._outfmt_acknowledged = False
        if context and "tools" in context:
            self._allowed_tools = list(context.get("tools") or [])
        else:
            self._allowed_tools = []

        self._reset_output_meter()

        url = self._build_ws_url()
        is_preview = getattr(self.config, 'api_version', 'ga').lower() == 'preview'
        headers = [
            ("api-key", self.config.api_key),
        ]

        logger.info(
            "Connecting to Azure OpenAI Realtime",
            url=url,
            call_id=call_id,
            api_version="preview" if is_preview else "ga",
            resource=self.config.resource_name,
        )
        try:
            self.websocket = await websockets.connect(url, additional_headers=headers)
        except Exception:
            logger.error("Failed to connect to Azure OpenAI Realtime", call_id=call_id, exc_info=True)
            raise

        logger.debug("Waiting for session.created from Azure OpenAI...", call_id=call_id)
        try:
            first_message = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=5.0
            )
            first_event = json.loads(first_message)
            if first_event.get("type") == "session.created":
                session_data = first_event.get("session", {})
                logger.info(
                    "Received session.created - session ready",
                    call_id=call_id,
                    session_id=session_data.get("id"),
                    model=session_data.get("model"),
                )
            else:
                logger.warning(
                    "Unexpected first event (expected session.created)",
                    call_id=call_id,
                    event_type=first_event.get("type")
                )
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session.created", call_id=call_id)
            raise RuntimeError("Azure OpenAI did not send session.created within 5s")
        except Exception as exc:
            logger.error(
                "Error receiving session.created",
                call_id=call_id,
                error=str(exc),
                exc_info=True
            )
            raise

        await self._send_session_update()
        self._log_session_assumptions()

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        try:
            logger.debug("Waiting for Azure OpenAI session.updated ACK before greeting...", call_id=call_id)
            await asyncio.wait_for(self._session_ack_event.wait(), timeout=2.0)
            logger.info(
                "Azure OpenAI session.updated ACK received - session configured",
                call_id=call_id,
                acknowledged=self._outfmt_acknowledged,
                output_format=self._provider_output_format,
                sample_rate=self._active_output_sample_rate_hz,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Azure OpenAI session.updated ACK timeout - proceeding anyway",
                call_id=call_id,
                note="Session may not be fully configured"
            )

        try:
            if (self.config.greeting or "").strip():
                logger.info("Sending explicit greeting (after session ACK)", call_id=call_id)
                await self._send_explicit_greeting()
            else:
                await self._ensure_response_request()
        except Exception:
            logger.debug("Initial response.create request failed", call_id=call_id, exc_info=True)

        try:
            async with self._pacer_lock:
                self._outbuf.clear()
            self._pacer_running = False
            self._pacer_start_ts = 0.0
            self._pacer_underruns = 0
            self._fallback_pcm24k_done = False
            if self._pacer_task and not self._pacer_task.done():
                self._pacer_task.cancel()
        except Exception:
            logger.debug("Failed to reset pacer state on session start", exc_info=True)

        logger.info("Azure OpenAI Realtime session established", call_id=call_id)

    async def send_audio(self, audio_chunk: bytes, sample_rate: int = None, encoding: str = None):
        if not audio_chunk:
            return
        if not self.websocket or self.websocket.state.name != "OPEN":
            logger.debug("Dropping inbound audio: websocket not ready", call_id=self._call_id)
            return

        try:
            if not self._input_info_logged:
                try:
                    logger.info(
                        "Azure OpenAI input config",
                        call_id=self._call_id,
                        input_encoding=self.config.input_encoding,
                        input_sample_rate_hz=self.config.input_sample_rate_hz,
                        provider_input_sample_rate_hz=self.config.provider_input_sample_rate_hz,
                        engine_provided_encoding=encoding,
                        engine_provided_sample_rate=sample_rate,
                    )
                    self._input_info_logged = True
                except Exception:
                    pass

            if encoding and sample_rate:
                if encoding.lower().strip() in ("linear16", "pcm16", "slin16"):
                    pcm16 = audio_chunk
                    provider_rate = sample_rate
                else:
                    logger.warning(
                        "Azure OpenAI Realtime: unexpected encoding from engine, converting",
                        call_id=self._call_id,
                        encoding=encoding,
                        sample_rate=sample_rate
                    )
                    pcm16 = self._convert_inbound_audio(audio_chunk)
                    provider_rate = int(getattr(self.config, "provider_input_sample_rate_hz", 0) or 24000)
            else:
                pcm16 = self._convert_inbound_audio(audio_chunk)
                provider_rate = int(getattr(self.config, "provider_input_sample_rate_hz", 0) or 24000)

            if not pcm16:
                return

            try:
                if self._in_audio_burst and self._pacer_underruns == 0:
                    return
            except Exception:
                pass

            await self._send_audio_to_openai(pcm16)

        except ConnectionClosedError:
            logger.warning("Azure OpenAI Realtime socket closed while sending audio", call_id=self._call_id)
            await self._reconnect_with_backoff()
        except Exception:
            logger.error("Failed to send audio to Azure OpenAI Realtime", call_id=self._call_id, exc_info=True)

    async def cancel_response(self):
        if not self.websocket or self.websocket.state.name != "OPEN":
            return
        if not self._pending_response:
            logger.debug("No pending response to cancel", call_id=self._call_id)
            return
        try:
            cancel_payload = {
                "type": "response.cancel",
                "event_id": f"cancel-{uuid.uuid4()}",
            }
            await self._send_json(cancel_payload)
            logger.info("Sent response.cancel to Azure OpenAI (barge-in)", call_id=self._call_id)
            self._pending_response = False
        except Exception:
            logger.error("Failed to send response.cancel", call_id=self._call_id, exc_info=True)

    async def _handle_function_call(self, event_data: Dict[str, Any]):
        try:
            context = {
                'call_id': self._call_id,
                'caller_channel_id': getattr(self, '_caller_channel_id', None),
                'bridge_id': getattr(self, '_bridge_id', None),
                'called_number': getattr(self, '_called_number', None),
                'session_store': getattr(self, '_session_store', None),
                'ari_client': getattr(self, '_ari_client', None),
                'config': getattr(self, '_full_config', None),
                'allowed_tools': self._allowed_tools,
                'websocket': self.websocket,
                'is_ga': self._is_ga,
            }
            result = await self.tool_adapter.handle_tool_call_event(event_data, context)

            item = event_data.get("item", {})
            function_name = item.get("name")
            if function_name == "hangup_call" and result:
                if result.get("will_hangup"):
                    self._hangup_after_response = True
                    logger.info(
                        "Hangup tool executed - next response will trigger hangup",
                        call_id=self._call_id,
                        function_name=function_name,
                        farewell=result.get("message")
                    )

            await self.tool_adapter.send_tool_result(result, context)

            if function_name == "hangup_call" and result and result.get("will_hangup"):
                farewell_text = str(result.get("message") or "").strip()
                if farewell_text and self.websocket and self.websocket.state.name == "OPEN":
                    try:
                        await self._send_json(
                            {
                                "type": "session.update",
                                "event_id": f"sess-tools-none-{uuid.uuid4()}",
                                "session": self._ga_session_type({"tool_choice": "none"}),
                            }
                        )
                    except Exception:
                        logger.debug(
                            "Failed to disable tool_choice for farewell response",
                            call_id=self._call_id,
                            exc_info=True,
                        )

                    try:
                        farewell_response: Dict[str, Any] = {
                            "instructions": (
                                "Say the following sentence to the user exactly, then stop. "
                                f"Do not call any tools: {farewell_text}"
                            ),
                        }
                        if not self._is_ga:
                            farewell_response["modalities"] = self._response_modalities
                            farewell_response["input"] = []
                        await self._send_json(
                            {
                                "type": "response.create",
                                "event_id": f"resp-farewell-{uuid.uuid4()}",
                                "response": farewell_response,
                            }
                        )
                        self._pending_response = True
                        logger.info(
                            "Farewell response.create sent (tools disabled)",
                            call_id=self._call_id,
                            farewell_preview=farewell_text[:80],
                        )
                    except Exception:
                        logger.debug("Failed to send farewell response.create", call_id=self._call_id, exc_info=True)

            try:
                session_store = getattr(self, '_session_store', None)
                if session_store and self._call_id and function_name:
                    from datetime import datetime
                    session = await session_store.get_by_call_id(self._call_id)
                    if session:
                        tool_record = {
                            "name": function_name,
                            "params": item.get("arguments", {}),
                            "result": result.get("status", "unknown") if isinstance(result, dict) else "success",
                            "message": result.get("message", "") if isinstance(result, dict) else str(result),
                            "timestamp": datetime.now().isoformat(),
                            "duration_ms": 0,
                        }
                        if not hasattr(session, 'tool_calls') or session.tool_calls is None:
                            session.tool_calls = []
                        session.tool_calls.append(tool_record)
                        await session_store.upsert_call(session)
                        logger.debug("Tool call logged to session", call_id=self._call_id, tool=function_name)
            except Exception as log_err:
                logger.debug(f"Failed to log tool call to session: {log_err}", call_id=self._call_id)

        except Exception as e:
            logger.error(
                "Function call handling failed",
                call_id=self._call_id,
                error=str(e),
                exc_info=True
            )
            try:
                item = event_data.get("item", {})
                call_id_field = item.get("call_id")
                if call_id_field:
                    error_response = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id_field,
                            "output": json.dumps({
                                "status": "error",
                                "message": f"Tool execution failed: {str(e)}",
                                "error": str(e)
                            })
                        }
                    }
                    if self.websocket and self.websocket.state.name == "OPEN":
                        await self._send_json(error_response)
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")

    async def stop_session(self):
        if self._closing or self._closed:
            return
        self._closing = True
        try:
            if self._receive_task:
                self._receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receive_task
            if self._keepalive_task:
                self._keepalive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._keepalive_task
            self._cancel_farewell_timeout()
            if self._greeting_vad_task and not self._greeting_vad_task.done():
                self._greeting_vad_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._greeting_vad_task
            bg_tasks = list(self._background_tasks)
            for task in bg_tasks:
                if not task.done():
                    task.cancel()
            if bg_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(*bg_tasks, return_exceptions=True)
            if self.websocket and self.websocket.state.name == "OPEN":
                await self.websocket.close()
            await self._emit_audio_done()
        finally:
            try:
                self._pacer_running = False
                if self._pacer_task:
                    self._pacer_task.cancel()
            except Exception:
                pass
            previous_call_id = self._call_id
            self._receive_task = None
            self._keepalive_task = None
            self._greeting_vad_task = None
            self._background_tasks.clear()
            self.websocket = None
            self._call_id = None
            self._closing = False
            self._closed = True
            self._pending_response = False
            self._in_audio_burst = False
            self._input_resample_state = None
            self._output_resample_state = None
            self._transcript_buffer = ""
            logger.info("Azure OpenAI Realtime session stopped")
            self._clear_metrics(previous_call_id)

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "name": "AzureOpenAIRealtimeProvider",
            "type": "cloud",
            "deployment": self.config.deployment,
            "voice": self.config.voice,
            "supported_codecs": self.supported_codecs,
        }

    def is_ready(self) -> bool:
        return bool(self.config.api_key) and bool(self.config.resource_name)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _is_ga(self) -> bool:
        """True when using the GA API (model= query param)."""
        return getattr(self.config, 'api_version', 'ga').lower() != 'preview'

    def _ga_session_type(self, session: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_ga and "type" not in session:
            session["type"] = "realtime"
        return session

    @property
    def _modalities_key(self) -> str:
        return "output_modalities" if self._is_ga else "modalities"

    @property
    def _response_modalities(self) -> list:
        if self._is_ga:
            return ["audio"]
        return [m for m in (self.config.response_modalities or []) if m in ("audio", "text")] or ["audio"]

    def _build_ws_url(self) -> str:
        """Build the Azure OpenAI Realtime WebSocket URL.

        GA:     wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>
        Preview: wss://<resource>.openai.azure.com/openai/realtime?api-version=<ver>&deployment=<deployment>
        """
        resource = (self.config.resource_name or "").strip().rstrip(".")
        is_preview = not self._is_ga

        if is_preview:
            base = f"wss://{resource}.openai.azure.com/openai/realtime"
            api_ver = (self.config.preview_api_version or "2025-04-01-preview").strip()
            deployment = (self.config.deployment or "").strip()
            return f"{base}?api-version={api_ver}&deployment={deployment}"
        else:
            base = f"wss://{resource}.openai.azure.com/openai/v1/realtime"
            deployment = (self.config.deployment or "").strip()
            return f"{base}?model={deployment}"

    async def _send_session_update(self):
        output_modalities = [m for m in (self.config.response_modalities or []) if m in ("audio", "text")]
        if not output_modalities:
            output_modalities = ["audio"]

        output_enc = (self.config.output_encoding or "linear16").lower()
        input_enc = (getattr(self.config, "provider_input_encoding", None) or "linear16").lower()

        if self._is_ga:
            def _ga_audio_fmt(enc: str) -> str:
                enc = enc.lower()
                if enc in ("ulaw", "mulaw", "g711_ulaw", "mu-law"):
                    return "audio/pcmu"
                elif enc in ("alaw", "g711_alaw"):
                    return "audio/pcma"
                return "audio/pcm"

            in_fmt = _ga_audio_fmt(input_enc)
            out_fmt = _ga_audio_fmt(output_enc)
            in_rate = 8000 if in_fmt in ("audio/pcmu", "audio/pcma") else 24000
            out_rate = 8000 if out_fmt in ("audio/pcmu", "audio/pcma") else 24000

            audio_input: Dict[str, Any] = {
                "format": {"type": in_fmt, "rate": in_rate},
                "transcription": {"model": "whisper-1"},
            }
            td_config: Dict[str, Any] = {
                "type": "server_vad",
                "create_response": True,
                "interrupt_response": True,
            }
            if getattr(self.config, "turn_detection", None):
                try:
                    td = self.config.turn_detection
                    td_config.update({
                        "type": td.type,
                        "silence_duration_ms": td.silence_duration_ms,
                        "threshold": td.threshold,
                        "prefix_padding_ms": td.prefix_padding_ms,
                    })
                except Exception:
                    logger.debug("Failed to build turn_detection for GA", call_id=self._call_id, exc_info=True)
            audio_input["turn_detection"] = td_config

            session: Dict[str, Any] = {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": audio_input,
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.config.voice,
                    },
                },
            }
        else:
            def _beta_audio_fmt(enc: str) -> str:
                enc = enc.lower()
                if enc in ("ulaw", "mulaw", "g711_ulaw", "mu-law"):
                    return "g711_ulaw"
                elif enc in ("alaw", "g711_alaw"):
                    return "g711_alaw"
                return "pcm16"

            in_fmt = _beta_audio_fmt(input_enc)
            out_fmt = _beta_audio_fmt(output_enc)

            session: Dict[str, Any] = {
                "modalities": output_modalities,
                "input_audio_format": in_fmt,
                "output_audio_format": out_fmt,
                "voice": self.config.voice,
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
            }
            if getattr(self.config, "turn_detection", None):
                try:
                    td = self.config.turn_detection
                    session["turn_detection"] = {
                        "type": td.type,
                        "silence_duration_ms": td.silence_duration_ms,
                        "threshold": td.threshold,
                        "prefix_padding_ms": td.prefix_padding_ms,
                    }
                except Exception:
                    logger.debug("Failed to include turn_detection in session.update", call_id=self._call_id, exc_info=True)

        self._ga_session_type(session)

        if self.config.instructions:
            audio_forcing_prefix = (
                "IMPORTANT: You are a voice-based AI assistant. "
                "ALWAYS respond with AUDIO speech, never text-only. "
                "Every response MUST include spoken audio output. "
            )
            session["instructions"] = audio_forcing_prefix + self.config.instructions
        else:
            session["instructions"] = (
                "IMPORTANT: You are a voice-based AI assistant. "
                "ALWAYS respond with AUDIO speech, never text-only. "
                "Every response MUST include spoken audio output. "
            )

        try:
            tools = self.tool_adapter.get_tools_config(list(self._allowed_tools or []))
            if tools:
                session["tools"] = tools
                session["tool_choice"] = "auto"
                logger.info(
                    f"Azure OpenAI session configured with {len(tools)} tools",
                    call_id=self._call_id,
                )
        except Exception as e:
            logger.warning(
                f"Failed to add tools to Azure OpenAI session: {e}",
                call_id=self._call_id,
                exc_info=True,
            )

        payload: Dict[str, Any] = {
            "type": "session.update",
            "event_id": f"sess-{uuid.uuid4()}",
            "session": session,
        }

        logger.info(
            "Azure OpenAI session.update payload",
            call_id=self._call_id,
            output_audio_format=session.get("output_audio_format"),
            input_audio_format=session.get("input_audio_format"),
            modalities=session.get("modalities"),
        )

        await self._send_json(payload)

    async def _send_explicit_greeting(self):
        greeting = (self.config.greeting or "").strip()
        if not greeting or not self.websocket or self.websocket.state.name != "OPEN":
            return

        logger.info("Disabling turn_detection for greeting playback", call_id=self._call_id)

        if not self._is_ga:
            disable_vad_payload: Dict[str, Any] = {
                "type": "session.update",
                "event_id": f"sess-disable-vad-{uuid.uuid4()}",
                "session": self._ga_session_type({
                    "turn_detection": None
                })
            }
            await self._send_json(disable_vad_payload)

        await asyncio.sleep(0.1)

        if self._is_ga:
            response_payload: Dict[str, Any] = {
                "type": "response.create",
                "event_id": f"resp-{uuid.uuid4()}",
                "response": {
                    "instructions": f"Please greet the user with the following: {greeting}",
                },
            }
        else:
            response_payload: Dict[str, Any] = {
                "type": "response.create",
                "event_id": f"resp-{uuid.uuid4()}",
                "response": {
                    "modalities": self._response_modalities,
                    "instructions": f"Please greet the user with the following: {greeting}",
                    "input": [],
                },
            }

        logger.info(
            "Sending greeting response.create",
            call_id=self._call_id,
            greeting_preview=greeting[:50] + "..." if len(greeting) > 50 else greeting,
        )

        await self._send_json(response_payload)
        self._pending_response = True

        logger.info("Greeting sent - will re-enable VAD after completion", call_id=self._call_id)

        if self._greeting_vad_task and not self._greeting_vad_task.done():
            self._greeting_vad_task.cancel()
        self._greeting_vad_task = asyncio.create_task(self._greeting_vad_fallback())
        self._greeting_vad_task.add_done_callback(_log_provider_task_exception)
        self._background_tasks.add(self._greeting_vad_task)
        self._greeting_vad_task.add_done_callback(self._background_tasks.discard)

    async def _greeting_vad_fallback(self):
        try:
            await asyncio.sleep(5.0)
            if not self._greeting_completed:
                logger.warning(
                    "VAD fallback - greeting completion not detected, re-enabling VAD",
                    call_id=self._call_id
                )
                self._greeting_completed = True
                await self._re_enable_vad()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("VAD fallback failed", call_id=self._call_id, exc_info=True)

    async def _re_enable_vad(self):
        if not self.websocket or self.websocket.state.name != "OPEN":
            return
        turn_detection_config = None
        if getattr(self.config, "turn_detection", None):
            try:
                td = self.config.turn_detection
                turn_detection_config = {
                    "type": td.type,
                    "silence_duration_ms": td.silence_duration_ms,
                    "threshold": td.threshold,
                    "prefix_padding_ms": td.prefix_padding_ms,
                }
            except Exception:
                logger.debug("Failed to build turn_detection config, using defaults",
                             call_id=self._call_id, exc_info=True)

        if self._is_ga:
            logger.info(
                "GA mode: skipping turn_detection re-enable (server manages VAD)",
                call_id=self._call_id,
            )
        else:
            session_update = {}
            if turn_detection_config:
                session_update["turn_detection"] = turn_detection_config
            else:
                session_update["turn_detection"] = {"type": "server_vad"}

            enable_vad_payload: Dict[str, Any] = {
                "type": "session.update",
                "event_id": f"sess-enable-vad-{uuid.uuid4()}",
                "session": self._ga_session_type(session_update)
            }
            await self._send_json(enable_vad_payload)
            logger.info(
                "Turn_detection re-enabled after greeting",
                call_id=self._call_id,
                config=turn_detection_config if turn_detection_config else "defaults"
            )

    async def _ensure_response_request(self):
        if self._pending_response or not self.websocket or self.websocket.state.name != "OPEN":
            return
        resp_obj: Dict[str, Any] = {}
        if not self._is_ga:
            resp_obj[self._modalities_key] = self._response_modalities
        resp_obj["metadata"] = {"call_id": self._call_id}
        if self.config.instructions:
            resp_obj["instructions"] = self.config.instructions

        response_payload: Dict[str, Any] = {
            "type": "response.create",
            "event_id": f"resp-{uuid.uuid4()}",
            "response": resp_obj,
        }
        await self._send_json(response_payload)
        self._pending_response = True

    def _start_farewell_timeout(self):
        self._cancel_farewell_timeout()
        self._farewell_timeout_task = asyncio.create_task(self._farewell_timeout_handler())
        logger.debug("Farewell timeout started (5s fallback)", call_id=self._call_id)

    def _cancel_farewell_timeout(self):
        if self._farewell_timeout_task and not self._farewell_timeout_task.done():
            self._farewell_timeout_task.cancel()
            logger.debug("Farewell timeout cancelled", call_id=self._call_id)
            self._farewell_timeout_task = None

    async def _farewell_timeout_handler(self):
        try:
            await asyncio.sleep(5.0)
            logger.warning(
                "Farewell timeout expired - no audio within 5s, triggering hangup",
                call_id=self._call_id
            )
            try:
                if self.on_event:
                    await self.on_event({
                        "type": "HangupReady",
                        "call_id": self._call_id,
                        "reason": "farewell_timeout",
                        "had_audio": False
                    })
            except Exception as e:
                logger.error(
                    "Failed to emit HangupReady event from timeout",
                    call_id=self._call_id,
                    error=str(e),
                    exc_info=True
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Farewell timeout handler error",
                call_id=self._call_id,
                error=str(e),
                exc_info=True
            )

    async def _send_json(self, payload: Dict[str, Any]):
        if not self.websocket or self.websocket.state.name != "OPEN":
            return
        try:
            ptype = payload.get("type")
            if ptype and not ptype.startswith("input_audio_buffer."):
                logger.debug("Azure OpenAI send", call_id=self._call_id, type=ptype)
        except Exception:
            pass
        message = json.dumps(payload)
        async with self._send_lock:
            await self.websocket.send(message)

    async def _cancel_response(self, response_id: str):
        if not self.websocket or self.websocket.state.name != "OPEN":
            return
        try:
            cancel_payload = {
                "type": "response.cancel",
                "event_id": f"cancel-{uuid.uuid4()}",
                "response_id": response_id
            }
            await self._send_json(cancel_payload)
            logger.debug("Sent response.cancel to Azure OpenAI", call_id=self._call_id, response_id=response_id)
            try:
                await self._emit_audio_done()
            except Exception:
                logger.debug("Failed emitting AgentAudioDone during barge-in cancel", call_id=self._call_id, exc_info=True)
            try:
                async with self._pacer_lock:
                    self._outbuf.clear()
            except Exception:
                logger.debug("Failed clearing pacer buffer during barge-in cancel", call_id=self._call_id, exc_info=True)
            try:
                self._pacer_running = False
                if self._pacer_task and not self._pacer_task.done():
                    self._pacer_task.cancel()
            except Exception:
                logger.debug("Failed stopping pacer during barge-in cancel", call_id=self._call_id, exc_info=True)
        except Exception:
            logger.error(
                "Failed to cancel Azure OpenAI response",
                call_id=self._call_id,
                response_id=response_id,
                exc_info=True
            )

    async def _emit_provider_barge_in(self, *, event_type: str) -> None:
        try:
            now = time.time()
            if now - float(self._last_barge_in_emit_ts or 0.0) < 0.25:
                return
            self._last_barge_in_emit_ts = now
            await self.on_event(
                {
                    "type": "ProviderBargeIn",
                    "call_id": self._call_id,
                    "provider": "azure_openai_realtime",
                    "event": event_type,
                }
            )
        except Exception:
            logger.debug("Failed to emit ProviderBargeIn", call_id=self._call_id, exc_info=True)

    async def _send_audio_to_openai(self, pcm16: bytes):
        vad_enabled = getattr(self.config, "turn_detection", None) is not None
        if vad_enabled:
            try:
                audio_b64 = base64.b64encode(pcm16).decode("ascii")
                await self._send_json({"type": "input_audio_buffer.append", "audio": audio_b64})
            except Exception:
                logger.error("Failed to append input audio buffer (VAD)", call_id=self._call_id, exc_info=True)
        else:
            async with self._audio_lock:
                self._pending_audio_provider_rate.extend(pcm16)
                bytes_per_ms = int(self.config.provider_input_sample_rate_hz * 2 / 1000)
                commit_threshold_ms = 160
                commit_threshold_bytes = bytes_per_ms * commit_threshold_ms

                if len(self._pending_audio_provider_rate) >= commit_threshold_bytes:
                    chunk = bytes(self._pending_audio_provider_rate)
                    self._pending_audio_provider_rate.clear()
                    audio_b64 = base64.b64encode(chunk).decode("ascii")
                    try:
                        await self._send_json({"type": "input_audio_buffer.append", "audio": audio_b64})
                        self._last_commit_ts = time.monotonic()
                        logger.info(
                            "Azure OpenAI appended input audio (auto-commit on speech_stopped)",
                            call_id=self._call_id,
                            ms=len(chunk) // bytes_per_ms,
                            bytes=len(chunk),
                        )
                    except Exception:
                        logger.error("Failed to append input audio buffer", call_id=self._call_id, exc_info=True)

    def _convert_inbound_audio(self, audio_chunk: bytes) -> Optional[bytes]:
        fmt_raw = getattr(self.config, "input_encoding", None) or "slin16"
        fmt = fmt_raw.strip().lower()
        try:
            self.config.input_encoding = fmt
        except Exception:
            pass

        valid_encodings = {
            "ulaw", "mulaw", "g711_ulaw", "mu-law",
            "slin16", "linear16", "pcm16",
        }
        if fmt not in valid_encodings:
            logger.warning("Unsupported input encoding for Azure OpenAI Realtime", encoding=fmt_raw)
            fmt = "slin16"
            try:
                self.config.input_encoding = fmt
            except Exception:
                pass

        chunk_len = len(audio_chunk)
        if chunk_len == 160:
            actual_format = "ulaw"
            inferred_rate = 8000
        elif chunk_len == 320:
            actual_format = "pcm16"
            inferred_rate = 8000
        elif chunk_len == 640:
            actual_format = "pcm16"
            inferred_rate = 16000
        else:
            actual_format = "pcm16" if fmt in ("slin16", "linear16", "pcm16") else "ulaw"
            inferred_rate = int(getattr(self.config, "input_sample_rate_hz", 0) or 0) or 8000

        if actual_format == "ulaw":
            source_rate = 8000
        else:
            declared_rate = int(getattr(self.config, "input_sample_rate_hz", 0) or 0)
            source_rate = declared_rate or inferred_rate or 8000

        if actual_format == "ulaw":
            pcm_src = mulaw_to_pcm16le(audio_chunk)
        else:
            pcm_src = audio_chunk

        provider_rate = int(getattr(self.config, "provider_input_sample_rate_hz", 0) or 0)
        if provider_rate and provider_rate != source_rate:
            pcm_provider_rate, self._input_resample_state = resample_audio(
                pcm_src, source_rate, provider_rate, state=self._input_resample_state,
            )
            return pcm_provider_rate

        self._input_resample_state = None
        return pcm_src

    async def _receive_loop(self):
        assert self.websocket is not None
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode Azure OpenAI Realtime payload", payload_preview=message[:64])
                    continue
                await self._handle_event(event)
        except asyncio.CancelledError:
            pass
        except (ConnectionClosedError, ConnectionClosedOK):
            logger.info("Azure OpenAI Realtime connection closed", call_id=self._call_id)
        except Exception:
            logger.error("Azure OpenAI Realtime receive loop error", call_id=self._call_id, exc_info=True)
        finally:
            await self._emit_audio_done()
            self._pending_response = False
            try:
                if not self._closing and not self._closed and self._call_id:
                    if not self._reconnect_task or self._reconnect_task.done():
                        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())
            except Exception:
                logger.debug("Failed to schedule Azure OpenAI reconnect", call_id=self._call_id, exc_info=True)

    async def _handle_event(self, event: Dict[str, Any]):
        event_type = event.get("type")

        if event_type == "error":
            error_code = event.get("error", {}).get("code")
            if error_code == "response_cancel_not_active":
                logger.debug(
                    "Response already completed (cannot cancel)",
                    call_id=self._call_id,
                    response_id=self._current_response_id
                )
                return
            logger.error("Azure OpenAI Realtime error event", call_id=self._call_id, error_event=event)
            return

        if event_type == "response.created":
            response = event.get("response", {})
            response_id = response.get("id")
            if response_id:
                self._current_response_id = response_id
                try:
                    self._audio_seen_response_ids.discard(response_id)
                except Exception:
                    pass
                if not self._greeting_completed and self._greeting_response_id is None:
                    self._greeting_response_id = response_id
                    logger.info(
                        "Greeting response created - protected from barge-in",
                        call_id=self._call_id,
                        response_id=response_id
                    )
                elif self._hangup_after_response:
                    self._farewell_response_id = response_id
                    logger.info(
                        "Farewell response created - will trigger hangup on completion",
                        call_id=self._call_id,
                        response_id=response_id
                    )
                    self._start_farewell_timeout()
                    self._farewell_waiting_for_audio_done = True
            return

        if event_type == "response.delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "output_audio.delta":
                audio_b64 = delta.get("audio")
                if audio_b64:
                    await self._handle_output_audio(audio_b64)
            elif delta_type == "output_audio.done":
                await self._emit_audio_done()
            elif delta_type == "output_text.delta":
                text = delta.get("text")
                if text:
                    await self._emit_transcript(text, is_final=False)
            elif delta_type == "output_text.done":
                if self._transcript_buffer:
                    await self._emit_transcript("", is_final=True)
            return

        if event_type == "response.output_audio.delta":
            delta = event.get("delta")
            audio_b64 = (
                event.get("audio")
                or (delta if isinstance(delta, str) else (delta or {}).get("audio"))
            )
            if audio_b64:
                await self._handle_output_audio(audio_b64)
            else:
                logger.debug("Missing audio in response.output_audio.delta", call_id=self._call_id)
            return

        if event_type == "response.output_audio.done":
            await self._emit_audio_done()
            return

        if event_type == "response.audio.delta":
            audio_b64 = event.get("delta")
            if audio_b64:
                if not self._in_audio_burst:
                    self._in_audio_burst = True
                    self._response_audio_start_time = time.time()
                await self._handle_output_audio(audio_b64)
            else:
                logger.debug("Missing audio in response.audio.delta", call_id=self._call_id)
            return

        if event_type == "response.audio.done":
            if self._in_audio_burst:
                self._in_audio_burst = False
            await self._emit_audio_done()
            return

        if event_type == "response.audio_transcript.delta":
            delta = event.get("delta")
            text = event.get("text")
            if text is None:
                if isinstance(delta, dict):
                    text = delta.get("text")
                elif isinstance(delta, str):
                    text = delta
            if text:
                await self._emit_transcript(text, is_final=False)
            return

        if event_type == "response.audio_transcript.done":
            if self._transcript_buffer:
                await self._track_conversation("assistant", self._transcript_buffer)
                await self._emit_transcript("", is_final=True)
            return

        if event_type in ("response.completed", "response.error", "response.cancelled", "response.done"):
            current_response_id = self._current_response_id
            had_audio_for_response = bool(
                current_response_id and current_response_id in self._audio_seen_response_ids
            )
            self._response_audio_start_time = None
            await self._emit_audio_done()

            if event_type in ("response.completed", "response.done") and not had_audio_for_response:
                response_data = event.get("response", {})
                output_items = response_data.get("output", [])
                status = response_data.get("status")
                status_details = response_data.get("status_details")
                logger.warning(
                    "Response completed without audio output - investigating",
                    call_id=self._call_id,
                    event_type=event_type,
                    response_status=status,
                    status_details=status_details,
                    output_items_count=len(output_items),
                    output_types=[item.get("type") for item in output_items] if output_items else [],
                )

            if event_type == "response.error":
                logger.error("Azure OpenAI Realtime response error", call_id=self._call_id, error=event.get("error"))
            elif event_type == "response.cancelled":
                logger.info("Azure OpenAI response cancelled (barge-in)", call_id=self._call_id, response_id=self._current_response_id)

            if (self._current_response_id == self._greeting_response_id and
                not self._greeting_completed and
                event_type in ("response.completed", "response.done")):
                self._greeting_completed = True
                logger.info(
                    "Greeting response completed - re-enabling turn_detection",
                    call_id=self._call_id,
                    had_audio=had_audio_for_response
                )
                await self._re_enable_vad()
                try:
                    if self.on_event and self._call_id:
                        await self.on_event(
                            {
                                "type": "ClearTtsGating",
                                "call_id": self._call_id,
                                "reason": "greeting_completed",
                            }
                        )
                except Exception:
                    logger.debug("Failed to emit ClearTtsGating event", call_id=self._call_id, exc_info=True)

            if (self._farewell_response_id is not None and
                self._current_response_id == self._farewell_response_id and
                event_type in ("response.completed", "response.done")):
                self._cancel_farewell_timeout()
                if had_audio_for_response:
                    logger.info(
                        "Farewell response completed with audio - waiting for output_audio.done",
                        call_id=self._call_id,
                        response_id=self._current_response_id
                    )
                else:
                    logger.warning(
                        "Farewell response completed WITHOUT audio - triggering immediate hangup",
                        call_id=self._call_id,
                        response_id=self._current_response_id
                    )
                    try:
                        if self.on_event:
                            await self.on_event({
                                "type": "HangupReady",
                                "call_id": self._call_id,
                                "reason": "farewell_no_audio",
                                "had_audio": False
                            })
                    except Exception as e:
                        logger.error(
                            "Failed to emit HangupReady event for no-audio farewell",
                            call_id=self._call_id,
                            error=str(e),
                            exc_info=True,
                        )
                if not had_audio_for_response:
                    self._farewell_waiting_for_audio_done = False
                    self._farewell_response_id = None
                self._hangup_after_response = False

            try:
                if current_response_id:
                    self._audio_seen_response_ids.discard(current_response_id)
            except Exception:
                pass

            self._pending_response = False
            self._current_response_id = None
            if self._transcript_buffer:
                await self._emit_transcript("", is_final=True)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(
                    "User transcript received",
                    call_id=self._call_id,
                    transcript_preview=transcript[:100] if len(transcript) > 100 else transcript
                )
                await self._emit_transcript(transcript, is_final=True)
                await self._track_conversation("user", transcript)
            return

        if event_type == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.warning(
                "User transcription failed",
                call_id=self._call_id,
                error_type=error.get("type"),
                error_message=error.get("message"),
            )
            return

        if event_type == "response.output_text.delta":
            delta = event.get("delta") or {}
            text = delta.get("text")
            if text:
                await self._emit_transcript(text, is_final=False)
            return

        if event_type and event_type.startswith("input_audio_buffer"):
            if event_type == "input_audio_buffer.speech_stopped":
                self._turn_start_time = time.time()
                self._turn_first_audio_received = False
                logger.debug("Turn latency timer started (speech_stopped)", call_id=self._call_id)
            elif event_type == "input_audio_buffer.speech_started" and self._current_response_id:
                if self._current_response_id == self._greeting_response_id and not self._greeting_completed:
                    logger.info(
                        "Barge-in blocked - protecting greeting response",
                        call_id=self._call_id,
                        response_id=self._current_response_id
                    )
                elif self._response_audio_start_time:
                    elapsed = time.time() - self._response_audio_start_time
                    if elapsed < self._min_response_time_before_interrupt:
                        logger.info(
                            "Barge-in blocked - response too young",
                            call_id=self._call_id,
                            response_id=self._current_response_id,
                            elapsed_seconds=round(elapsed, 2),
                            min_required=self._min_response_time_before_interrupt
                        )
                    else:
                        logger.info(
                            "User interruption detected, cancelling response",
                            call_id=self._call_id,
                            response_id=self._current_response_id,
                            elapsed_seconds=round(elapsed, 2)
                        )
                        await self._cancel_response(self._current_response_id)
                        await self._emit_provider_barge_in(event_type=event_type)
                else:
                    logger.info(
                        "User interruption detected (no audio), cancelling response",
                        call_id=self._call_id,
                        response_id=self._current_response_id
                    )
                    await self._cancel_response(self._current_response_id)
                    await self._emit_provider_barge_in(event_type=event_type)
            else:
                if event_type == "input_audio_buffer.speech_started":
                    if self._greeting_response_id and not self._greeting_completed:
                        logger.info(
                            "Barge-in blocked - protecting greeting response",
                            call_id=self._call_id,
                            response_id=self._greeting_response_id,
                        )
                    else:
                        try:
                            async with self._pacer_lock:
                                self._outbuf.clear()
                        except Exception:
                            logger.debug("Failed to clear egress buffer on barge-in", call_id=self._call_id, exc_info=True)
                        try:
                            await self._emit_audio_done()
                        except Exception:
                            logger.debug("Failed to stop egress pacer on barge-in", call_id=self._call_id, exc_info=True)
                        logger.info(
                            "User speech started (no active response); requesting platform flush",
                            call_id=self._call_id,
                            event_type=event_type,
                        )
                        await self._emit_provider_barge_in(event_type=event_type)
                else:
                    logger.info("Azure OpenAI input_audio_buffer ack", call_id=self._call_id, event_type=event_type)
            return

        if event_type == "response.output_audio_transcript.delta":
            delta = event.get("delta")
            text = None
            if isinstance(delta, dict):
                text = delta.get("text")
            elif isinstance(delta, str):
                text = delta
            if text:
                await self._emit_transcript(text, is_final=False)
            return

        if event_type == "response.output_audio_transcript.done":
            if self._transcript_buffer:
                await self._track_conversation("assistant", self._transcript_buffer)
                await self._emit_transcript("", is_final=True)
            return

        if event_type == "session.updated":
            try:
                session = event.get("session", {})
                input_format = session.get("input_audio_format", "pcm16")
                output_format = session.get("output_audio_format", "pcm16")
                format_map = {
                    'pcm16': ('pcm16', 24000),
                    'g711_ulaw': ('g711_ulaw', 8000),
                    'g711_alaw': ('g711_alaw', 8000),
                }
                if output_format in format_map:
                    fmt, rate = format_map[output_format]
                    self._provider_output_format = fmt
                    self._active_output_sample_rate_hz = rate
                    self._outfmt_acknowledged = True

                logger.info(
                    "Azure OpenAI session.updated ACK received",
                    call_id=self._call_id,
                    input_format=input_format,
                    output_format=output_format,
                    sample_rate=self._active_output_sample_rate_hz,
                    acknowledged=self._outfmt_acknowledged,
                )

                if hasattr(self, '_session_ack_event') and self._session_ack_event:
                    self._session_ack_event.set()

            except Exception as exc:
                logger.error(
                    "Failed to process session.updated event",
                    call_id=self._call_id,
                    error=str(exc),
                    exc_info=True
                )
            return

        if event_type == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                call_id_field = item.get("call_id")
                function_name = item.get("name")
                logger.info(
                    "Azure OpenAI function call detected",
                    call_id=self._call_id,
                    function_call_id=call_id_field,
                    function_name=function_name,
                )
                task = asyncio.create_task(self._handle_function_call(event))
                task.add_done_callback(_log_provider_task_exception)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return

        logger.debug("Unhandled Azure OpenAI Realtime event", event_type=event_type)

    async def _handle_output_audio(self, audio_b64: str):
        try:
            raw_bytes = base64.b64decode(audio_b64)
        except Exception:
            logger.warning("Invalid base64 audio payload from Azure OpenAI", call_id=self._call_id)
            return

        if not raw_bytes:
            return

        try:
            if self._current_response_id:
                self._audio_seen_response_ids.add(self._current_response_id)
        except Exception:
            pass

        if self._turn_start_time is not None and not self._turn_first_audio_received:
            self._turn_first_audio_received = True
            turn_latency_ms = (time.time() - self._turn_start_time) * 1000
            if self._session_store and self._call_id:
                try:
                    call_id_copy = self._call_id
                    latency_copy = turn_latency_ms
                    async def save_latency():
                        try:
                            session = await self._session_store.get_by_call_id(call_id_copy)
                            if session:
                                session.turn_latencies_ms.append(latency_copy)
                                await self._session_store.upsert_call(session)
                                logger.debug("Turn latency saved to session", call_id=call_id_copy, latency_ms=round(latency_copy, 1))
                        except Exception as e:
                            logger.debug("Failed to save turn latency", call_id=call_id_copy, error=str(e))
                    task = asyncio.create_task(save_latency())
                    task.add_done_callback(_log_provider_task_exception)
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except Exception as e:
                    logger.debug("Failed to create latency save task", error=str(e))
            logger.info("Turn latency recorded", call_id=self._call_id, latency_ms=round(turn_latency_ms, 1))

        self._update_output_meter(len(raw_bytes))

        target_enc = (self.config.target_encoding or "").lower()
        if (
            self._outfmt_acknowledged
            and self._provider_output_format in ("g711_ulaw", "ulaw", "mulaw", "g711", "mu-law")
            and target_enc in ("ulaw", "mulaw", "g711_ulaw", "mu-law")
            and int(self.config.target_sample_rate_hz or 0) == 8000
            and int(round(self._active_output_sample_rate_hz or 8000)) == 8000
        ):
            outbound = raw_bytes
        else:
            effective_fmt = self._provider_output_format
            if not self._outfmt_acknowledged:
                inferred = None
                try:
                    l = len(raw_bytes)
                    if l % 2 == 1:
                        inferred = "ulaw"
                    else:
                        win_pcm = raw_bytes[: min(640, l - (l % 2))]
                        rms_pcm = audioop.rms(win_pcm, 2) if win_pcm else 0
                        try:
                            win_mulaw_pcm16 = mulaw_to_pcm16le(raw_bytes[: min(320, l)])
                        except Exception:
                            win_mulaw_pcm16 = b""
                        rms_ulaw = audioop.rms(win_mulaw_pcm16, 2) if win_mulaw_pcm16 else 0
                        if rms_ulaw > max(50, int(1.5 * (rms_pcm or 1))):
                            inferred = "ulaw"
                        else:
                            inferred = "pcm16"
                except Exception:
                    inferred = None
                self._inferred_provider_encoding = inferred or self._inferred_provider_encoding or "pcm16"
                effective_fmt = self._inferred_provider_encoding
                if not self._inference_logged:
                    try:
                        logger.info(
                            "Azure OpenAI output format not ACKed; using inferred decode path",
                            call_id=self._call_id,
                            inferred=effective_fmt,
                            bytes=len(raw_bytes),
                        )
                    except Exception:
                        pass
                    self._inference_logged = True

            if not self._outfmt_acknowledged and not self._inference_logged:
                logger.warning(
                    "Processing audio without format ACK - using inference fallback",
                    call_id=self._call_id,
                    inferred_format=effective_fmt,
                    note="Azure OpenAI should send session.updated ACK."
                )

            if effective_fmt in ("g711_ulaw", "ulaw", "mulaw", "g711", "mu-law"):
                try:
                    pcm_provider_output = mulaw_to_pcm16le(raw_bytes)
                except Exception:
                    logger.warning("Failed to convert μ-law provider output to PCM16", call_id=self._call_id, exc_info=True)
                    return
            else:
                pcm_provider_output = raw_bytes

            target_rate = self.config.target_sample_rate_hz
            if not self._outfmt_acknowledged and effective_fmt in ("g711_ulaw", "ulaw", "mulaw", "g711", "mu-law"):
                source_rate = 8000
            else:
                source_rate = int(round(self._active_output_sample_rate_hz or self.config.output_sample_rate_hz or 0))
                if not source_rate:
                    source_rate = self.config.output_sample_rate_hz
            pcm_target, self._output_resample_state = resample_audio(
                pcm_provider_output,
                source_rate,
                target_rate,
                state=self._output_resample_state,
            )

            outbound = convert_pcm16le_to_target_format(pcm_target, self.config.target_encoding)
            if not outbound:
                return

        try:
            async with self._pacer_lock:
                self._outbuf.extend(outbound)
        except Exception:
            logger.debug("Failed appending to pacer buffer", call_id=self._call_id, exc_info=True)

        if self._egress_pacer_enabled:
            await self._ensure_pacer_started()
        else:
            if self.on_event:
                if not self._first_output_chunk_logged:
                    logger.info(
                        "Azure OpenAI Realtime first audio chunk",
                        call_id=self._call_id,
                        bytes=len(outbound),
                        target_encoding=self.config.target_encoding,
                    )
                    self._first_output_chunk_logged = True
                self._in_audio_burst = True
                try:
                    await self.on_event(
                        {
                            "type": "AgentAudio",
                            "data": outbound,
                            "streaming_chunk": True,
                            "call_id": self._call_id,
                            "encoding": (self.config.target_encoding or "slin16"),
                            "sample_rate": self.config.target_sample_rate_hz,
                        }
                    )
                except Exception:
                    logger.error("Failed to emit AgentAudio event", call_id=self._call_id, exc_info=True)

    async def _emit_audio_done(self):
        if not self.on_event or not self._call_id:
            return
        try:
            if self._in_audio_burst:
                await self.on_event(
                    {
                        "type": "AgentAudioDone",
                        "streaming_done": True,
                        "call_id": self._call_id,
                    }
                )
        except Exception:
            logger.error("Failed to emit AgentAudioDone event", call_id=self._call_id, exc_info=True)
        finally:
            self._in_audio_burst = False
            try:
                self._pacer_running = False
                if self._pacer_task and not self._pacer_task.done():
                    self._pacer_task.cancel()
            except Exception:
                logger.debug("Failed to pause pacer on AgentAudioDone", call_id=self._call_id, exc_info=True)
            self._output_resample_state = None
            self._first_output_chunk_logged = False

        if self._farewell_waiting_for_audio_done and self._farewell_response_id is not None:
            self._cancel_farewell_timeout()
            self._farewell_waiting_for_audio_done = False
            self._farewell_response_id = None
            try:
                await self.on_event(
                    {
                        "type": "HangupReady",
                        "call_id": self._call_id,
                        "reason": "farewell_completed",
                        "had_audio": True,
                    }
                )
            except Exception:
                logger.error("Failed to emit HangupReady after output_audio.done", call_id=self._call_id, exc_info=True)

    async def _emit_transcript(self, text: str, *, is_final: bool):
        if not self.on_event or not self._call_id:
            return
        if text:
            self._transcript_buffer += text
        payload = {
            "type": "Transcript",
            "call_id": self._call_id,
            "text": text or self._transcript_buffer,
            "is_final": is_final,
        }
        try:
            await self.on_event(payload)
        except Exception:
            logger.error("Failed to emit transcript event", call_id=self._call_id, exc_info=True)
        if is_final:
            self._transcript_buffer = ""

    async def _track_conversation(self, role: str, text: str):
        import time
        if not self._call_id or not text:
            return
        if not hasattr(self, '_session_store') or not self._session_store:
            return
        try:
            session = await self._session_store.get_by_call_id(self._call_id)
            if session:
                session.conversation_history.append({
                    "role": role,
                    "content": text,
                    "timestamp": time.time()
                })
                await self._session_store.upsert_call(session)
        except Exception as e:
            logger.error(
                "Failed to track conversation",
                call_id=self._call_id,
                error=str(e),
                exc_info=True
            )

    async def _keepalive_loop(self):
        try:
            while self.websocket and self.websocket.state.name == "OPEN":
                await asyncio.sleep(_KEEPALIVE_INTERVAL_SEC)
                if not self.websocket or self.websocket.state.name != "OPEN":
                    break
                try:
                    async with self._send_lock:
                        if self.websocket and self.websocket.state.name == "OPEN":
                            await self.websocket.ping()
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.debug("Azure OpenAI Realtime keepalive failed", call_id=self._call_id, exc_info=True)
                    break
        except asyncio.CancelledError:
            pass

    async def _reconnect_with_backoff(self):
        call_id = self._call_id
        if not call_id:
            return
        backoff = 0.5
        for attempt in range(1, 6):
            if self._closing or self._closed:
                return
            try:
                url = self._build_ws_url()
                headers = [
                    ("api-key", self.config.api_key),
                ]
                logger.info("Reconnecting to Azure OpenAI Realtime", call_id=call_id, attempt=attempt)
                self.websocket = await websockets.connect(url, additional_headers=headers)
                self._pending_response = False
                self._in_audio_burst = False
                self._first_output_chunk_logged = False
                await self._send_session_update()
                self._log_session_assumptions()
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                logger.info("Azure OpenAI Realtime reconnected", call_id=call_id)
                return
            except Exception:
                logger.warning("Azure OpenAI Realtime reconnect failed", call_id=call_id, attempt=attempt, exc_info=True)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(6.0, backoff * 2)
        logger.error("Azure OpenAI Realtime reconnection exhausted attempts", call_id=call_id)

    # ------------------------------------------------------------------ #
    # Metrics and session metadata helpers
    # ------------------------------------------------------------------ #

    def _reset_output_meter(self) -> None:
        self._output_meter_start_ts = 0.0
        self._output_meter_last_log_ts = 0.0
        self._output_meter_bytes = 0
        self._output_rate_warned = False
        self._provider_reported_output_rate = None
        try:
            self._active_output_sample_rate_hz = float(self.config.output_sample_rate_hz)
        except Exception:
            self._active_output_sample_rate_hz = None

    def _log_session_assumptions(self) -> None:
        call_id = self._call_id
        if not call_id:
            return
        assumed_output = int(getattr(self.config, "output_sample_rate_hz", 0) or 0)
        try:
            _AZURE_ASSUMED_OUTPUT_RATE.set(assumed_output)
        except Exception:
            pass
        info_payload = {
            "input_encoding": str(getattr(self.config, "input_encoding", "") or ""),
            "input_sample_rate_hz": str(getattr(self.config, "input_sample_rate_hz", "") or ""),
            "provider_input_encoding": str(getattr(self.config, "provider_input_encoding", "") or ""),
            "provider_input_sample_rate_hz": str(getattr(self.config, "provider_input_sample_rate_hz", "") or ""),
            "output_encoding": self._session_output_encoding,
            "output_sample_rate_hz": str(int(self._active_output_sample_rate_hz or getattr(self.config, "output_sample_rate_hz", "") or 0)),
            "target_encoding": str(getattr(self.config, "target_encoding", "") or ""),
            "target_sample_rate_hz": str(getattr(self.config, "target_sample_rate_hz", "") or ""),
        }
        try:
            _AZURE_SESSION_AUDIO_INFO.info(info_payload)
        except Exception:
            pass
        try:
            logger.info(
                "Azure OpenAI Realtime session assumptions",
                call_id=call_id,
                input_encoding=info_payload["input_encoding"],
                input_sample_rate_hz=info_payload["input_sample_rate_hz"],
                provider_input_sample_rate_hz=info_payload["provider_input_sample_rate_hz"],
                output_sample_rate_hz=info_payload["output_sample_rate_hz"],
                target_encoding=info_payload["target_encoding"],
                target_sample_rate_hz=info_payload["target_sample_rate_hz"],
            )
        except Exception:
            logger.debug("Failed to log Azure OpenAI session assumptions", exc_info=True)

    def _handle_session_info_event(self, event: Dict[str, Any]) -> None:
        call_id = self._call_id
        if not call_id:
            return
        session_data = event.get("session") or {}
        output_meta = session_data.get("output_audio_format") or {}
        provider_rate = self._extract_sample_rate(output_meta)
        provider_encoding = self._extract_encoding(output_meta)
        if provider_rate:
            self._provider_reported_output_rate = provider_rate
            try:
                _AZURE_PROVIDER_OUTPUT_RATE.set(provider_rate)
            except Exception:
                pass
            try:
                self._active_output_sample_rate_hz = float(provider_rate)
            except Exception:
                self._active_output_sample_rate_hz = provider_rate
        enc_norm = (provider_encoding or "").lower()
        if enc_norm in ("g711_ulaw", "ulaw", "mulaw", "mu-law") and int(provider_rate or 0) == 8000:
            self._outfmt_acknowledged = True
            self._provider_output_format = "g711_ulaw"
            self._session_output_bytes_per_sample = 1
            self._session_output_encoding = "g711_ulaw"
        else:
            self._outfmt_acknowledged = False
            self._provider_output_format = "pcm16"
            self._session_output_bytes_per_sample = 2
            self._session_output_encoding = "pcm16"
        info_payload = {
            "input_encoding": str(getattr(self.config, "input_encoding", "") or ""),
            "input_sample_rate_hz": str(getattr(self.config, "input_sample_rate_hz", "") or ""),
            "provider_input_encoding": str(getattr(self.config, "provider_input_encoding", "") or ""),
            "provider_input_sample_rate_hz": str(getattr(self.config, "provider_input_sample_rate_hz", "") or ""),
            "output_encoding": provider_encoding or self._session_output_encoding,
            "output_sample_rate_hz": str(provider_rate or self._active_output_sample_rate_hz or getattr(self.config, "output_sample_rate_hz", "") or ""),
            "target_encoding": str(getattr(self.config, "target_encoding", "") or ""),
            "target_sample_rate_hz": str(getattr(self.config, "target_sample_rate_hz", "") or ""),
        }
        try:
            _AZURE_SESSION_AUDIO_INFO.info(info_payload)
        except Exception:
            pass

    def _update_output_meter(self, chunk_bytes: int) -> None:
        if not chunk_bytes or not self._call_id:
            return
        now = time.monotonic()
        if not self._output_meter_start_ts:
            self._output_meter_start_ts = now
            self._output_meter_last_log_ts = now
        self._output_meter_bytes += chunk_bytes
        elapsed = max(1e-6, now - self._output_meter_start_ts)
        bytes_per_sample = max(1, self._session_output_bytes_per_sample)
        measured_rate = (self._output_meter_bytes / bytes_per_sample) / elapsed
        try:
            target_is_ulaw = str(getattr(self.config, "target_encoding", "") or "").lower() in (
                "ulaw", "mulaw", "g711_ulaw", "mu-law",
            )
        except Exception:
            target_is_ulaw = False
        try:
            _AZURE_MEASURED_OUTPUT_RATE.set(measured_rate)
        except Exception:
            pass
        try:
            assumed_now = float(self._active_output_sample_rate_hz or getattr(self.config, "output_sample_rate_hz", 0) or 0)
        except Exception:
            assumed_now = float(getattr(self.config, "output_sample_rate_hz", 0) or 0)
        if elapsed >= 0.25 and assumed_now > 0:
            try:
                drift_now = abs(measured_rate - assumed_now) / assumed_now
            except Exception:
                drift_now = 0.0
            if drift_now > 0.10 and not self._output_rate_warned:
                self._output_rate_warned = True
                logger.debug(
                    "Azure OpenAI output rate drift detected (expected for real-time streaming)",
                    call_id=self._call_id,
                    measured_rate_hz=round(measured_rate, 2),
                    configured_rate_hz=assumed_now,
                    note="Measured rate reflects playback speed, not sample rate. Ignoring.",
                )
        if now - self._output_meter_last_log_ts >= 1.0:
            self._output_meter_last_log_ts = now
            assumed = float(self._active_output_sample_rate_hz or getattr(self.config, "output_sample_rate_hz", 0) or 0)
            reported = self._provider_reported_output_rate
            log_payload = {
                "call_id": self._call_id,
                "assumed_output_sample_rate_hz": assumed or None,
                "provider_reported_sample_rate_hz": reported,
                "measured_output_sample_rate_hz": round(measured_rate, 2),
                "window_seconds": round(elapsed, 2),
                "bytes_window": self._output_meter_bytes,
            }
            try:
                logger.info(
                    "Azure OpenAI Realtime output rate check",
                    **{k: v for k, v in log_payload.items() if v is not None},
                )
            except Exception:
                logger.debug("Failed to log Azure OpenAI output rate check", exc_info=True)
            try:
                if not self._fallback_pcm24k_done:
                    if self._outfmt_acknowledged and self._provider_output_format in ("g711_ulaw", "g711_alaw"):
                        return
                    window_anchor = self._pacer_start_ts if self._pacer_start_ts > 0.0 else self._output_meter_start_ts
                    window = now - window_anchor if window_anchor > 0.0 else elapsed
                    if window >= 10.0 and measured_rate and measured_rate < 7600.0:
                        asyncio.create_task(self._switch_to_pcm24k_output())
                        self._fallback_pcm24k_done = True
            except Exception:
                logger.debug("PCM24k fallback evaluation error", exc_info=True)

    async def _ensure_pacer_started(self) -> None:
        if self._pacer_running:
            return
        if not self.on_event or not self._call_id:
            return
        self._pacer_running = True
        self._pacer_start_ts = time.monotonic()
        try:
            clear_buffer_payload = {
                "type": "input_audio_buffer.clear",
                "event_id": f"clear-echo-{uuid.uuid4()}",
            }
            await self._send_json(clear_buffer_payload)
            logger.debug("Cleared Azure OpenAI input buffer for echo prevention", call_id=self._call_id)
        except Exception:
            logger.debug("Failed to clear input buffer", call_id=self._call_id, exc_info=True)
        try:
            if self._pacer_task and not self._pacer_task.done():
                self._pacer_task.cancel()
        except Exception:
            pass
        self._pacer_task = asyncio.create_task(self._pacer_loop())

    async def _pacer_loop(self) -> None:
        call_id = self._call_id
        if not call_id or not self.on_event:
            self._pacer_running = False
            return
        chunk_bytes, silence_factory = self._pacer_params()
        warmup_bytes = int(max(0, self._egress_pacer_warmup_ms) / 20) * chunk_bytes
        try:
            while self.websocket and self.websocket.state.name == "OPEN" and self._pacer_running:
                async with self._pacer_lock:
                    buf_len = len(self._outbuf)
                if buf_len >= warmup_bytes or not self._egress_pacer_enabled:
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("Pacer warm-up error", call_id=call_id, exc_info=True)
        try:
            while self.websocket and self.websocket.state.name == "OPEN" and self._pacer_running:
                chunk = b""
                async with self._pacer_lock:
                    if len(self._outbuf) >= chunk_bytes:
                        chunk = bytes(self._outbuf[:chunk_bytes])
                        del self._outbuf[:chunk_bytes]
                if not chunk:
                    self._pacer_underruns += 1
                    chunk = silence_factory(chunk_bytes)
                has_buffered_audio = len(self._outbuf) > 0
                is_first_real_chunk = chunk and self._pacer_underruns == 0
                if has_buffered_audio or is_first_real_chunk:
                    self._in_audio_burst = True
                    if not self._first_output_chunk_logged:
                        try:
                            logger.info(
                                "Azure OpenAI Realtime first paced audio chunk",
                                call_id=call_id,
                                bytes=len(chunk),
                                target_encoding=self.config.target_encoding,
                            )
                        except Exception:
                            pass
                        self._first_output_chunk_logged = True
                try:
                    await self.on_event(
                        {
                            "type": "AgentAudio",
                            "data": chunk,
                            "streaming_chunk": True,
                            "call_id": call_id,
                            "encoding": (self.config.target_encoding or "slin16"),
                            "sample_rate": self.config.target_sample_rate_hz,
                        }
                    )
                except Exception:
                    logger.error("Failed to emit paced AgentAudio", call_id=call_id, exc_info=True)
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("Pacer loop error", call_id=call_id, exc_info=True)
        finally:
            self._pacer_running = False
            self._in_audio_burst = False

    def _pacer_params(self) -> tuple:
        enc = (self.config.target_encoding or "ulaw").lower()
        rate = int(self.config.target_sample_rate_hz or 8000)
        if enc in ("ulaw", "mulaw", "g711_ulaw", "mu-law"):
            bytes_per_sample = 1
            chunk_bytes = int(rate / 50) * bytes_per_sample
            def silence(n: int) -> bytes:
                return bytes([0xFF]) * max(0, n)
            return chunk_bytes, silence
        bytes_per_sample = 2
        chunk_bytes = int(rate / 50) * bytes_per_sample
        def silence(n: int) -> bytes:
            return b"\x00" * max(0, n)
        return chunk_bytes, silence

    async def _switch_to_pcm24k_output(self) -> None:
        if not self.websocket or self.websocket.state.name != "OPEN":
            return
        call_id = self._call_id
        try:
            logger.warning(
                "Switching Azure OpenAI output to PCM16@24k due to sustained low measured rate",
                call_id=call_id,
            )
        except Exception:
            pass
        if self._is_ga:
            pcm_session = {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}}
        else:
            pcm_session = {"output_audio_format": "pcm16"}
        payload: Dict[str, Any] = {
            "type": "session.update",
            "event_id": f"sess-{uuid.uuid4()}",
            "session": self._ga_session_type(pcm_session),
        }
        try:
            await self._send_json(payload)
            self._provider_output_format = "pcm16"
            self._session_output_bytes_per_sample = 2
            try:
                self._active_output_sample_rate_hz = float(24000)
            except Exception:
                self._active_output_sample_rate_hz = 24000.0
            self._reset_output_meter()
        except Exception:
            logger.debug("Failed to switch Azure OpenAI session to PCM16@24k", call_id=call_id, exc_info=True)

    @staticmethod
    def _extract_sample_rate(fmt: Any) -> Optional[int]:
        if isinstance(fmt, str):
            if "@" in fmt:
                try:
                    return int(float(fmt.split("@", 1)[1]))
                except (IndexError, ValueError):
                    return None
            return None
        if not isinstance(fmt, dict):
            return None
        for key in ("sample_rate", "sample_rate_hz", "rate"):
            value = fmt.get(key)
            if value is None:
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_encoding(fmt: Any) -> Optional[str]:
        if isinstance(fmt, str):
            if "@" in fmt:
                return fmt.split("@", 1)[0].strip().lower()
            return fmt.lower()
        if not isinstance(fmt, dict):
            return None
        for key in ("encoding", "format", "type"):
            value = fmt.get(key)
            if isinstance(value, str) and value.strip():
                return value.lower()
        return None

    def _clear_metrics(self, call_id: Optional[str]) -> None:
        self._reset_output_meter()

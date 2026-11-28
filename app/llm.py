"""One place that talks to the LLM.

* Streams: yields text as it arrives, and collects tool calls from the stream.
* Falls back: if the main provider fails before sending anything, the same
  request goes to the fallback model (Ollama). The answer says which model ran.
* Counts: tokens and cost of every call.
"""
import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"id","name","arguments"}
    usage: Usage = field(default_factory=Usage)
    model: str = ""


def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "missing", timeout=timeout, max_retries=2)


def _cost(model: str, prompt: int, completion: int) -> float:
    if model != settings.llm_model:  # the local fallback costs nothing per token
        return 0.0
    return (prompt * settings.llm_price_in + completion * settings.llm_price_out) / 1_000_000


def _stream_once(client: OpenAI, model: str, messages, tools):
    """Generator: yields text pieces as they arrive, returns the full LLMReply."""
    kwargs = {"model": model, "messages": messages, "stream": True,
              "stream_options": {"include_usage": True}, "max_tokens": 1500}
    if tools:
        kwargs["tools"] = tools
    reply = LLMReply(model=model)
    calls: dict[int, dict] = {}
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.usage:
            reply.usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens,
                                _cost(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens))
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            reply.text += delta.content
            yield delta.content
        for tc in delta.tool_calls or []:  # tool calls arrive in pieces; glue them by index
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    reply.tool_calls = [calls[i] for i in sorted(calls)]
    return reply


def stream(messages: list[dict], tools: list[dict] | None = None):
    """Generator: yields text pieces; its return value is the LLMReply.

    Use it as:  reply = yield from stream(messages, tools)
    """
    if settings.llm_fake:
        from app.fake_llm import fake_stream
        return (yield from fake_stream(messages, tools))

    sent_any = False
    try:
        gen = _stream_once(_client(settings.llm_base_url, settings.llm_api_key, 60),
                           settings.llm_model, messages, tools)
        while True:
            try:
                piece = next(gen)
            except StopIteration as stop:
                return stop.value
            sent_any = True
            yield piece
    except Exception as e:
        if sent_any:  # half an answer is already on the screen: do not mix two models
            raise
        log.warning("main LLM failed (%s); trying fallback %s", e, settings.fallback_model)
        try:
            return (yield from _stream_once(_client(settings.fallback_base_url, "ollama", 180),
                                            settings.fallback_model, messages, tools))
        except Exception as e2:
            raise RuntimeError(f"main LLM failed ({e}) and fallback failed ({e2})") from e2


def parse_arguments(raw: str) -> dict:
    return json.loads(raw) if raw.strip() else {}
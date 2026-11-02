"""The model reply is JSON, and citations have to name a real block."""

import json

import pytest

from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.generation.parse import extract_json_object, interpret_answer, request_json
from rag.generation.prompt import SYSTEM_PROMPT
from rag.generation.types import INSUFFICIENT_CONTEXT


def test_system_prompt_forbids_outside_knowledge() -> None:
    assert "outside knowledge" in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT
    assert INSUFFICIENT_CONTEXT in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT


def test_fenced_json_and_a_short_preface_both_parse() -> None:
    fenced = '```json\n{"sufficient": true, "answer": "Yes [1].", "citations": [1]}\n```'
    assert extract_json_object(fenced)["answer"] == "Yes [1]."
    preface = 'Here you go:\n{"sufficient": false, "answer": "nope", "citations": []}\n'
    assert extract_json_object(preface)["sufficient"] is False


def test_non_object_replies_are_rejected() -> None:
    with pytest.raises(GenerationError, match="JSON object"):
        extract_json_object("not json")
    with pytest.raises(GenerationError, match="JSON object"):
        extract_json_object("[1, 2]")
    with pytest.raises(GenerationError, match="missing answer"):
        interpret_answer({"sufficient": True, "citations": []}, context_size=1)


def test_sufficient_answer_keeps_citation_order_from_the_text() -> None:
    payload = {
        "sufficient": True,
        "answer": "See [2] then [1].",
        "citations": [1],
    }
    draft = interpret_answer(payload, context_size=2)
    assert draft.sufficient is True
    assert draft.citation_numbers == (2, 1)
    assert draft.answer == "See [2] then [1]."
    assert draft.insufficient_reason is None


def test_declared_citation_is_kept_when_the_text_omits_the_marker() -> None:
    payload = {
        "sufficient": True,
        "answer": "Forking copies the session [1].",
        "citations": [1, 2],
    }
    draft = interpret_answer(payload, context_size=2)
    assert draft.citation_numbers == (1, 2)


def test_invalid_markers_are_removed() -> None:
    payload = {
        "sufficient": True,
        "answer": "Forking copies the session [1] [9].",
        "citations": [1, 9],
    }
    draft = interpret_answer(payload, context_size=1)
    assert draft.sufficient is True
    assert draft.citation_numbers == (1,)
    assert draft.answer == "Forking copies the session [1]."
    assert "[9]" not in draft.answer


def test_model_refusal_becomes_the_fixed_sentence() -> None:
    payload = {
        "sufficient": False,
        "answer": "I think the secret is 42 [1].",
        "citations": [1],
    }
    draft = interpret_answer(payload, context_size=1)
    assert draft.sufficient is False
    assert draft.insufficient_reason == "model"
    assert draft.answer == INSUFFICIENT_CONTEXT
    assert draft.citation_numbers == ()
    assert "42" not in draft.answer


def test_uncited_answer_is_not_shown() -> None:
    payload = {
        "sufficient": True,
        "answer": "Forking copies the session from memory.",
        "citations": [9],
    }
    draft = interpret_answer(payload, context_size=1)
    assert draft.sufficient is False
    assert draft.insufficient_reason == "uncited"
    assert draft.answer == INSUFFICIENT_CONTEXT
    assert "memory" not in draft.answer


def test_citation_types_are_strict() -> None:
    payload = {"sufficient": True, "answer": "Yes [1].", "citations": [True]}
    with pytest.raises(GenerationError, match="integers"):
        interpret_answer(payload, context_size=1)
    payload["citations"] = ["1"]
    with pytest.raises(GenerationError, match="integers"):
        interpret_answer(payload, context_size=1)


def test_request_json_retries_once_then_fails() -> None:
    llm = _ScriptedLLM(["LEAKED-REPLY", "still no"])
    with pytest.raises(GenerationError, match="could not be read") as caught:
        request_json(
            llm,
            [LLMMessage(role="user", content="question")],
            _echo,
            max_tokens=32,
        )
    assert "LEAKED-REPLY" not in str(caught.value)
    assert len(llm.calls) == 2
    roles = [message.role for message in llm.calls[1][0]]
    assert roles == ["user", "assistant", "user"]
    assert llm.calls[1][0][1].content == "LEAKED-REPLY"
    assert "LEAKED-REPLY" not in llm.calls[1][0][2].content


def test_request_json_accepts_the_second_reply() -> None:
    body = json.dumps({"ok": True})
    llm = _ScriptedLLM(["nope", body])
    value, responses = request_json(
        llm,
        [LLMMessage(role="user", content="question")],
        _echo,
        max_tokens=16,
    )
    assert value == {"ok": True}
    assert len(responses) == 2
    assert llm.calls[0][1] == 16


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[LLMMessage], int | None]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        self.calls.append((list(messages), max_tokens))
        content = self._replies.pop(0)
        return LLMResponse(
            content=content,
            model="scripted",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


def _echo(payload: dict[str, object]) -> dict[str, object]:
    return payload

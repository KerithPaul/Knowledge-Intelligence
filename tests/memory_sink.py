"""In-memory span sink for tests. It does not talk to Langfuse."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRow:
    name: str
    observation: str
    attributes: dict[str, Any]
    level: str
    status_message: str | None
    output: Any


@dataclass
class MemorySink:
    """Records each closed span. `fail_open` and `fail_close` raise instead."""

    rows: list[MemoryRow] = field(default_factory=list)
    fail_open: bool = False
    fail_close: bool = False

    def open(self, name: str, *, observation: str) -> "_Closer":
        if self.fail_open:
            raise RuntimeError("sink open failed")
        return _Closer(self, name, observation)


class _Closer:
    def __init__(self, sink: MemorySink, name: str, observation: str) -> None:
        self._sink = sink
        self._name = name
        self._observation = observation

    def close(
        self,
        *,
        attributes: dict[str, Any],
        level: str,
        status_message: str | None,
        output: Any,
    ) -> None:
        if self._sink.fail_close:
            raise RuntimeError("sink close failed")
        self._sink.rows.append(
            MemoryRow(
                name=self._name,
                observation=self._observation,
                attributes=dict(attributes),
                level=level,
                status_message=status_message,
                output=output,
            )
        )

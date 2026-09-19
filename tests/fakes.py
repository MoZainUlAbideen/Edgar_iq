"""A fake GroqClient-shaped object for testing agents without real API
calls. Returns canned responses in order, one per .complete() call, so a
test can script exactly what "the planner said" vs "the drafter said"."""

from __future__ import annotations


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []  # (system, user) per call, for assertions

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of canned responses")
        return self._responses.pop(0)

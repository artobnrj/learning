"""Record landmark streams to JSON-lines and replay them through the engine.

Recording only stores landmarks (not video), so sessions are tiny, private
and perfect for regression-testing threshold changes:

    gesturectl run --record session.jsonl
    gesturectl replay session.jsonl --config tweaked.yaml
"""
from __future__ import annotations

import json
from typing import Iterator, List, Tuple

from .landmarks import Hand


class Recorder:
    def __init__(self, path: str):
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, t: float, hands: List[Hand]) -> None:
        self._fh.write(json.dumps({"t": round(t, 4), "hands": [h.to_json() for h in hands]}) + "\n")

    def close(self) -> None:
        self._fh.close()


def read_session(path: str) -> Iterator[Tuple[float, List[Hand]]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                yield d["t"], [Hand.from_json(h) for h in d["hands"]]

"""复投观察期档案。

观察期是复投判决的唯一证据来源，因此档案**只增不改**：每个观察窗口在评估
完成时即刻封存，条目携带当时算出的水平与 z 值以及前序条目的哈希，构成
哈希链；结案（pass/fail）后拒绝再写入。后续即使判定参数调整、群体基线
重算或平台重启，也只能依据流水重放档案——任何改写都会破坏哈希链而被
:meth:`ObservationDossier.verify_integrity` 识破，这正是「观察期数据不被
后面的重算覆盖」的落实方式。
"""

from __future__ import annotations

import hashlib
import json

from ..errors import InterlockError


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ObservationDossier:
    """一个簇的复投观察期档案（只增不改，哈希链封存）。"""

    def __init__(self, rack_id: str, started_window: int, required_windows: int, z_limit: float) -> None:
        self._rack_id = rack_id
        self._started_window = int(started_window)
        self._required_windows = max(1, int(required_windows))
        self._z_limit = float(z_limit)
        self._entries: list[dict] = []
        self._final: dict | None = None

    @property
    def started_window(self) -> int:
        return self._started_window

    @property
    def required_windows(self) -> int:
        return self._required_windows

    @property
    def z_limit(self) -> float:
        return self._z_limit

    @property
    def sealed(self) -> int:
        """返回已封存窗口数。"""

        return len(self._entries)

    @property
    def finalized(self) -> bool:
        """档案是否已结案。"""

        return self._final is not None

    @property
    def verdict(self) -> str | None:
        """返回结案结论（pass/fail），未结案为 None。"""

        return self._final["verdict"] if self._final is not None else None

    def seal_window(
        self,
        window_index: int,
        resistance_level: float,
        temp_rise_level: float,
        z_r: float,
        z_t: float,
    ) -> dict:
        """封存一个观察窗口；结案后拒绝再写入。"""

        if self._final is not None:
            raise InterlockError("观察档案已结案，禁止再写入", component="joint")
        entry = {
            "seq": len(self._entries),
            "window_index": int(window_index),
            "resistance_level": float(resistance_level),
            "temp_rise_level": float(temp_rise_level),
            "z_r": float(z_r),
            "z_t": float(z_t),
            "passed": float(z_r) < self._z_limit and float(z_t) < self._z_limit,
            "prev_hash": self._entries[-1]["hash"] if self._entries else "0" * 64,
        }
        entry["hash"] = _digest(entry)
        self._entries.append(entry)
        return dict(entry)

    def ready(self) -> bool:
        """观察窗口是否封满。"""

        return len(self._entries) >= self._required_windows

    def finalize(self, verdict: str, window_index: int) -> dict:
        """结案：pass 复投 / fail 重新停用；不允许重复结案。"""

        if self._final is not None:
            raise InterlockError("观察档案不允许重复结案", component="joint")
        if verdict not in ("pass", "fail"):
            raise InterlockError(f"未知的观察结论: {verdict}", component="joint")
        self._final = {
            "verdict": verdict,
            "window_index": int(window_index),
            "sealed": len(self._entries),
            "hash": _digest({"final": verdict, "tail": self._entries[-1]["hash"] if self._entries else "0" * 64}),
        }
        return dict(self._final)

    def verify_integrity(self) -> bool:
        """重放哈希链，核对档案自封存以来未被改写。"""

        prev = "0" * 64
        for seq, entry in enumerate(self._entries):
            if entry.get("seq") != seq or entry.get("prev_hash") != prev:
                return False
            if _digest({key: value for key, value in entry.items() if key != "hash"}) != entry.get("hash"):
                return False
            prev = entry["hash"]
        if self._final is not None:
            expect = _digest({"final": self._final["verdict"], "tail": prev})
            if self._final.get("hash") != expect:
                return False
        return True

    def as_dict(self) -> dict:
        """返回档案全貌（含封存条目与完整性结论）。"""

        return {
            "rack_id": self._rack_id,
            "started_window": self._started_window,
            "required_windows": self._required_windows,
            "z_limit": self._z_limit,
            "sealed": len(self._entries),
            "finalized": self.finalized,
            "verdict": self.verdict,
            "integrity": self.verify_integrity(),
            "entries": [dict(entry) for entry in self._entries],
        }

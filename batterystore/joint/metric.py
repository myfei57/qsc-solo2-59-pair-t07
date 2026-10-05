"""评估窗口特征。

连接点发热近似 I²·R：同一簇在不同出力下的温升不能直接相比，因此窗口内的
温升采样一律按 (I_rated/I)² 折算到额定电流再比较；电流太低的采样发热信号
太弱，不参与温升统计。内阻与电流无关，直接取窗口中位数压住尖峰。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import StoreError
from .stats import median


@dataclass(frozen=True)
class JointSample:
    """一次连接点采样。"""

    resistance_mohm: float
    temp_rise_c: float
    current_a: float


@dataclass(frozen=True)
class WindowMetric:
    """一个评估窗口的特征。"""

    rack_id: str
    window_index: int
    resistance_mohm: float
    temp_rise_c: float
    samples: int

    def as_dict(self) -> dict:
        return {
            "rack_id": self.rack_id,
            "window_index": self.window_index,
            "resistance_mohm": round(self.resistance_mohm, 4),
            "temp_rise_c": round(self.temp_rise_c, 3),
            "samples": self.samples,
        }


def build_window_metric(
    rack_id: str,
    window_index: int,
    samples: list[JointSample],
    rated_current_a: float,
    min_current_ratio: float,
) -> WindowMetric:
    """把窗口内采样聚合成一个特征，温升折算到额定电流。"""

    if not samples:
        raise StoreError("评估窗口内没有采样", component="joint")
    rises = [
        sample.temp_rise_c * (rated_current_a / sample.current_a) ** 2
        for sample in samples
        if abs(sample.current_a) >= min_current_ratio * rated_current_a
    ]
    if not rises:
        raise StoreError("窗口内电流过低，温升无法归一到额定电流", component="joint")
    return WindowMetric(
        rack_id=rack_id,
        window_index=window_index,
        resistance_mohm=median([sample.resistance_mohm for sample in samples]),
        temp_rise_c=median(rises),
        samples=len(samples),
    )

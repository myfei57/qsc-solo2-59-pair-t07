"""日特征聚合。

把一天的有效样本压成两个中位数：等效连接电阻与归一化温升系数。用中位数
而不是均值——连接压降是毫欧级小信号，少数离群样本（启停、阶跃、通信毛刺）
不该左右全天的判定依据。数据质量门也落在这里：有效样本不足、温度传感器
卡死（全天量程接近零而电流明显在变化）都标记为无效日，判定层只使用质量
为 ``OK`` 的日特征。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..config import JointLimits
from .sample import JointSample, sample_metrics

QUALITY_OK = "OK"


@dataclass(frozen=True)
class DayFeature:
    """一簇一天的连接点特征。"""

    rack_id: str
    day: str
    r_mohm: float
    k_norm: float
    n_samples: int
    quality: str
    version: int

    def as_dict(self) -> dict:
        return {
            "rack_id": self.rack_id,
            "day": self.day,
            "r_mohm": round(self.r_mohm, 4),
            "k_norm": round(self.k_norm, 2),
            "n_samples": self.n_samples,
            "quality": self.quality,
            "version": self.version,
        }


def aggregate_day(
    rack_id: str,
    day: str,
    samples: list[JointSample],
    limits: JointLimits,
    version: int = 1,
) -> DayFeature:
    """聚合一天的有效样本，返回带质量标记的日特征。

    ``version`` 标识这是该日的第几次计算：重算只产生更高版本的新记录，
    不覆盖旧值——观察期数据不被重算覆盖靠的就是这条规矩。
    """

    rows = sample_metrics(samples, limits)
    quality = QUALITY_OK
    if samples:
        temps = [sample.t_conn_c for sample in samples]
        currents = [sample.current_a for sample in samples]
        if max(temps) - min(temps) < 0.2 and max(currents) - min(currents) > 0.2 * limits.rated_current_a:
            quality = "STUCK_SENSOR"
    if not rows:
        if quality == QUALITY_OK:
            quality = "NO_VALID_SAMPLES"
        return DayFeature(rack_id, day, float("nan"), float("nan"), 0, quality, version)
    if len(rows) < limits.min_samples_day and quality == QUALITY_OK:
        quality = "INSUFFICIENT_SAMPLES"
    return DayFeature(
        rack_id,
        day,
        statistics.median(row[1] for row in rows),
        statistics.median(row[2] for row in rows),
        len(rows),
        quality,
        version,
    )

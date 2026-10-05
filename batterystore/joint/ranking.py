"""劣化判定口径：同型簇相对位次（本组件选定的口径）。

两种候选口径——"按固定上升幅度判"与"拿同型簇的相对位次判"——在环境温度
变化时结论会不一致，本组件选定**相对位次**，理由如下：

1. 铜铝连接件的电阻温度系数约为 +0.4%/℃，季节温差 30℃ 就带来约 12% 的
   表观电阻漂移，与早期劣化信号同量级。固定门限必然在夏天误报、冬天漏报；
   要补救就得再叠一层环境温度补偿模型，而补偿模型本身又引入传感器布点、
   舱体通风差异等新的误差源。
2. 相对位次在同一时刻的簇间截面里比较：环境温度、负载水平、整站同步老化
   对同型簇是共模量，被中位数抵消，剩下的离群者就是"比同胞劣化得更快的
   那一台"——这正是检修要找的对象。
3. 固定幅度口径依赖每簇各自的历史基线，检修、复紧、更换传感器后基线都要
   重立，运维上是个长期漏洞；相对位次不依赖自身历史，新簇投运第一天就能
   参与判定（持续性窗口除外）。

该口径的已知盲区是**共模劣化**（所有连接点同步变差时无人离群）。盲区由
独立于本口径的两道防线兜底：连接点绝对温度保护（``protect_t_conn_c``，
见 monitor 的保护扫描）与定期力矩／微欧计复检。同型簇数量不足 ``min_group``
时判定暂停（宁缺毋滥），不退回固定幅度口径。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..config import JointLimits
from .features import QUALITY_OK, DayFeature

FLAG_OUTLIER = "OUTLIER"
FLAG_WATCH = "WATCH"
FLAG_NORMAL = "NORMAL"
FLAG_INVALID = "INVALID"

VERDICT_DEGRADED = "DEGRADED"
VERDICT_WATCH = "WATCH"
VERDICT_NORMAL = "NORMAL"
VERDICT_SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class DayScore:
    """一簇一天的相对位次得分。"""

    rack_id: str
    day: str
    z_r: float
    z_k: float
    flag: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "rack_id": self.rack_id,
            "day": self.day,
            "z_r": round(self.z_r, 3),
            "z_k": round(self.z_k, 3),
            "flag": self.flag,
            "reason": self.reason,
        }


def robust_zscores(values: dict[str, float]) -> dict[str, float]:
    """对同型簇当日取值求稳健 z 分数（相对中位数、以 MAD 为尺度）。

    同型簇高度一致时 MAD 接近零，微小差异会被放大成假离群，因此给尺度
    一个"中位数 2%"的下限。
    """

    median = statistics.median(values.values())
    mad = statistics.median(abs(value - median) for value in values.values())
    scale = max(1.4826 * mad, 0.02 * abs(median), 1e-9)
    return {rack_id: (value - median) / scale for rack_id, value in values.items()}


def trend_slope(values: list[float]) -> float:
    """Theil–Sen 稳健斜率（每天变化量），观察期判断指标是否仍在爬升。"""

    if len(values) < 2:
        return 0.0
    slopes = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            slopes.append((values[right] - values[left]) / (right - left))
    return statistics.median(slopes)


def rank_day(features: dict[str, DayFeature], limits: JointLimits) -> dict[str, DayScore]:
    """对同型簇一天的特征做相对位次判定，返回每簇的日标记。

    内阻与温升两个相互独立的物理量互相印证：两个都明显离群、或其中一个
    极端离群，才记为离群日，避免单一传感器故障直接触发停用。
    """

    usable = {rack_id: feature for rack_id, feature in features.items() if feature.quality == QUALITY_OK}
    scores: dict[str, DayScore] = {}
    if len(usable) < limits.min_group:
        for rack_id, feature in features.items():
            reason = "INSUFFICIENT_PEERS" if feature.quality == QUALITY_OK else feature.quality
            scores[rack_id] = DayScore(rack_id, feature.day, 0.0, 0.0, FLAG_INVALID, reason)
        return scores
    z_r = robust_zscores({rack_id: feature.r_mohm for rack_id, feature in usable.items()})
    z_k = robust_zscores({rack_id: feature.k_norm for rack_id, feature in usable.items()})
    for rack_id, feature in features.items():
        if feature.quality != QUALITY_OK:
            scores[rack_id] = DayScore(rack_id, feature.day, 0.0, 0.0, FLAG_INVALID, feature.quality)
            continue
        pair = (z_r[rack_id], z_k[rack_id])
        if min(pair) >= limits.z_on or max(pair) >= limits.z_strong:
            flag = FLAG_OUTLIER
        elif max(pair) >= limits.z_watch:
            flag = FLAG_WATCH
        else:
            flag = FLAG_NORMAL
        scores[rack_id] = DayScore(rack_id, feature.day, pair[0], pair[1], flag)
    return scores


def evaluate_persistence(flags: list[str], limits: JointLimits) -> str:
    """把最近若干天的日标记汇总成判定结论。

    长期走势靠持续性体现：窗口内离群天数达到阈值才判劣化，单日毛刺不会
    触发停用；有效天数不足时宁可暂停判定，也不凑数下结论。
    """

    valid = [flag for flag in flags if flag != FLAG_INVALID]
    if len(valid) < limits.persist_days:
        return VERDICT_SUSPENDED
    outliers = valid.count(FLAG_OUTLIER)
    if outliers >= limits.persist_days:
        return VERDICT_DEGRADED
    if outliers + valid.count(FLAG_WATCH) >= limits.persist_days:
        return VERDICT_WATCH
    return VERDICT_NORMAL

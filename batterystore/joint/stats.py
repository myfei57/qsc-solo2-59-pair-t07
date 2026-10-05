"""稳健统计：中位数、MAD 尺度与 Theil–Sen 斜率。

连接点劣化是以周/月为尺度的缓慢漂移，原始采样里混着量测噪声与偶发尖峰。
用中位数与 MAD（绝对中位差）代替均值/标准差、用点对斜率中位数代替最小二乘，
单次异常采样就不会把长期趋势带偏。全部只用标准库，保持交付树零依赖。
"""

from __future__ import annotations

import statistics


def median(values: list[float]) -> float:
    """返回序列中位数。"""

    seq = [float(value) for value in values]
    if not seq:
        raise ValueError("空序列没有中位数")
    return float(statistics.median(seq))


def mad_sigma(values: list[float]) -> float:
    """返回 MAD 换算的稳健标准差（1.4826·MAD）。"""

    seq = [float(value) for value in values]
    center = median(seq)
    return 1.4826 * float(statistics.median([abs(value - center) for value in seq]))


def theil_sen_slope(values: list[float]) -> float:
    """返回序列对序号 0..n-1 的 Theil–Sen 稳健斜率。"""

    seq = [float(value) for value in values]
    count = len(seq)
    if count < 2:
        return 0.0
    slopes = [(seq[j] - seq[i]) / (j - i) for i in range(count) for j in range(i + 1, count)]
    return float(statistics.median(slopes))

"""项目路径与情感分类的公共规则。"""
from pathlib import Path
import os

# 数据路径可指向原项目，代码及配置保留在本工程中。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AMAZON_PROJECT_ROOT", PACKAGE_ROOT)).expanduser().resolve()
CONFIG_DIR = PACKAGE_ROOT / "config"


def classify_scalar(score, threshold=0.1):
    """严格大于/小于阈值；边界归为接近中性。"""
    if score > threshold:
        return "Positive"
    if score < -threshold:
        return "Negative"
    return "Near-neutral"


def classify_series(scores, threshold=0.1):
    """保留索引，与CSV round_trip读取配合使用。"""
    import pandas as pd
    labels = pd.Series("Near-neutral", index=scores.index)
    labels.loc[scores > threshold] = "Positive"
    labels.loc[scores < -threshold] = "Negative"
    return labels

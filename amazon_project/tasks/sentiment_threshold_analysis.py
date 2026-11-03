"""任务22：sentiment_threshold_analysis。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from amazon_project.common import classify_series

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    BASE_DIR = PROJECT_ROOT
    INPUT_FILE = (
        BASE_DIR / "data_clean" / "review_sentiment_unique_texts.csv"
    )

    # 单独保存第二周修改结果
    TABLE_DIR = (
        BASE_DIR / "reports" / "week2" / "revision" / "tables"
    )
    FIGURE_DIR = (
        BASE_DIR / "reports" / "week2" / "revision" / "figures"
    )

    for folder in (TABLE_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 保留CSV中浮点数的精度，避免边界得分读取后改变分类
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"text_id": "string"},
        float_precision="round_trip",
    )

    scores = pd.to_numeric(df["polarity"], errors="coerce")

    invalid = scores.isna() | ~scores.between(-1, 1)
    if invalid.any():
        raise ValueError(
            f"发现 {invalid.sum()} 条缺失或异常情感得分，请先检查。"
        )

    if df.empty:
        raise ValueError("输入文件没有数据。")

    if df["text_id"].isna().any() or df["text_id"].duplicated().any():
        raise ValueError("text_id存在缺失或重复，请先检查。")

    df["polarity"] = scores
    n = len(df)

    LABELS = ["Negative", "Near-neutral", "Positive"]
    COLORS = {
        "Negative": "#D65F5F",
        "Near-neutral": "#999999",
        "Positive": "#59A14F",
    }


    def classify(threshold):
        return classify_series(scores, threshold)


    # 1. 比较0至0.30之间的分类阈值
    thresholds = [i / 100 for i in range(0, 31, 2)]
    rows = []

    for threshold in thresholds:
        counts = classify(threshold).value_counts()

        for label in LABELS:
            count = int(counts.get(label, 0))
            rows.append({
                "threshold": threshold,
                "sentiment_label": label,
                "text_count": count,
                "share_percent": count / n * 100,
            })

    sensitivity = pd.DataFrame(rows)

    sensitivity.to_csv(
        TABLE_DIR / "sentiment_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. 核对阈值从0.1变为0.2时，哪些分类发生变化
    labels_01 = classify(0.1)
    labels_02 = classify(0.2)

    transition = pd.crosstab(
        labels_01,
        labels_02,
    ).reindex(
        index=LABELS,
        columns=LABELS,
        fill_value=0,
    )

    transition.index.name = "label_at_0.1"
    transition.columns.name = "label_at_0.2"

    transition.to_csv(
        TABLE_DIR / "sentiment_transition_0.1_to_0.2.csv",
        encoding="utf-8-sig",
    )

    positive_to_neutral = int(
        (
            (labels_01 == "Positive")
            & (labels_02 == "Near-neutral")
        ).sum()
    )

    negative_to_neutral = int(
        (
            (labels_01 == "Negative")
            & (labels_02 == "Near-neutral")
        ).sum()
    )

    # 根据原始数量计算，避免用已四舍五入的百分比相减
    positive_share_drop = positive_to_neutral / n * 100

    # 3. 保存得分概况，辅助解释分类结果
    score_summary = (
        scores.describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        .rename_axis("statistic")
        .reset_index(name="polarity")
    )

    score_summary.to_csv(
        TABLE_DIR / "sentiment_score_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4. 得分分布图
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.hist(
        scores,
        bins=40,
        range=(-1, 1),
        color="#4C78A8",
        edgecolor="white",
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
        label="Zero",
    )

    ax.axvline(
        0.1,
        color="#F28E2B",
        linestyle="--",
        linewidth=1.5,
        label="Thresholds: +/-0.1",
    )
    ax.axvline(
        -0.1,
        color="#F28E2B",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        0.2,
        color="#9467BD",
        linestyle=":",
        linewidth=1.8,
        label="Thresholds: +/-0.2",
    )
    ax.axvline(
        -0.2,
        color="#9467BD",
        linestyle=":",
        linewidth=1.8,
    )

    ax.set_title(
        f"Sentiment score distribution "
        f"(n={n:,} unique text cells)"
    )
    ax.set_xlabel("TextBlob polarity")
    ax.set_ylabel("Number of text cells")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "15_sentiment_score_distribution.png",
        dpi=200,
    )
    plt.close(fig)

    # 5. 分类占比随阈值变化的图
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label in LABELS:
        part = sensitivity.loc[
            sensitivity["sentiment_label"] == label
        ]

        ax.plot(
            part["threshold"],
            part["share_percent"],
            marker="o",
            markersize=4,
            color=COLORS[label],
            label=label,
        )

    ax.set_title(
        "Sensitivity of sentiment classification to threshold"
    )
    ax.set_xlabel("Symmetric threshold t")
    ax.set_ylabel("Share of text cells (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.text(
        0.5,
        0.015,
        "Positive: score > t | Negative: score < -t | "
        "Near-neutral: -t <= score <= t",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        FIGURE_DIR / "16_sentiment_threshold_sensitivity.png",
        dpi=200,
    )
    plt.close(fig)

    # 6. 输出核对结果
    print("=== 情感得分检查 ===")
    print(f"文本数量：{n:,}")
    print(f"平均得分：{scores.mean():.4f}")
    print(f"得分中位数：{scores.median():.4f}")

    print("\n=== 三个主要阈值的分类结果 ===")
    selected = sensitivity.loc[
        sensitivity["threshold"].isin([0.0, 0.1, 0.2])
    ]
    print(
        selected.round({"share_percent": 2}).to_string(index=False)
    )

    print("\n=== 阈值从0.1提高到0.2的分类变化 ===")
    print(transition.to_string())

    print(f"\n正向转为接近中性：{positive_to_neutral}份")
    print(f"负向转为接近中性：{negative_to_neutral}份")
    print(f"正向占比下降：{positive_share_drop:.2f}个百分点")

    print("\n统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print(
        "完成。本步骤生成的修改版统计表和图片已更新，"
        "输入数据及此前的第二周结果未修改。"
    )

if __name__ == "__main__":
    run()

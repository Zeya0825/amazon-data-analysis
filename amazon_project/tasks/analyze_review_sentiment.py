"""任务20：analyze_review_sentiment。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from amazon_project.common import classify_scalar as classify
    from importlib.metadata import version
    import json

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from textblob import TextBlob
    from textblob.sentiments import PatternAnalyzer

    BASE_DIR = PROJECT_ROOT
    CLEAN_DIR = BASE_DIR / "data_clean"
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "figures"

    for folder in (CLEAN_DIR, TABLE_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 一、读取文本，去除空文本和完全重复的清理后文本
    df = pd.read_csv(
        CLEAN_DIR / "product_review_text_analysis.csv",
        dtype={
            "product_id": "string",
            "review_text_clean": "string"
        }
    )

    df["review_text_clean"] = (
        df["review_text_clean"].fillna("").str.strip()
    )
    df["source_row"] = pd.to_numeric(
        df["source_row"], errors="raise"
    )

    texts = (
        df.loc[df["review_text_clean"].ne("")]
        .sort_values("source_row")
        .drop_duplicates("review_text_clean", keep="first")
        .copy()
        .reset_index(drop=True)
    )

    if texts.empty:
        raise ValueError("没有可分析的文本。")

    # 商品编号用于追溯样例，不代表文本只属于这一个商品
    texts["text_id"] = [
        f"T{i:04d}" for i in range(1, len(texts) + 1)
    ]

    # 二、对完整清理后文本评分
    # 不使用关键词步骤中过滤停用词后的结果，保留否定词和上下文
    analyzer = PatternAnalyzer()
    polarities = []

    print(f"开始分析 {len(texts):,} 份不同文本……")

    for i, text in enumerate(texts["review_text_clean"], start=1):
        sentiment = TextBlob(text, analyzer=analyzer).sentiment
        polarities.append(float(sentiment.polarity))

        if i % 200 == 0:
            print(f"已分析 {i:,} 份")

    texts["polarity"] = polarities
    assert texts["polarity"].between(-1, 1).all()


    texts["sentiment_label"] = texts["polarity"].map(classify)

    # 三、保存完整分析数据，仅在本地使用
    texts.to_csv(
        CLEAN_DIR / "review_sentiment_unique_texts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 四、分类统计
    label_order = ["Negative", "Near-neutral", "Positive"]

    counts = (
        texts["sentiment_label"]
        .value_counts()
        .reindex(label_order, fill_value=0)
    )

    summary = pd.DataFrame({
        "sentiment_label": label_order,
        "text_count": counts.to_numpy()
    })
    summary["share_percent"] = (
        summary["text_count"] / len(texts) * 100
    )

    assert summary["text_count"].sum() == len(texts)

    summary.to_csv(
        TABLE_DIR / "review_sentiment_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    score_summary = (
        texts["polarity"]
        .describe()
        .rename_axis("statistic")
        .reset_index(name="polarity")
    )
    score_summary.to_csv(
        TABLE_DIR / "review_sentiment_score_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 五、检查分界变化对分类数量的影响
    # 这不是准确率评估，只是展示结果对分类规则的依赖
    threshold_rows = []

    for threshold in [0.0, 0.1, 0.2]:
        labels = texts["polarity"].map(
            lambda score: classify(score, threshold)
        )
        for label in label_order:
            count = int(labels.eq(label).sum())
            threshold_rows.append({
                "threshold": threshold,
                "sentiment_label": label,
                "text_count": count,
                "share_percent": count / len(texts) * 100
            })

    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(
        TABLE_DIR / "sentiment_threshold_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 保存实际方法和版本
    settings = {
        "textblob_version": version("textblob"),
        "analyzer": "PatternAnalyzer",
        "analysis_unit": "unique_cleaned_review_text_cell",
        "text_count": len(texts),
        "positive_rule": "polarity > 0.1",
        "negative_rule": "polarity < -0.1",
        "near_neutral_rule": "-0.1 <= polarity <= 0.1",
        "manual_sample_rule": "up to 3 random texts per predicted class",
        "random_seed": 42
    }
    (TABLE_DIR / "sentiment_method.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 六、生成供人工阅读的抽查样本
    # 每类最多抽3份；没有该类就不抽
    sample_parts = []

    for label in label_order:
        group = texts.loc[texts["sentiment_label"].eq(label)]
        if not group.empty:
            sample_parts.append(
                group.sample(n=min(3, len(group)), random_state=42)
            )

    sample = pd.concat(sample_parts, ignore_index=True)

    # 完整样本文本只保存到本地data_clean
    sample[[
        "text_id", "product_id", "source_row",
        "polarity", "sentiment_label", "review_text_clean"
    ]].to_csv(
        CLEAN_DIR / "sentiment_manual_review_texts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 人工判断表不附整份文本，之后填写判断和理由
    review_sheet = sample[[
        "text_id", "product_id", "source_row",
        "polarity", "sentiment_label"
    ]].copy()
    review_sheet["human_label"] = ""
    review_sheet["agreement"] = ""
    review_sheet["notes"] = ""

    review_path = TABLE_DIR / "sentiment_manual_review.csv"

    # 避免重跑脚本时覆盖已经填写的人工判断
    if not review_path.exists():
        review_sheet.to_csv(
            review_path, index=False, encoding="utf-8-sig"
        )
    else:
        print("已有人工抽查表，保留原文件，没有覆盖。")

    # 七、绘制分类图
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        summary["sentiment_label"],
        summary["text_count"],
        color=["#E15759", "#BAB0AC", "#59A14F"]
    )

    for bar, share in zip(bars, summary["share_percent"]):
        ax.annotate(
            f"{int(bar.get_height()):,}\n({share:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )

    ax.set_title(
        f"TextBlob sentiment of unique text cells (n={len(texts):,})"
    )
    ax.set_ylabel("Number of text cells")
    ax.set_ylim(0, max(summary["text_count"].max() * 1.22, 1))
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Thresholds: -0.1 and +0.1. "
        "Automated text labels, not customer satisfaction rates.",
        ha="center",
        fontsize=9
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        FIGURE_DIR / "14_review_sentiment_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 八、打印结果
    print("\n=== 自动情感分类结果 ===")
    print(summary.round(2).to_string(index=False))

    print("\n=== 情感得分统计 ===")
    print(score_summary.round(4).to_string(index=False))

    print("\n=== 不同分类分界的结果 ===")
    print(threshold_table.round(2).to_string(index=False))

    print("\n=== 待人工抽查样本 ===")
    print(sample[[
        "text_id", "product_id", "polarity", "sentiment_label"
    ]].round(4).to_string(index=False))

    print("\n人工抽查尚未完成，不能据此报告模型准确率。")
    print("图片：14_review_sentiment_distribution.png")
    print("统计表位置：", TABLE_DIR)
    print("人工阅读文本：", CLEAN_DIR / "sentiment_manual_review_texts.csv")
    print("人工判断表：", review_path)
    print("完成，输入数据没有修改。")

if __name__ == "__main__":
    run()

"""任务24：inspect_threshold_texts。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from amazon_project.common import classify_series

    import pandas as pd


    BASE_DIR = PROJECT_ROOT
    INPUT_FILE = (
        BASE_DIR / "data_clean" / "review_sentiment_unique_texts.csv"
    )

    TABLE_DIR = (
        BASE_DIR / "reports" / "week2" / "revision" / "tables"
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "text_id": "string",
            "product_id": "string",
            "review_text_clean": "string",
        },
        float_precision="round_trip",
    )

    required = {
        "text_id", "product_id", "polarity", "review_text_clean"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}")

    scores = pd.to_numeric(df["polarity"], errors="coerce")

    if df.empty or scores.isna().any() or not scores.between(-1, 1).all():
        raise ValueError("情感得分为空、缺失或超出范围。")

    if df["text_id"].isna().any() or df["text_id"].duplicated().any():
        raise ValueError("文本编号存在缺失或重复。")

    df["polarity"] = scores


    def classify(threshold):
        return classify_series(scores, threshold)


    df["label_at_0.1"] = classify(0.1)
    df["label_at_0.2"] = classify(0.2)

    positive_changed = df.loc[
        (df["label_at_0.1"] == "Positive")
        & (df["label_at_0.2"] == "Near-neutral")
    ].copy()

    negative_changed = df.loc[
        (df["label_at_0.1"] == "Negative")
        & (df["label_at_0.2"] == "Near-neutral")
    ].copy()

    # 在正向变中性的文本中，按得分排序选取5个不同位置：
    # 最低、约25%、约50%、约75%、最高。
    # 这是解释性样例，不是用于估计准确率的随机样本。
    ordered = positive_changed.sort_values(
        ["polarity", "text_id"]
    )

    if ordered.empty:
        positive_examples = ordered.copy()
    else:
        positions = sorted({
            round((len(ordered) - 1) * fraction)
            for fraction in [0, 0.25, 0.5, 0.75, 1]
        })
        positive_examples = ordered.iloc[positions].copy()

    positive_examples["selection_reason"] = (
        "score-spread examples: Positive to Near-neutral"
    )

    # 负向变中性的文本数量少，全部保留
    negative_examples = negative_changed.sort_values(
        ["polarity", "text_id"]
    ).copy()
    negative_examples["selection_reason"] = (
        "all Negative to Near-neutral cases"
    )

    examples = pd.concat(
        [positive_examples, negative_examples],
        ignore_index=True,
    )

    columns = [
        "text_id",
        "product_id",
        "polarity",
        "label_at_0.1",
        "label_at_0.2",
        "selection_reason",
        "review_text_clean",
    ]

    # 含完整文本的样例留在data_clean，便于阅读
    examples_path = (
        BASE_DIR / "data_clean" / "threshold_text_examples.csv"
    )
    examples[columns].to_csv(
        examples_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 提交结果目录只保存样例编号和得分等信息
    examples[
        [c for c in columns if c != "review_text_clean"]
    ].to_csv(
        TABLE_DIR / "threshold_example_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== 分类变化核对 ===")
    print(f"正向变为接近中性：{len(positive_changed)}份")
    print(f"负向变为接近中性：{len(negative_changed)}份")
    print(f"本次选取阅读样例：{len(examples)}份")

    print("\n=== 样例编号与得分 ===")
    print(
        examples[
            ["text_id", "polarity", "label_at_0.1", "label_at_0.2"]
        ].to_string(index=False)
    )

    print("\n=== 样例文本预览（每份最多800字符）===")
    for _, row in examples.iterrows():
        text = row["review_text_clean"]
        text = "" if pd.isna(text) else str(text)

        print("\n文本编号：", row["text_id"])
        print("情感得分：", repr(row["polarity"]))
        print("分类变化：", row["label_at_0.1"], "->", row["label_at_0.2"])
        print(text[:800])

        if len(text) > 800:
            print("【预览已截断，完整文本保存在样例CSV中】")

    print("\n完整文本样例：", examples_path)
    print("样例索引：", TABLE_DIR / "threshold_example_index.csv")
    print("完成。没有修改输入数据。")

if __name__ == "__main__":
    run()

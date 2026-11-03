"""任务18：analyze_review_length。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import re
    import html

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BASE_DIR = PROJECT_ROOT
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "figures"
    CLEAN_DIR = BASE_DIR / "data_clean"

    for folder in (TABLE_DIR, FIGURE_DIR, CLEAN_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 一、取回每个商品对应的原始评论文本
    raw = pd.read_csv(
        BASE_DIR / "data_raw" / "sales" / "amazon.csv",
        dtype="string"
    )
    raw["source_row"] = range(1, len(raw) + 1)

    products = pd.read_csv(
        CLEAN_DIR / "products_unique.csv",
        dtype={"product_id": "string"}
    )
    products["source_row"] = pd.to_numeric(
        products["source_row"], errors="raise"
    ).astype(int)

    assert products["product_id"].is_unique

    df = products[
        ["source_row", "product_id", "main_category", "rating"]
    ].merge(
        raw[["source_row", "product_id", "review_content"]],
        on=["source_row", "product_id"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    assert df["_merge"].eq("both").all(), "部分商品没有匹配到原始行。"
    df = df.drop(columns="_merge")

    # 二、清理正文
    # 不按逗号拆分，不删除否定词，保留后续情感分析所需的信息
    def clean_text(value):
        if pd.isna(value):
            return ""

        text = html.unescape(str(value))
        # 逗号可能连接下一条评论，所以网址匹配在逗号处停止
        text = re.sub(
            r"https?://[^\s,<>]+", " ", text, flags=re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["review_text_clean"] = df["review_content"].map(clean_text)
    df["text_character_count"] = df["review_text_clean"].str.len()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    nonempty = df["review_text_clean"].ne("")
    valid_rating = df["rating"].between(1, 5)

    # 商品层面保留每个商品；不因文本重复而删除商品
    paired = df.loc[nonempty & valid_rating].copy()

    if len(paired) < 2:
        raise ValueError("可用于分析的商品数量不足。")

    # 三、比较两种统计口径
    # 第二种只作检查：相同清理后文本保留原始行号最小的商品
    unique_text_pairs = (
        paired.sort_values("source_row")
        .drop_duplicates("review_text_clean", keep="first")
    )

    def correlation_summary(data, name):
        x = data["text_character_count"]
        y = data["rating"]

        # 对排名计算普通相关系数，得到Spearman相关系数
        # 无需额外安装scipy
        if len(data) < 2 or x.nunique() < 2 or y.nunique() < 2:
            coefficient = float("nan")
        else:
            coefficient = x.rank().corr(y.rank())

        return {
            "analysis_scope": name,
            "product_count": len(data),
            "spearman_correlation": coefficient
        }

    correlations = pd.DataFrame([
        correlation_summary(paired, "all_valid_products"),
        correlation_summary(
            unique_text_pairs,
            "unique_cleaned_text_first_source_row"
        )
    ])

    correlations.to_csv(
        TABLE_DIR / "review_length_rating_correlation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 四、清理后文本的长度统计
    length_summary = (
        df.loc[nonempty, "text_character_count"]
        .describe()
        .rename_axis("statistic")
        .reset_index(name="character_count")
    )
    length_summary.to_csv(
        TABLE_DIR / "cleaned_review_text_length_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    checks = pd.DataFrame([{
        "total_products": len(df),
        "nonempty_cleaned_text": int(nonempty.sum()),
        "empty_cleaned_text": int((~nonempty).sum()),
        "unique_nonempty_cleaned_text": int(
            df.loc[nonempty, "review_text_clean"].nunique()
        ),
        "valid_length_rating_pairs": len(paired),
    }])
    checks.to_csv(
        TABLE_DIR / "review_length_analysis_checks.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 五、保存本地分析数据，后续关键词和情感分析会用到
    # 这是处理后的数据集，不放进交给老师的代码与结果压缩包
    df.drop(columns="review_content").to_csv(
        CLEAN_DIR / "product_review_text_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 六、画图
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 5), squeeze=False
    )
    ax_length, ax_relation = axes.ravel()

    ax_length.hist(
        df.loc[nonempty, "text_character_count"],
        bins=30,
        color="#4C78A8",
        edgecolor="white"
    )
    ax_length.set_title(
        f"Text length by product (n={int(nonempty.sum()):,})"
    )
    ax_length.set_xlabel("Characters in cleaned review-text cell")
    ax_length.set_ylabel("Number of products")

    ax_relation.scatter(
        paired["text_character_count"],
        paired["rating"],
        s=18,
        alpha=0.3,
        color="#59A14F"
    )
    ax_relation.set_xscale("log")
    ax_relation.set_ylim(1, 5.1)
    ax_relation.set_title(
        f"Text length and product rating (n={len(paired):,})"
    )
    ax_relation.set_xlabel("Text characters (log scale)")
    ax_relation.set_ylabel("Product rating")

    for ax in (ax_length, ax_relation):
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Each text cell may contain multiple reviews; "
        "ratings are product-level.",
        ha="center",
        fontsize=10
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        FIGURE_DIR / "11_review_text_length_and_rating.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("=== 评论文本清理与配对检查 ===")
    print(checks.to_string(index=False))

    print("\n=== 清理后整格评论文本的字符数 ===")
    print(length_summary.round(2).to_string(index=False))

    print("\n=== 文本长度与商品评分的相关性 ===")
    print(correlations.round(4).to_string(index=False))

    print("\n图片：11_review_text_length_and_rating.png")
    print("统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print(
        "本地分析数据：",
        CLEAN_DIR / "product_review_text_analysis.csv"
    )
    print("完成，原始文件和第一周清洗文件没有修改。")

if __name__ == "__main__":
    run()

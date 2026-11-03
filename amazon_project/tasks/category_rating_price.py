"""任务25：category_rating_price。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    BASE_DIR = PROJECT_ROOT
    INPUT_FILE = BASE_DIR / "data_clean" / "products_unique.csv"

    TABLE_DIR = BASE_DIR / "reports" / "week2" / "revision" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "revision" / "figures"

    for folder in (TABLE_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"product_id": "string", "category": "string"},
        float_precision="round_trip",
    )

    required = {"product_id", "category", "rating", "discounted_price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}")

    if df.empty:
        raise ValueError("商品表为空。")

    if df["product_id"].isna().any() or df["product_id"].duplicated().any():
        raise ValueError("商品编号缺失或不唯一，请先检查。")

    # 从完整类别路径中提取第一级类别
    df["main_category"] = (
        df["category"]
        .str.split("|", regex=False)
        .str[0]
        .str.strip()
        .fillna("Unknown")
        .replace("", "Unknown")
    )

    # 保留有效评分，缺失或异常值不参与评分统计
    rating = pd.to_numeric(df["rating"], errors="coerce")
    price = pd.to_numeric(df["discounted_price"], errors="coerce")

    invalid_rating = rating.notna() & ~rating.between(1, 5)
    invalid_price = price.notna() & (
        (price < 0) | price.isin([float("inf"), float("-inf")])
    )

    if invalid_rating.any() or invalid_price.any():
        raise ValueError(
            f"发现异常评分{int(invalid_rating.sum())}条，"
            f"异常价格{int(invalid_price.sum())}条，请先检查。"
        )

    df["rating_numeric"] = rating
    df["price_numeric"] = price

    category_order = (
        df["main_category"].value_counts().index.tolist()
    )

    # 1. 各类别评分概况
    rating_rows = []

    for category in category_order:
        part = df.loc[df["main_category"] == category]
        valid = part["rating_numeric"].dropna()
        valid_n = len(valid)
        high_n = int((valid >= 4).sum())

        rating_rows.append({
            "main_category": category,
            "product_count": len(part),
            "valid_rating_count": valid_n,
            "missing_or_non_numeric_rating_count": len(part) - valid_n,
            "mean_product_rating": valid.mean(),
            "median_product_rating": valid.median(),
            "min_product_rating": valid.min(),
            "max_product_rating": valid.max(),
            "rating_ge_4_count": high_n,
            "rating_ge_4_percent": (
                high_n / valid_n * 100 if valid_n else float("nan")
            ),
        })

    rating_summary = pd.DataFrame(rating_rows)

    rating_summary.to_csv(
        TABLE_DIR / "category_rating_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. 各类别评分分组
    # 与第二周原来的评分分组保持一致
    GROUPS = ["1 to <2", "2 to <3", "3 to <4", "4 to <4.5", "4.5 to 5"]


    def rating_group(value):
        if pd.isna(value):
            return pd.NA
        if value < 2:
            return GROUPS[0]
        if value < 3:
            return GROUPS[1]
        if value < 4:
            return GROUPS[2]
        if value < 4.5:
            return GROUPS[3]
        return GROUPS[4]


    df["rating_group"] = df["rating_numeric"].map(rating_group)

    distribution_rows = []

    for category in category_order:
        part = df.loc[df["main_category"] == category]
        valid_n = int(part["rating_numeric"].notna().sum())
        counts = part["rating_group"].value_counts()

        for group in GROUPS:
            count = int(counts.get(group, 0))
            distribution_rows.append({
                "main_category": category,
                "rating_group": group,
                "product_count": count,
                "valid_rating_count": valid_n,
                "share_percent": (
                    count / valid_n * 100 if valid_n else float("nan")
                ),
            })

    distribution = pd.DataFrame(distribution_rows)

    distribution.to_csv(
        TABLE_DIR / "category_rating_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3. 各类别价格表，使用折后价格
    price_rows = []

    for category in category_order:
        part = df.loc[df["main_category"] == category]
        valid = part["price_numeric"].dropna()

        price_rows.append({
            "main_category": category,
            "product_count": len(part),
            "valid_price_count": len(valid),
            "missing_or_non_numeric_price_count": len(part) - len(valid),
            "min_price_inr": valid.min(),
            "q1_price_inr": valid.quantile(0.25),
            "median_price_inr": valid.median(),
            "mean_price_inr": valid.mean(),
            "q3_price_inr": valid.quantile(0.75),
            "max_price_inr": valid.max(),
        })

    price_summary = pd.DataFrame(price_rows)

    price_summary.to_csv(
        TABLE_DIR / "category_price_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4. 核对统计总数
    if int(rating_summary["product_count"].sum()) != len(df):
        raise ValueError("类别商品总数核对失败。")

    if int(distribution["product_count"].sum()) != int(rating.notna().sum()):
        raise ValueError("评分分组总数核对失败。")

    # 5. 三个商品数量最多的类别：评分分布图
    # 全部类别仍保留在CSV中，避免小样本类别挤占图表
    main_categories = category_order[:3]
    colors = ["#BAB0AC", "#E15759", "#F28E2B", "#4E79A7", "#59A14F"]

    fig, axes = plt.subplots(
        1, len(main_categories),
        figsize=(15, 5.5),
        sharey=True,
        squeeze=False,
    )

    for ax, category in zip(axes[0], main_categories):
        part = (
            distribution.loc[distribution["main_category"] == category]
            .set_index("rating_group")
            .reindex(GROUPS)
        )
        valid_n = int(part["valid_rating_count"].iloc[0])
        shares = part["share_percent"].fillna(0).tolist()

        ax.bar(range(len(GROUPS)), shares, color=colors)

        for position, (_, row) in enumerate(part.iterrows()):
            if valid_n:
                label = f'{int(row["product_count"])}\n({row["share_percent"]:.1f}%)'
                ax.text(
                    position,
                    shares[position] + 1,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(f"{category}\nValid ratings: {valid_n}")
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS, rotation=30, ha="right")
        ax.set_ylim(0, 110)
        ax.set_xlabel("Product rating")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Share within valid ratings (%)")
    fig.suptitle("Product rating distributions by main category")
    fig.text(
        0.5, 0.02,
        "Each product counts once; missing ratings are excluded. "
        "Labels show product counts and within-category percentages.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(
        FIGURE_DIR / "17_rating_distribution_by_category.png",
        dpi=200,
    )
    plt.close(fig)

    # 6. 将完整类别价格表保存成图片，方便放入报告
    display = price_summary[
        [
            "main_category",
            "valid_price_count",
            "min_price_inr",
            "q1_price_inr",
            "median_price_inr",
            "mean_price_inr",
            "q3_price_inr",
            "max_price_inr",
        ]
    ].copy()

    display.columns = [
        "Category", "Valid n", "Min", "Q1",
        "Median", "Mean", "Q3", "Max",
    ]

    display["Valid n"] = display["Valid n"].astype(str)

    for column in ["Min", "Q1", "Median", "Mean", "Q3", "Max"]:
        display[column] = display[column].map(
            lambda value: f"{value:,.2f}" if pd.notna(value) else "-"
        )

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    ax.set_title(
        "Discounted price by category (INR)",
        fontsize=15,
        pad=18,
    )

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.26, 0.07] + [0.11] * 6,
        bbox=[0, 0.14, 1, 0.78],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#315A7D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F4F8")

        if column == 0 and row > 0:
            cell.set_text_props(ha="left")

    fig.text(
        0.5, 0.055,
        "Q1/Q3: 25th/75th percentiles. "
        "Categories with very few products are descriptive only.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(
        FIGURE_DIR / "18_category_price_table.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 7. 输出结果
    print("=== 记录核对 ===")
    print("商品总数：", len(df))
    print("类别数量：", len(category_order))
    print("有效评分数量：", int(rating.notna().sum()))
    print("评分缺失或非数字数量：", int(rating.isna().sum()))
    print("有效价格数量：", int(price.notna().sum()))

    print("\n=== 各类别评分统计 ===")
    print(rating_summary.round(3).to_string(index=False))

    print("\n=== 各类别价格统计（INR）===")
    print(price_summary.round(2).to_string(index=False))

    print("\n已生成图片：")
    print("17_rating_distribution_by_category.png")
    print("18_category_price_table.png")
    print("\n统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print("完成。输入数据和原有第二周结果没有修改。")

if __name__ == "__main__":
    run()

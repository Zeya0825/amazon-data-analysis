"""任务15：analyze_ratings_discounts。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 一、设置路径
    BASE_DIR = PROJECT_ROOT
    INPUT_PATH = BASE_DIR / "data_clean" / "products_unique.csv"
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "figures"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 二、读取每商品一行的分析表
    df = pd.read_csv(INPUT_PATH, dtype={"product_id": "string"})

    assert df["product_id"].notna().all(), "发现缺失商品编号，请检查。"
    assert df["product_id"].is_unique, "商品编号不唯一，请检查输入文件。"

    rating = pd.to_numeric(df["rating"], errors="coerce")
    discount = pd.to_numeric(
        df["discount_percentage"], errors="coerce"
    )

    # 缺失值不填成0，超出范围的值单独计数
    rating_outside = rating.notna() & ~rating.between(1, 5)
    discount_outside = discount.notna() & ~discount.between(0, 1)

    valid_rating = rating[rating.between(1, 5)].copy()
    valid_discount = discount[discount.between(0, 1)].copy()

    if valid_rating.empty or valid_discount.empty:
        raise ValueError("有效评分或折扣为空，请检查输入文件。")

    # 三、保存有效数量、缺失数量和范围检查结果
    coverage = pd.DataFrame([
        {
            "field": "rating",
            "total_products": len(df),
            "valid_products": len(valid_rating),
            "missing_or_non_numeric": int(rating.isna().sum()),
            "outside_range": int(rating_outside.sum()),
        },
        {
            "field": "discount_percentage",
            "total_products": len(df),
            "valid_products": len(valid_discount),
            "missing_or_non_numeric": int(discount.isna().sum()),
            "outside_range": int(discount_outside.sum()),
        },
    ])

    coverage.to_csv(
        TABLE_DIR / "rating_discount_coverage.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 四、基本统计
    # 折扣转成百分数，例如0.60转成60
    discount_percent = valid_discount * 100

    summary = pd.DataFrame({
        "product_rating": valid_rating.describe(),
        "discount_percent": discount_percent.describe(),
    })
    summary.index.name = "statistic"

    summary.to_csv(
        TABLE_DIR / "rating_discount_statistics.csv",
        encoding="utf-8-sig"
    )

    # 五、评分分组
    # 左闭右开：[1,2)、[2,3)、[3,4)、[4,4.5)、[4.5,5]
    rating_labels = [
        "1 to <2",
        "2 to <3",
        "3 to <4",
        "4 to <4.5",
        "4.5 to 5",
    ]

    rating_group = pd.cut(
        valid_rating,
        bins=[1, 2, 3, 4, 4.5, np.inf],
        labels=rating_labels,
        right=False
    )

    rating_counts = (
        rating_group.value_counts(sort=False)
        .reindex(rating_labels, fill_value=0)
    )

    rating_table = pd.DataFrame({
        "rating_group": rating_labels,
        "product_count": rating_counts.to_numpy(),
    })
    rating_table["share_percent"] = (
        rating_table["product_count"] / len(valid_rating) * 100
    )

    assert rating_table["product_count"].sum() == len(valid_rating)

    rating_table.to_csv(
        TABLE_DIR / "product_rating_groups.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 六、评分分组图
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        rating_table["rating_group"],
        rating_table["product_count"],
        color="#4C78A8"
    )

    for bar, share in zip(bars, rating_table["share_percent"]):
        ax.annotate(
            f"{int(bar.get_height()):,}\n({share:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        f"Product rating groups (valid n={len(valid_rating):,})"
    )
    ax.set_xlabel("Product rating group")
    ax.set_ylabel("Number of products")
    ax.set_ylim(0, max(rating_counts.max() * 1.22, 1))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "07_product_rating_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 七、折扣分组统计
    discount_labels = [
        "0 to <20%",
        "20 to <40%",
        "40 to <60%",
        "60 to <80%",
        "80 to 100%",
    ]

    discount_group = pd.cut(
        valid_discount,
        bins=[0, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=discount_labels,
        right=False
    )

    discount_counts = (
        discount_group.value_counts(sort=False)
        .reindex(discount_labels, fill_value=0)
    )

    discount_table = pd.DataFrame({
        "discount_group": discount_labels,
        "product_count": discount_counts.to_numpy(),
    })
    discount_table["share_percent"] = (
        discount_table["product_count"] / len(valid_discount) * 100
    )

    assert discount_table["product_count"].sum() == len(valid_discount)

    discount_table.to_csv(
        TABLE_DIR / "product_discount_groups.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 八、折扣分布图
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        discount_percent,
        bins=np.arange(0, 101, 10),
        color="#F28E2B",
        edgecolor="white"
    )

    median_discount = discount_percent.median()
    ax.axvline(
        median_discount,
        color="#B22222",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_discount:.1f}%"
    )

    ax.set_title(
        f"Discount distribution (valid n={len(valid_discount):,})"
    )
    ax.set_xlabel("Price reduction (%)")
    ax.set_ylabel("Number of products")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "08_discount_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 九、输出结果
    high_rating_count = int((valid_rating >= 4).sum())
    high_rating_share = high_rating_count / len(valid_rating) * 100

    pd.DataFrame([{
        "threshold": 4,
        "valid_rating_products": len(valid_rating),
        "products_rating_ge_4": high_rating_count,
        "share_percent": high_rating_share,
    }]).to_csv(
        TABLE_DIR / "high_rating_product_share.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=== 有效数据与缺失检查 ===")
    print(coverage.to_string(index=False))

    print("\n=== 评分与折扣基本统计 ===")
    print(summary.round(3).to_string())
    print("注意：discount_percent列使用百分数，50表示降价50%。")

    print("\n=== 商品评分分组 ===")
    print(rating_table.round(2).to_string(index=False))

    print(
        f"\n评分达到4分及以上的商品：{high_rating_count:,}个"
        f"，占有效评分商品的{high_rating_share:.2f}%"
    )

    print("\n=== 商品折扣分组 ===")
    print(discount_table.round(2).to_string(index=False))

    print("\n已生成图片：")
    print("07_product_rating_distribution.png")
    print("08_discount_distribution.png")
    print("统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print("完成。原始文件和第一周清洗文件没有修改。")

if __name__ == "__main__":
    run()

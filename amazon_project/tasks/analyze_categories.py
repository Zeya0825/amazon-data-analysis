"""任务13：analyze_categories。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_dir = PROJECT_ROOT
    week2_dir = project_dir / "reports" / "week2"
    tables_dir = week2_dir / "tables"
    figures_dir = week2_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 使用每商品一行的表，避免重复计算商品
    products = pd.read_csv(
        project_dir / "data_clean" / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    assert products["product_id"].is_unique, "商品编号不唯一，请检查输入文件。"

    # 缺失类别单独归为Unknown，避免统计时被漏掉
    products["main_category"] = (
        products["main_category"].astype("string").fillna("Unknown")
    )

    # 按一级类别统计
    summary = (
        products.groupby("main_category")
        .agg(
            product_count=("product_id", "size"),
            median_price_inr=("discounted_price", "median"),
            valid_rating_products=("rating", "count"),
            mean_product_rating=("rating", "mean"),
            median_discount_ratio=("discount_percentage", "median"),
        )
        .reset_index()
        .sort_values("product_count", ascending=False)
    )

    summary["product_share"] = (
        summary["product_count"] / len(products)
    )

    assert summary["product_count"].sum() == len(products)

    # 保存完整精度的统计结果
    summary.to_csv(
        tables_dir / "category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 图4：不同类别的商品数量
    plot_data = summary.sort_values("product_count", ascending=True)

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(plot_data) * 0.5))
    )

    bars = ax.barh(
        plot_data["main_category"],
        plot_data["product_count"],
        color="#4C78A8",
    )

    ax.bar_label(
        bars,
        labels=[
            f"{int(row.product_count):,} ({row.product_share:.1%})"
            for row in plot_data.itertuples()
        ],
        padding=5,
    )

    ax.set_xlim(0, plot_data["product_count"].max() * 1.25)
    ax.set_xlabel("Number of distinct products")
    ax.set_ylabel("")
    ax.set_title(
        f"Product categories in this dataset (n={len(products):,})"
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    figure_path = figures_dir / "04_product_category_counts.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=== 商品类别统计 ===")
    display = summary.copy()

    display["product_share_percent"] = (
        display["product_share"] * 100
    ).round(2)

    print(
        display[
            [
                "main_category",
                "product_count",
                "product_share_percent",
                "median_price_inr",
                "valid_rating_products",
                "mean_product_rating",
                "median_discount_ratio",
            ]
        ].round(3).to_string(index=False)
    )

    print("\n类别数量：", len(summary))
    print("各类别商品数量之和：", summary["product_count"].sum())
    print("统计表：", tables_dir / "category_summary.csv")
    print("图片：", figure_path)

if __name__ == "__main__":
    run()

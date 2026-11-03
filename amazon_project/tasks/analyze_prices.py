"""任务14：analyze_prices。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import numpy as np
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

    products = pd.read_csv(
        project_dir / "data_clean" / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    assert products["product_id"].is_unique

    products["discounted_price"] = pd.to_numeric(
        products["discounted_price"], errors="coerce"
    )
    prices = products["discounted_price"].dropna()

    # 1. 保存价格描述统计
    statistics = prices.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )

    statistics.rename_axis("statistic").reset_index(
        name="value_inr"
    ).to_csv(
        tables_dir / "price_descriptive_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. 全部商品价格分布
    # 左图为普通刻度；右图用对数刻度展示正价格
    positive_prices = prices[prices > 0]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.8),
        squeeze=False
    )
    axes = axes.ravel()

    axes[0].hist(
        prices, bins=30, color="#4C78A8", edgecolor="white"
    )
    axes[0].set_title(f"Price distribution: linear scale (n={len(prices):,})")
    axes[0].set_xlabel("Discounted price (INR)")
    axes[0].set_ylabel("Number of products")

    if not positive_prices.empty:
        low = positive_prices.min()
        high = positive_prices.max()

        if low == high:
            bins = [low / 1.1, high * 1.1]
        else:
            bins = np.geomspace(low, high, 31)

        axes[1].hist(
            positive_prices,
            bins=bins,
            color="#F28E2B",
            edgecolor="white",
        )
        axes[1].set_xscale("log")

    axes[1].set_title(
        f"Price distribution: log scale (n={len(positive_prices):,})"
    )
    axes[1].set_xlabel("Discounted price (INR, log scale)")
    axes[1].set_ylabel("Number of products")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "05_price_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. 比较商品数量最多的三个类别
    top_categories = (
        products["main_category"].value_counts().head(3).index.tolist()
    )

    price_groups = []
    labels = []
    category_rows = []

    for category in top_categories:
        values = products.loc[
            products["main_category"] == category,
            "discounted_price",
        ].dropna()

        # 对数坐标只显示正价格，明确记录被排除的数量
        positive = values[values > 0]
        price_groups.append(positive.to_numpy())
        labels.append(f"{category}\n(n={len(positive):,})")

        category_rows.append({
            "main_category": category,
            "valid_price_count": len(values),
            "non_positive_count": int((values <= 0).sum()),
            "median_price_inr": values.median(),
            "q1_price_inr": values.quantile(0.25),
            "q3_price_inr": values.quantile(0.75),
        })

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.boxplot(
        price_groups,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#A8C5DF"},
        medianprops={"color": "#D55E00", "linewidth": 2},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.4,
        },
    )

    ax.set_yscale("log")
    ax.set_ylabel("Discounted price (INR, log scale)")
    ax.set_title("Price spread in the three largest categories")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "06_price_by_main_category.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    category_summary = pd.DataFrame(category_rows)
    category_summary.to_csv(
        tables_dir / "main_category_price_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== 全部商品价格统计 ===")
    print(statistics.round(2).to_string())

    print("\n=== 对数价格图的范围检查 ===")
    print("缺失价格数量：", products["discounted_price"].isna().sum())
    print("非正价格数量（不进入对数图）：", int((prices <= 0).sum()))

    print("\n=== 三个主要类别的价格统计 ===")
    print(category_summary.round(2).to_string(index=False))

    print("\n已生成：")
    print("05_price_distribution.png")
    print("06_price_by_main_category.png")
    print("图片位置：", figures_dir)

if __name__ == "__main__":
    run()

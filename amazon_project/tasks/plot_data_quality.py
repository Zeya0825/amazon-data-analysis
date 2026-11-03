"""任务12：plot_data_quality。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd
    import matplotlib

    # 将图直接保存为文件，不弹出图窗
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    project_dir = PROJECT_ROOT
    week2_dir = project_dir / "reports" / "week2"
    tables_dir = week2_dir / "tables"
    figures_dir = week2_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 图中文字使用英文，避免不同电脑出现中文字体乱码
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })

    def save_figure(fig, filename):
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("已保存：", path.name)


    # 图1：相同1,465行数据清洗前后的缺失数量
    comparison = pd.read_csv(
        tables_dir / "missing_values_before_after.csv"
    )

    # 仅展示清洗前或清洗后存在缺失的字段
    selected = comparison[
        (comparison["raw_missing_count"] > 0)
        | (comparison["clean_missing_count"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = list(range(len(selected)))
    width = 0.35

    before = ax.bar(
        [i - width / 2 for i in x],
        selected["raw_missing_count"],
        width,
        label="Before cleaning",
        color="#4C78A8",
    )

    after = ax.bar(
        [i + width / 2 for i in x],
        selected["clean_missing_count"],
        width,
        label="After cleaning",
        color="#F28E2B",
    )

    ax.bar_label(before, padding=3, fmt="%.0f")
    ax.bar_label(after, padding=3, fmt="%.0f")

    ax.set_xticks(x)
    ax.set_xticklabels(selected["column_name"])
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values before and after cleaning\nSame 1,465 rows")
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, "01_missing_before_after.png")


    # 图2：按商品统计重复记录情况
    products = pd.read_csv(
        project_dir / "data_clean" / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    record_count = pd.to_numeric(
        products["source_record_count"], errors="raise"
    )

    # 兼容CSV读取为布尔值或文字的情况
    has_difference = (
        products["has_key_differences"]
        .astype("string")
        .str.lower()
        .eq("true")
        .fillna(False)
    )

    single_count = int((record_count == 1).sum())

    repeated_same = int(
        ((record_count > 1) & ~has_difference).sum()
    )

    repeated_different = int(
        ((record_count > 1) & has_difference).sum()
    )

    labels = [
        "One source record",
        "Multiple records:\nno key-field differences",
        "Multiple records:\nkey-field differences",
    ]
    counts = [single_count, repeated_same, repeated_different]

    assert sum(counts) == len(products)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(
        labels,
        counts,
        color=["#4C78A8", "#72B7B2", "#F28E2B"],
    )

    ax.bar_label(
        bars,
        labels=[
            f"{n:,} ({n / len(products):.1%})"
            for n in counts
        ],
        padding=5,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_xlabel("Number of distinct products")
    ax.set_title(f"Source-record status of {len(products):,} products")
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, "02_duplicate_product_status.png")


    # 图3：统计极端值占比
    outliers = pd.read_csv(
        tables_dir / "statistical_outliers_summary.csv"
    )

    outliers["flagged_percent"] = (
        outliers["flagged_rows"] / outliers["valid_rows"] * 100
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    bars = ax.bar(
        outliers["field"],
        outliers["flagged_percent"],
        color=["#4C78A8", "#F28E2B"],
        width=0.5,
    )

    ax.bar_label(
        bars,
        labels=[
            f"{int(row.flagged_rows)}/{int(row.valid_rows)}"
            f" ({row.flagged_percent:.1f}%)"
            for row in outliers.itertuples()
        ],
        padding=5,
    )

    ax.set_ylim(0, outliers["flagged_percent"].max() * 1.35)
    ax.set_ylabel("Flagged share of non-missing values (%)")
    ax.set_title("Values flagged by the 1.5 × IQR rule\nFlagged only — no deletion")
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, "03_statistical_outlier_share.png")

    print("\n完成，共生成3张数据质量图。")
    print("图片文件夹：", figures_dir)

if __name__ == "__main__":
    run()

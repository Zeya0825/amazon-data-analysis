"""任务16：analyze_historical_ratings。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from collections import Counter

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    from matplotlib.patches import Patch

    # 一、路径
    BASE_DIR = PROJECT_ROOT
    INPUT_PATH = BASE_DIR / "data_clean" / "ratings_unique.csv"
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "figures"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 二、分批统计
    year_parts = []
    score_counts = Counter()
    total_rows = 0
    invalid_rows = 0
    earliest = None
    latest = None

    print("开始分批统计历史评分，请等待完成提示……")

    reader = pd.read_csv(
        INPUT_PATH,
        usecols=["rating", "timestamp"],
        chunksize=200_000
    )

    for batch_no, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        rating = pd.to_numeric(chunk["rating"], errors="coerce")
        timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")

        dates = pd.to_datetime(
            timestamp, unit="s", utc=True, errors="coerce"
        )

        # 历史评分分布按1、2、3、4、5分统计
        valid = dates.notna() & rating.isin([1, 2, 3, 4, 5])
        invalid_rows += int((~valid).sum())

        if valid.any():
            valid_dates = dates.loc[valid]

            batch_min = valid_dates.min()
            batch_max = valid_dates.max()

            earliest = (
                batch_min if earliest is None
                else min(earliest, batch_min)
            )
            latest = (
                batch_max if latest is None
                else max(latest, batch_max)
            )

            part = pd.DataFrame({
                "year": valid_dates.dt.year,
                "rating": rating.loc[valid],
            })
            part["low_rating"] = part["rating"].le(2).astype(int)

            grouped = part.groupby("year").agg(
                rating_count=("rating", "size"),
                rating_sum=("rating", "sum"),
                low_rating_count=("low_rating", "sum")
            )
            year_parts.append(grouped)

            score_counts.update(
                part["rating"].astype(int).value_counts().to_dict()
            )

        if batch_no % 10 == 0:
            print(f"已处理 {total_rows:,} 行")

    if not year_parts:
        raise ValueError("没有可以分析的有效评分记录。")

    # 三、合并各批次结果
    annual = (
        pd.concat(year_parts)
        .groupby(level=0)
        .sum()
        .sort_index()
    )

    # 补齐范围内的年份；没有记录的年份数量为0、平均分为空
    annual = annual.reindex(
        range(earliest.year, latest.year + 1),
        fill_value=0
    )
    annual.index.name = "year"

    denominator = annual["rating_count"].replace(0, float("nan"))
    annual["avg_rating"] = annual["rating_sum"] / denominator
    annual["low_rating_ratio"] = (
        annual["low_rating_count"] / denominator
    )

    annual = annual.reset_index()

    # 标记没有覆盖完整日历年的首尾年份
    start_partial = (earliest.month, earliest.day) != (1, 1)
    end_partial = (latest.month, latest.day) != (12, 31)

    annual["partial_boundary_year"] = (
        (annual["year"].eq(earliest.year) & start_partial)
        | (annual["year"].eq(latest.year) & end_partial)
    )

    # 说明实际观察范围，不把中间年份视为全平台完整数据
    annual["observed_start_utc"] = [
        str(max(
            earliest,
            pd.Timestamp(year=int(y), month=1, day=1, tz="UTC")
        ).date())
        for y in annual["year"]
    ]
    annual["observed_end_utc"] = [
        str(min(
            latest,
            pd.Timestamp(year=int(y), month=12, day=31, tz="UTC")
        ).date())
        for y in annual["year"]
    ]

    valid_rows = int(annual["rating_count"].sum())
    assert valid_rows + invalid_rows == total_rows

    annual.to_csv(
        TABLE_DIR / "historical_ratings_by_year.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 四、历史评分分布
    distribution = pd.DataFrame({
        "rating": [1, 2, 3, 4, 5],
        "rating_count": [score_counts.get(i, 0) for i in range(1, 6)]
    })
    distribution["share_percent"] = (
        distribution["rating_count"] / valid_rows * 100
    )

    assert distribution["rating_count"].sum() == valid_rows

    distribution.to_csv(
        TABLE_DIR / "historical_rating_distribution.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 五、记录分析范围和检查结果
    coverage = pd.DataFrame([{
        "input_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "earliest_utc": str(earliest),
        "latest_utc": str(latest),
    }])
    coverage.to_csv(
        TABLE_DIR / "historical_analysis_coverage.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 六、年度评分数量及平均分图
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 8), squeeze=False
    )
    ax_count, ax_rating = axes.ravel()

    colors = [
        "#F28E2B" if partial else "#4C78A8"
        for partial in annual["partial_boundary_year"]
    ]

    ax_count.bar(
        annual["year"], annual["rating_count"], color=colors
    )
    ax_count.set_title("Historical rating records by year")
    ax_count.set_ylabel("Number of rating records")
    ax_count.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax_count.legend(handles=[
        Patch(color="#4C78A8", label="Interior year"),
        Patch(color="#F28E2B", label="Partial boundary year")
    ])

    ax_rating.plot(
        annual["year"],
        annual["avg_rating"],
        marker="o",
        color="#4C78A8"
    )

    partial = annual["partial_boundary_year"]
    ax_rating.scatter(
        annual.loc[partial, "year"],
        annual.loc[partial, "avg_rating"],
        color="#F28E2B",
        zorder=3
    )

    ax_rating.set_title("Average rating by year")
    ax_rating.set_xlabel("Year (UTC)")
    ax_rating.set_ylabel("Average rating")
    ax_rating.set_ylim(1, 5)

    for ax in (ax_count, ax_rating):
        ax.set_xticks(annual["year"])
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        f"Observed dates: {earliest.date()} to {latest.date()}. "
        "Rating activity is not sales.",
        ha="center", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        FIGURE_DIR / "09_historical_ratings_by_year.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 七、1～5分分布图
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        distribution["rating"],
        distribution["rating_count"],
        color="#59A14F"
    )

    for bar, share in zip(bars, distribution["share_percent"]):
        ax.annotate(
            f"{int(bar.get_height()):,}\n({share:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(f"Historical rating distribution (n={valid_rows:,})")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of rating records")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, distribution["rating_count"].max() * 1.2)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "10_historical_rating_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 八、打印结果
    overall_avg = annual["rating_sum"].sum() / valid_rows
    overall_low = annual["low_rating_count"].sum() / valid_rows

    print("\n=== 记录核对 ===")
    print(f"输入记录数：{total_rows:,}")
    print(f"有效记录数：{valid_rows:,}")
    print(f"时间或评分异常记录数：{invalid_rows:,}")
    print("最早时间：", earliest)
    print("最晚时间：", latest)

    print("\n=== 年度历史评分统计 ===")
    print(annual[[
        "year", "rating_count", "avg_rating",
        "low_rating_ratio", "partial_boundary_year"
    ]].round(4).to_string(index=False))

    print("\n=== 历史评分分布 ===")
    print(distribution.round(2).to_string(index=False))

    print(f"\n全部有效记录的平均评分：{overall_avg:.4f}")
    print(f"低评分（1～2分）记录占比：{overall_low:.2%}")

    print("\n统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print("处理完成，输入文件没有修改。")

if __name__ == "__main__":
    run()

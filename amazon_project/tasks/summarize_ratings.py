"""任务09：summarize_ratings。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import sqlite3
    import pandas as pd

    project_dir = PROJECT_ROOT
    data_dir = project_dir / "data_clean"

    database_path = data_dir / "ratings_dedup.sqlite"

    if not database_path.exists():
        raise FileNotFoundError("找不到数据库，请先完成第08步。")

    # 以只读方式打开上一步生成的数据库
    conn = sqlite3.connect(
        database_path.resolve().as_uri() + "?mode=ro",
        uri=True,
    )

    try:
        print("开始按商品汇总评分，请稍等……")

        summary = pd.read_sql_query(
            """
            SELECT
                product_id,
                COUNT(*) AS historical_rating_count,
                COUNT(DISTINCT user_id) AS historical_rater_count,
                AVG(rating) AS historical_avg_rating,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END)
                    AS historical_low_rating_count,
                MIN(rating_datetime_utc) AS first_rating_datetime_utc,
                MAX(rating_datetime_utc) AS last_rating_datetime_utc
            FROM ratings
            GROUP BY product_id
            """,
            conn,
        )

    finally:
        conn.close()

    # 本项目将1—2分定义为低评分
    summary["historical_low_rating_ratio"] = (
        summary["historical_low_rating_count"]
        / summary["historical_rating_count"]
    )

    summary_path = data_dir / "historical_product_ratings_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 商品评分汇总结果 ===")
    print(f"商品数量：{len(summary):,}")
    print("商品编号是否唯一：", summary["product_id"].is_unique)
    print(
        "汇总后的评分总条数：",
        f"{summary['historical_rating_count'].sum():,}",
    )

    print("\n评分记录最多的前5个商品：")
    print(
        summary.nlargest(5, "historical_rating_count")[
            [
                "product_id",
                "historical_rating_count",
                "historical_rater_count",
                "historical_avg_rating",
                "historical_low_rating_ratio",
            ]
        ].to_string(index=False)
    )

    # 小范围关联示例：只保留两份数据共有的商品
    products = pd.read_csv(
        data_dir / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    matched = products.merge(
        summary,
        on="product_id",
        how="inner",
        validate="one_to_one",
    )

    matched_path = data_dir / "matched_products_example.csv"
    matched.to_csv(
        matched_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 小范围关联示例 ===")
    print("匹配到的商品数：", len(matched))
    print(
        "这些商品在历史数据中的评分总条数：",
        f"{matched['historical_rating_count'].sum():,}",
    )

    print("\n评分汇总表：", summary_path)
    print("关联示例表：", matched_path)
    print("处理完成。")

if __name__ == "__main__":
    run()

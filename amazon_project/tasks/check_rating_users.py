"""任务31：check_rating_users。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import sqlite3

    import pandas as pd


    ROOT = PROJECT_ROOT
    INPUT = ROOT / "data_clean" / "ratings_unique.csv"
    LOCAL_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"

    CHUNK_SIZE = 200_000


    def main():
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        # 使用本地数据库累计计数，避免把全部评分一次放入内存。
        database_path = LOCAL_DIR / "recommendation_user_counts.sqlite"

        connection = sqlite3.connect(database_path)

        try:
            # 只重建本脚本自己的计数表，方便重复运行。
            connection.execute("DROP TABLE IF EXISTS user_counts")
            connection.execute("""
                CREATE TABLE user_counts (
                    user_id TEXT PRIMARY KEY,
                    rating_count INTEGER NOT NULL
                )
            """)
            connection.commit()

            total_rows = 0
            missing_user_rows = 0

            print("开始分批统计用户评分数量，请等待完成提示……")

            reader = pd.read_csv(
                INPUT,
                usecols=["user_id"],
                dtype={"user_id": "string"},
                chunksize=CHUNK_SIZE,
            )

            for batch_number, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)

                missing = (
                    chunk["user_id"].isna()
                    | chunk["user_id"].fillna("").str.strip().eq("")
                )
                missing_user_rows += int(missing.sum())

                counts = chunk.loc[~missing, "user_id"].value_counts()

                records = (
                    (str(user_id), int(count))
                    for user_id, count in counts.items()
                )

                with connection:
                    connection.executemany("""
                        INSERT INTO user_counts (user_id, rating_count)
                        VALUES (?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            rating_count =
                                user_counts.rating_count
                                + excluded.rating_count
                    """, records)

                if batch_number % 5 == 0:
                    print(
                        f"已处理 {total_rows:,} 行",
                        flush=True,
                    )

            totals = connection.execute("""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(rating_count), 0),
                    COALESCE(MIN(rating_count), 0),
                    COALESCE(MAX(rating_count), 0)
                FROM user_counts
            """).fetchone()

            user_count, counted_rows, min_count, max_count = totals

            if counted_rows + missing_user_rows != total_rows:
                raise ValueError("记录数量核对失败。")

            if user_count == 0:
                raise ValueError("没有有效用户，无法继续统计。")

            summary = pd.DataFrame([{
                "input_rating_rows": total_rows,
                "missing_user_rows": missing_user_rows,
                "counted_rating_rows": counted_rows,
                "unique_users": user_count,
                "mean_ratings_per_user": counted_rows / user_count,
                "min_ratings_per_user": min_count,
                "max_ratings_per_user": max_count,
            }])

            # 查看不同活跃程度的用户数量。
            ranges = [
                ("1", 1, 1),
                ("2-4", 2, 4),
                ("5-9", 5, 9),
                ("10-19", 10, 19),
                ("20-49", 20, 49),
                ("50-99", 50, 99),
                ("100+", 100, None),
            ]

            distribution_rows = []

            for label, lower, upper in ranges:
                if upper is None:
                    result = connection.execute("""
                        SELECT COUNT(*), COALESCE(SUM(rating_count), 0)
                        FROM user_counts
                        WHERE rating_count >= ?
                    """, (lower,)).fetchone()
                else:
                    result = connection.execute("""
                        SELECT COUNT(*), COALESCE(SUM(rating_count), 0)
                        FROM user_counts
                        WHERE rating_count BETWEEN ? AND ?
                    """, (lower, upper)).fetchone()

                users, ratings = result

                distribution_rows.append({
                    "ratings_per_user_group": label,
                    "user_count": users,
                    "user_share_percent": users / user_count * 100,
                    "rating_rows": ratings,
                    "rating_share_percent": ratings / counted_rows * 100,
                })

            distribution = pd.DataFrame(distribution_rows)

            # 比较几个可能的用户筛选条件，暂不决定最终采用哪个。
            threshold_rows = []

            for threshold in [5, 10, 20, 30, 50]:
                users, ratings = connection.execute("""
                    SELECT COUNT(*), COALESCE(SUM(rating_count), 0)
                    FROM user_counts
                    WHERE rating_count >= ?
                """, (threshold,)).fetchone()

                threshold_rows.append({
                    "minimum_ratings_per_user": threshold,
                    "eligible_users": users,
                    "eligible_rating_rows": ratings,
                    "user_share_percent": users / user_count * 100,
                    "rating_share_percent": ratings / counted_rows * 100,
                })

            thresholds = pd.DataFrame(threshold_rows)

            summary.to_csv(
                TABLE_DIR / "recommendation_user_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            distribution.to_csv(
                TABLE_DIR / "recommendation_user_activity_distribution.csv",
                index=False,
                encoding="utf-8-sig",
            )
            thresholds.to_csv(
                TABLE_DIR / "recommendation_user_threshold_comparison.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print("\n=== 用户评分概况 ===")
            print(summary.round(4).to_string(index=False))

            print("\n=== 用户活跃程度分组 ===")
            print(distribution.round(4).to_string(index=False))

            print("\n=== 不同筛选条件对应的数据规模 ===")
            print(thresholds.round(4).to_string(index=False))

            print(
                "\n记录数核对：",
                counted_rows + missing_user_rows == total_rows,
            )
            print("统计表位置：", TABLE_DIR)
            print("本地用户计数数据库：", database_path)
            print(
                "\n完成。本步骤统计的是评分记录数，不是购买次数。"
            )
            print(
                "尚未抽取建模数据、划分训练集或训练模型，"
                "输入文件没有修改。"
            )

        finally:
            connection.close()


    main()


if __name__ == "__main__":
    run()

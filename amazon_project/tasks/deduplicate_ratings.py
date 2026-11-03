"""任务08：deduplicate_ratings。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import sqlite3
    import pandas as pd

    project_dir = PROJECT_ROOT
    data_dir = project_dir / "data_clean"
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "ratings_clean.csv"
    output_path = data_dir / "ratings_unique.csv"
    database_path = data_dir / "ratings_dedup.sqlite"

    # 数据库用于在硬盘上检查整份文件的重复记录
    conn = sqlite3.connect(database_path)

    try:
        # 重跑时重新构建本脚本的中间表，不修改输入CSV
        conn.execute("DROP TABLE IF EXISTS ratings")

        conn.execute("""
            CREATE TABLE ratings (
                source_row INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                rating REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                rating_datetime_utc TEXT NOT NULL,
                UNIQUE (user_id, product_id, rating, timestamp)
            )
        """)
        conn.commit()

        total_rows = 0
        kept_rows = 0

        print("开始检查全量重复记录，可能需要几分钟或更久……")

        for batch_number, chunk in enumerate(
            pd.read_csv(
                input_path,
                dtype={
                    "source_row": "int64",
                    "user_id": "string",
                    "product_id": "string",
                    "rating": "float64",
                    "timestamp": "int64",
                    "rating_datetime_utc": "string",
                },
                chunksize=200_000,
            ),
            start=1,
        ):
            columns = [
                "source_row",
                "user_id",
                "product_id",
                "rating",
                "timestamp",
                "rating_datetime_utc",
            ]

            changes_before = conn.total_changes

            # 四项完全相同的记录只保留首次插入的一条
            with conn:
                conn.executemany(
                    """
                    INSERT INTO ratings
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, product_id, rating, timestamp)
                    DO NOTHING
                    """,
                    chunk[columns].itertuples(index=False, name=None),
                )

            total_rows += len(chunk)
            kept_rows += conn.total_changes - changes_before

            if batch_number % 5 == 0:
                print(f"已检查 {total_rows:,} 行")

        duplicate_rows = total_rows - kept_rows

        print("\n重复检查完成，正在导出CSV……")

        # 分批导出，不一次性载入全部记录
        first_batch = True

        for chunk in pd.read_sql_query(
            """
            SELECT source_row, user_id, product_id, rating,
                   timestamp, rating_datetime_utc
            FROM ratings
            """,
            conn,
            chunksize=200_000,
        ):
            chunk.to_csv(
                output_path,
                mode="w" if first_batch else "a",
                header=first_batch,
                index=False,
                encoding="utf-8",
            )
            first_batch = False

        # 保存统计结果，后面写周报时可以直接查
        summary = pd.DataFrame({
            "item": [
                "input_rows",
                "unique_rows",
                "duplicate_rows_removed",
            ],
            "value": [total_rows, kept_rows, duplicate_rows],
        })

        summary.to_csv(
            report_dir / "ratings_dedup_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n=== 评分去重结果 ===")
        print(f"输入记录数：{total_rows:,}")
        print(f"去重后记录数：{kept_rows:,}")
        print(f"删除的完全重复记录数：{duplicate_rows:,}")
        print("记录数核对：", total_rows == kept_rows + duplicate_rows)
        print("去重数据：", output_path)
        print("统计结果：", report_dir / "ratings_dedup_summary.csv")
        print("处理完成，原始文件和上一份清洗文件均保留。")

    finally:
        conn.close()

if __name__ == "__main__":
    run()

"""任务07：clean_ratings。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT
    input_path = (
        project_dir / "data_raw" / "reviews" / "ratings_Electronics (1).csv"
    )

    output_dir = project_dir / "data_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ratings_clean.csv"
    issues_path = output_dir / "ratings_issues.csv"

    columns = ["user_id", "product_id", "rating", "timestamp"]

    total_rows = 0
    saved_rows = 0
    issue_rows = 0
    first_batch = True

    print("开始分批处理评分数据，请等待“处理完成”提示……")

    for batch_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            header=None,
            names=columns,
            dtype="string",
            chunksize=200_000,
        ),
        start=1,
    ):
        for column in columns:
            chunk[column] = chunk[column].str.strip().replace("", pd.NA)

        # 原始文件没有表头，因此第一条记录就是第1行
        chunk.insert(
            0,
            "source_row",
            range(total_rows + 1, total_rows + len(chunk) + 1),
        )
        total_rows += len(chunk)

        # 转换数字，但保留原始文字供异常记录检查
        ratings = pd.to_numeric(chunk["rating"], errors="coerce")
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce")

        # 仅接受整数秒时间戳
        valid_timestamp = (
            timestamps.notna() & (timestamps % 1 == 0)
        ).fillna(False)

        dates = pd.to_datetime(
            timestamps.where(valid_timestamp),
            unit="s",
            errors="coerce",
            utc=True,
        )

        # 为问题记录标记原因
        chunk["missing_id"] = (
            chunk["user_id"].isna() | chunk["product_id"].isna()
        )
        chunk["invalid_rating"] = (
            ~ratings.between(1, 5).fillna(False)
        )
        chunk["invalid_timestamp"] = (
            ~valid_timestamp | dates.isna()
        )

        bad = chunk[
            ["missing_id", "invalid_rating", "invalid_timestamp"]
        ].any(axis=1)

        # 问题记录单独保存，不直接丢弃
        issues = chunk.loc[bad].copy()
        issues.to_csv(
            issues_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False,
            encoding="utf-8",
        )

        # 合格记录规范格式后保存
        clean = chunk.loc[
            ~bad, ["source_row", "user_id", "product_id"]
        ].copy()

        clean["rating"] = ratings.loc[~bad].astype("float64")
        clean["timestamp"] = timestamps.loc[~bad].astype("int64")
        clean["rating_datetime_utc"] = (
            dates.loc[~bad].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        clean.to_csv(
            output_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False,
            encoding="utf-8",
        )

        saved_rows += len(clean)
        issue_rows += len(issues)
        first_batch = False

        if batch_number % 10 == 0:
            print(f"已处理 {total_rows:,} 行")

    print("\n=== 评分数据处理结果 ===")
    print(f"原始记录数：{total_rows:,}")
    print(f"保存到清洗表的记录数：{saved_rows:,}")
    print(f"单独保存的问题记录数：{issue_rows:,}")
    print("记录数是否核对一致：", total_rows == saved_rows + issue_rows)
    print("清洗文件：", output_path)
    print("问题记录文件：", issues_path)
    print("\n处理完成。尚未去重，原始文件未修改。")

if __name__ == "__main__":
    run()

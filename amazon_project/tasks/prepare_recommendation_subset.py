"""任务32：prepare_recommendation_subset。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import sqlite3
    import json

    import pandas as pd


    ROOT = PROJECT_ROOT
    INPUT = ROOT / "data_clean" / "ratings_unique.csv"
    LOCAL_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"

    MIN_USER_RATINGS = 20
    CHUNK_SIZE = 200_000


    def main():
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        database_path = LOCAL_DIR / "recommendation_user_counts.sqlite"

        if not database_path.exists():
            raise FileNotFoundError("未找到用户计数数据库，请先运行第31步。")

        with sqlite3.connect(database_path) as connection:
            eligible = pd.read_sql_query(
                """
                SELECT user_id, rating_count
                FROM user_counts
                WHERE rating_count >= ?
                ORDER BY user_id
                """,
                connection,
                params=(MIN_USER_RATINGS,),
            )

        if eligible.empty:
            raise ValueError("没有符合条件的用户。")

        selected_users = set(eligible["user_id"])
        expected_rows = int(eligible["rating_count"].sum())

        print("=== 本次抽取范围 ===")
        print("每位用户至少评分记录数：", MIN_USER_RATINGS)
        print(f"符合条件的用户数：{len(selected_users):,}")
        print(f"预计抽取记录数：{expected_rows:,}")
        print("\n开始分批读取历史评分……", flush=True)

        parts = []
        scanned_rows = 0

        reader = pd.read_csv(
            INPUT,
            usecols=[
                "source_row", "user_id",
                "product_id", "rating", "timestamp",
            ],
            dtype={
                "source_row": "int64",
                "user_id": "string",
                "product_id": "string",
                "rating": "float64",
                "timestamp": "int64",
            },
            chunksize=CHUNK_SIZE,
        )

        for batch_number, chunk in enumerate(reader, start=1):
            scanned_rows += len(chunk)

            selected = chunk[
                chunk["user_id"].isin(selected_users)
            ].copy()

            if not selected.empty:
                parts.append(selected)

            if batch_number % 5 == 0:
                print(f"已扫描 {scanned_rows:,} 行", flush=True)

        if not parts:
            raise ValueError("没有抽取到评分记录。")

        df = pd.concat(parts, ignore_index=True)
        del parts

        if len(df) != expected_rows:
            raise ValueError(
                f"抽取数量不一致：预计{expected_rows}，实际{len(df)}。"
            )

        if (
            df["product_id"].isna().any()
            or df["product_id"].str.strip().eq("").any()
        ):
            raise ValueError("商品编号存在缺失。")

        if not df["rating"].between(1, 5).all():
            raise ValueError("发现缺失或超出范围的评分。")

        dates = pd.to_datetime(
            df["timestamp"],
            unit="s",
            utc=True,
            errors="coerce",
        )
        if dates.isna().any():
            raise ValueError("发现无法转换的评分时间。")

        user_counts = df.groupby("user_id").size()
        product_counts = df.groupby("product_id").size()

        if (
            len(user_counts) != len(selected_users)
            or user_counts.min() < MIN_USER_RATINGS
        ):
            raise ValueError("抽取后的用户数量或评分数量不符合条件。")

        # 检查用户—商品组合重复情况，不在这一步直接删除。
        pair_columns = ["user_id", "product_id"]
        duplicate_mask = df.duplicated(pair_columns, keep=False)
        extra_pair_rows = int(
            df.duplicated(pair_columns, keep="first").sum()
        )

        duplicate_pairs = (
            df.loc[duplicate_mask]
            .groupby(pair_columns)
            .agg(
                record_count=("rating", "size"),
                distinct_rating_count=("rating", "nunique"),
                distinct_timestamp_count=("timestamp", "nunique"),
            )
            .reset_index()
        )

        unique_pairs = len(df) - extra_pair_rows
        density_percent = (
            unique_pairs
            / (len(user_counts) * len(product_counts))
            * 100
        )

        summary = pd.DataFrame([{
            "minimum_user_rating_records": MIN_USER_RATINGS,
            "source_rows_scanned": scanned_rows,
            "selected_rating_rows": len(df),
            "selected_users": len(user_counts),
            "selected_products": len(product_counts),
            "min_records_per_user": int(user_counts.min()),
            "max_records_per_user": int(user_counts.max()),
            "mean_records_per_user": float(user_counts.mean()),
            "products_with_one_record": int(
                product_counts.eq(1).sum()
            ),
            "repeated_user_product_pairs": len(duplicate_pairs),
            "extra_user_product_rows": extra_pair_rows,
            "observed_pair_density_percent": density_percent,
            "earliest_time_utc": dates.min().isoformat(),
            "latest_time_utc": dates.max().isoformat(),
        }])

        # 建模数据留在本地，不放进提交给老师的数据结果目录。
        output_path = LOCAL_DIR / "recommendation_ratings_subset.csv"
        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        duplicate_pairs.to_csv(
            LOCAL_DIR / "recommendation_duplicate_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            TABLE_DIR / "recommendation_subset_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        product_distribution = pd.DataFrame([
            {
                "records_per_product_group": label,
                "product_count": int(mask.sum()),
                "rating_rows": int(product_counts[mask].sum()),
            }
            for label, mask in [
                ("1", product_counts.eq(1)),
                ("2-4", product_counts.between(2, 4)),
                ("5-9", product_counts.between(5, 9)),
                ("10-19", product_counts.between(10, 19)),
                ("20+", product_counts.ge(20)),
            ]
        ])

        product_distribution.to_csv(
            TABLE_DIR / "recommendation_subset_product_distribution.csv",
            index=False,
            encoding="utf-8-sig",
        )

        config = {
            "input_file": INPUT.name,
            "minimum_user_rating_records": MIN_USER_RATINGS,
            "selection_scope": "full_observed_history",
            "random_user_sampling": False,
            "product_filter_applied": False,
            "duplicate_user_product_rows_removed": False,
            "note": (
                "Exploratory active-user subset selected using total "
                "observed rating counts. Not representative of all users "
                "and not a prospective deployment simulation."
            ),
        }

        with open(
            TABLE_DIR / "recommendation_subset_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        print("\n=== 推荐模型候选数据 ===")
        print(f"评分记录数：{len(df):,}")
        print(f"用户数：{len(user_counts):,}")
        print(f"商品数：{len(product_counts):,}")
        print(f"每位用户最少记录数：{user_counts.min():,}")
        print(f"每位用户最多记录数：{user_counts.max():,}")
        print(f"平均每位用户记录数：{user_counts.mean():.4f}")

        print("\n=== 用户—商品重复检查 ===")
        print("有多条记录的用户—商品组合数：", len(duplicate_pairs))
        print("重复组合的多余记录数：", extra_pair_rows)
        print(f"用户—商品组合覆盖比例：{density_percent:.6f}%")

        print("\n=== 商品收到评分的数量分布 ===")
        print(product_distribution.to_string(index=False))

        print("\n=== 时间范围 ===")
        print("最早时间：", dates.min())
        print("最晚时间：", dates.max())

        print("\n抽取记录数核对：", len(df) == expected_rows)
        print("本地候选数据：", output_path)
        print("统计表：", TABLE_DIR)
        print(
            "完成。尚未划分训练集、测试集或训练推荐模型，"
            "输入文件没有修改。"
        )


    main()


if __name__ == "__main__":
    run()

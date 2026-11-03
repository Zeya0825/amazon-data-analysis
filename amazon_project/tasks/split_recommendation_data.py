"""任务33：split_recommendation_data。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json
    import math

    import numpy as np
    import pandas as pd


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"

    TEST_RATIO = 0.20
    RANDOM_SEED = 42


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            DATA_DIR / "recommendation_ratings_subset.csv",
            dtype={
                "source_row": "int64",
                "user_id": "string",
                "product_id": "string",
                "rating": "float64",
                "timestamp": "int64",
            },
        )

        if df.empty:
            raise ValueError("输入数据为空。")

        if df[["user_id", "product_id"]].isna().any().any():
            raise ValueError("用户或商品编号存在缺失。")

        if not df["source_row"].is_unique:
            raise ValueError("source_row不唯一，请检查输入。")

        if df.duplicated(["user_id", "product_id"]).any():
            raise ValueError("存在重复用户—商品组合，请先处理。")

        if not df["rating"].between(1, 5).all():
            raise ValueError("评分缺失或超出1至5分范围。")

        # 固定输入顺序，再进行随机划分，便于重复运行核对。
        df = df.sort_values(
            ["user_id", "source_row"]
        ).reset_index(drop=True)

        rng = np.random.default_rng(RANDOM_SEED)
        is_test = np.zeros(len(df), dtype=bool)

        print("开始按用户划分训练集和测试集……")

        for user_id, indices in df.groupby(
            "user_id", sort=True
        ).indices.items():
            count = len(indices)

            if count < 2:
                raise ValueError(f"用户 {user_id} 的记录不足2条。")

            # 向上取整，所以最终测试比例可能略高于20%。
            test_count = min(
                count - 1,
                max(1, math.ceil(count * TEST_RATIO)),
            )

            selected = rng.choice(
                indices,
                size=test_count,
                replace=False,
            )
            is_test[selected] = True

        train = df.loc[~is_test].copy()
        test = df.loc[is_test].copy()

        train_users = set(train["user_id"])
        train_products = set(train["product_id"])

        # 标记测试中的用户、商品是否在训练集出现过。
        test["user_seen_in_train"] = test["user_id"].isin(train_users)
        test["product_seen_in_train"] = test["product_id"].isin(
            train_products
        )

        # 检查两部分是否有记录或用户—商品组合重叠。
        overlap_rows = len(
            set(train["source_row"]) & set(test["source_row"])
        )

        overlap_pairs = len(
            train[["user_id", "product_id"]].merge(
                test[["user_id", "product_id"]],
                on=["user_id", "product_id"],
                how="inner",
            )
        )

        row_count_ok = len(train) + len(test) == len(df)
        all_users_have_train = bool(test["user_seen_in_train"].all())

        if (
            not row_count_ok
            or overlap_rows != 0
            or overlap_pairs != 0
            or not all_users_have_train
        ):
            raise ValueError("训练集与测试集划分检查失败。")

        # 用户层面的划分统计留在本地。
        user_split = pd.concat(
            [
                train.groupby("user_id").size().rename("train_count"),
                test.groupby("user_id").size().rename("test_count"),
            ],
            axis=1,
        ).fillna(0).astype(int).reset_index()

        train.to_csv(
            DATA_DIR / "recommendation_train.csv",
            index=False,
            encoding="utf-8-sig",
        )
        test.to_csv(
            DATA_DIR / "recommendation_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        user_split.to_csv(
            DATA_DIR / "recommendation_user_split_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows = []

        for name, part in [("train", train), ("test", test)]:
            summary_rows.append({
                "split": name,
                "rating_rows": len(part),
                "share_percent": len(part) / len(df) * 100,
                "unique_users": part["user_id"].nunique(),
                "unique_products": part["product_id"].nunique(),
                "mean_rating": part["rating"].mean(),
            })

        summary = pd.DataFrame(summary_rows)

        coverage_rows = []
        for label, mask in [
            ("seen_product", test["product_seen_in_train"]),
            ("unseen_product", ~test["product_seen_in_train"]),
        ]:
            part = test.loc[mask]
            coverage_rows.append({
                "test_group": label,
                "rating_rows": len(part),
                "share_of_test_percent": len(part) / len(test) * 100,
                "unique_products": part["product_id"].nunique(),
                "unique_users": part["user_id"].nunique(),
            })

        coverage = pd.DataFrame(coverage_rows)

        checks = pd.DataFrame([{
            "input_rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
            "row_count_matches": row_count_ok,
            "overlapping_source_rows": overlap_rows,
            "overlapping_user_product_pairs": overlap_pairs,
            "all_test_users_seen_in_train": all_users_have_train,
            "minimum_train_rows_per_user": int(
                user_split["train_count"].min()
            ),
            "minimum_test_rows_per_user": int(
                user_split["test_count"].min()
            ),
        }])

        summary.to_csv(
            TABLE_DIR / "recommendation_split_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        coverage.to_csv(
            TABLE_DIR / "recommendation_test_product_coverage.csv",
            index=False,
            encoding="utf-8-sig",
        )
        checks.to_csv(
            TABLE_DIR / "recommendation_split_checks.csv",
            index=False,
            encoding="utf-8-sig",
        )

        config = {
            "method": "per_user_random_holdout",
            "test_ratio_target": TEST_RATIO,
            "test_count_rounding": "ceil",
            "random_seed": RANDOM_SEED,
            "input_order": ["user_id", "source_row"],
            "unseen_test_products_retained": True,
            "limitations": [
                "Random historical holdout, not future-time prediction.",
                "Active-user cohort selected using full-history counts.",
                "No test ratings may be used for model training.",
                "Unseen products must be reported separately in evaluation.",
            ],
        }

        with open(
            TABLE_DIR / "recommendation_split_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        print("\n=== 训练集与测试集 ===")
        print(summary.round(4).to_string(index=False))

        print("\n=== 测试集商品覆盖情况 ===")
        print(coverage.round(4).to_string(index=False))
        print("seen_product：商品在训练集中出现过")
        print("unseen_product：商品没有在训练集中出现过")

        print("\n=== 划分检查 ===")
        print("总记录数核对：", row_count_ok)
        print("训练集与测试集重叠记录数：", overlap_rows)
        print("重叠用户—商品组合数：", overlap_pairs)
        print("测试用户均在训练集中出现：", all_users_have_train)
        print(
            "每位用户最少训练记录数：",
            user_split["train_count"].min(),
        )
        print(
            "每位用户最少测试记录数：",
            user_split["test_count"].min(),
        )

        print("\n训练集：", DATA_DIR / "recommendation_train.csv")
        print("测试集：", DATA_DIR / "recommendation_test.csv")
        print("统计表：", TABLE_DIR)
        print("完成。尚未训练模型，输入候选数据没有修改。")


    main()


if __name__ == "__main__":
    run()

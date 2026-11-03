"""任务38：analyze_recommendation_popularity。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "revision" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "revision" / "figures"

    GROUPS = ["Unseen", "1", "2-4", "5-9", "10-19", "20+"]
    RELEVANT_THRESHOLD = 4.0


    def popularity_group(counts):
        return pd.cut(
            counts,
            bins=[-1, 0, 1, 4, 9, 19, np.inf],
            labels=GROUPS,
            right=True,
        )


    def save_csv(df, filename):
        df.to_csv(
            TABLE_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        read_options = {
            "dtype": {
                "user_id": "string",
                "product_id": "string",
            },
            "float_precision": "round_trip",
        }

        train = pd.read_csv(
            DATA_DIR / "recommendation_train.csv",
            **read_options,
        )
        test = pd.read_csv(
            DATA_DIR / "recommendation_test.csv",
            **read_options,
        )
        predictions = pd.read_csv(
            DATA_DIR / "recommendation_test_predictions.csv",
            **read_options,
        )

        prediction_columns = [
            "pred_GlobalMean", "pred_UserMean", "pred_SVD"
        ]

        for name, frame in [("train", train), ("test", test)]:
            if frame[["user_id", "product_id"]].isna().any().any():
                raise ValueError(f"{name}存在缺失的用户或商品编号。")

            frame["rating"] = pd.to_numeric(
                frame["rating"], errors="raise"
            )
            if not frame["rating"].between(1, 5).all():
                raise ValueError(f"{name}存在异常评分。")

            if frame.duplicated(["user_id", "product_id"]).any():
                raise ValueError(f"{name}存在重复用户—商品组合。")

        required = [
            "user_id", "product_id", "rating"
        ] + prediction_columns

        if not set(required).issubset(predictions.columns):
            raise ValueError("逐条预测文件缺少所需字段。")

        if len(predictions) != len(test):
            raise ValueError("预测文件与测试集行数不一致。")

        pred_part = predictions[required].rename(
            columns={"rating": "saved_actual_rating"}
        )

        evaluated = test[
            ["user_id", "product_id", "rating"]
        ].merge(
            pred_part,
            on=["user_id", "product_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        if not evaluated["_merge"].eq("both").all():
            raise ValueError("部分测试记录没有对应预测。")

        for column in ["saved_actual_rating"] + prediction_columns:
            evaluated[column] = pd.to_numeric(
                evaluated[column], errors="raise"
            )
            if not np.isfinite(
                evaluated[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"{column}存在缺失或非有限数值。")

        if not np.allclose(
            evaluated["rating"],
            evaluated["saved_actual_rating"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("预测文件中的真实评分与测试集不一致。")

        # 热门程度只由训练集确定。
        item_counts = train.groupby("product_id").size()

        items = item_counts.rename(
            "train_rating_count"
        ).reset_index()
        items["popularity_group"] = popularity_group(
            items["train_rating_count"]
        )

        evaluated["train_rating_count"] = (
            evaluated["product_id"]
            .map(item_counts)
            .fillna(0)
            .astype(int)
        )
        evaluated["popularity_group"] = popularity_group(
            evaluated["train_rating_count"]
        )
        evaluated["relevant"] = (
            evaluated["rating"] >= RELEVANT_THRESHOLD
        )

        total_relevant = int(evaluated["relevant"].sum())
        seen_mask = evaluated["train_rating_count"].gt(0)
        seen_test_count = int(seen_mask.sum())
        seen_relevant_count = int(
            evaluated.loc[seen_mask, "relevant"].sum()
        )

        summary_rows = []

        for group in GROUPS:
            catalog_part = items.loc[
                items["popularity_group"].eq(group)
            ]
            test_part = evaluated.loc[
                evaluated["popularity_group"].eq(group)
            ]

            relevant_count = int(test_part["relevant"].sum())

            summary_rows.append({
                "popularity_group": group,
                "train_catalog_products": len(catalog_part),
                "train_catalog_share_percent":
                    len(catalog_part) / len(items) * 100,
                "train_rating_rows": int(
                    catalog_part["train_rating_count"].sum()
                ),
                "test_rows": len(test_part),
                "test_share_percent":
                    len(test_part) / len(evaluated) * 100,
                "test_unique_products":
                    test_part["product_id"].nunique(),
                "relevant_test_rows": relevant_count,
                "share_of_all_relevant_percent":
                    relevant_count / total_relevant * 100
                    if total_relevant else np.nan,
                "share_of_seen_relevant_percent":
                    relevant_count / seen_relevant_count * 100
                    if group != "Unseen" and seen_relevant_count
                    else np.nan,
                "within_group_relevant_percent":
                    relevant_count / len(test_part) * 100
                    if len(test_part) else np.nan,
            })

        summary = pd.DataFrame(summary_rows)

        error_rows = []

        for group in GROUPS:
            part = evaluated.loc[
                evaluated["popularity_group"].eq(group)
            ]

            if part.empty:
                continue

            for model_name, column in [
                ("GlobalMean", "pred_GlobalMean"),
                ("UserMean", "pred_UserMean"),
                ("SVD", "pred_SVD"),
            ]:
                errors = (
                    part[column].to_numpy(dtype=float)
                    - part["rating"].to_numpy(dtype=float)
                )

                error_rows.append({
                    "popularity_group": group,
                    "model": model_name,
                    "test_rows": len(part),
                    "MAE": float(np.mean(np.abs(errors))),
                    "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                })

        errors_table = pd.DataFrame(error_rows)

        # 训练评分次数最多的前10%商品：
        # 固定选取数量，次数相同的商品按编号排序。
        sorted_items = items.sort_values(
            ["train_rating_count", "product_id"],
            ascending=[False, True],
        )
        head_count = int(np.ceil(len(items) * 0.10))
        head_ids = set(
            sorted_items.head(head_count)["product_id"]
        )

        head_test = evaluated["product_id"].isin(head_ids)
        head_relevant_count = int(
            (head_test & evaluated["relevant"]).sum()
        )

        concentration = pd.DataFrame([{
            "train_catalog_products": len(items),
            "head_product_count": head_count,
            "head_catalog_share_percent":
                head_count / len(items) * 100,
            "head_train_rating_share_percent":
                sorted_items.head(head_count)[
                    "train_rating_count"
                ].sum() / len(train) * 100,
            "seen_test_rows": seen_test_count,
            "seen_relevant_test_rows": seen_relevant_count,
            "head_share_of_seen_test_percent":
                int(head_test.sum()) / seen_test_count * 100,
            "head_share_of_seen_relevant_percent":
                head_relevant_count / seen_relevant_count * 100
                if seen_relevant_count else np.nan,
        }])

        save_csv(
            summary,
            "recommendation_popularity_distribution.csv",
        )
        save_csv(
            errors_table,
            "recommendation_errors_by_popularity.csv",
        )
        save_csv(
            concentration,
            "recommendation_head_concentration.csv",
        )

        # 图1：商品目录占比与测试相关记录占比。
        seen_summary = summary.loc[
            summary["popularity_group"].ne("Unseen")
        ]
        x = np.arange(len(seen_summary))
        width = 0.38

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(
            x - width / 2,
            seen_summary["train_catalog_share_percent"],
            width,
            label="Share of training catalog",
        )
        ax.bar(
            x + width / 2,
            seen_summary["share_of_seen_relevant_percent"],
            width,
            label="Share of relevant seen-item test records",
        )
        ax.set_xticks(
            x, seen_summary["popularity_group"].astype(str)
        )
        ax.set_xlabel("Number of training ratings per item")
        ax.set_ylabel("Share (%)")
        ax.set_title("Item popularity and held-out relevant records")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "08_popularity_and_test_relevance.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 图2：不同热门程度下的评分预测误差。
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for model_name in ["GlobalMean", "UserMean", "SVD"]:
            part = (
                errors_table.loc[
                    errors_table["model"].eq(model_name)
                ]
                .set_index("popularity_group")
                .reindex(GROUPS)
            )
            ax.plot(
                GROUPS,
                part["RMSE"],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel("Number of training ratings per item")
        ax.set_ylabel("RMSE")
        ax.set_title("Rating prediction error by item popularity")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "09_rating_error_by_popularity.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        config = {
            "relevant_rating_threshold": RELEVANT_THRESHOLD,
            "popularity_source": "training rating counts only",
            "groups": GROUPS,
            "head_definition": (
                "Top ceil(10% of training catalog) items by "
                "training rating count; product_id breaks ties."
            ),
            "scope": (
                "All test records; this is not the restricted "
                "user cohort used in the earlier Top-N evaluation."
            ),
            "notes": [
                "Rating frequency is not sales volume.",
                "Unseen items are reported separately.",
                "Relevant means held-out rating >= 4.",
                "This step does not calculate new Top-N metrics.",
                "No model retraining or input modification.",
            ],
        }

        (TABLE_DIR / "recommendation_popularity_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("=== 数据核对 ===")
        print(f"训练记录数：{len(train):,}")
        print(f"测试记录数：{len(evaluated):,}")
        print(f"训练集商品数：{len(items):,}")
        print(f"测试集中评分达到4分的记录数：{total_relevant:,}")
        print("预测文件与测试集核对通过。")

        print("\n=== 商品热门程度与测试记录分布 ===")
        print(summary.round(4).to_string(index=False))

        print("\n=== 不同热门程度下的评分预测误差 ===")
        print(errors_table.round(4).to_string(index=False))

        print("\n=== 训练集中最热门的前10%商品 ===")
        print(concentration.round(4).to_string(index=False))

        print(f"\n统计表：{TABLE_DIR}")
        print(f"图片：{FIGURE_DIR}")
        print("完成。本步没有重新训练模型，也没有修改输入文件。")


    main()


if __name__ == "__main__":
    run()

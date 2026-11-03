"""任务36：recommend_products_demo。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import argparse
    import json

    import numpy as np
    import pandas as pd
    from surprise import dump


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"
    MODEL_PATH = ROOT / "models" / "week3" / "recommendation_svd.model"


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--user",
            default=None,
            help="用户编号；不填写时使用训练集中排序后的第一位用户",
        )
        parser.add_argument("--top", type=int, default=10)
        args = parser.parse_args()

        if args.top < 1:
            raise ValueError("推荐数量必须大于0。")

        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        train = pd.read_csv(
            DATA_DIR / "recommendation_train.csv",
            usecols=["user_id", "product_id", "rating"],
            dtype={
                "user_id": "string",
                "product_id": "string",
                "rating": "float64",
            },
        )
        _, model = dump.load(str(MODEL_PATH))

        users = set(train["user_id"])
        user = args.user if args.user is not None else sorted(users)[0]
        known_user = user in users

        history = train[train["user_id"] == user]
        rated_items = set(history["product_id"])

        item_stats = (
            train.groupby("product_id")
            .agg(
                train_rating_count=("rating", "size"),
                train_mean_rating=("rating", "mean"),
            )
            .reset_index()
        )

        candidates = item_stats[
            ~item_stats["product_id"].isin(rated_items)
        ].copy()

        if candidates.empty:
            raise ValueError("没有可推荐的候选商品。")

        top_n = min(args.top, len(candidates))

        print("=== 推荐原型 ===")
        print("用户编号：", user)
        print("训练中是否出现过该用户：", known_user)
        print("该用户训练评分记录数：", len(history))
        print(f"候选商品数：{len(candidates):,}")
        print("本次推荐数量：", top_n)

        outputs = []

        # 热门程度使用训练集中的评分次数，不是销量。
        popular = candidates.sort_values(
            ["train_rating_count", "product_id"],
            ascending=[False, True],
        ).head(top_n).copy()

        popular["method"] = (
            "Popularity" if known_user else "Popularity_new_user"
        )
        popular["ranking_score"] = popular["train_rating_count"]
        popular["predicted_rating"] = np.nan
        popular["rank"] = range(1, len(popular) + 1)
        outputs.append(popular)

        if known_user:
            print("\n正在计算SVD推荐……", flush=True)

            scores = []
            for number, item in enumerate(
                candidates["product_id"], start=1
            ):
                score = model.predict(
                    str(user), str(item), clip=False
                ).est
                scores.append(float(score))

                if number % 25_000 == 0:
                    print(f"已计算 {number:,} 个商品", flush=True)

            candidates["ranking_score"] = scores

            if not np.isfinite(
                candidates["ranking_score"].to_numpy()
            ).all():
                raise ValueError("预测分出现异常。")

            svd = candidates.sort_values(
                ["ranking_score", "product_id"],
                ascending=[False, True],
            ).head(top_n).copy()

            svd["method"] = "SVD"
            # 排序用未截断分数；展示的预测评分限制在1至5分。
            svd["predicted_rating"] = svd["ranking_score"].clip(1, 5)
            svd["rank"] = range(1, len(svd) + 1)
            outputs.append(svd)

        else:
            print("\n这是未见用户，使用热门商品作为默认推荐。")

        result = pd.concat(outputs, ignore_index=True)
        result["user_id"] = user

        if result["product_id"].isin(rated_items).any():
            raise ValueError("推荐结果包含该用户训练中已评分的商品。")

        result = result[
            [
                "user_id", "method", "rank", "product_id",
                "ranking_score", "predicted_rating",
                "train_rating_count", "train_mean_rating",
            ]
        ]

        # 固定两个演示输出名称，避免多次运行产生大量文件。
        suffix = "known_user" if known_user else "new_user"

        output_path = TABLE_DIR / f"recommendation_demo_{suffix}.csv"
        result.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        config = {
            "user_id": user,
            "known_user": known_user,
            "training_history_rows": len(history),
            "candidate_count": len(candidates),
            "requested_top_n": args.top,
            "returned_per_method": top_n,
            "candidate_source": "all_training_products",
            "excluded_items": "user_training_rated_items",
            "test_data_used": False,
            "model_retrained": False,
            "tie_breaking": "product_id_ascending",
            "note": (
                "Training-state demonstration, not a new evaluation. "
                "Held-out test items are not excluded because test data "
                "is not read. Rating counts are not sales counts."
            ),
        }

        with open(
            TABLE_DIR / f"recommendation_demo_{suffix}_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        for method, group in result.groupby("method", sort=False):
            print(f"\n=== {method} 推荐结果 ===")
            columns = ["rank", "product_id", "train_rating_count"]
            if method == "SVD":
                columns += ["predicted_rating"]
            print(group[columns].round(4).to_string(index=False))

        print("\n已检查：推荐结果不包含该用户训练中已评分的商品。")
        print("结果文件：", output_path)
        print("完成。没有读取测试集，也没有重新训练模型。")


    main()


if __name__ == "__main__":
    run()

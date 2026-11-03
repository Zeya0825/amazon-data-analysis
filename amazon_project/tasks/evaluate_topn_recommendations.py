"""任务35：evaluate_topn_recommendations。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from surprise import dump


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "figures"
    MODEL_PATH = ROOT / "models" / "week3" / "recommendation_svd.model"

    SEED = 42
    UNRATED_SAMPLE_SIZE = 100
    RELEVANT_RATING = 4.0
    K_VALUES = [5, 10]


    def read_data(path):
        return pd.read_csv(
            path,
            usecols=["user_id", "product_id", "rating"],
            dtype={
                "user_id": "string",
                "product_id": "string",
                "rating": "float64",
            },
        )


    def main():
        for folder in [TABLE_DIR, FIGURE_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        train = read_data(DATA_DIR / "recommendation_train.csv")
        test = read_data(DATA_DIR / "recommendation_test.csv")
        _, model = dump.load(str(MODEL_PATH))

        popularity = train.groupby("product_id").size().to_dict()
        catalog = np.array(sorted(popularity), dtype=object)
        catalog_set = set(catalog)

        train_items = (
            train.groupby("user_id")["product_id"].agg(set).to_dict()
        )
        test_groups = {
            user: group
            for user, group in test.groupby("user_id", sort=True)
        }

        rng = np.random.default_rng(SEED)
        user_results = []
        candidate_checks = []

        skipped_no_known_test = 0
        skipped_no_relevant = 0
        example_rows = []

        print("开始评估抽样候选集上的推荐效果……", flush=True)

        for number, user in enumerate(sorted(test_groups), start=1):
            group = test_groups[user]
            known_test = group[group["product_id"].isin(catalog_set)]

            if known_test.empty:
                skipped_no_known_test += 1
                continue

            relevant = set(
                known_test.loc[
                    known_test["rating"] >= RELEVANT_RATING,
                    "product_id",
                ]
            )

            # 没有已知喜欢商品时，召回率分母为0。
            # 本次两种方法均只评价至少有一个可命中目标的用户。
            if not relevant:
                skipped_no_relevant += 1
                continue

            # 排除用户在训练集、测试集中所有已评分商品，
            # 再抽取未评分商品。
            rated_items = train_items[user] | set(group["product_id"])
            available_count = len(catalog) - len(
                rated_items & catalog_set
            )

            if available_count < UNRATED_SAMPLE_SIZE:
                raise ValueError("可抽取的未评分商品不足100个。")

            sampled = set()
            while len(sampled) < UNRATED_SAMPLE_SIZE:
                draws = rng.choice(
                    catalog,
                    size=UNRATED_SAMPLE_SIZE * 2,
                    replace=False,
                )
                for item in draws:
                    if item not in rated_items:
                        sampled.add(item)
                    if len(sampled) == UNRATED_SAMPLE_SIZE:
                        break

            candidates = np.array(
                sorted(set(known_test["product_id"]) | sampled),
                dtype=object,
            )

            # 先随机排列，分数相同时按同一随机顺序打破平局。
            candidates = candidates[rng.permutation(len(candidates))]

            if set(candidates) & train_items[user]:
                raise ValueError("候选商品中包含该用户的训练商品。")

            # 排序使用未截断的预测分，避免超过5分的结果全部并列。
            svd_scores = np.array([
                model.predict(str(user), str(item), clip=False).est
                for item in candidates
            ])
            popular_scores = np.array([
                popularity[item] for item in candidates
            ])

            if not np.isfinite(svd_scores).all():
                raise ValueError("SVD预测分存在异常。")

            scores_by_model = {
                "Popularity": popular_scores,
                "SVD": svd_scores,
            }

            candidate_checks.append({
                "user_id": user,
                "candidate_count": len(candidates),
                "known_test_items": len(known_test),
                "sampled_unrated_items": len(sampled),
                "relevant_test_items": len(relevant),
            })

            for model_name, scores in scores_by_model.items():
                order = np.argsort(-scores, kind="stable")

                for k in K_VALUES:
                    top_items = candidates[order[:k]]
                    hits = len(set(top_items) & relevant)

                    user_results.append({
                        "user_id": user,
                        "model": model_name,
                        "k": k,
                        "hits": hits,
                        "relevant_count": len(relevant),
                        "precision_at_k": hits / k,
                        "recall_at_k": hits / len(relevant),
                        "hit_rate_at_k": int(hits > 0),
                    })

                # 固定保存最先遇到的3位合格用户，不挑效果好的用户。
                if len(candidate_checks) <= 3:
                    for rank, index in enumerate(order[:10], start=1):
                        item = candidates[index]
                        example_rows.append({
                            "user_id": user,
                            "model": model_name,
                            "rank": rank,
                            "product_id": item,
                            "ranking_score": float(scores[index]),
                            "is_known_relevant": item in relevant,
                            "candidate_source": (
                                "sampled_unrated"
                                if item in sampled
                                else "held_out_rating"
                            ),
                        })

            if number % 1000 == 0:
                print(
                    f"已检查 {number:,} 位用户，"
                    f"纳入评估 {len(candidate_checks):,} 位",
                    flush=True,
                )

        if not user_results:
            raise ValueError("没有符合评估条件的用户。")

        details = pd.DataFrame(user_results)
        candidates_df = pd.DataFrame(candidate_checks)

        # 每位用户先计算指标，再取平均，即用户等权平均。
        metrics = (
            details.groupby(["model", "k"], as_index=False)
            .agg(
                evaluated_users=("user_id", "size"),
                precision_at_k=("precision_at_k", "mean"),
                recall_at_k=("recall_at_k", "mean"),
                hit_rate_at_k=("hit_rate_at_k", "mean"),
            )
        )

        scope = pd.DataFrame([{
            "total_test_users": len(test_groups),
            "evaluated_users": len(candidate_checks),
            "excluded_no_seen_test_items": skipped_no_known_test,
            "excluded_no_relevant_seen_test_items": skipped_no_relevant,
            "test_rows_on_unseen_products": int(
                (~test["product_id"].isin(catalog_set)).sum()
            ),
            "relevant_rating_threshold": RELEVANT_RATING,
            "sampled_unrated_items_per_user": UNRATED_SAMPLE_SIZE,
            "min_candidate_count": int(
                candidates_df["candidate_count"].min()
            ),
            "mean_candidate_count": float(
                candidates_df["candidate_count"].mean()
            ),
            "max_candidate_count": int(
                candidates_df["candidate_count"].max()
            ),
        }])

        if (
            len(candidate_checks)
            + skipped_no_known_test
            + skipped_no_relevant
            != len(test_groups)
        ):
            raise ValueError("用户数量核对失败。")

        metrics.to_csv(
            TABLE_DIR / "recommendation_topn_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        scope.to_csv(
            TABLE_DIR / "recommendation_topn_scope.csv",
            index=False, encoding="utf-8-sig",
        )
        pd.DataFrame(example_rows).to_csv(
            TABLE_DIR / "recommendation_topn_examples.csv",
            index=False, encoding="utf-8-sig",
        )

        details.to_csv(
            DATA_DIR / "recommendation_topn_user_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        candidates_df.to_csv(
            DATA_DIR / "recommendation_candidate_counts.csv",
            index=False, encoding="utf-8-sig",
        )

        config = {
            "seed": SEED,
            "k_values": K_VALUES,
            "relevant_rating_threshold": RELEVANT_RATING,
            "sampled_unrated_items_per_user": UNRATED_SAMPLE_SIZE,
            "candidate_catalog": "training_products_only",
            "candidates": "all_seen_test_items_plus_sampled_unrated_items",
            "evaluated_users": "at_least_one_relevant_seen_test_item",
            "aggregation": "macro_average_equal_weight_per_user",
            "tie_breaking": "shared_random_candidate_order",
            "svd_ranking_score": "unclipped_prediction",
            "popularity_score": "training_rating_count",
            "limitations": [
                "Sampled-candidate evaluation, not full-catalog evaluation.",
                "Unrated items have unknown relevance; they are not confirmed negatives.",
                "Recall denominator excludes unseen-product test positives.",
                "Results concern selected active users with evaluable positives.",
                "Example lists come from evaluation candidates, not full-catalog serving.",
            ],
        }
        with open(
            TABLE_DIR / "recommendation_topn_config.json",
            "w", encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric, title in zip(
            axes,
            ["precision_at_k", "recall_at_k"],
            ["Precision@K", "Recall@K"],
        ):
            for model_name, color in [
                ("Popularity", "#7195B5"),
                ("SVD", "#398675"),
            ]:
                part = metrics[metrics["model"] == model_name]
                ax.plot(
                    part["k"], part[metric],
                    marker="o", label=model_name, color=color,
                )
            ax.set_xticks(K_VALUES)
            ax.set_xlabel("K")
            ax.set_ylabel(title)
            ax.set_title(title + " on sampled candidates")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "05_recommendation_topn_metrics.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(fig)

        print("\n=== 推荐评估范围 ===")
        print(scope.round(4).to_string(index=False))
        print("\n=== 推荐指标（0至1）===")
        print(metrics.round(4).to_string(index=False))
        print("\n统计表：", TABLE_DIR)
        print("图片：", FIGURE_DIR / "05_recommendation_topn_metrics.png")
        print("完成。没有重新训练模型，也没有修改训练集和测试集。")


    main()


if __name__ == "__main__":
    run()

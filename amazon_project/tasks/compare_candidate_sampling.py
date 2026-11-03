"""任务39：compare_candidate_sampling。保留原始分析口径与输出文件名。"""

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
    TABLE_DIR = ROOT / "reports" / "week3" / "revision" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "revision" / "figures"

    SEED = 42
    MAX_USERS = 2000
    KS = [5, 10]
    SCENARIOS = [
        ("Uniform100", 100, False),
        ("Popularity100", 100, True),
        ("Uniform500", 500, False),
    ]


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        options = {
            "dtype": {"user_id": "string", "product_id": "string"},
            "float_precision": "round_trip",
        }

        train = pd.read_csv(
            DATA_DIR / "recommendation_train.csv", **options
        )
        test = pd.read_csv(
            DATA_DIR / "recommendation_test.csv", **options
        )
        _, model = dump.load(
            str(ROOT / "models" / "week3" / "recommendation_svd.model")
        )

        for frame in [train, test]:
            frame["rating"] = pd.to_numeric(
                frame["rating"], errors="raise"
            )
            if not frame["rating"].between(1, 5).all():
                raise ValueError("发现异常评分。")
            if frame[["user_id", "product_id"]].isna().any().any():
                raise ValueError("发现缺失的用户或商品编号。")
            if frame.duplicated(["user_id", "product_id"]).any():
                raise ValueError("发现重复的用户—商品组合。")

        counts = train.groupby("product_id").size()
        catalog = np.array(sorted(counts.index), dtype=object)
        catalog_set = set(catalog)

        if model.trainset.n_ratings != len(train):
            raise ValueError("模型训练记录数与当前训练集不一致。")

        model_items = {
            model.trainset.to_raw_iid(i)
            for i in model.trainset.all_items()
        }
        if model_items != catalog_set:
            raise ValueError("模型商品目录与当前训练集不一致。")

        train_sets = (
            train.groupby("user_id")["product_id"].agg(set).to_dict()
        )
        test_groups = {
            uid: part for uid, part in test.groupby("user_id")
        }

        # 所有设置使用同一批符合条件的用户。
        eligible = []
        for uid in sorted(test_groups):
            part = test_groups[uid]
            seen = part["product_id"].isin(catalog_set)
            relevant = seen & part["rating"].ge(4)

            if uid in train_sets and relevant.any():
                eligible.append(uid)

        if not eligible:
            raise ValueError("没有符合条件的评估用户。")

        user_rng = np.random.default_rng(SEED)
        selected = sorted(
            user_rng.choice(
                np.array(eligible, dtype=object),
                size=min(MAX_USERS, len(eligible)),
                replace=False,
            ).tolist()
        )

        # 本次将训练评分次数最多的前10%定义为头部商品，
        # 其余90%定义为长尾商品，方便明确统计口径。
        item_table = counts.rename("count").reset_index()
        item_table = item_table.sort_values(
            ["count", "product_id"], ascending=[False, True]
        )
        head_size = int(np.ceil(len(catalog) * 0.10))
        head_ids = set(item_table.head(head_size)["product_id"])

        user_rows = []
        scope_rows = []

        coverage = {
            (name, method, k): set()
            for name, _, _ in SCENARIOS
            for method in ["Popularity", "SVD"]
            for k in KS
        }

        print(f"符合条件的用户：{len(eligible):,}")
        print(f"本次共同评估用户：{len(selected):,}")
        print("开始候选集对照实验，请等待完成提示……", flush=True)

        for position, uid in enumerate(selected):
            part = test_groups[uid]
            all_test_items = set(part["product_id"])
            known_test_items = all_test_items & catalog_set
            relevant_items = set(
                part.loc[part["rating"].ge(4), "product_id"]
            ) & catalog_set

            head_relevant = relevant_items & head_ids
            tail_relevant = relevant_items - head_ids

            # 未评分候选排除该用户全部训练、测试商品。
            excluded = train_sets[uid] | all_test_items
            pool = np.array(
                [item for item in catalog if item not in excluded],
                dtype=object,
            )
            if len(pool) < 500:
                raise ValueError("可抽样商品不足500件。")

            uniform_rng = np.random.default_rng(
                np.random.SeedSequence([SEED, position, 1])
            )
            uniform500 = uniform_rng.choice(
                pool, size=500, replace=False
            )
            # 两种等概率候选设置采用嵌套抽样。
            uniform100 = uniform500[:100]

            weights = counts.reindex(pool).to_numpy(dtype=float)
            weights /= weights.sum()

            weighted_rng = np.random.default_rng(
                np.random.SeedSequence([SEED, position, 2])
            )
            weighted100 = weighted_rng.choice(
                pool, size=100, replace=False, p=weights
            )

            samples = {
                "Uniform100": uniform100,
                "Popularity100": weighted100,
                "Uniform500": uniform500,
            }

            # 同一用户、同一商品只计算一次SVD得分。
            union_items = set(known_test_items)
            for sample in samples.values():
                union_items.update(sample.tolist())

            svd_scores = {
                item: model.predict(uid, item, clip=False).est
                for item in sorted(union_items)
            }

            for scenario_index, (name, n_extra, weighted) in enumerate(
                SCENARIOS
            ):
                sample = samples[name]
                candidate_set = known_test_items | set(sample)
                candidates = np.array(
                    sorted(candidate_set), dtype=object
                )

                # 两种算法共用随机打乱后的次序处理同分商品。
                tie_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [SEED, position, 10 + scenario_index]
                    )
                )
                candidates = candidates[
                    tie_rng.permutation(len(candidates))
                ]

                if set(candidates) & train_sets[uid]:
                    raise ValueError("候选集中出现训练已评分商品。")

                sampled_counts = counts.reindex(
                    sample
                ).to_numpy(dtype=float)

                scope_rows.append({
                    "user_id": uid,
                    "scenario": name,
                    "candidate_count": len(candidates),
                    "sampled_unrated_count": n_extra,
                    "sampled_mean_train_count":
                        float(sampled_counts.mean()),
                    "sampled_head_share":
                        float(np.mean([x in head_ids for x in sample])),
                })

                score_arrays = {
                    "Popularity": counts.reindex(
                        candidates
                    ).to_numpy(dtype=float),
                    "SVD": np.array(
                        [svd_scores[item] for item in candidates]
                    ),
                }

                for method, scores in score_arrays.items():
                    order = np.argsort(-scores, kind="stable")

                    for k in KS:
                        recommended = candidates[order[:k]].tolist()
                        rec_set = set(recommended)
                        hits = len(rec_set & relevant_items)

                        head_hits = len(rec_set & head_relevant)
                        tail_hits = len(rec_set & tail_relevant)
                        tail_slots = sum(
                            item not in head_ids for item in recommended
                        )

                        coverage[(name, method, k)].update(rec_set)

                        user_rows.append({
                            "user_id": uid,
                            "scenario": name,
                            "model": method,
                            "k": k,
                            "hits": hits,
                            "precision": hits / k,
                            "recall": hits / len(relevant_items),
                            "hit_rate": int(hits > 0),
                            "head_hits": head_hits,
                            "head_relevant": len(head_relevant),
                            "tail_hits": tail_hits,
                            "tail_relevant": len(tail_relevant),
                            "tail_slots": tail_slots,
                        })

            if (position + 1) % 200 == 0:
                print(
                    f"已完成 {position + 1:,}/{len(selected):,} 位用户",
                    flush=True,
                )

        details = pd.DataFrame(user_rows)
        scope = pd.DataFrame(scope_rows)

        results = []

        for (scenario, method, k), group in details.groupby(
            ["scenario", "model", "k"], sort=False
        ):
            head_denominator = group["head_relevant"].sum()
            tail_denominator = group["tail_relevant"].sum()

            results.append({
                "scenario": scenario,
                "model": method,
                "k": k,
                "evaluated_users": len(group),
                "precision_at_k": group["precision"].mean(),
                "recall_at_k": group["recall"].mean(),
                "hit_rate_at_k": group["hit_rate"].mean(),
                "head_recall_micro":
                    group["head_hits"].sum() / head_denominator
                    if head_denominator else np.nan,
                "tail_recall_micro":
                    group["tail_hits"].sum() / tail_denominator
                    if tail_denominator else np.nan,
                "tail_recommendation_share":
                    group["tail_slots"].sum() / (len(group) * k),
                "catalog_coverage":
                    len(coverage[(scenario, method, k)]) / len(catalog),
            })

        result = pd.DataFrame(results)

        scope_summary = scope.groupby("scenario", sort=False).agg(
            evaluated_users=("user_id", "size"),
            min_candidate_count=("candidate_count", "min"),
            mean_candidate_count=("candidate_count", "mean"),
            max_candidate_count=("candidate_count", "max"),
            mean_sampled_train_count=(
                "sampled_mean_train_count", "mean"
            ),
            mean_sampled_head_share=("sampled_head_share", "mean"),
        ).reset_index()

        result.to_csv(
            TABLE_DIR / "candidate_sampling_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        scope_summary.to_csv(
            TABLE_DIR / "candidate_sampling_scope.csv",
            index=False, encoding="utf-8-sig",
        )

        # 用户级中间结果留在本地，不需要交给老师。
        details.to_csv(
            DATA_DIR / "candidate_sampling_user_metrics.csv",
            index=False, encoding="utf-8-sig",
        )

        config = {
            "seed": SEED,
            "eligible_users": len(eligible),
            "evaluated_users": len(selected),
            "maximum_users": MAX_USERS,
            "k": KS,
            "relevant_threshold": 4,
            "scenarios": [name for name, _, _ in SCENARIOS],
            "head_definition": (
                "Top ceil(10% of training catalog) by training "
                "rating count, product_id breaks ties"
            ),
            "tail_definition": "Remaining 90% of training catalog",
            "notes": [
                "Same sampled users in all scenarios.",
                "Same candidates for both models within a scenario.",
                "All seen test items retained in every candidate set.",
                "Sampled unrated items have unknown relevance.",
                "Overall precision/recall/hit rate are user macro means.",
                "Head/tail recall are pooled hits divided by pooled relevant counts.",
                "Weighted sampling uses training counts, not actual exposure logs.",
                "No retraining, tuning or full-catalog ranking evaluation.",
            ],
        }
        (TABLE_DIR / "candidate_sampling_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        scenario_names = [name for name, _, _ in SCENARIOS]
        x = np.arange(len(scenario_names))
        width = 0.36

        for ax, metric, title in [
            (axes[0], "recall_at_k", "Recall@10"),
            (axes[1], "tail_recall_micro", "Long-tail pooled recall@10"),
        ]:
            for index, method in enumerate(["Popularity", "SVD"]):
                part = result.loc[
                    result["model"].eq(method) & result["k"].eq(10)
                ].set_index("scenario").reindex(scenario_names)

                ax.bar(
                    x + (index - 0.5) * width,
                    part[metric],
                    width,
                    label=method,
                )

            ax.set_xticks(x, scenario_names, rotation=15)
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "10_candidate_sampling_comparison.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(fig)

        print("\n=== 候选集范围 ===")
        print(scope_summary.round(4).to_string(index=False))

        print("\n=== 推荐效果对比 ===")
        print(result[
            [
                "scenario", "model", "k", "evaluated_users",
                "precision_at_k", "recall_at_k", "hit_rate_at_k",
            ]
        ].round(4).to_string(index=False))

        print("\n=== 头部、长尾及覆盖情况 ===")
        print(result[
            [
                "scenario", "model", "k",
                "head_recall_micro", "tail_recall_micro",
                "tail_recommendation_share", "catalog_coverage",
            ]
        ].round(4).to_string(index=False))

        print(f"\n统计表：{TABLE_DIR}")
        print(f"图片：{FIGURE_DIR}")
        print("完成。模型、训练集和测试集均未修改。")


    main()


if __name__ == "__main__":
    run()

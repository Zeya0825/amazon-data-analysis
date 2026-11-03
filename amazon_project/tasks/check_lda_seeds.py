"""任务37：check_lda_seeds。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from itertools import combinations
    import json
    import time

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel
    from scipy.optimize import linear_sum_assignment


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    MODEL_DIR = ROOT / "models" / "week3" / "revision"
    TABLE_DIR = ROOT / "reports" / "week3" / "revision" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "revision" / "figures"

    SEEDS = [42, 123, 2026]
    TOPIC_COUNTS = [3, 4, 5, 6]
    PASSES = 15
    ITERATIONS = 100
    TOP_WORDS = 10


    def save_table(rows, filename):
        pd.DataFrame(rows).to_csv(
            TABLE_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )


    def main():
        for folder in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            DATA_DIR / "topic_model_input.csv",
            dtype={"text_id": "string", "product_id": "string"},
            float_precision="round_trip",
        )
        dictionary = Dictionary.load(
            str(DATA_DIR / "topic_dictionary.dict")
        )

        texts = [
            json.loads(value)
            for value in df["topic_tokens_json"]
        ]

        if not texts or not len(dictionary):
            raise ValueError("文本或词典为空。")

        for words in texts:
            if not isinstance(words, list) or not words:
                raise ValueError("发现空文本或格式错误的词语列表。")
            if any(word not in dictionary.token2id for word in words):
                raise ValueError("文本中的词语与词典不一致。")

        corpus = [dictionary.doc2bow(words) for words in texts]

        print("=== 本次实验 ===")
        print(f"文本数量：{len(texts):,}")
        print(f"词典大小：{len(dictionary):,}")
        print(f"主题数量：{TOPIC_COUNTS}")
        print(f"随机种子：{SEEDS}")
        print("共训练12个模型，请等待完成提示……", flush=True)

        metric_rows = []
        topic_rows = []
        alignment_rows = []
        stability_rows = []

        completed = 0

        for k in TOPIC_COUNTS:
            # 保存同一主题数量下各个种子的前10词，供配对比较。
            seed_topic_words = {}

            for seed in SEEDS:
                start = time.perf_counter()

                print(
                    f"\n正在训练：主题数={k}，seed={seed}",
                    flush=True,
                )

                model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=k,
                    random_state=seed,
                    passes=PASSES,
                    iterations=ITERATIONS,
                    update_every=0,
                    alpha="symmetric",
                    eta="symmetric",
                    eval_every=None,
                )

                coherence_model = CoherenceModel(
                    model=model,
                    texts=texts,
                    dictionary=dictionary,
                    coherence="c_v",
                    topn=TOP_WORDS,
                    processes=1,
                )

                per_topic_scores = (
                    coherence_model.get_coherence_per_topic()
                )
                coherence = float(np.mean(per_topic_scores))

                word_lists = []
                for topic_id in range(k):
                    words = [
                        word
                        for word, weight in model.show_topic(
                            topic_id, topn=TOP_WORDS
                        )
                    ]
                    word_lists.append(words)

                    topic_rows.append({
                        "num_topics": k,
                        "random_seed": seed,
                        "topic_number": topic_id + 1,
                        "topic_coherence_cv": float(
                            per_topic_scores[topic_id]
                        ),
                        "top_words": ", ".join(words),
                    })

                seed_topic_words[seed] = word_lists

                all_words = [
                    word for words in word_lists for word in words
                ]
                diversity = len(set(all_words)) / len(all_words)

                model.save(
                    str(MODEL_DIR / f"lda_k{k}_seed{seed}.model")
                )

                metric_rows.append({
                    "num_topics": k,
                    "random_seed": seed,
                    "text_count": len(texts),
                    "vocabulary_size": len(dictionary),
                    "coherence_cv": coherence,
                    "top_word_diversity": diversity,
                    "elapsed_seconds": time.perf_counter() - start,
                })

                # 每完成一次就保存，方便查看进度。
                save_table(metric_rows, "lda_seed_metrics.csv")
                save_table(topic_rows, "lda_seed_topics.csv")

                completed += 1
                print(
                    f"已完成 {completed}/12："
                    f"一致性={coherence:.4f}，"
                    f"不同关键词占比={diversity:.4f}",
                    flush=True,
                )

            # 不同模型的“主题1”不一定表示同一内容。
            # 根据关键词重合程度，寻找整体最匹配的主题配对。
            for seed_a, seed_b in combinations(SEEDS, 2):
                topics_a = seed_topic_words[seed_a]
                topics_b = seed_topic_words[seed_b]

                similarities = np.zeros((k, k))

                for i in range(k):
                    for j in range(k):
                        words_a = set(topics_a[i])
                        words_b = set(topics_b[j])

                        similarities[i, j] = (
                            len(words_a & words_b)
                            / len(words_a | words_b)
                        )

                rows, cols = linear_sum_assignment(-similarities)
                matched_scores = similarities[rows, cols]

                for i, j in zip(rows, cols):
                    alignment_rows.append({
                        "num_topics": k,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "topic_a": int(i + 1),
                        "topic_b": int(j + 1),
                        "top10_jaccard": float(similarities[i, j]),
                        "words_a": ", ".join(topics_a[i]),
                        "words_b": ", ".join(topics_b[j]),
                    })

                stability_rows.append({
                    "num_topics": k,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "mean_matched_top10_jaccard": float(
                        np.mean(matched_scores)
                    ),
                    "min_matched_top10_jaccard": float(
                        np.min(matched_scores)
                    ),
                })

        metrics = pd.DataFrame(metric_rows)
        stability = pd.DataFrame(stability_rows)

        summary = metrics.groupby("num_topics").agg(
            run_count=("random_seed", "size"),
            coherence_mean=("coherence_cv", "mean"),
            coherence_std=("coherence_cv", "std"),
            coherence_min=("coherence_cv", "min"),
            coherence_max=("coherence_cv", "max"),
            diversity_mean=("top_word_diversity", "mean"),
        ).reset_index()

        stability_summary = stability.groupby("num_topics").agg(
            mean_pairwise_top10_jaccard=(
                "mean_matched_top10_jaccard", "mean"
            ),
            lowest_pair_mean_jaccard=(
                "mean_matched_top10_jaccard", "min"
            ),
        ).reset_index()

        summary = summary.merge(
            stability_summary,
            on="num_topics",
            validate="one_to_one",
        )

        save_table(alignment_rows, "lda_seed_topic_alignment.csv")
        save_table(stability_rows, "lda_seed_pair_stability.csv")

        summary.to_csv(
            TABLE_DIR / "lda_seed_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 图1：三个种子各自的结果，以及均值和标准差。
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for seed in SEEDS:
            part = metrics.loc[
                metrics["random_seed"].eq(seed)
            ].sort_values("num_topics")

            ax.plot(
                part["num_topics"],
                part["coherence_cv"],
                marker="o",
                alpha=0.7,
                label=f"Seed {seed}",
            )

        ax.errorbar(
            summary["num_topics"],
            summary["coherence_mean"],
            yerr=summary["coherence_std"],
            fmt="ks--",
            capsize=5,
            linewidth=2,
            label="Mean +/- sample SD",
        )

        ax.set_xticks(TOPIC_COUNTS)
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("C_v coherence")
        ax.set_title("LDA coherence across three random seeds")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "06_lda_seed_coherence.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 图2：配对后的关键词稳定性。
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for seed_a, seed_b in combinations(SEEDS, 2):
            part = stability.loc[
                stability["seed_a"].eq(seed_a)
                & stability["seed_b"].eq(seed_b)
            ].sort_values("num_topics")

            ax.plot(
                part["num_topics"],
                part["mean_matched_top10_jaccard"],
                marker="o",
                label=f"{seed_a} vs {seed_b}",
            )

        ax.set_xticks(TOPIC_COUNTS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("Mean matched top-10 Jaccard similarity")
        ax.set_title("Topic keyword stability after matching")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "07_lda_seed_topic_stability.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        config = {
            "seeds": SEEDS,
            "topic_counts": TOPIC_COUNTS,
            "passes": PASSES,
            "iterations": ITERATIONS,
            "update_every": 0,
            "alpha": "symmetric",
            "eta": "symmetric",
            "eval_every": None,
            "coherence": "c_v",
            "topn": TOP_WORDS,
            "coherence_processes": 1,
            "text_count": len(texts),
            "vocabulary_size": len(dictionary),
            "topic_matching": (
                "One-to-one matching maximizing total "
                "top-10 keyword Jaccard similarity"
            ),
            "notes": [
                "Same input texts and dictionary for all runs.",
                "Coherence is descriptive, not classification accuracy.",
                "Sample SD across three seeds is not a confidence interval.",
                "Keyword matching does not prove semantic equivalence.",
                "Original models and reports are not overwritten.",
            ],
        }

        (TABLE_DIR / "lda_seed_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n=== 12次实验结果 ===")
        print(
            metrics[
                [
                    "num_topics",
                    "random_seed",
                    "coherence_cv",
                    "top_word_diversity",
                ]
            ].round(4).to_string(index=False)
        )

        print("\n=== 不同主题数量的汇总 ===")
        print(summary.round(4).to_string(index=False))

        print("\n=== 五主题模型：不同种子的关键词 ===")
        topic_table = pd.DataFrame(topic_rows)

        for seed in SEEDS:
            print(f"\nseed={seed}")
            selected = topic_table.loc[
                topic_table["num_topics"].eq(5)
                & topic_table["random_seed"].eq(seed)
            ]
            for row in selected.itertuples(index=False):
                print(f"主题 {row.topic_number}：{row.top_words}")

        print("\n说明：关键词相似度越接近1，重合程度越高。")
        print("主题是否含义相同，还需要结合关键词和原文判断。")
        print(f"\n统计表：{TABLE_DIR}")
        print(f"图片：{FIGURE_DIR}")
        print(f"新增模型：{MODEL_DIR}")
        print("完成。原有输入数据、模型和第三周结果均未修改。")


    main()


if __name__ == "__main__":
    run()

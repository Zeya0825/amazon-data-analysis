"""任务28：train_lda_candidates。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json
    import time

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    MODEL_DIR = ROOT / "models" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "figures"

    TOPIC_COUNTS = [3, 4, 5, 6]
    RANDOM_SEED = 42
    PASSES = 15
    ITERATIONS = 100
    TOP_WORDS = 10


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
            raise ValueError("文本或词典为空，请先检查第27步。")

        for words in texts:
            if not isinstance(words, list) or not words:
                raise ValueError("发现格式错误或空的词语列表。")
            if any(word not in dictionary.token2id for word in words):
                raise ValueError("文本词语与词典不一致，请重新运行第27步。")

        corpus = [dictionary.doc2bow(words) for words in texts]

        print("=== 建模数据 ===")
        print(f"文本数量：{len(texts):,}")
        print(f"词典大小：{len(dictionary):,}")
        print(f"待比较的主题数：{TOPIC_COUNTS}")
        print("开始训练，请等待全部模型完成……", flush=True)

        metric_rows = []
        keyword_rows = []
        overview_rows = []

        for topic_count in TOPIC_COUNTS:
            start = time.perf_counter()
            print(
                f"\n正在训练 {topic_count} 个主题的模型……",
                flush=True,
            )

            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=topic_count,
                random_state=RANDOM_SEED,
                passes=PASSES,
                iterations=ITERATIONS,
                update_every=0,
                alpha="symmetric",
                eta="symmetric",
                eval_every=None,
            )

            print("训练完成，正在计算主题一致性……", flush=True)

            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v",
                topn=TOP_WORDS,
                processes=1,
            )
            per_topic_scores = coherence_model.get_coherence_per_topic()
            coherence_score = float(
                sum(per_topic_scores) / len(per_topic_scores)
            )

            # 保存候选模型，之后选定主题数时可以直接使用。
            model.save(
                str(MODEL_DIR / f"lda_k{topic_count}.model")
            )

            all_top_words = []

            print(f"\n=== {topic_count} 个主题：关键词 ===")
            for topic_id in range(topic_count):
                word_weights = model.show_topic(
                    topic_id, topn=TOP_WORDS
                )
                words = [word for word, weight in word_weights]
                all_top_words.extend(words)

                # 展示时从1编号，内部模型仍从0编号。
                topic_number = topic_id + 1

                overview_rows.append({
                    "num_topics": topic_count,
                    "topic_number": topic_number,
                    "topic_coherence_cv": float(
                        per_topic_scores[topic_id]
                    ),
                    "top_words": ", ".join(words),
                })

                for rank, (word, weight) in enumerate(
                    word_weights, start=1
                ):
                    keyword_rows.append({
                        "num_topics": topic_count,
                        "topic_number": topic_number,
                        "word_rank": rank,
                        "word": word,
                        "word_probability": float(weight),
                    })

                print(
                    f"主题 {topic_number}："
                    + ", ".join(words)
                )

            # 前10词中不同词语占比，仅辅助查看主题之间的重复程度。
            word_diversity = (
                len(set(all_top_words)) / len(all_top_words)
            )

            elapsed = time.perf_counter() - start

            metric_rows.append({
                "num_topics": topic_count,
                "text_count": len(texts),
                "vocabulary_size": len(dictionary),
                "coherence_cv": coherence_score,
                "top_word_diversity": word_diversity,
                "random_seed": RANDOM_SEED,
                "passes": PASSES,
                "iterations": ITERATIONS,
                "elapsed_seconds": elapsed,
                "evaluation_scope": "same_corpus_descriptive",
            })

            print(f"主题一致性：{coherence_score:.4f}")
            print(f"前10词的不同词语占比：{word_diversity:.4f}")
            print(f"本模型用时：{elapsed:.1f} 秒", flush=True)

            # 每完成一个模型就保存一次结果。
            pd.DataFrame(metric_rows).to_csv(
                TABLE_DIR / "lda_candidate_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(keyword_rows).to_csv(
                TABLE_DIR / "lda_candidate_keywords.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(overview_rows).to_csv(
                TABLE_DIR / "lda_candidate_topic_overview.csv",
                index=False,
                encoding="utf-8-sig",
            )

        metrics = pd.DataFrame(metric_rows)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            metrics["num_topics"],
            metrics["coherence_cv"],
            marker="o",
            linewidth=2,
            color="#2C6EAA",
        )

        for row in metric_rows:
            ax.annotate(
                f'{row["coherence_cv"]:.4f}',
                (row["num_topics"], row["coherence_cv"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
            )

        ax.set_xticks(TOPIC_COUNTS)
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("C_v coherence")
        ax.set_title("LDA candidate comparison")
        ax.grid(alpha=0.25)
        ax.margins(y=0.25)

        fig.text(
            0.5, 0.02,
            "Descriptive comparison on the same corpus; not accuracy.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(
            FIGURE_DIR / "01_lda_candidate_coherence.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        config = {
            "topic_counts": TOPIC_COUNTS,
            "random_seed": RANDOM_SEED,
            "passes": PASSES,
            "iterations": ITERATIONS,
            "alpha": "symmetric",
            "eta": "symmetric",
            "update_every": 0,
            "coherence": "c_v",
            "coherence_topn": TOP_WORDS,
            "coherence_processes": 1,
            "text_count": len(texts),
            "vocabulary_size": len(dictionary),
            "note": (
                "Exploratory topic comparison on the same corpus. "
                "No final topic count selected automatically. "
                "One random seed used in this initial comparison."
            ),
        }
        with open(
            TABLE_DIR / "lda_training_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        print("\n=== 候选模型比较结果 ===")
        print(
            metrics[
                [
                    "num_topics",
                    "coherence_cv",
                    "top_word_diversity",
                    "elapsed_seconds",
                ]
            ].round(4).to_string(index=False)
        )

        print("\n模型保存位置：", MODEL_DIR)
        print("统计表位置：", TABLE_DIR)
        print("图片：", FIGURE_DIR / "01_lda_candidate_coherence.png")
        print(
            "\n完成。尚未确定最终主题数，"
            "需要结合关键词和原文判断。输入文件没有修改。"
        )


    main()


if __name__ == "__main__":
    run()

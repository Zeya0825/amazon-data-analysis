"""任务29：inspect_topic_examples。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json

    import numpy as np
    import pandas as pd

    from gensim.corpora import Dictionary
    from gensim.models import LdaModel


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    MODEL_DIR = ROOT / "models" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"

    # 每个主题选取3份代表文本。
    EXAMPLES_PER_TOPIC = 3

    # 优先选择至少有20个保留词的文本，
    # 避免极短文本虽然主题权重高，却没有足够内容可阅读。
    MIN_EXAMPLE_TOKENS = 20


    def inspect_model(df, corpus, topic_count):
        model = LdaModel.load(
            str(MODEL_DIR / f"lda_k{topic_count}.model")
        )

        # 固定本次推断的随机状态，方便重复运行核对。
        model.random_state = np.random.RandomState(42)

        distributions = []
        for bow in corpus:
            topic_weights = dict(
                model.get_document_topics(
                    bow,
                    minimum_probability=0.0,
                )
            )
            distributions.append([
                topic_weights.get(topic_id, 0.0)
                for topic_id in range(topic_count)
            ])

        weights = np.asarray(distributions)

        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("主题权重之和异常，请检查模型输出。")

        assignments = df[
            ["text_id", "product_id", "topic_token_count"]
        ].copy()

        # 展示编号从1开始，与第28步一致。
        assignments["dominant_topic"] = weights.argmax(axis=1) + 1
        assignments["dominant_topic_weight"] = weights.max(axis=1)

        for topic_id in range(topic_count):
            assignments[f"topic_{topic_id + 1}_weight"] = (
                weights[:, topic_id]
            )

        assignments.to_csv(
            DATA_DIR / f"lda_k{topic_count}_text_assignments.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows = []
        example_rows = []

        for topic_number in range(1, topic_count + 1):
            topic_id = topic_number - 1
            mask = assignments["dominant_topic"].eq(topic_number)
            assigned = assignments.loc[mask]

            keywords = ", ".join(
                word for word, weight in model.show_topic(
                    topic_id, topn=10
                )
            )

            summary_rows.append({
                "num_topics": topic_count,
                "topic_number": topic_number,
                "dominant_text_count": len(assigned),
                "dominant_text_share_percent": (
                    len(assigned) / len(df) * 100
                ),
                "mean_topic_weight_all_texts": float(
                    weights[:, topic_id].mean()
                ),
                "top_words": keywords,
            })

            eligible = assigned[
                assigned["topic_token_count"] >= MIN_EXAMPLE_TOKENS
            ]

            selected = eligible.sort_values(
                ["dominant_topic_weight", "text_id"],
                ascending=[False, True],
            ).head(EXAMPLES_PER_TOPIC)

            # 候选模型若有小主题，允许少于3份，不重复补样。
            for rank, (index, row) in enumerate(
                selected.iterrows(), start=1
            ):
                example_rows.append({
                    "num_topics": topic_count,
                    "topic_number": topic_number,
                    "example_rank": rank,
                    "text_id": row["text_id"],
                    "product_id": row["product_id"],
                    "main_category": df.loc[index, "main_category"],
                    "topic_token_count": int(row["topic_token_count"]),
                    "dominant_topic_weight": float(
                        row["dominant_topic_weight"]
                    ),
                    "polarity": df.loc[index, "polarity"],
                    "top_words": keywords,
                    "review_text_clean": df.loc[
                        index, "review_text_clean"
                    ],
                })

        summary = pd.DataFrame(summary_rows)
        examples = pd.DataFrame(example_rows)

        summary.to_csv(
            TABLE_DIR / f"lda_k{topic_count}_topic_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if not examples.empty:
            # 完整正文保存在本地分析目录。
            examples.to_csv(
                DATA_DIR / f"lda_k{topic_count}_reading_examples.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # 报告结果目录只保存样例索引。
            examples.drop(columns=["review_text_clean"]).to_csv(
                TABLE_DIR / f"lda_k{topic_count}_example_index.csv",
                index=False,
                encoding="utf-8-sig",
            )

        return summary, examples


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            DATA_DIR / "topic_model_input.csv",
            dtype={"text_id": "string", "product_id": "string"},
            float_precision="round_trip",
        ).reset_index(drop=True)

        dictionary = Dictionary.load(
            str(DATA_DIR / "topic_dictionary.dict")
        )
        texts = [
            json.loads(value)
            for value in df["topic_tokens_json"]
        ]
        corpus = [dictionary.doc2bow(words) for words in texts]

        for topic_count in [4, 5]:
            summary, examples = inspect_model(
                df, corpus, topic_count
            )

            print(f"\n=== {topic_count}主题模型：文本分布 ===")
            print(
                summary[
                    [
                        "topic_number",
                        "dominant_text_count",
                        "dominant_text_share_percent",
                        "mean_topic_weight_all_texts",
                    ]
                ].round(4).to_string(index=False)
            )

            if examples.empty:
                print("没有符合长度条件的阅读样例。")
                continue

            # 4主题全部展示；5主题重点展示安装与服务候选主题。
            # 第28步的5主题模型中，该主题编号为5。
            shown = (
                examples
                if topic_count == 4
                else examples[examples["topic_number"] == 5]
            )

            print(f"\n=== {topic_count}主题模型：阅读样例 ===")

            for _, row in shown.iterrows():
                print(
                    f"\n主题{row['topic_number']} "
                    f"样例{row['example_rank']} "
                    f"文本编号：{row['text_id']}"
                )
                print("商品类别：", row["main_category"])
                print(
                    "主要主题权重："
                    f"{row['dominant_topic_weight']:.4f}"
                )
                print("关键词：", row["top_words"])

                text = str(row["review_text_clean"])
                print(text[:1000])

                if len(text) > 1000:
                    print("【预览已截断，完整文本已保存到CSV】")

        print("\n统计表位置：", TABLE_DIR)
        print("完整阅读样例位置：", DATA_DIR)
        print(
            "\n完成。代表样例按主题权重选取，不是随机抽样；"
            "主题权重不是模型准确率。"
        )
        print("输入数据和已训练模型没有修改。")


    main()


if __name__ == "__main__":
    run()

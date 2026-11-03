"""任务30：analyze_topic_sentiment。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from amazon_project.common import classify_series
    import json

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "figures"

    TOPIC_NAMES = {
        1: "日常使用与操作体验（混合主题）",
        2: "清洁设备与空气处理",
        3: "手机性能与充电配件",
        4: "音频设备与电子产品体验",
        5: "安装设置与售后服务（含打印机使用）",
    }

    LABELS = ["Negative", "Near-neutral", "Positive"]
    THRESHOLDS = [0.1, 0.2]


    def classify(scores, threshold):
        return classify_series(scores, threshold).to_numpy()


    def main():
        for folder in [TABLE_DIR, FIGURE_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        texts = pd.read_csv(
            DATA_DIR / "topic_model_input.csv",
            dtype={"text_id": "string", "product_id": "string"},
            float_precision="round_trip",
        )

        assignments = pd.read_csv(
            DATA_DIR / "lda_k5_text_assignments.csv",
            dtype={"text_id": "string", "product_id": "string"},
            float_precision="round_trip",
        )

        if texts["text_id"].isna().any():
            raise ValueError("文本编号存在缺失。")

        if set(texts["text_id"]) != set(assignments["text_id"]):
            raise ValueError("文本表与主题分配表的编号不一致。")

        df = texts.merge(
            assignments[
                ["text_id", "dominant_topic", "dominant_topic_weight"]
            ],
            on="text_id",
            how="left",
            validate="one_to_one",
        )

        df["polarity"] = pd.to_numeric(
            df["polarity"], errors="coerce"
        )

        if (
            df["polarity"].isna().any()
            or not df["polarity"].between(-1, 1).all()
        ):
            raise ValueError("情感得分缺失或超出范围，请检查输入。")

        if not df["dominant_topic"].isin(TOPIC_NAMES).all():
            raise ValueError("出现未知主题编号。")

        df["topic_name"] = df["dominant_topic"].map(TOPIC_NAMES)

        for threshold in THRESHOLDS:
            df[f"label_t{threshold:.1f}"] = classify(
                df["polarity"], threshold
            )

        # 检查0.1阈值是否与之前保存的结果一致。
        if "sentiment_label" in df.columns:
            mismatch = (
                df["sentiment_label"] != df["label_t0.1"]
            ).sum()
            print("与原情感标签不一致的文本数：", int(mismatch))
            if mismatch:
                raise ValueError(
                    "情感标签与第二周结果不一致，请先核对。"
                )

        summary_rows = []
        sentiment_rows = []

        for topic_number, topic_name in TOPIC_NAMES.items():
            group = df[df["dominant_topic"] == topic_number]
            count = len(group)

            summary_rows.append({
                "topic_number": topic_number,
                "topic_name": topic_name,
                "text_count": count,
                "text_share_percent": count / len(df) * 100,
                "mean_polarity": group["polarity"].mean(),
                "median_polarity": group["polarity"].median(),
                "mean_dominant_topic_weight": (
                    group["dominant_topic_weight"].mean()
                ),
                "median_token_count": group["topic_token_count"].median(),
            })

            for threshold in THRESHOLDS:
                label_column = f"label_t{threshold:.1f}"
                counts = group[label_column].value_counts()

                for label in LABELS:
                    label_count = int(counts.get(label, 0))
                    sentiment_rows.append({
                        "topic_number": topic_number,
                        "topic_name": topic_name,
                        "threshold": threshold,
                        "sentiment_label": label,
                        "text_count": label_count,
                        "topic_total_texts": count,
                        "share_within_topic_percent": (
                            label_count / count * 100
                            if count else np.nan
                        ),
                    })

        summary = pd.DataFrame(summary_rows)
        sentiment = pd.DataFrame(sentiment_rows)

        if summary["text_count"].sum() != len(df):
            raise ValueError("各主题文本数量之和不一致。")

        for threshold in THRESHOLDS:
            total = sentiment.loc[
                sentiment["threshold"] == threshold, "text_count"
            ].sum()
            if total != len(df):
                raise ValueError("情感分类数量核对失败。")

        summary.to_csv(
            TABLE_DIR / "topic_sentiment_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        sentiment.to_csv(
            TABLE_DIR / "topic_sentiment_threshold_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 保存不含评论正文的本地关联结果。
        df[
            [
                "text_id",
                "product_id",
                "dominant_topic",
                "topic_name",
                "dominant_topic_weight",
                "polarity",
                "label_t0.1",
                "label_t0.2",
            ]
        ].to_csv(
            DATA_DIR / "topic_sentiment_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 图片1：主要主题的文本数量。
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = np.arange(len(summary))
        bars = ax.bar(
            positions,
            summary["text_count"],
            color="#3978A8",
        )
        ax.set_xticks(
            positions,
            [f"Topic {i}" for i in summary["topic_number"]],
        )
        ax.set_ylabel("Number of text cells")
        ax.set_title("Dominant topic distribution")
        ax.bar_label(bars, padding=3)
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "02_topic_text_counts.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 图片2：两种阈值下，各主题的情感构成。
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5), sharey=True
        )
        colors = {
            "Negative": "#C65A5A",
            "Near-neutral": "#AAB2BD",
            "Positive": "#4A9D82",
        }

        for ax, threshold in zip(axes, THRESHOLDS):
            subset = sentiment[
                sentiment["threshold"] == threshold
            ]
            pivot = subset.pivot(
                index="topic_number",
                columns="sentiment_label",
                values="share_within_topic_percent",
            ).reindex(
                index=list(TOPIC_NAMES),
                columns=LABELS,
            )

            bottom = np.zeros(len(pivot))

            for label in LABELS:
                values = pivot[label].to_numpy()
                ax.bar(
                    positions,
                    values,
                    bottom=bottom,
                    label=label,
                    color=colors[label],
                )
                bottom += values

            ax.set_xticks(
                positions,
                [f"T{i}" for i in TOPIC_NAMES],
            )
            ax.set_ylim(0, 100)
            ax.set_xlabel("Dominant topic")
            ax.set_title(f"Threshold = {threshold:.1f}")

        axes[0].set_ylabel("Share within topic (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle("Whole-text sentiment by dominant topic")
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
        fig.savefig(
            FIGURE_DIR / "03_topic_sentiment_thresholds.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        decision = {
            "selected_num_topics": 5,
            "selection_reason": (
                "Selected for exploratory reporting because it separates "
                "installation/setup/service content, while retaining "
                "coherence close to the four-topic candidate."
            ),
            "coherence_k4_rounded": 0.4519,
            "coherence_k5_rounded": 0.4433,
            "topic_names": TOPIC_NAMES,
            "thresholds": THRESHOLDS,
            "limitations": [
                "Topic names are provisional human interpretations.",
                "Topics may mix product categories and concerns.",
                "Sentiment is measured for the whole text cell, not aspects.",
                "Representative examples are not random accuracy samples.",
                "Candidate comparison used one random seed.",
            ],
        }

        with open(
            TABLE_DIR / "topic_selection_notes.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(decision, file, ensure_ascii=False, indent=2)

        print("\n=== 主题与情感得分汇总 ===")
        print(
            summary[
                [
                    "topic_number",
                    "text_count",
                    "text_share_percent",
                    "mean_polarity",
                    "median_polarity",
                ]
            ].round(4).to_string(index=False)
        )

        for threshold in THRESHOLDS:
            print(f"\n=== 阈值 {threshold:.1f}：各主题情感数量 ===")
            print(
                sentiment[
                    sentiment["threshold"] == threshold
                ].pivot(
                    index="topic_number",
                    columns="sentiment_label",
                    values="text_count",
                ).reindex(columns=LABELS).to_string()
            )

        print("\n总文本数：", len(df))
        print("主题数量核对：", summary["text_count"].sum() == len(df))
        print("\n统计表：", TABLE_DIR)
        print("图片：", FIGURE_DIR)
        print("完成。输入数据和模型没有修改。")


    main()


if __name__ == "__main__":
    run()

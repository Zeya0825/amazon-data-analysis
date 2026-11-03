"""任务42：summarize_product_review_actions。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json
    from amazon_project.common import CONFIG_DIR
    import hashlib
    import re

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    # 路径以项目目录为准，不依赖运行命令时所在的位置。
    ROOT = PROJECT_ROOT
    INPUT = ROOT / "data_clean/week4/product_review_evidence_texts.csv"
    TABLES = ROOT / "reports/week4/tables"
    FIGURES = ROOT / "reports/week4/figures"

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 以下是阅读样例正文后的分析记录，不是自动情感分类结果。
    # 同一商品可以涉及多个问题；保留正向或不同意见，避免只摘取负面反馈。
    ANALYSIS = json.loads((CONFIG_DIR / "product_review_annotations.json").read_text(encoding="utf-8"))


    def normalize_text(value):
        """与提取步骤一致，只合并空白，不按逗号拆分评论。"""
        return re.sub(r"\s+", " ", str(value)).strip()


    def main():
        if not INPUT.exists():
            raise FileNotFoundError(f"找不到输入文件：{INPUT}")

        source = pd.read_csv(
            INPUT,
            dtype={"product_id": "string"},
            float_precision="round_trip",
        )

        required = {
            "reading_order", "source_row", "product_id", "main_category",
            "rating", "rating_count", "review_content_raw", "review_text_hash",
        }
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"输入文件缺少字段：{sorted(missing)}")

        annotations = pd.DataFrame(ANALYSIS)

        if source["product_id"].duplicated().any():
            raise ValueError("输入文件存在重复商品编号，请先检查。")

        if set(source["product_id"]) != set(annotations["product_id"]):
            raise ValueError("本次商品清单与阅读分析记录不一致，请勿直接套用。")

        # 重新计算正文哈希，确认输入文本与提取时的哈希一致。
        normalized = source["review_content_raw"].fillna("").map(normalize_text)
        if normalized.eq("").any():
            raise ValueError("发现空评论正文，请先检查。")

        calculated_hash = normalized.map(
            lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
        if not calculated_hash.eq(source["review_text_hash"]).all():
            raise ValueError("正文哈希核对失败，请检查输入文件是否被修改。")

        base_columns = [
            "reading_order", "source_row", "product_id", "main_category",
            "rating", "rating_count", "review_text_hash",
        ]
        result = source[base_columns].merge(
            annotations, on="product_id", how="left", validate="one_to_one"
        ).sort_values("reading_order")

        result["evidence_status"] = "评论样例中的反馈，尚未实物验证"
        result["action_status"] = "建议，尚未执行"

        # 保留15个商品的分析记录；统计时对重复正文去重。
        result["included_in_text_summary"] = ~result[
            "review_text_hash"
        ].duplicated(keep="first")

        # 相同正文应使用相同的问题分类。
        labels_per_text = result.groupby("review_text_hash")[
            "issue_types"
        ].nunique()
        if labels_per_text.gt(1).any():
            raise ValueError("相同正文被分配了不同分类，请核对分析记录。")

        unique_texts = result.loc[
            result["included_in_text_summary"]
        ].copy()

        expanded = (
            unique_texts[["review_text_hash", "issue_types"]]
            .assign(issue_type=lambda x: x["issue_types"].str.split("|"))
            .explode("issue_type")
        )

        issue_order = [
            "功能表现与耐用性",
            "商品说明与适配",
            "操作与结构设计",
            "售后与评价核查",
        ]

        counts = expanded.groupby("issue_type")[
            "review_text_hash"
        ].nunique().reindex(issue_order, fill_value=0)

        summary = counts.rename("unique_text_count").reset_index()
        summary["total_unique_texts"] = len(unique_texts)
        summary["share_of_selected_texts_percent"] = (
            summary["unique_text_count"] / len(unique_texts) * 100
        ).round(2)
        summary["counting_note"] = (
            "同一正文可涉及多个问题；仅描述所选样例，不代表总体问题率"
        )

        actions_path = TABLES / "product_review_actions.csv"
        summary_path = TABLES / "product_review_issue_summary.csv"
        figure_path = FIGURES / "01_selected_review_issue_types.png"

        result.to_csv(actions_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        # Windows优先使用微软雅黑；没有时尝试其他中文字体。
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.barh(
            summary["issue_type"],
            summary["unique_text_count"],
            color="#3576A8",
            height=0.6,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, len(unique_texts) + 1)
        ax.set_xticks(range(0, len(unique_texts) + 1, 2))
        ax.set_xlabel("涉及该问题的不同评论正文数量")
        ax.set_title(
            f"所选商品评论中的问题类型（{len(unique_texts)}份不同正文）",
            pad=16,
        )
        ax.bar_label(bars, padding=5, fmt="%d")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        fig.text(
            0.03, 0.04,
            "注：15个商品对应14份不同正文；同一正文可归入多个问题类型。\n"
            "样例按低评分规则选取，图中数量不代表全部商品的问题发生率。",
            fontsize=10,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print("=== 阅读分析核对 ===")
        print("商品分析记录数：", len(result))
        print("不同评论正文数：", len(unique_texts))
        print(
            "汇总时排除的重复正文记录数：",
            int((~result["included_in_text_summary"]).sum()),
        )

        print("\n=== 样例问题类型汇总 ===")
        print(summary[
            ["issue_type", "unique_text_count", "share_of_selected_texts_percent"]
        ].to_string(index=False))

        print("\n分析表：", actions_path)
        print("汇总表：", summary_path)
        print("图片：", figure_path)
        print("\n完成。输入文件没有修改。")
        print("分类来自阅读整理，建议尚未执行，未进行实物验证。")


    main()


if __name__ == "__main__":
    run()

"""任务43：build_business_action_plan。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json
    from amazon_project.common import CONFIG_DIR

    import pandas as pd


    ROOT = PROJECT_ROOT
    TABLES = ROOT / "reports/week4/tables"
    INPUT = TABLES / "product_review_actions.csv"
    TABLES.mkdir(parents=True, exist_ok=True)

    # 排序表示建议的处理顺序，不是模型计算的风险排名。
    # 责任角色为建议分工，不代表已有实际团队接受任务。
    ACTION_RULES = json.loads((CONFIG_DIR / "business_action_rules.json").read_text(encoding="utf-8"))


    def main():
        if not INPUT.exists():
            raise FileNotFoundError(f"找不到输入文件：{INPUT}")

        source = pd.read_csv(
            INPUT,
            dtype={
                "product_id": "string",
                "review_text_hash": "string",
            },
            keep_default_na=False,
        )

        required = {
            "product_id", "review_text_hash", "issue_types",
            "review_findings", "suggested_action",
        }
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"分析表缺少字段：{sorted(missing)}")

        if source.empty or source["product_id"].duplicated().any():
            raise ValueError("分析表为空或商品编号不唯一。")

        if source["review_text_hash"].eq("").any():
            raise ValueError("存在空文本哈希，无法核对不同正文数量。")

        # 将多问题标签展开，便于关联行动建议。
        expanded = source.assign(
            issue_type=source["issue_types"].str.split("|", regex=False)
        ).explode("issue_type")

        rule_types = {row["issue_type"] for row in ACTION_RULES}
        unknown = set(expanded["issue_type"]) - rule_types
        if unknown:
            raise ValueError(f"存在没有对应行动规则的问题类型：{unknown}")

        rows = []
        for rule in ACTION_RULES:
            matched = expanded.loc[
                expanded["issue_type"].eq(rule["issue_type"])
            ]

            row = dict(rule)
            row["sample_product_count"] = matched["product_id"].nunique()
            row["unique_evidence_text_count"] = (
                matched["review_text_hash"].nunique()
            )
            row["sample_product_ids"] = "|".join(
                matched["product_id"].drop_duplicates()
            )
            row["action_status"] = "建议，尚未执行"
            row["baseline_value"] = "待补充数据"
            row["target_value"] = "待建立基线后确定"
            rows.append(row)

        plan = pd.DataFrame(rows).sort_values(
            ["suggested_order", "action_id"]
        )

        # 每条映射可追溯到具体商品及其阅读记录。
        mapping = expanded[
            [
                "product_id", "review_text_hash", "issue_type",
                "review_findings", "suggested_action",
            ]
        ].merge(
            plan[["issue_type", "action_id", "action_name"]],
            on="issue_type",
            how="left",
            validate="many_to_one",
        )

        plan_path = TABLES / "business_action_plan.csv"
        mapping_path = TABLES / "business_action_product_mapping.csv"

        plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
        mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")

        print("=== 行动建议与样例证据 ===")
        print(
            plan[
                [
                    "action_id",
                    "suggested_order",
                    "action_name",
                    "sample_product_count",
                    "unique_evidence_text_count",
                ]
            ].to_string(index=False)
        )

        print("\n=== 核对 ===")
        print("输入商品数：", source["product_id"].nunique())
        print("不同评论正文数：", source["review_text_hash"].nunique())
        print("行动建议数：", len(plan))
        print("商品—行动关联数：", len(mapping))
        print("所有关联均有行动编号：", mapping["action_id"].notna().all())

        print("\n行动建议表：", plan_path)
        print("商品与行动对应表：", mapping_path)
        print("\n完成。没有修改输入文件。")
        print("处理顺序为建议安排，不是风险评分。")
        print("行动尚未执行，改善指标尚未计算，目标值没有预设。")


    main()


if __name__ == "__main__":
    run()

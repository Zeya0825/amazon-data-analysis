"""任务40：build_product_review_list。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json

    import pandas as pd


    ROOT = PROJECT_ROOT
    INPUT_PATH = ROOT / "data_clean" / "products_unique.csv"
    TABLE_DIR = ROOT / "reports" / "week4" / "tables"

    RATING_CUTOFF = 4.0
    MIN_RATING_COUNT = 100
    MAX_EXAMPLES = 15


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            INPUT_PATH,
            dtype={
                "product_id": "string",
                "product_name": "string",
                "main_category": "string",
            },
            float_precision="round_trip",
        )

        required_columns = {
            "source_row",
            "product_id",
            "product_name",
            "main_category",
            "rating",
            "rating_count",
            "has_key_differences",
            "different_fields",
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"缺少需要的字段：{sorted(missing_columns)}"
            )

        if df["product_id"].isna().any():
            raise ValueError("商品编号存在缺失。")

        if df["product_id"].duplicated().any():
            raise ValueError("商品编号不唯一，请检查商品分析表。")

        # 将评分和评分人数转成数字，保留缺失值。
        df["rating"] = pd.to_numeric(
            df["rating"], errors="coerce"
        )
        df["rating_count"] = pd.to_numeric(
            df["rating_count"], errors="coerce"
        )

        # 显式解析布尔值，避免把字符串“False”误当成True。
        difference_flags = (
            df["has_key_differences"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        valid_flags = difference_flags.isin(
            ["true", "false", "1", "0"]
        )

        if not valid_flags.all():
            raise ValueError(
                "has_key_differences存在缺失或无法识别的值。"
            )

        df["has_key_differences"] = difference_flags.isin(
            ["true", "1"]
        )

        df["different_fields"] = (
            df["different_fields"].fillna("").astype(str)
        )

        df["valid_rating"] = df["rating"].between(1, 5)
        df["valid_rating_count"] = (
            df["rating_count"].notna()
            & df["rating_count"].ge(0)
            & df["rating_count"].mod(1).eq(0)
        )

        usable = df["valid_rating"] & df["valid_rating_count"]
        low_rating = df["rating"].lt(RATING_CUTOFF)
        enough_ratings = df["rating_count"].ge(MIN_RATING_COUNT)

        # 分组互斥，每件商品只进入一个状态。
        df["review_status"] = "not_selected_by_current_rule"

        df.loc[
            ~usable, "review_status"
        ] = "check_missing_or_invalid_values"

        df.loc[
            usable & low_rating & ~enough_ratings,
            "review_status",
        ] = "low_rating_limited_count"

        selected = usable & low_rating & enough_ratings

        df.loc[
            selected & df["has_key_differences"],
            "review_status",
        ] = "check_key_field_differences_first"

        df.loc[
            selected & ~df["has_key_differences"],
            "review_status",
        ] = "review_text_next"

        reasons = {
            "not_selected_by_current_rule":
                "未进入本次筛选范围，不代表不存在问题。",
            "check_missing_or_invalid_values":
                "评分或评分人数缺失/异常，先核查数据。",
            "low_rating_limited_count":
                "评分低于4分，但评分人数不足100，单独保留观察。",
            "check_key_field_differences_first":
                "满足评分和人数条件，但重复记录关键字段不同，先核查。",
            "review_text_next":
                "评分低于4分且评分人数不少于100，下一步阅读评论。",
        }

        df["selection_reason"] = df["review_status"].map(reasons)

        # 数值较低的评分在前；同分时评分人数较多的在前。
        # 此顺序只是文本阅读顺序，不是质量排名。
        sort_columns = ["rating", "rating_count", "product_id"]
        sort_ascending = [True, False, True]

        output_columns = [
            "source_row",
            "product_id",
            "product_name",
            "main_category",
            "rating",
            "rating_count",
            "has_key_differences",
            "different_fields",
            "review_status",
            "selection_reason",
        ]

        reading_list = (
            df.loc[df["review_status"].eq("review_text_next")]
            .sort_values(sort_columns, ascending=sort_ascending)
            .head(MAX_EXAMPLES)
            [output_columns]
            .copy()
        )

        reading_list.insert(
            0, "reading_order", range(1, len(reading_list) + 1)
        )

        issue_list = (
            df.loc[
                df["review_status"].isin([
                    "check_missing_or_invalid_values",
                    "check_key_field_differences_first",
                ])
            ]
            .sort_values(
                sort_columns,
                ascending=sort_ascending,
                na_position="last",
            )
            [output_columns]
            .copy()
        )

        status_order = list(reasons)

        status_summary = (
            df["review_status"]
            .value_counts()
            .reindex(status_order, fill_value=0)
            .rename_axis("review_status")
            .reset_index(name="product_count")
        )
        status_summary["share_percent"] = (
            status_summary["product_count"] / len(df) * 100
        )
        status_summary["explanation"] = (
            status_summary["review_status"].map(reasons)
        )

        category_rows = []

        for category, part in df.groupby(
            "main_category", dropna=False
        ):
            valid_count = int(part["valid_rating"].sum())

            low_count = int(
                (
                    part["valid_rating"]
                    & part["rating"].lt(RATING_CUTOFF)
                ).sum()
            )

            category_rows.append({
                "main_category": category,
                "product_count": len(part),
                "valid_rating_products": valid_count,
                "rating_below_4_products": low_count,
                "rating_below_4_percent":
                    low_count / valid_count * 100
                    if valid_count else None,
                "review_text_next_products": int(
                    part["review_status"].eq(
                        "review_text_next"
                    ).sum()
                ),
                "check_key_differences_products": int(
                    part["review_status"].eq(
                        "check_key_field_differences_first"
                    ).sum()
                ),
                "low_rating_limited_count_products": int(
                    part["review_status"].eq(
                        "low_rating_limited_count"
                    ).sum()
                ),
            })

        category_summary = pd.DataFrame(category_rows).sort_values(
            "product_count", ascending=False
        )

        # 比较人数门槛变化会选出多少商品，
        # 避免把100这个门槛当作唯一正确的选择。
        sensitivity_rows = []

        for minimum_count in [50, 100, 500]:
            condition = (
                usable
                & low_rating
                & df["rating_count"].ge(minimum_count)
            )

            sensitivity_rows.append({
                "rating_cutoff_exclusive": RATING_CUTOFF,
                "minimum_rating_count": minimum_count,
                "selected_before_difference_check": int(
                    condition.sum()
                ),
                "with_key_differences": int(
                    (condition & df["has_key_differences"]).sum()
                ),
                "without_key_differences": int(
                    (condition & ~df["has_key_differences"]).sum()
                ),
            })

        sensitivity = pd.DataFrame(sensitivity_rows)

        outputs = {
            "product_review_selection_summary.csv":
                status_summary,
            "product_review_category_summary.csv":
                category_summary,
            "product_review_reading_list.csv":
                reading_list,
            "product_review_data_checks.csv":
                issue_list,
            "product_review_threshold_comparison.csv":
                sensitivity,
        }

        for filename, frame in outputs.items():
            frame.to_csv(
                TABLE_DIR / filename,
                index=False,
                encoding="utf-8-sig",
            )

        config = {
            "input_file": "data_clean/products_unique.csv",
            "rating_cutoff_exclusive": RATING_CUTOFF,
            "minimum_rating_count": MIN_RATING_COUNT,
            "maximum_reading_examples": MAX_EXAMPLES,
            "reading_order": [
                "rating ascending",
                "rating_count descending",
                "product_id ascending",
            ],
            "notes": [
                "Thresholds are exploratory, not industry standards.",
                "Rating counts do not guarantee representativeness.",
                "Key-field differences are checked separately.",
                "Reading order is not a product quality ranking.",
                "The first-source-row selection rule is inherited.",
                "No review text has been read or classified in this step.",
                "Input data is not modified.",
            ],
        }

        (TABLE_DIR / "product_review_selection_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("=== 商品数量核对 ===")
        print(f"输入商品数：{len(df):,}")
        print(
            "状态分组数量核对：",
            int(status_summary["product_count"].sum()) == len(df),
        )

        print("\n=== 筛选结果 ===")
        print(
            status_summary[
                ["review_status", "product_count", "share_percent"]
            ].round(4).to_string(index=False)
        )

        print("\n=== 各类别概况 ===")
        print(category_summary.round(4).to_string(index=False))

        print("\n=== 不同评分人数门槛 ===")
        print(sensitivity.to_string(index=False))

        print("\n=== 下一步阅读清单 ===")
        print(f"本次输出：{len(reading_list)}个商品")

        if not reading_list.empty:
            print(
                reading_list[
                    [
                        "reading_order",
                        "product_id",
                        "main_category",
                        "rating",
                        "rating_count",
                    ]
                ].to_string(index=False)
            )
        else:
            print("当前规则没有筛选出可直接进入阅读的商品。")

        print(f"\n结果位置：{TABLE_DIR}")
        print("完成。原始文件和前三周数据均未修改。")
        print("本步骤仅生成待核查清单，尚未判断具体商品问题。")


    main()


if __name__ == "__main__":
    run()

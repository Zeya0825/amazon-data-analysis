"""任务41：extract_product_review_evidence。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import hashlib
    import re

    import pandas as pd


    ROOT = PROJECT_ROOT
    TABLE_DIR = ROOT / "reports" / "week4" / "tables"
    LOCAL_DIR = ROOT / "data_clean" / "week4"

    READING_PATH = TABLE_DIR / "product_review_reading_list.csv"
    RAW_PATH = ROOT / "data_raw" / "sales" / "amazon.csv"


    def normalize_spaces(value):
        """只统一空白字符，用于检查文本重复，不改变评论含义。"""
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()


    def text_hash(value):
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


    def main():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

        reading = pd.read_csv(
            READING_PATH,
            dtype={"product_id": "string"},
            float_precision="round_trip",
        )

        raw = pd.read_csv(
            RAW_PATH,
            dtype="string",
        )

        required = {
            "product_id",
            "review_title",
            "review_content",
            "about_product",
        }
        missing = required - set(raw.columns)

        if missing:
            raise ValueError(f"原始商品表缺少字段：{sorted(missing)}")

        if reading["product_id"].isna().any():
            raise ValueError("阅读清单存在缺失商品编号。")

        if reading["product_id"].duplicated().any():
            raise ValueError("阅读清单的商品编号不唯一。")

        # 与第一周一致：第一条数据记录的source_row为1。
        # 这是CSV记录编号，不是文本文件的物理行号。
        raw.insert(0, "source_row", range(1, len(raw) + 1))
        raw["product_id"] = raw["product_id"].str.strip()

        reading["source_row"] = pd.to_numeric(
            reading["source_row"], errors="raise"
        )

        if (
            reading["source_row"].isna().any()
            or not reading["source_row"].mod(1).eq(0).all()
        ):
            raise ValueError("阅读清单source_row存在缺失或非整数。")

        reading["source_row"] = reading["source_row"].astype(int)

        selected_raw = raw[
            [
                "source_row",
                "product_id",
                "about_product",
                "review_title",
                "review_content",
            ]
        ].rename(columns={
            "product_id": "raw_product_id",
            "about_product": "about_product_raw",
            "review_title": "review_title_raw",
            "review_content": "review_content_raw",
        })

        # 按保留的来源行号提取，避免重复商品关联到错误记录。
        evidence = reading.merge(
            selected_raw,
            on="source_row",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        if not evidence["_merge"].eq("both").all():
            raise ValueError("部分来源行号无法在原始文件中找到。")

        same_id = evidence["product_id"].eq(
            evidence["raw_product_id"]
        ).fillna(False)

        if not same_id.all():
            raise ValueError(
                "来源行号对应的商品编号不一致，请停止并检查文件版本。"
            )

        evidence = evidence.drop(
            columns=["_merge", "raw_product_id"]
        ).sort_values("reading_order")

        evidence["review_text_normalized"] = (
            evidence["review_content_raw"].map(normalize_spaces)
        )

        evidence["review_character_count"] = (
            evidence["review_text_normalized"].str.len()
        )

        evidence["review_text_hash"] = (
            evidence["review_text_normalized"].map(text_hash)
        )

        # 统计这15个样例内部是否共用完全相同的正文。
        nonempty = evidence["review_character_count"].gt(0)

        hash_counts = (
            evidence.loc[nonempty, "review_text_hash"].value_counts()
        )

        evidence["same_text_products_in_selection"] = (
            evidence["review_text_hash"]
            .map(hash_counts)
            .fillna(0)
            .astype(int)
        )

        evidence["shares_text_with_selected_product"] = (
            evidence["same_text_products_in_selection"].gt(1)
        )

        # 完整正文用于阅读，留在本地分析目录。
        evidence_path = LOCAL_DIR / "product_review_evidence_texts.csv"
        evidence.to_csv(
            evidence_path,
            index=False,
            encoding="utf-8-sig",
        )

        index_columns = [
            "reading_order",
            "source_row",
            "product_id",
            "product_name",
            "main_category",
            "rating",
            "rating_count",
            "review_character_count",
            "review_text_hash",
            "same_text_products_in_selection",
            "shares_text_with_selected_product",
        ]

        index = evidence[index_columns].copy()
        index.to_csv(
            TABLE_DIR / "product_review_evidence_index.csv",
            index=False,
            encoding="utf-8-sig",
        )

        unique_texts = evidence.loc[
            nonempty, "review_text_hash"
        ].nunique()

        summary = pd.DataFrame([{
            "selected_products": len(evidence),
            "matched_source_rows": len(evidence),
            "nonempty_review_products": int(nonempty.sum()),
            "empty_review_products": int((~nonempty).sum()),
            "unique_nonempty_texts": unique_texts,
            "repeated_text_rows_excluding_first":
                int(nonempty.sum()) - unique_texts,
            "products_sharing_text_in_selection": int(
                evidence["shares_text_with_selected_product"].sum()
            ),
        }])

        summary.to_csv(
            TABLE_DIR / "product_review_evidence_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("=== 评论提取核对 ===")
        print(summary.to_string(index=False))

        print("\n=== 阅读索引 ===")
        print(
            index[
                [
                    "reading_order",
                    "product_id",
                    "rating",
                    "rating_count",
                    "review_character_count",
                    "same_text_products_in_selection",
                ]
            ].to_string(index=False)
        )

        repeated = evidence.loc[
            evidence["shares_text_with_selected_product"],
            [
                "product_id",
                "review_text_hash",
                "same_text_products_in_selection",
            ],
        ]

        print("\n=== 样例中的相同评论正文 ===")
        if repeated.empty:
            print("没有发现空白统一后完全相同的非空正文。")
        else:
            print(repeated.to_string(index=False))

        print(f"\n完整阅读文本：{evidence_path}")
        print(
            "阅读索引：",
            TABLE_DIR / "product_review_evidence_index.csv",
        )
        print(
            "核对汇总：",
            TABLE_DIR / "product_review_evidence_summary.csv",
        )

        print("\n完成。原始文件和此前分析结果未修改。")
        print("本步骤只提取和核对文本，尚未判断问题类型。")
        print("商品介绍属于商家描述，不能直接当作用户反馈。")


    main()


if __name__ == "__main__":
    run()

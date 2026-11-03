"""任务17：inspect_review_text。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    BASE_DIR = PROJECT_ROOT
    RAW_PATH = BASE_DIR / "data_raw" / "sales" / "amazon.csv"
    UNIQUE_PATH = BASE_DIR / "data_clean" / "products_unique.csv"
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 读取原始商品表，保留文本
    raw = pd.read_csv(RAW_PATH, dtype="string")
    raw["source_row"] = range(1, len(raw) + 1)

    products = pd.read_csv(
        UNIQUE_PATH,
        dtype={"product_id": "string"}
    )

    assert products["product_id"].is_unique, "商品分析表中存在重复编号。"

    # 按第一周选择的原始行，取回对应的评论文本
    # 使用商品编号和原始行号共同匹配，避免重复商品造成错配
    products["source_row"] = pd.to_numeric(
        products["source_row"], errors="raise"
    ).astype(int)

    text_columns = [
        "source_row", "product_id",
        "review_id", "review_title", "review_content"
    ]

    reviews = products[
        ["source_row", "product_id", "rating"]
    ].merge(
        raw[text_columns],
        on=["source_row", "product_id"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    assert reviews["_merge"].eq("both").all(), "部分商品未匹配到原始行。"

    # 仅用于检查：合并连续空白，保留正文中的标点
    text = (
        reviews["review_content"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    nonempty = text.ne("")
    nonempty_text = text[nonempty]

    # 这里是整格文本的长度，不是单条独立评论的长度
    text_length = nonempty_text.str.len()

    # 逗号仅作为结构线索，不据此判断评论条数
    review_ids = reviews["review_id"].fillna("").str.strip()

    checks = pd.DataFrame([
        {
            "check": "selected_products",
            "count": len(reviews)
        },
        {
            "check": "nonempty_review_text",
            "count": int(nonempty.sum())
        },
        {
            "check": "empty_review_text",
            "count": int((~nonempty).sum())
        },
        {
            "check": "unique_nonempty_text_after_whitespace_normalization",
            "count": int(nonempty_text.nunique())
        },
        {
            "check": "repeated_text_rows_excluding_first",
            "count": int(nonempty_text.duplicated().sum())
        },
        {
            "check": "review_id_cells_containing_comma",
            "count": int(review_ids.str.contains(",", regex=False).sum())
        },
    ])

    checks.to_csv(
        TABLE_DIR / "review_text_structure_checks.csv",
        index=False,
        encoding="utf-8-sig"
    )

    length_summary = (
        text_length.describe()
        .rename_axis("statistic")
        .reset_index(name="character_count")
    )

    length_summary.to_csv(
        TABLE_DIR / "review_text_length_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=== 评论文本结构检查 ===")
    print(checks.to_string(index=False))

    print("\n=== 每个商品所附整格评论文本的字符数 ===")
    print(length_summary.round(2).to_string(index=False))

    print("\n=== 前3个非空文本样例（只显示开头）===")

    for idx in reviews.index[nonempty][:3]:
        row = reviews.loc[idx]

        print("\n商品编号：", row["product_id"])
        print("商品评分：", row["rating"])
        print("评论编号字段前150字符：", str(row["review_id"])[:150])
        print("评论标题字段前200字符：", str(row["review_title"])[:200])
        print("评论正文字段前500字符：", text.loc[idx][:500])

    print("\n统计表保存位置：", TABLE_DIR)
    print("完成。本步骤只检查，没有拆分评论或修改数据。")

if __name__ == "__main__":
    run()

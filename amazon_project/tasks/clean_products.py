"""任务05：clean_products。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT
    input_path = project_dir / "data_raw" / "sales" / "amazon.csv"

    output_dir = project_dir / "data_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype="string")

    # 去掉首尾空格，将空字符串视为缺失
    for column in df.columns:
        df[column] = df[column].str.strip().replace("", pd.NA)

    # 记录原始数据行号：第一条数据为第1行，不包括表头
    df.insert(0, "source_row", range(1, len(df) + 1))

    # 保留原始数字文字，方便检查清洗前后的差异
    numeric_columns = [
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count",
    ]

    for column in numeric_columns:
        df[column + "_raw"] = df[column]

    # 价格：去掉印度卢比符号和千位分隔符
    for column in ["discounted_price", "actual_price"]:
        text = (
            df[column]
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[column] = pd.to_numeric(text, errors="coerce")

    df["currency"] = "INR"

    # 折扣：64%转换为0.64
    df["discount_percentage"] = pd.to_numeric(
        df["discount_percentage"].str.replace("%", "", regex=False),
        errors="coerce",
    ) / 100

    # 评分：无法转换或超出1—5范围的值设为缺失
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    valid_rating = df["rating"].between(1, 5).fillna(False)
    df.loc[~valid_rating, "rating"] = pd.NA

    # 评分人数：去掉逗号；负数或非整数设为缺失
    df["rating_count"] = pd.to_numeric(
        df["rating_count"].str.replace(",", "", regex=False),
        errors="coerce",
    )

    valid_count = (
        df["rating_count"].notna()
        & (df["rating_count"] >= 0)
        & (df["rating_count"] % 1 == 0)
    ).fillna(False)

    df.loc[~valid_count, "rating_count"] = pd.NA
    df["rating_count"] = df["rating_count"].astype("Int64")

    # 提取一级类别，保留原来的完整类别列
    df["main_category"] = df["category"].str.split("|", regex=False).str[0]

    # 标记重复商品，不在这一步删除
    df["is_repeated_product"] = df["product_id"].duplicated(keep=False)

    # 价格和折扣异常先标记，等待检查
    df["price_needs_check"] = (
        df["discounted_price"].isna()
        | df["actual_price"].isna()
        | (df["discounted_price"] < 0)
        | (df["actual_price"] <= 0)
        | (df["discounted_price"] > df["actual_price"])
    ).fillna(True)

    df["discount_needs_check"] = (
        ~df["discount_percentage"].between(0, 1).fillna(False)
    )

    output_path = output_dir / "products_clean_all_rows.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=== 商品数据清洗结果 ===")
    print("保留行数：", len(df))
    print("不同商品数：", df["product_id"].nunique())
    print("清洗后的评分缺失数量：", df["rating"].isna().sum())
    print("清洗后的评分人数缺失数量：", df["rating_count"].isna().sum())
    print("需要检查的价格记录数：", df["price_needs_check"].sum())
    print("需要检查的折扣记录数：", df["discount_needs_check"].sum())

    print("\n清洗后前5行：")
    print(
        df[
            [
                "product_id",
                "discounted_price",
                "actual_price",
                "discount_percentage",
                "rating",
                "rating_count",
            ]
        ].head().to_string(index=False)
    )

    print("\n保存位置：", output_path)
    print("原始文件未修改。")

if __name__ == "__main__":
    run()

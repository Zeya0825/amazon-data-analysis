"""任务10：missing_values。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT

    # 第二周的结果单独保存
    week2_dir = project_dir / "reports" / "week2"
    tables_dir = week2_dir / "tables"
    figures_dir = week2_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始商品数据
    raw = pd.read_csv(
        project_dir / "data_raw" / "sales" / "amazon.csv",
        dtype="string",
    )

    # 将空格或空字符串也视为缺失
    for column in raw.columns:
        raw[column] = raw[column].str.strip().replace("", pd.NA)

    # 读取第一周生成的清洗数据
    clean_all = pd.read_csv(
        project_dir / "data_clean" / "products_clean_all_rows.csv",
        dtype={"product_id": "string"},
    )

    products = pd.read_csv(
        project_dir / "data_clean" / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    # 数据质量检查中的“缺失”不包括以下字段的空白：
    # different_fields为空，表示未发现关键字段差异
    # *_raw是追溯用的原始文字备份，不重复纳入主字段统计
    product_columns = [
        "product_id",
        "product_name",
        "category",
        "main_category",
        "discounted_price",
        "actual_price",
        "currency",
        "discount_percentage",
        "rating",
        "rating_count",
    ]

    def missing_table(df, table_name, columns):
        """统计指定字段的缺失数量和缺失比例。"""
        rows = []

        for column in columns:
            missing_count = int(df[column].isna().sum())

            rows.append({
                "table_name": table_name,
                "column_name": column,
                "total_rows": len(df),
                "missing_count": missing_count,
                "non_missing_count": len(df) - missing_count,
                "missing_ratio": (
                    missing_count / len(df) if len(df) else 0
                ),
            })

        return pd.DataFrame(rows)

    # 原始表：检查全部16个原有字段
    raw_stats = missing_table(
        raw, "raw_products", raw.columns.tolist()
    )

    # 完整清洗表：检查相同的16个字段，便于前后比较
    clean_stats = missing_table(
        clean_all, "clean_products_all_rows", raw.columns.tolist()
    )

    # 每商品一行的分析表：检查主要业务字段
    unique_stats = missing_table(
        products, "products_unique", product_columns
    )

    all_stats = pd.concat(
        [raw_stats, clean_stats, unique_stats],
        ignore_index=True,
    )

    all_stats.to_csv(
        tables_dir / "missing_values_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 清洗前后对比：使用相同的1,465行，避免混入去重影响
    comparison = raw_stats[
        ["column_name", "missing_count"]
    ].rename(columns={"missing_count": "raw_missing_count"})

    comparison = comparison.merge(
        clean_stats[["column_name", "missing_count"]].rename(
            columns={"missing_count": "clean_missing_count"}
        ),
        on="column_name",
        validate="one_to_one",
    )

    comparison["change"] = (
        comparison["clean_missing_count"]
        - comparison["raw_missing_count"]
    )

    comparison.to_csv(
        tables_dir / "missing_values_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== 三张商品表的行数 ===")
    print("原始商品表：", len(raw))
    print("完整清洗表：", len(clean_all))
    print("每商品一行分析表：", len(products))

    print("\n=== 存在缺失值的字段 ===")
    has_missing = all_stats[all_stats["missing_count"] > 0].copy()

    # 只调整终端显示，CSV中保留原始比例
    has_missing["missing_percent"] = (
        has_missing["missing_ratio"] * 100
    ).round(4)

    print(
        has_missing[
            [
                "table_name",
                "column_name",
                "total_rows",
                "missing_count",
                "missing_percent",
            ]
        ].to_string(index=False)
    )

    print("\n=== 清洗前后缺失数量发生变化的字段 ===")
    print(
        comparison[comparison["change"] != 0].to_string(index=False)
    )

    print("\n结果保存位置：", tables_dir)
    print("完成。第一周数据文件没有修改。")

if __name__ == "__main__":
    run()

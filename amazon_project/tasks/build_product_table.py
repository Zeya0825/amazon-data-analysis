"""任务06：build_product_table。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT
    data_dir = project_dir / "data_clean"
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        data_dir / "products_clean_all_rows.csv",
        dtype={"product_id": "string"},
    )

    # 检查同一商品的关键字段是否存在差异
    check_columns = [
        "product_name",
        "category",
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count",
    ]

    grouped = df.groupby("product_id", sort=False)

    # 每个商品在原数据中有多少条记录
    record_counts = grouped.size().rename("source_record_count")

    # 缺失与非缺失也算作差异
    field_counts = grouped[check_columns].nunique(dropna=False)

    different_fields = field_counts.apply(
        lambda row: ", ".join(row.index[row > 1]),
        axis=1,
    ).rename("different_fields")

    summary = pd.concat(
        [record_counts, different_fields], axis=1
    ).reset_index()

    summary["has_key_differences"] = summary["different_fields"].ne("")

    # 输出全部重复商品的检查清单
    duplicate_report = summary[
        summary["source_record_count"] > 1
    ]

    duplicate_report.to_csv(
        report_dir / "duplicate_products_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 按原始行号排序，每个商品取首次出现的记录
    products = (
        df.sort_values("source_row")
        .drop_duplicates(subset="product_id", keep="first")
    )

    # 商品分析表只保留商品相关字段
    # 评论、用户等信息仍保存在完整清洗表中
    keep_columns = [
        "source_row",
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
        "price_needs_check",
        "discount_needs_check",
    ]

    products = products[keep_columns].merge(
        summary,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    products["selection_rule"] = "first_source_row"

    output_path = data_dir / "products_unique.csv"
    products.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=== 商品分析表结果 ===")
    print("商品分析表行数：", len(products))
    print("商品编号是否唯一：", products["product_id"].is_unique)
    print("有多条原始记录的商品数：", len(duplicate_report))
    print(
        "重复记录中关键字段存在差异的商品数：",
        duplicate_report["has_key_differences"].sum(),
    )

    print("\n重复商品检查清单前5行：")
    print(duplicate_report.head().to_string(index=False))

    print("\n商品分析表保存位置：", output_path)
    print(
        "重复商品检查清单保存位置：",
        report_dir / "duplicate_products_check.csv",
    )
    print("完整清洗表和原始文件均未修改。")

if __name__ == "__main__":
    run()

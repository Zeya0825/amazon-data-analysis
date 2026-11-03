"""任务11：check_anomalies。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT
    tables_dir = project_dir / "reports" / "week2" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 从原始表检查规则问题，避免清洗后看不到原来的异常
    raw = pd.read_csv(
        project_dir / "data_raw" / "sales" / "amazon.csv",
        dtype="string",
    )

    for column in raw.columns:
        raw[column] = raw[column].str.strip().replace("", pd.NA)

    raw.insert(0, "source_row", range(1, len(raw) + 1))

    def parse_price(series):
        return pd.to_numeric(
            series.str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    price = parse_price(raw["discounted_price"])
    original_price = parse_price(raw["actual_price"])

    discount = pd.to_numeric(
        raw["discount_percentage"].str.replace("%", "", regex=False),
        errors="coerce",
    ) / 100

    rating = pd.to_numeric(raw["rating"], errors="coerce")

    count = pd.to_numeric(
        raw["rating_count"].str.replace(",", "", regex=False),
        errors="coerce",
    )

    summary_rows = []
    detail_frames = []

    def add_check(name, field, mask, eligible):
        """记录检查结果；缺失值与格式异常分别统计。"""
        mask = mask.fillna(False)
        eligible = eligible.fillna(False)

        checked_rows = int(eligible.sum())
        flagged_rows = int((mask & eligible).sum())

        summary_rows.append({
            "check_name": name,
            "field": field,
            "checked_rows": checked_rows,
            "flagged_rows": flagged_rows,
            "flagged_ratio": (
                flagged_rows / checked_rows if checked_rows else None
            ),
        })

        if flagged_rows:
            detail = raw.loc[
                mask & eligible, ["source_row", "product_id"]
            ].copy()
            detail["check_name"] = name
            detail["field"] = field
            detail["raw_value"] = raw.loc[mask & eligible, field]
            detail_frames.append(detail)

    # 一、非空内容是否能够转为数字
    numeric_fields = {
        "discounted_price": price,
        "actual_price": original_price,
        "discount_percentage": discount,
        "rating": rating,
        "rating_count": count,
    }

    for field, values in numeric_fields.items():
        add_check(
            "non_numeric_value",
            field,
            values.isna(),
            raw[field].notna(),
        )

    # 二、数值范围检查；缺失值不重复计作范围异常
    add_check(
        "negative_discounted_price",
        "discounted_price",
        price < 0,
        price.notna(),
    )

    add_check(
        "non_positive_actual_price",
        "actual_price",
        original_price <= 0,
        original_price.notna(),
    )

    add_check(
        "discounted_above_actual",
        "discounted_price",
        price > original_price,
        price.notna() & original_price.notna(),
    )

    add_check(
        "discount_outside_0_1",
        "discount_percentage",
        ~discount.between(0, 1),
        discount.notna(),
    )

    add_check(
        "rating_outside_1_5",
        "rating",
        ~rating.between(1, 5),
        rating.notna(),
    )

    add_check(
        "negative_or_fractional_rating_count",
        "rating_count",
        (count < 0) | (count % 1 != 0),
        count.notna(),
    )

    # 三、标示折扣是否与价格大致一致
    # 容许1个百分点差异，避免把显示时的整数舍入当成错误
    calculated_discount = 1 - price / original_price.where(
        original_price > 0
    )

    add_check(
        "discount_difference_over_1_percentage_point",
        "discount_percentage",
        (discount - calculated_discount).abs() > 0.01,
        price.notna()
        & original_price.notna()
        & (original_price > 0)
        & discount.notna(),
    )

    rule_summary = pd.DataFrame(summary_rows)
    rule_summary.to_csv(
        tables_dir / "anomaly_checks_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if detail_frames:
        details = pd.concat(detail_frames, ignore_index=True)
    else:
        details = pd.DataFrame(columns=[
            "source_row", "product_id",
            "check_name", "field", "raw_value",
        ])

    details.to_csv(
        tables_dir / "anomaly_checks_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 四、在每商品一行的分析表中标记极端值
    products = pd.read_csv(
        project_dir / "data_clean" / "products_unique.csv",
        dtype={"product_id": "string"},
    )

    # 使用四分位距IQR方法，只作探索性标记，不删除记录
    iqr_rows = []

    for field in ["discounted_price", "rating_count"]:
        values = pd.to_numeric(products[field], errors="coerce")
        valid = values.dropna()

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged = (values < lower) | (values > upper)

        iqr_rows.append({
            "field": field,
            "valid_rows": len(valid),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "flagged_rows": int(flagged.sum()),
            "action": "flag_only_no_deletion",
        })

    iqr_summary = pd.DataFrame(iqr_rows)
    iqr_summary.to_csv(
        tables_dir / "statistical_outliers_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== 原始商品数据的规则检查 ===")
    print(
        rule_summary[
            ["check_name", "field", "checked_rows", "flagged_rows"]
        ].to_string(index=False)
    )

    print("\n=== 被规则标记的记录 ===")
    print(details.to_string(index=False))

    print("\n=== 商品分析表的统计极端值检查 ===")
    print(iqr_summary.to_string(index=False))

    print("\n保存位置：", tables_dir)
    print("完成。这里只检查和标记，没有修改或删除商品数据。")

if __name__ == "__main__":
    run()

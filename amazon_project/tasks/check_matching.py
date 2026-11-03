"""任务04：check_matching。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    project_dir = PROJECT_ROOT

    reviews_path = (
        project_dir / "data_raw" / "reviews" / "ratings_Electronics (1).csv"
    )
    sales_path = project_dir / "data_raw" / "sales" / "amazon.csv"

    # 商品表中的不同商品编号
    sales = pd.read_csv(sales_path, dtype="string")
    sales_ids = set(sales["product_id"].dropna().str.strip())
    sales_ids.discard("")

    # 用集合记录商品编号，重复编号只保留一次
    reviews_ids = set()
    matched_ids = set()
    matched_rows = 0

    earliest = None
    latest = None
    invalid_time_count = 0

    print("开始检查商品编号匹配情况和评分时间……")

    for batch_number, chunk in enumerate(
        pd.read_csv(
            reviews_path,
            header=None,
            names=["user_id", "product_id", "rating", "timestamp"],
            dtype="string",
            chunksize=200_000,
        ),
        start=1,
    ):
        product_ids = chunk["product_id"].str.strip()
        current_ids = set(product_ids.dropna())
        current_ids.discard("")

        reviews_ids.update(current_ids)
        matched_ids.update(current_ids & sales_ids)
        matched_rows += int(product_ids.isin(sales_ids).sum())

        # 将秒单位的 Unix 时间戳转成日期，统一按 UTC 解释
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce")
        dates = pd.to_datetime(
            timestamps, unit="s", errors="coerce", utc=True
        )

        invalid_time_count += int(dates.isna().sum())
        valid_dates = dates.dropna()

        if not valid_dates.empty:
            current_min = valid_dates.min()
            current_max = valid_dates.max()

            earliest = (
                current_min if earliest is None
                else min(earliest, current_min)
            )
            latest = (
                current_max if latest is None
                else max(latest, current_max)
            )

        if batch_number % 10 == 0:
            print(f"已检查 {batch_number} 批")

    print("\n=== 商品编号匹配结果 ===")
    print(f"评分数据中的不同商品数：{len(reviews_ids):,}")
    print(f"商品表中的不同商品数：{len(sales_ids):,}")
    print(f"两份数据共有的商品数：{len(matched_ids):,}")
    print(f"共有商品对应的评分记录数：{matched_rows:,}")

    if sales_ids:
        match_rate = len(matched_ids) / len(sales_ids)
        print(f"商品表的商品匹配率：{match_rate:.2%}")

    print("匹配到的商品编号示例：", sorted(matched_ids)[:10])

    print("\n=== 评分时间范围（UTC）===")
    print("最早评分时间：", earliest)
    print("最晚评分时间：", latest)
    print("缺失或无法转换的时间数量：", invalid_time_count)

    print("\n检查完成，原始文件未修改。")

if __name__ == "__main__":
    run()

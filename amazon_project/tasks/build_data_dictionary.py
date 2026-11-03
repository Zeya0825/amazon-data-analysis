"""任务21：build_data_dictionary。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import pandas as pd

    BASE_DIR = PROJECT_ROOT
    OUTPUT_DIR = BASE_DIR / "reports" / "week2" / "tables"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    def add(table, grain, field, dtype, meaning, unit, note):
        rows.append({
            "table_name": table,
            "row_meaning": grain,
            "field_name": field,
            "logical_type": dtype,
            "meaning": meaning,
            "unit_or_range": unit,
            "notes": note,
        })

    # 一、原始商品表：16个字段
    raw_fields = [
        ("product_id", "文本", "商品编号", "无",
         "可能重复，按文本读取"),
        ("product_name", "文本", "商品名称", "无", ""),
        ("category", "文本", "商品分类路径", "无",
         "多级类别以竖线分隔"),
        ("discounted_price", "文本", "展示的折后价格", "INR",
         "原始值含货币符号或千位分隔符，不是销售金额"),
        ("actual_price", "文本", "展示的原价", "INR",
         "不是已核实的历史成交价"),
        ("discount_percentage", "文本", "标示的降价百分比", "0%～100%",
         "降价60%表示按原价40%出售"),
        ("rating", "文本", "商品层面评分", "通常1～5",
         "发现一个竖线异常值，清洗后转为缺失"),
        ("rating_count", "文本", "商品页面报告的评分数量", "次",
         "有2条缺失，不是销量或本文件提供的评论条数"),
        ("about_product", "文本", "商品介绍", "无", ""),
        ("user_id", "文本", "所附评论的用户编号字段", "无",
         "一格可能含多个用户，不能直接当作独立用户记录"),
        ("user_name", "文本", "所附评论的用户名称字段", "无",
         "一格可能含多个名称"),
        ("review_id", "文本", "所附评论的编号字段", "无",
         "一格可能含多个评论编号"),
        ("review_title", "文本", "所附评论标题", "无",
         "可能汇总多个标题"),
        ("review_content", "文本", "所附评论正文", "字符",
         "可能汇总多条评论，不能直接按逗号拆成独立评论"),
        ("img_link", "文本", "商品图片链接", "URL", ""),
        ("product_link", "文本", "商品页面链接", "URL", ""),
    ]

    for field, dtype, meaning, unit, note in raw_fields:
        add("amazon.csv", "一条原始商品记录，商品编号可能重复",
            field, dtype, meaning, unit, note)

    # 二、原始历史评分表
    rating_fields = [
        ("user_id", "文本", "评分用户编号", "无",
         "不表示购买行为"),
        ("product_id", "文本", "被评分商品编号", "无",
         "保留前导零"),
        ("rating", "数值", "该条记录的评分", "1～5",
         "与商品表的商品层面评分口径不同"),
        ("timestamp", "整数", "评分时间的Unix时间戳", "秒",
         "按UTC转换"),
    ]

    for field, dtype, meaning, unit, note in rating_fields:
        add("ratings_Electronics (1).csv", "一条用户对商品的历史评分记录",
            field, dtype, meaning, unit, note)

    # 三、商品分析表的主要字段
    product_fields = [
        ("source_row", "整数", "对应原始商品记录的行序号", "从1开始",
         "不包括CSV表头"),
        ("product_id", "文本", "商品编号", "无",
         "本表唯一；重复商品选择原始行序号最小的记录"),
        ("product_name", "文本", "商品名称", "无", ""),
        ("category", "文本", "完整分类路径", "无", ""),
        ("main_category", "文本", "一级商品类别", "无",
         "从完整分类路径提取"),
        ("discounted_price", "数值", "清洗后的展示折后价格", "INR",
         "不能乘评分数量估算销售额"),
        ("actual_price", "数值", "清洗后的展示原价", "INR", ""),
        ("currency", "文本", "货币单位", "INR", "印度卢比"),
        ("discount_percentage", "数值", "降价比例", "0～1",
         "0.49表示降价49%"),
        ("rating", "数值，可缺失", "商品评分", "1～5",
         "统计平均评分时排除缺失；每个商品等权"),
        ("rating_count", "整数，可缺失", "页面报告的评分数量", "次",
         "不是销量，不用0填补缺失"),
        ("source_record_count", "整数", "该商品在原始表中的记录数量", "条", ""),
        ("different_fields", "文本，可空", "重复记录中不同的已检查关键字段", "字段名",
         "空白不一定是数据缺失，也可能表示未发现关键字段差异"),
        ("has_key_differences", "布尔值", "重复记录是否存在关键字段差异",
         "True或False", "33个商品标记为True"),
        ("selection_rule", "文本", "重复商品代表记录的选取规则", "无",
         "first_source_row，不表示最新记录"),
        ("price_needs_check", "布尔值", "是否被价格规则标记", "True或False",
         "具体规则见清洗代码"),
        ("discount_needs_check", "布尔值", "是否被折扣规则标记", "True或False",
         "具体规则见清洗代码"),
    ]

    # 仅记录本地文件中实际存在的字段
    product_columns = pd.read_csv(
        BASE_DIR / "data_clean" / "products_unique.csv", nrows=0
    ).columns

    for field, dtype, meaning, unit, note in product_fields:
        if field in product_columns:
            add("products_unique.csv", "一个商品",
                field, dtype, meaning, unit, note)

    # 四、清洗去重后的历史评分表
    for field, dtype, meaning, unit, note in rating_fields:
        add("ratings_unique.csv", "一条去除完全重复后的历史评分记录",
            field, dtype, meaning, unit, note)

    add("ratings_unique.csv", "一条去除完全重复后的历史评分记录",
        "source_row", "整数", "对应原始评分记录的行序号",
        "从1开始", "用于追溯原始记录")

    add("ratings_unique.csv", "一条去除完全重复后的历史评分记录",
        "rating_datetime_utc", "日期时间", "转换后的评分时间",
        "UTC", "范围为1998-12-04至2014-07-23，首尾年份不完整")

    # 五、第二周评论分析主要字段
    text_fields = [
        ("product_id", "文本", "商品编号", "无", ""),
        ("source_row", "整数", "选用记录的原始行序号", "从1开始", ""),
        ("main_category", "文本", "一级商品类别", "无", ""),
        ("rating", "数值，可缺失", "商品层面评分", "1～5",
         "不是整格评论对应的某位用户评分"),
        ("review_text_clean", "文本", "去网址、HTML标签并整理空白后的正文",
         "无", "保留否定词；一格可能含多条评论"),
        ("text_character_count", "整数", "清理后整格文本的字符数",
         "字符", "不是单条独立评论长度"),
    ]

    for field, dtype, meaning, unit, note in text_fields:
        add("product_review_text_analysis.csv", "一个商品所附的整格评论文本",
            field, dtype, meaning, unit, note)

    for field, dtype, meaning, unit, note in [
        ("text_id", "文本", "本次分析分配的文本编号", "无",
         "不是原始评论编号"),
        ("polarity", "数值", "TextBlob自动情感得分", "-1～1",
         "不是概率或模型准确率"),
        ("sentiment_label", "文本", "自动情感分类", "三个类别",
         ">0.1偏正面；<-0.1偏负面；其余倾向不明显"),
    ]:
        add("review_sentiment_unique_texts.csv", "一份不同的非空清理后文本",
            field, dtype, meaning, unit, note)

    dictionary = pd.DataFrame(rows)

    assert not dictionary.duplicated(
        ["table_name", "field_name"]
    ).any()

    output_path = OUTPUT_DIR / "data_dictionary.csv"
    dictionary.to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )

    print("=== 数据字典生成结果 ===")
    print(
        dictionary.groupby("table_name", sort=False)
        .size()
        .rename("documented_fields")
        .to_string()
    )
    print("\n总字段说明条数：", len(dictionary))
    print("保存位置：", output_path)
    print("完成，没有修改数据文件。")

if __name__ == "__main__":
    run()

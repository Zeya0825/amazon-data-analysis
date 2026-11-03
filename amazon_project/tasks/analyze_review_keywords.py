"""任务19：analyze_review_keywords。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from collections import Counter
    import re

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS

    BASE_DIR = PROJECT_ROOT
    TABLE_DIR = BASE_DIR / "reports" / "week2" / "tables"
    FIGURE_DIR = BASE_DIR / "reports" / "week2" / "figures"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 一、读取上一步生成的文本分析数据
    df = pd.read_csv(
        BASE_DIR / "data_clean" / "product_review_text_analysis.csv",
        dtype={"product_id": "string", "review_text_clean": "string"}
    )

    texts = df["review_text_clean"].fillna("").str.strip()
    texts = texts[texts.ne("")].drop_duplicates()
    document_count = len(texts)

    if document_count == 0:
        raise ValueError("没有可分析的非空文本。")

    # 二、设置停用词
    # 停用词：the、and等常见但对关键词展示帮助较小的词
    # 此列表仅用于关键词展示，不用于后面的情感分析
    stop_words = {word.lower() for word in STOPWORDS}
    stop_words.update({
        "product", "products", "amazon",
        "www", "http", "https", "com",
        "jpg", "jpeg", "png", "html"
    })

    # 保存实际使用的停用词，便于复现
    pd.DataFrame({
        "stopword": sorted(stop_words)
    }).to_csv(
        TABLE_DIR / "keyword_stopwords.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 三、统计英文单词
    # 总出现次数：一个词重复出现会重复计数
    # 文本出现数：同一份文本中，一个词最多计一次
    total_frequency = Counter()
    document_frequency = Counter()
    documents_with_tokens = 0

    for text in texts:
        words = re.findall(
            r"[a-z]+(?:'[a-z]+)?",
            text.lower().replace("’", "'")
        )

        words = [
            word for word in words
            if len(word) >= 3 and word not in stop_words
        ]

        total_frequency.update(words)
        document_frequency.update(set(words))

        if words:
            documents_with_tokens += 1

    if not document_frequency:
        raise ValueError("过滤后没有关键词，请检查文本或停用词。")

    keywords = pd.DataFrame([
        {
            "word": word,
            "document_count": count,
            "document_share_percent": count / document_count * 100,
            "total_occurrences": total_frequency[word],
        }
        for word, count in document_frequency.items()
    ])

    keywords = keywords.sort_values(
        ["document_count", "total_occurrences", "word"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    keywords.to_csv(
        TABLE_DIR / "review_keyword_frequencies.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([{
        "unique_nonempty_texts": document_count,
        "texts_with_retained_english_words": documents_with_tokens,
        "distinct_retained_words": len(keywords),
        "retained_word_occurrences": sum(total_frequency.values())
    }]).to_csv(
        TABLE_DIR / "review_keyword_analysis_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 四、前20个关键词柱状图
    top20 = keywords.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        top20["word"],
        top20["document_count"],
        color="#4C78A8"
    )

    for bar in bars:
        ax.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}",
            va="center",
            fontsize=9
        )

    ax.set_title(
        f"Top 20 words across unique review-text cells (n={document_count:,})"
    )
    ax.set_xlabel("Number of text cells mentioning the word")
    ax.set_ylabel("")
    ax.set_xlim(0, top20["document_count"].max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "12_review_top_keywords.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 五、词云：字的大小按文本出现数确定
    frequencies = dict(zip(
        keywords["word"], keywords["document_count"]
    ))

    cloud = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        max_words=80,
        colormap="viridis",
        random_state=42,
        collocations=False
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Review keyword cloud: sized by text-cell frequency",
        fontsize=14,
        pad=15
    )

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "13_review_keyword_wordcloud.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("=== 关键词分析范围 ===")
    print("不同的非空文本数量：", document_count)
    print("包含保留英文词的文本数量：", documents_with_tokens)
    print("保留的不同单词数量：", len(keywords))

    print("\n=== 前20个关键词 ===")
    print(keywords.head(20).round(2).to_string(index=False))

    print("\n已生成：")
    print("12_review_top_keywords.png")
    print("13_review_keyword_wordcloud.png")
    print("统计表位置：", TABLE_DIR)
    print("图片位置：", FIGURE_DIR)
    print("完成，输入数据没有修改。")

if __name__ == "__main__":
    run()

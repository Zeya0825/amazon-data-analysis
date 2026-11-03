"""任务27：prepare_topic_texts。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    from collections import Counter
    import json
    import re

    import pandas as pd
    from gensim.corpora import Dictionary
    from gensim.parsing.preprocessing import STOPWORDS


    ROOT = PROJECT_ROOT
    INPUT = ROOT / "data_clean" / "review_sentiment_unique_texts.csv"

    LOCAL_DIR = ROOT / "data_clean" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"

    for folder in [LOCAL_DIR, TABLE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    # 初步设置：至少出现在5份文本中，
    # 且不超过全部文本的80%。
    MIN_DOCUMENTS = 5
    MAX_DOCUMENT_SHARE = 0.80

    # 去掉对区分具体主题帮助较小的泛泛用词。
    # quality、price、delivery等具体内容词保留。
    EXTRA_STOPWORDS = {
        "product", "products", "amazon",
        "good", "nice", "great", "excellent",
        "bad", "best", "awesome", "okay", "ok",
        "really", "like", "buy", "bought", "purchase",
        "purchased", "use", "used", "using",
    }

    stopwords = set(STOPWORDS) | EXTRA_STOPWORDS


    def tokenize(text):
        # 仅分析英文词语，不把其他语言误当作已分析。
        words = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
        return [word for word in words if word not in stopwords]


    print("开始准备主题模型文本……")

    df = pd.read_csv(
        INPUT,
        dtype={"text_id": "string", "product_id": "string"},
        float_precision="round_trip",
    )

    required = {
        "text_id", "product_id",
        "review_text_clean", "polarity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少字段：{sorted(missing)}")

    if df["text_id"].isna().any() or not df["text_id"].is_unique:
        raise ValueError("text_id存在缺失或重复，请先检查。")

    texts = df["review_text_clean"].fillna("").astype(str)
    tokens_before = [tokenize(text) for text in texts]

    dictionary = Dictionary(tokens_before)
    vocabulary_before = len(dictionary)

    dictionary.filter_extremes(
        no_below=MIN_DOCUMENTS,
        no_above=MAX_DOCUMENT_SHARE,
        keep_n=None,
    )
    dictionary.compactify()

    if len(dictionary) == 0:
        raise ValueError("筛选后没有剩余词语，请检查文本和筛选条件。")

    vocabulary = set(dictionary.token2id)
    tokens_after = [
        [word for word in words if word in vocabulary]
        for words in tokens_before
    ]
    keep_mask = [len(words) > 0 for words in tokens_after]

    if not any(keep_mask):
        raise ValueError("筛选后没有可用于建模的文本。")

    # 每份文本的处理情况：不包含评论正文。
    checks = df[["text_id", "product_id"]].copy()
    checks["tokens_before_dictionary_filter"] = [
        len(words) for words in tokens_before
    ]
    checks["tokens_after_dictionary_filter"] = [
        len(words) for words in tokens_after
    ]
    checks["included_in_lda"] = keep_mask

    checks.to_csv(
        TABLE_DIR / "topic_text_preparation_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 本地建模数据保留ID，便于之后回看原文。
    prepared = df.loc[keep_mask].copy()
    kept_tokens = [
        words for words, keep in zip(tokens_after, keep_mask) if keep
    ]
    prepared["topic_tokens_json"] = [
        json.dumps(words, ensure_ascii=False)
        for words in kept_tokens
    ]
    prepared["topic_token_count"] = [
        len(words) for words in kept_tokens
    ]

    prepared.to_csv(
        LOCAL_DIR / "topic_model_input.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary.save(str(LOCAL_DIR / "topic_dictionary.dict"))

    # 统计每个词出现于多少份文本，以及总出现次数。
    document_counts = Counter()
    occurrence_counts = Counter()

    for words in kept_tokens:
        document_counts.update(set(words))
        occurrence_counts.update(words)

    word_table = pd.DataFrame([
        {
            "word": word,
            "document_count": document_counts[word],
            "document_share_percent": (
                document_counts[word] / len(kept_tokens) * 100
            ),
            "total_occurrences": occurrence_counts[word],
        }
        for word in vocabulary
    ]).sort_values(
        ["document_count", "total_occurrences", "word"],
        ascending=[False, False, True],
    )

    word_table.to_csv(
        TABLE_DIR / "topic_vocabulary_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([{
        "input_text_count": len(df),
        "nonempty_original_text_count": int(
            texts.str.strip().ne("").sum()
        ),
        "texts_with_tokens_before_filter": sum(
            bool(words) for words in tokens_before
        ),
        "included_text_count": len(prepared),
        "excluded_empty_after_filter": len(df) - len(prepared),
        "vocabulary_before_filter": vocabulary_before,
        "vocabulary_after_filter": len(dictionary),
        "min_document_count": MIN_DOCUMENTS,
        "max_document_share": MAX_DOCUMENT_SHARE,
    }])

    summary.to_csv(
        TABLE_DIR / "topic_preparation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 记录规则，后续报告可以准确说明处理方法。
    config = {
        "input_file": INPUT.name,
        "analysis_unit": "one_unique_whole_review_text_cell",
        "token_pattern": r"\b[a-z]{3,}\b",
        "min_document_count": MIN_DOCUMENTS,
        "max_document_share": MAX_DOCUMENT_SHARE,
        "extra_stopwords": sorted(EXTRA_STOPWORDS),
        "all_stopwords_used": sorted(stopwords),
        "stemming_or_lemmatization": False,
        "note": (
            "English tokens only; no comma-based review splitting. "
            "Topic preprocessing does not change sentiment scores."
        ),
    }

    with open(
        TABLE_DIR / "topic_preparation_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("\n=== 文本与词语数量 ===")
    print(summary.to_string(index=False))

    print("\n=== 保留词语的前20名 ===")
    print(
        word_table.head(20).to_string(
            index=False,
            formatters={
                "document_share_percent": lambda value: f"{value:.2f}"
            },
        )
    )

    print("\n=== 前3份文本的词语预览 ===")
    for text_id, words in zip(
        prepared["text_id"].head(3),
        kept_tokens[:3],
    ):
        print(f"\n{text_id}：")
        print(", ".join(words[:35]))

    print("\n本地建模数据：", LOCAL_DIR / "topic_model_input.csv")
    print("模型词典：", LOCAL_DIR / "topic_dictionary.dict")
    print("统计结果：", TABLE_DIR)
    print("完成。尚未训练模型，输入文件没有修改。")

if __name__ == "__main__":
    run()

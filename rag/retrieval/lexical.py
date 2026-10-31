from collections import Counter
import logging
import math
import re

import jieba
from rank_bm25 import BM25Okapi

STOPWORDS = set("的 了 是 在 和 与 或 及 请 帮 我 一下 怎么 如何 一个 哪个 什么 可以 这个 这些 需要 进行 通过 对应 查询 接口 数据".split())
IDENTIFIER = re.compile(r"[A-Za-z0-9_]+(?:(?:>>|[./:-])[A-Za-z0-9_]+)*")
CJK = re.compile(r"[\u3400-\u9fff]+")


class LexicalIndex:
    def __init__(self, chunks):
        jieba.setLogLevel(logging.WARNING)
        self.chunks = chunks
        self.tokenizer = jieba.Tokenizer()
        for chunk in chunks:
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,60}", chunk.api_path):
                self.tokenizer.add_word(term, freq=10000)
        corpus = [self.tokenize(f"{c.title} {c.title} {c.heading} {c.api_path} {c.text}") or ["空"] for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        # 使用常见的 log(1 + ...) 平滑 IDF；避免很小的业务分类内
        # 半数文档出现的精确字段得到 0 分，或高频词出现负分。
        frequencies = Counter(term for tokens in corpus for term in set(tokens))
        count = len(corpus)
        self.bm25.idf = {term: math.log1p((count - freq + 0.5) / (freq + 0.5))
                         for term, freq in frequencies.items()}

    def tokenize(self, text):
        tokens = []
        for match in IDENTIFIER.finditer(text):
            original = match.group()
            tokens.append(original.lower())
            # data>>list>>shipment_id 既保留完整路径，也保留 shipment_id；
            # 不能只剩下 shipment 和 id，否则精确字段检索会退化。
            tokens.extend(part.lower() for part in re.split(r">>|[/.:-]", original)
                          if part and part.lower() != original.lower())
            split = re.sub(r"([a-z])([A-Z])", r"\1 \2", original)
            tokens.extend(part.lower() for part in re.split(r"[_>/. :\-]+", split)
                          if part and part.lower() != original.lower())
        for match in CJK.finditer(text):
            tokens.extend(word for word in self.tokenizer.cut_for_search(match.group())
                          if word.strip() and word not in STOPWORDS)
        return tokens

    def search(self, query, limit, category=None):
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        result = []
        for i in indices:
            chunk = self.chunks[i]
            if scores[i] <= 0:
                continue
            if category and chunk.category.split("/")[0] != category:
                continue
            result.append((chunk.chunk_id, float(scores[i])))
            if len(result) >= limit:
                break
        return result

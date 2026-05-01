import math
import os
import re
from collections import Counter

import torch


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_case_identifier(case_name):
    return str(int(case_name.split(".")[0])).zfill(6)


class BM25Okapi:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        df = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

        self.tfs = [Counter(doc) for doc in corpus_tokens]

    def score_doc(self, query_tokens, doc_index):
        score = 0.0
        if self.N == 0:
            return score

        tf = self.tfs[doc_index]
        dl = self.doc_lens[doc_index]
        denom_base = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            term_tf = tf[term]
            score += idf * (term_tf * (self.k1 + 1.0)) / (term_tf + denom_base)
        return score

    def score(self, query_tokens, doc_indices=None):
        if doc_indices is None:
            doc_indices = range(self.N)
        return [self.score_doc(query_tokens, i) for i in doc_indices]


def rowwise_minmax_normalize(score_tensor):
    row_min = score_tensor.min(dim=1, keepdim=True).values
    row_max = score_tensor.max(dim=1, keepdim=True).values
    denom = row_max - row_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return (score_tensor - row_min) / denom


def build_bm25_score_store(query_dir, doc_dir, query_ids=None, doc_ids=None, k1=1.5, b=0.75):
    if query_ids is None:
        query_ids = sorted([f for f in os.listdir(query_dir) if f.endswith(".txt")])
    if doc_ids is None:
        doc_ids = sorted([f for f in os.listdir(doc_dir) if f.endswith(".txt")])

    doc_texts = [read_text(os.path.join(doc_dir, doc_id)) for doc_id in doc_ids]
    corpus_tokens = [tokenize(text) for text in doc_texts]
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)

    score_rows = []
    for query_id in query_ids:
        query_tokens = tokenize(read_text(os.path.join(query_dir, query_id)))
        score_rows.append(bm25.score(query_tokens))

    score_tensor = torch.tensor(score_rows, dtype=torch.float32)
    normalized_scores = rowwise_minmax_normalize(score_tensor)

    normalized_query_ids = [normalize_case_identifier(query_id) for query_id in query_ids]
    normalized_doc_ids = [normalize_case_identifier(doc_id) for doc_id in doc_ids]

    return {
        "scores": normalized_scores,
        "query_ids": normalized_query_ids,
        "doc_ids": normalized_doc_ids,
        "query_to_index": {case_id: idx for idx, case_id in enumerate(normalized_query_ids)},
        "doc_to_index": {case_id: idx for idx, case_id in enumerate(normalized_doc_ids)},
    }

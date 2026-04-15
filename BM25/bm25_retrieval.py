import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

from torch_metrics import t_metrics, metric, yf_metric

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


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
            # BM25 idf with +1 to avoid negative values for very frequent terms
            self.idf[term] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

        # term frequencies per doc
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline for COLIEE 2017 retrieval")
    parser.add_argument("--query_dir", default="PromptCase/task1_test_2017/summary_test_2017_txt")
    parser.add_argument("--doc_dir", default="PromptCase/task1_test_2017/summary_test_2017_txt")
    parser.add_argument("--labels", default="label/task1_test_labels_2017.json")
    parser.add_argument("--candidates", default="label/test_2017_candidate_with_yearfilter.json")
    parser.add_argument(
        "--restrict_to_candidates",
        action="store_true",
        help="If set, restrict ranking pool to candidates (year-filter).",
    )
    parser.add_argument(
        "--filter_after_ranking",
        action="store_true",
        default=True,
        help="If set, rank over full pool then filter ranked list to year-filter candidates.",
    )
    parser.add_argument(
        "--no_filter_after_ranking",
        action="store_false",
        dest="filter_after_ranking",
        help="Disable year-filtering after ranking.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--results_dir", default="bm25_results")
    parser.add_argument("--output", default="bm25_predictions_2017_test.json")
    parser.add_argument("--log_path", default="BM25_2017_run.log", help="Optional path to append metrics log (CaseGNN-like format).")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    def resolve_output_path(path_value):
        if path_value is None:
            return None
        # Keep absolute paths and explicit relative directories untouched.
        if os.path.isabs(path_value) or os.path.dirname(path_value):
            return path_value
        return os.path.join(args.results_dir, path_value)

    output_path = resolve_output_path(args.output)
    log_path = resolve_output_path(args.log_path)

    label_dict = load_json(args.labels)
    candidate_dict = load_json(args.candidates) if args.candidates else None

    # Build doc pool
    if candidate_dict and args.restrict_to_candidates:
        doc_ids = sorted({doc for docs in candidate_dict.values() for doc in docs})
    else:
        doc_ids = sorted([f for f in os.listdir(args.doc_dir) if f.endswith('.txt')])

    doc_texts = []
    missing_docs = []
    for doc_id in doc_ids:
        path = os.path.join(args.doc_dir, doc_id)
        if not os.path.exists(path):
            missing_docs.append(doc_id)
            continue
        doc_texts.append(read_text(path))

    # Keep doc_ids aligned with doc_texts
    if missing_docs:
        doc_ids = [d for d in doc_ids if d not in set(missing_docs)]

    corpus_tokens = [tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(corpus_tokens, k1=args.k1, b=args.b)

    id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    predictions = {}
    missing_queries = []

    for query_id in label_dict.keys():
        query_path = os.path.join(args.query_dir, query_id)
        if not os.path.exists(query_path):
            missing_queries.append(query_id)
            continue
        query_tokens = tokenize(read_text(query_path))

        if candidate_dict and args.restrict_to_candidates:
            candidates = [c for c in candidate_dict[query_id] if c in id_to_index]
            doc_indices = [id_to_index[c] for c in candidates]
        else:
            candidates = doc_ids
            doc_indices = list(range(len(doc_ids)))

        scores = bm25.score(query_tokens, doc_indices)
        # Sort candidates by score desc
        ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        if candidate_dict and args.filter_after_ranking:
            cand_set = set(candidate_dict[query_id])
            ranked = [c for c in ranked if c in cand_set]
        predictions[query_id] = ranked[:args.topk]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # Evaluate
    ndcg, mrr, map_score, p_score = t_metrics(label_dict, predictions, args.topk)
    correct_pred, retri_cases, relevant_cases, micro_pre, micro_recall, micro_f, macro_pre, macro_recall, macro_f = metric(
        args.topk, predictions, label_dict
    )

    if args.candidates:
        yf_results = yf_metric(args.topk, args.candidates, predictions, label_dict)
        (yf_dict, correct_pred_yf, retri_cases_yf, relevant_cases_yf, micro_pre_yf, micro_recall_yf, micro_f_yf,
         macro_pre_yf, macro_recall_yf, macro_f_yf) = yf_results
        ndcg_score_yf, mrr_score_yf, map_score_yf, p_score_yf = t_metrics(label_dict, yf_dict, args.topk)
    else:
        correct_pred_yf = retri_cases_yf = relevant_cases_yf = 0
        micro_pre_yf = micro_recall_yf = micro_f_yf = 0.0
        macro_pre_yf = macro_recall_yf = macro_f_yf = 0.0
        ndcg_score_yf = mrr_score_yf = map_score_yf = p_score_yf = 0.0

    print("BM25 evaluation")
    print(f"Queries: {len(predictions)}")
    if missing_docs:
        print(f"Missing docs in pool: {len(missing_docs)}")
    if missing_queries:
        print(f"Missing query files: {len(missing_queries)}")
    print(f"NDCG@{args.topk}: {ndcg:.4f}")
    print(f"MRR@{args.topk}: {mrr:.4f}")
    print(f"MAP@{args.topk}: {map_score:.4f}")
    print(f"P@{args.topk}: {p_score:.4f}")
    print(f"Micro P/R/F: {micro_pre:.4f} / {micro_recall:.4f} / {micro_f:.4f}")
    print(f"Macro P/R/F: {macro_pre:.4f} / {macro_recall:.4f} / {macro_f:.4f}")
    if args.candidates:
        print(f"Year-filter Micro P/R/F: {micro_pre_yf:.4f} / {micro_recall_yf:.4f} / {micro_f_yf:.4f}")
        print(f"Year-filter Macro P/R/F: {macro_pre_yf:.4f} / {macro_recall_yf:.4f} / {macro_f_yf:.4f}")

    if log_path:
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return x

        log_lines = [
            "Method: BM25",
            f"TopK: {args.topk}",
            f"Queries: {len(predictions)}",
        ]
        if missing_docs:
            log_lines.append(f"Missing docs in pool: {len(missing_docs)}")
        if missing_queries:
            log_lines.append(f"Missing query files: {len(missing_queries)}")

        log_lines += [
            f"Correct Predictions:  {correct_pred}",
            f"Retrived Cases:  {retri_cases}",
            f"Relevant Cases:  {relevant_cases}",
            f"Micro Precision:  {micro_pre}",
            f"Micro Recall:  {micro_recall}",
            f"Micro F1:  {micro_f}",
            f"Macro Precision:  {macro_pre}",
            f"Macro Recall:  {macro_recall}",
            f"Macro F1:  {macro_f}",
            f"NDCG@{args.topk}:  {_to_float(ndcg)}",
            f"MRR@{args.topk}:  {_to_float(mrr)}",
            f"MAP:  {_to_float(map_score)}",
        ]

        if args.candidates:
            log_lines += [
                f"Correct Predictions yf:  {correct_pred_yf}",
                f"Retrived Cases yf:  {retri_cases_yf}",
                f"Relevant Cases yf:  {relevant_cases_yf}",
                f"Micro Precision yf:  {micro_pre_yf}",
                f"Micro Recall yf:  {micro_recall_yf}",
                f"Micro F1 yf:  {micro_f_yf}",
                f"Macro Precision yf:  {macro_pre_yf}",
                f"Macro Recall yf:  {macro_recall_yf}",
                f"Macro F1 yf:  {macro_f_yf}",
                f"NDCG@{args.topk} yf:  {_to_float(ndcg_score_yf)}",
                f"MRR@{args.topk} yf:  {_to_float(mrr_score_yf)}",
                f"MAP yf:  {_to_float(map_score_yf)}",
            ]

        log_lines.append("")  # trailing newline

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(l) for l in log_lines))

    print(f"Predictions saved to: {output_path}")
    if log_path:
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()

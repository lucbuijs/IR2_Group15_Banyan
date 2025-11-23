import numpy as np
import pandas as pd
import math
import json
import faiss
from collections import defaultdict

def load_glove(path):
    glove = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            glove[word] = vec
    return glove

def load_corpus(path):
    corpus = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj["_id"])
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            corpus[doc_id] = text
    return corpus

def load_queries(path):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            queries[qid] = obj["text"]
    return queries

def load_relevance(path):
    df = pd.read_csv(path, sep="\t")
    gt = defaultdict(set)
    for _, row in df.iterrows():
        if row["score"] > 0:
            q = str(row["query-id"])
            d = str(row["corpus-id"])
            gt[q].add(d)
    return gt

def embed_text(text, glove, dim=300):
    tokens = text.lower().split()
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

def recall_at_k(retrieved_list, relevant_set, k):
    retrieved_k = retrieved_list[:k]
    hits = len([d for d in retrieved_k if d in relevant_set])
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

def dcg(retrieved, relevant_set, k):
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            score += 1.0 / math.log2(i + 2)   # rank i -> position i+1 -> denom log2(i+2)
    return score

def ndcg_at_k(retrieved, relevant_set, k):
    ideal = dcg(list(relevant_set), relevant_set, k)
    if ideal == 0:
        return 0.0
    return dcg(retrieved, relevant_set, k) / ideal

def main():

    # Benchmark datasets:
    dataset = "quora"

    # dim of glove embeddings
    dim_glove = 100

    # ---- Paths ----
    GLOVE_PATH = f"../embeddings/glove.6B.{dim_glove}d.txt"
    CORPUS_PATH = f"../beir/{dataset}/corpus.jsonl"
    QUERIES_PATH = f"../beir/{dataset}/queries.jsonl"
    TEST_PATH = f"../beir/{dataset}/qrels/test.tsv"

    print("Loading GloVe ...")
    glove = load_glove(GLOVE_PATH)

    print(f"Loading BEIR {dataset} corpus ...")
    corpus = load_corpus(CORPUS_PATH)
    doc_ids = list(corpus.keys())

    print("Embedding documents ...")
    doc_vecs = np.vstack([embed_text(corpus[doc_id], glove, dim_glove) for doc_id in doc_ids])
    doc_vecs = np.ascontiguousarray(doc_vecs, dtype=np.float32)

    print("Building FAISS index ...")
    faiss.normalize_L2(doc_vecs)
    dim = doc_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_vecs)

    print("Loading queries ...")
    queries = load_queries(QUERIES_PATH)

    print("Loading ground-truth ...")
    ground_truth = load_relevance(TEST_PATH)

    retrieved_results = {}

    print("Retrieving for all queries ...")
    for qid, qtext in queries.items():
        qvec = embed_text(qtext, glove).reshape(1, -1)
        qvec = np.ascontiguousarray(qvec, dtype=np.float32)
        faiss.normalize_L2(qvec)

        D, I = index.search(qvec, 100)   # retrieve top 100
        retrieved_docs = [doc_ids[i] for i in I[0]]
        retrieved_results[qid] = retrieved_docs

    print("Evaluating ...")

    ndcg1_list, ndcg10_list = [], []
    rec1_list, rec10_list = [], []

    for qid, relevant in ground_truth.items():
        if qid not in retrieved_results:
            continue

        retrieved = retrieved_results[qid]

        ndcg1_list.append(ndcg_at_k(retrieved, relevant, 1))
        ndcg10_list.append(ndcg_at_k(retrieved, relevant, 10))
        rec1_list.append(recall_at_k(retrieved, relevant, 1))
        rec10_list.append(recall_at_k(retrieved, relevant, 10))

    print("\n===== FINAL RESULTS =====")
    print("NDCG@1  =", sum(ndcg1_list) / len(ndcg1_list))
    print("NDCG@10 =", sum(ndcg10_list) / len(ndcg10_list))
    print("R@1     =", sum(rec1_list) / len(rec1_list))
    print("R@10    =", sum(rec10_list) / len(rec10_list))

    print("\nDone.")

if __name__ == "__main__":
    main()


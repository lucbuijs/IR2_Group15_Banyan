import json
import random
import numpy as np

# --- Load corpus ---
def load_corpus(path):
    corpus = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(str(obj["_id"]))
    return corpus

# --- Load queries ---
def load_queries(path):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            queries[qid] = obj["text"]
    return queries

# --- Load relevance ---
def load_relevance(path):
    relevance = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            qid, cid, score = parts

            if i == 0 and qid == "query-id":
                continue

            relevance.setdefault(qid, {})[cid] = float(score)
    return relevance

def random_ranking(doc_ids, k):
    return random.sample(doc_ids, k)

def ndcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_rel_count = min(k, len(relevant))
    if ideal_rel_count == 0:
        return 0.0

    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(ideal_rel_count))
    return dcg / ideal_dcg

def recall_at_k(relevant, retrieved, k):
    if len(relevant) == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


def run_seed(seed, corpus, queries, relevance):
    random.seed(seed)

    K = 10
    ndcg1, ndcg10 = [], []
    r1, r10 = [], []

    for qid in queries:
        ret = random_ranking(corpus, K)
        rel = relevance.get(qid, {})

        ndcg1.append(ndcg_at_k(rel, ret, 1))
        ndcg10.append(ndcg_at_k(rel, ret, 10))
        r1.append(recall_at_k(rel, ret, 1))
        r10.append(recall_at_k(rel, ret, 10))

    return {
        "NDCG@1": np.mean(ndcg1),
        "NDCG@10": np.mean(ndcg10),
        "R@1": np.mean(r1),
        "R@10": np.mean(r10),
    }


def main():
    dataset = "arguana"

    print("Loading corpus...")
    corpus = load_corpus(f"../beir/{dataset}/corpus.jsonl")
    print(f"Loaded {len(corpus)} documents.")

    print("Loading queries...")
    queries = load_queries(f"../beir/{dataset}/queries.jsonl")
    print(f"Loaded {len(queries)} queries.")

    print("Loading relevance judgments...")
    relevance = load_relevance(f"../beir/{dataset}/qrels/test.tsv")
    print("Loaded relevance for", len(relevance), "queries.")

    print("\nRunning RANDOM baseline with 3 seeds...\n")

    all_results = []
    seeds = [0, 1, 2]

    for s in seeds:
        print(f"--- Seed {s} ---")
        res = run_seed(s, corpus, queries, relevance)
        all_results.append(res)

        print(f"NDCG@1  = {res['NDCG@1']:.4f}")
        print(f"NDCG@10 = {res['NDCG@10']:.4f}")
        print(f"R@1     = {res['R@1']:.4f}")
        print(f"R@10    = {res['R@10']:.4f}")
        print()

    # Average across seeds
    print("===== AVERAGE OVER SEEDS =====")
    for metric in ["NDCG@1", "NDCG@10", "R@1", "R@10"]:
        avg = np.mean([res[metric] for res in all_results])
        print(f"{metric}: {avg:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()

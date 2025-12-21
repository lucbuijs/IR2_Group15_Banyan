import numpy as np
import logging

from tqdm import tqdm
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


class BM25Regular:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: list of tokenized documents 
                e.g. [["hello", "world"], ["hello", "chatgpt"]]
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b

        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.N

        self.tf = []
        self.df = defaultdict(int)

        # NEW: inverted index
        self.inverted_index = defaultdict(list)

        self._initialize()

    def _initialize(self):
        """Compute TF, DF, and inverted index."""
        for doc_id, doc in enumerate(self.corpus):
            tf_doc = defaultdict(int)

            # compute term frequencies
            for word in doc:
                tf_doc[word] += 1
            
            self.tf.append(tf_doc)

            # update document frequencies + inverted index
            for word in tf_doc:
                self.df[word] += 1
                self.inverted_index[word].append(doc_id)
        
        # Precompute IDF
        self.idf = {
            word: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for word, df in self.df.items()
        }

    def score(self, query_tokens, doc_id):
        """Compute BM25 score for a given doc."""
        score = 0.0
        doc_tf = self.tf[doc_id]
        doc_length = self.doc_len[doc_id]

        for word in query_tokens:
            if word not in doc_tf:
                continue

            tf = doc_tf[word]
            idf = self.idf.get(word, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += idf * (numerator / denominator)

        return score
    
    def search(self, query_tokens, top_k=5):
        """Return top_k docs ranked by BM25 using inverted index."""
        
        # Collect **only documents that contain query terms**
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])

        # Score only relevant docs
        scores = []
        for doc_id in candidate_docs:
            scores.append((doc_id, self.score(query_tokens, doc_id)))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


dataset = "quora"
corpus_path = f"../../../../examples/retrieval/evaluation/lexical/datasets/{dataset}"
corpus, queries, qrels = GenericDataLoader(corpus_path).load(split="test")

# Tokenize corpus
tokenized_corpus = [
    (doc_id, (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower().split())
    for doc_id in tqdm(corpus, desc="Tokenizing Corpus")
]
doc_ids, tokenized_docs = zip(*tokenized_corpus)

# Initialize BM25
bm25 = BM25Regular(tokenized_docs)

# Retrieve documents
retriever_results = {}
for qid in tqdm(queries, desc="Retrieving Queries"):
    query_tokens = queries[qid].lower().split()
    top_docs = bm25.search(query_tokens)
    retriever_results[qid] = {doc_ids[doc_id]: score for doc_id, score in top_docs}

# Evaluate retrieval
retriever = EvaluateRetrieval(k_values=[1, 10])
logging.basicConfig(level=logging.INFO)
logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, retriever_results, retriever.k_values)
logging.info(f"NDCG: {ndcg}")
logging.info(f"MAP: {_map}")
logging.info(f"Recall: {recall}")
logging.info(f"Precision: {precision}")


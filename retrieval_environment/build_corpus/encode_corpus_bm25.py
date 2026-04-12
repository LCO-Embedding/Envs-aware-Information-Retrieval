import numpy as np
import scipy.sparse as sp
import json
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset

def fit_and_encode_bm25(corpus, k1=1.5, b=0.75, save_path_prefix="bm25_index"):

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.vocabulary_

    n_docs, n_terms = X.shape
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    avgdl = doc_lengths.mean()
    
    term_counts = np.array(X.getnnz(axis=0)).flatten()
    idf = np.log((n_docs - term_counts + 0.5) / (term_counts + 0.5) + 1)
    idf[idf < 0] = 0

    X = X.tocsr() 
    
    X_coo = X.tocoo()
    
    rows = X_coo.row
    cols = X_coo.col
    tfs = X_coo.data
    
    len_norm = 1 - b + (b * doc_lengths[rows] / avgdl)
    
    bm25_values = (tfs * (k1 + 1)) / (tfs + (k1 * len_norm))

    bm25_matrix = sp.csr_matrix((bm25_values, (rows, cols)), shape=X.shape)
    
    sp.save_npz(f"{save_path_prefix}_corpus.npz", bm25_matrix)
    
    metadata = {
        "vocab": vocab, 
        "idf_vector": idf.tolist(),
        "params": {"k1": k1, "b": b}
    }
    
    with open(f"{save_path_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f)
        

if __name__ == "__main__":

    data = load_dataset("./RAGBench-filtered","corpus")["train"]
    texts = data["text"]
    fit_and_encode_bm25(texts, save_path_prefix="bm25_index_RAGBench")
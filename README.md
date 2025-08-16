# Wikipedia Article Clustering using TF-IDF, Sentence Embeddings, and Various Clustering Algorithms

This project explores clustering techniques on a set of Wikipedia articles to find natural groupings based on content similarity. Two approaches are implemented:

1. **Silhouette-Driven Clustering** – Evaluates multiple algorithms and hyperparameters to select the clustering with the highest silhouette score.  
2. **Best Possible Clustering Approach** – Focuses on specific methods (Agglomerative Clustering, K-Means with embeddings) to achieve optimal clustering performance.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Approach 1: Silhouette-Driven Clustering](#approach-1-silhouette-driven-clustering)
4. [Approach 2: Best Possible Clustering](#approach-2-best-possible-clustering)
5. [Clustering Algorithms](#clustering-algorithms)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

---

## Dataset
We use a small set of Wikipedia articles:

```text
Data Science, Artificial Intelligence, Machine Learning,
European Central Bank, Bank, Financial Technology,
International Monetary Fund, Basketball, Swimming, Tennis
````

* Articles are fetched using the `wikipedia` Python library.
* Preprocessing includes lowercase conversion, punctuation removal, stop word removal, and lemmatization using `spaCy`.

---

## Data Preprocessing

1. **Text cleaning & normalization:**

   * Remove numbers and punctuation
   * Convert text to lowercase
   * Lemmatization and stop word removal
2. **Feature extraction:**

   * TF-IDF vectorization (`sklearn.TfidfVectorizer`)
   * Sentence embeddings using **Sentence-BERT** (`sentence-transformers`)

---

## Approach 1: Silhouette-Driven Clustering

* **Goal:** Test multiple clustering algorithms and hyperparameters to maximize silhouette score.

* **Dimensionality reduction:** UMAP applied to embeddings for better clustering.

* **Algorithms tested:**

  * KMeans
  * Agglomerative Clustering
  * DBSCAN, HDBSCAN, OPTICS
  * MeanShift
  * Affinity Propagation
  * Spectral Clustering
  * Gaussian Mixture

* **Meta-process:**

  1. Fit each algorithm with different hyperparameters
  2. Evaluate with silhouette score, Calinski-Harabasz score, and Davies-Bouldin index
  3. Select the best performing model

* **Visualization:** Barplot of silhouette scores for all algorithms.

---

## Approach 2: Best Possible Clustering

* Focused methods for high-quality clustering:

  1. **Agglomerative Clustering with TF-IDF vectors**

     * Tested linkage methods: `ward`, `single`, `complete`, `average`
     * Evaluated silhouette score for different `k` clusters
  2. **K-Means with Sentence-BERT embeddings**

     * Embeddings: `paraphrase-MiniLM-L6-v2` and `all-MiniLM-L6-v2`
     * Tested cluster counts from 2–10 and initializations (`k-means++`, `random`)
     * Selected best model based on silhouette score

* **Output:** Cluster assignments for each article with best hyperparameters.

---

## Clustering Algorithms

| Algorithm                 | Notes                                           |
| ------------------------- | ----------------------------------------------- |
| KMeans                    | Centroid-based clustering                       |
| Agglomerative Clustering  | Hierarchical clustering with different linkages |
| DBSCAN / HDBSCAN / OPTICS | Density-based clustering                        |
| MeanShift                 | Mode-seeking clustering                         |
| AffinityPropagation       | Message-passing clustering                      |
| SpectralClustering        | Graph-based clustering                          |
| GaussianMixture           | Probabilistic model-based clustering            |

---

## Evaluation Metrics

* **Silhouette Score:** Measures cohesion vs separation of clusters (primary metric)
* **Calinski-Harabasz Index:** Ratio of between-cluster variance to within-cluster variance
* **Davies-Bouldin Index:** Average similarity between each cluster and its most similar one (lower is better)

---

## Usage

1. Install dependencies:

```bash
pip install wikipedia spacy sentence-transformers umap-learn hdbscan matplotlib seaborn scikit-learn
python -m spacy download en_core_web_sm
```

2. Run the script for Approach 1 or Approach 2.
3. The output includes:

   * Best cluster assignments for each article
   * Silhouette scores
   * Visualization of clustering performance

---

## Dependencies

* Python >= 3.8
* pandas, numpy, re
* scikit-learn
* sentence-transformers
* spacy
* wikipedia
* umap-learn
* hdbscan
* matplotlib, seaborn
* scipy (for dendrograms and linkage)

---

## Notes

* Using embeddings with UMAP often improves clustering quality over raw TF-IDF vectors.
* Silhouette score is used to select optimal hyperparameters and cluster count.
* Both approaches demonstrate different strategies: exhaustive search vs focused optimization.

---

## Author

Reihan – Data Science & NLP Enthusiast



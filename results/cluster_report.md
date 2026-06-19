# Clustering diagnostic — all-MiniLM-L6-v2

Diagnostic over the SBERT embeddings (plan §2.7 / Phase 6), not a leaderboard entry. `subject_purity` is the size-weighted dominant-subject share per cluster (text recovering subjects with no metadata); `silhouette` is the sampled cosine score. Regenerate with `python scripts/run_clustering.py`.

| algorithm | config | n_clusters | n_noise | silhouette | subject_purity | largest_cluster_frac |
| --- | --- | --- | --- | --- | --- | --- |
| kmeans | n_clusters=100 | 100 | 0 | 0.1157 | 0.3281 | 0.024 |
| agglomerative | n_clusters=100, linkage=ward | 100 | 0 | 0.0821 | 0.3242 | 0.0256 |
| hdbscan | min_cluster_size=15 | 32 | 9955 | 0.6359 | 0.2818 | 0.0152 |

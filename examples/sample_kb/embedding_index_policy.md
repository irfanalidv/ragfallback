# Embedding and Index Policy

All document text is embedded using sentence-transformers/all-MiniLM-L6-v2, producing 384-dimensional vectors. Embedding dimensions must match between the model used at index time and the model used at query time. Dimension mismatches cause retrieval errors.

Index builds are triggered automatically when documents are added or updated. Full rebuilds are required when the embedding model is changed. Stale indices serving outdated content can be detected using the StaleIndexDetector module.

Vector indices are stored on disk and loaded into memory on startup. Memory footprint scales linearly with the number of documents and the embedding dimension. For large corpora, use FAISS with an IVF index to reduce query latency.

Metadata filters can be applied at query time to restrict retrieval to a subset of documents. Metadata values must be scalar types (string, integer, float, boolean). Lists and nested dicts must be serialized before indexing.

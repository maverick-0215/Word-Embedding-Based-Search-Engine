This project was done as the part of the data science course taught by Prof. Anirban Dasgupta, we implemented a semantic search engine using word embeddings, based on the MS MARCO dataset.

#### We implement and compare three retrieval pipelines:

Word2Vec baseline

Word2Vec + BERT re-ranking

Word2Vec + BM25 re-ranking

Our results show that **combining Word2Vec with Bert based re-ranking improves retrieval effectiveness**.


## Techniques & Libraries

| Task                 | Tools/Libraries                                       |
|----------------------|-------------------------------------------------------|
| Dataset loading      | Hugging Face `datasets`                               |
| Text preprocessing   | `nltk`                                                |
| Word embeddings      | `gensim` (Word2Vec)                                   |
| Similarity search    | `faiss`                                               |
| Sentence embeddings  | `sentence-transformers` (MiniLM BERT)                |
| Visualization        | `scikit-learn`, `matplotlib`                          |


## Results

| Retrieval Setup      | MRR@10  | Recall@10 
|----------------------|---------|-----------
| Word2Vec             | 0.1568  | 0.450     
| Word2Vec + BERT      | 0.1983  | 0.525     
| Word2Vec + BM25      | 0.1701  | 0.478     

Each query had limited(1-3) relevant passages, which limited the results

Semantic similarity and clustering also confirm the enhanced relevance from BERT re-ranking.

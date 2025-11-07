# Movie-Recommender-System
## Dataset
Source: Kaggle - Movies & Ratings for Recommendation System by Nicoleta Cilibiu
https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system?select=movies.csv

Deep learning recommendation models on the movie ratings dataset (treated as implicit feedback). Implemented Neural Collaborative Filtering (NCF) and Deep Structured Semantic Model (DSSM) with negative sampling. Evaluated models using Recall@K and NDCG@K.

## Data preprocessing
Load the ratings data and treat it as implicit feedback by setting every observed rating to 1.0. Encode the user and item IDs into indices for embedding layers. Sort interactions per user by time and perform a leave-one-out split: for each user, the last interaction (by timestamp) is held out for testing.

For each positive interaction in the training set, sample several negative examples (items the user has not interacted with). I used about 4 negatives per positive (common choice).

## NCF
The NCF model uses separate embeddings for users and items, concatenates them, and passes through a multi-layer perceptron. I used a simple MLP (2 FC layers) with ReLU activations and a sigmoid output.

## DSSM
The DSSM is a two-tower model: one neural network encodes the user, another encodes the item, and the model scores their similarity by dot product. I implemented each tower as an embedding plus a hidden layer, then compute a dot and sigmoid.

## Models training
I trained both models with binary cross-entropy loss on the positive/negative samples. Used Adam optimizer and moved data to GPU. Train for a few epochs, monitoring the training loss.

## Evaluation
After training, evaluate both models on the test interactions. For each user in the test set, rank all items by predicted score and check if the true test item is in the top-K. Exclude training items from ranking and compute Recall@K (0 or 1 per user, averaged) and NDCG@K (normalized discounted gain) at various K. NDCG@K uses the position of the test item to reward higher ranks.

## Results
- NCF > DSSM. 

- Recall@10: NCF 4% vs DSSM 1.8% → ~2.2× better (low numbers because evaluation is made for a full-catalog data, so the models score almost all items for each user).

- Recall@50: 11.15% vs 5.57% → ~2× better.

- NDCG mirrors this (NCF ~2× higher).

- NCF captures non-linear user–item interactions; the DSSM two-tower dot product is more constrained without rich side features.

# Possible upgrades
- Combine GMF (Generalized Matrix Factorization) and MLP for better results.

- Add movie genres or other content features to DSSM.

- Tune hyperparameters: embedding sizes, epoch counts.
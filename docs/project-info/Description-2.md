# 🛡️ 1. HTTP Request Analysis (Attack Detection)

## Goal
Detect suspicious HTTP requests (e.g., XSS, CSRF, SQLi)

## 🧠 Suggested Techniques

| Model / Technique | Purpose |
|-------------------|----------|
| ✅ Logistic Regression / Random Forest | Binary classification (normal vs. abnormal) |
| ✅ TF-IDF + Multinomial Naive Bayes | Text-based anomaly detection on request content |
| ✅ Autoencoder (shallow) | Unsupervised anomaly detection — reconstructs "normal" traffic |
| ✅ Isolation Forest | Detects outliers based on request metadata or parameter structures |

## 🧪 Features to Use
- Request path, method, header keys
- Query/body content (after TF-IDF vectorization)
- IP rate patterns, user-agent fingerprints

## ✅ Recommendation
Start with TF-IDF + Naive Bayes or Logistic Regression for initial labeling, then move to Autoencoder for anomaly score–based filtering.

# 🎯 2. Personalized Recommendations + Dynamic Deals

## Goal
Recommend products based on user behavior & context (region, age, favorites, etc.)

## 🧠 Suggested Techniques

| Model / Technique | Purpose |
|-------------------|----------|
| ✅ Item-based Collaborative Filtering | Recommend based on similar users/products |
| ✅ Matrix Factorization (SVD) | Reduce user-product matrix to latent space |
| ✅ k-NN or Cosine Similarity | Lightweight content-based filtering |
| ✅ Light Neural Network (MLP) | Personalized ranking model (e.g., user-product input vector → score) |

## 🧪 Features to Use
- User metadata (age, region)
- Behavior data: views, favorites, comment sentiment
- Product features: tags, category, price range

## ✅ Recommendation
Use Collaborative Filtering + Content-Based Hybrid, optionally enhanced with small MLP model (e.g., 2–3 dense layers) if training data permits.

# 🆕 3. New Product Automation (Tag Suggestion + Price Estimation)

## Goal
Auto-tag and price a newly added product based on similar entries.

## 🧠 Suggested Techniques

| Model / Technique | Purpose |
|-------------------|----------|
| ✅ K-Nearest Neighbors (k-NN) | Suggest tags based on similar items |
| ✅ TF-IDF + Cosine Similarity | Compare descriptions and assign existing tags |
| ✅ Multi-label Classification (Logistic Regression or shallow NN) | Predict multiple tags from text |
| ✅ Linear Regression / Random Forest Regressor | Predict price range from description, category, tags |
| ✅ BERT (optional, pretrained) | For tag suggestions, if you want to try a small deep NLP model |

## 🧪 Features to Use
- Product title & description (NLP preprocessed)
- Category, known tags
- Numerical features: weight, brand, etc. (if available)

## ✅ Recommendation
Start with TF-IDF + similarity for tags, and Linear/Random Forest Regressor for price. Later try multi-label classification or BERT embeddings for better NLP.

# Summary Table

| Feature | Classical ML | Light NN / DL | Notes |
|---------|--------------|---------------|-------|
| Request Analysis | Naive Bayes, LogReg, Isolation Forest | Autoencoder (shallow) | Lightweight, good for rule + anomaly combo |
| Recommendations | k-NN, SVD, Clustering | MLP (2–3 dense layers) | Use hybrid filtering; small NN boosts |
| Product Tagging & Pricing | TF-IDF + k-NN, Linear/Random Forest | Multi-label classification (shallow) | Can plug in BERT embeddings optionally |

# Tools/Libraries You Can Use

| Language | ML Libraries |
|----------|--------------|
| Go | gorse, golearn, onnxruntime-go | (If we later decide to use Golang not only as an API layer)
| Python | scikit-learn, lightfm, surprise, Tensorflow, transformers |

> ✅ If you're building in Go, you can train models in Python, export them using ONNX, and load them in Go with onnxruntime.


# Books/TextBooks/Articles

1. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Chapter 5, Predicting Online Ad Click-Through with Logistic Regression.
2. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Chapter 9, Mining the 20 Newsgroups Dataset with Text AnalysisTechniques.
3. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Chapter 10, Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling.
4. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - TF and tf-idf.
5. IBM Article - "What is an autoencoder?", November, 2023 - https://www.ibm.com/think/topics/autoencoder
6. Research Gate Research - "Isolation Forest", January 2009 - https://www.researchgate.net/publication/224384174_Isolation_Forest
7. "Isolation Forest", Original Research Paper, Fei Tony Liu, 2008 - https://www.lamda.nju.edu.cn/publication/icdm08b.pdf
8. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - multi-layer perceptron (MLP), Implementing neural networks with scikit-learn, Implementing neural networks with TensorFlow.
9. Research Gate Research - "Multilayer perceptron and neural networks", July 2009 - https://www.researchgate.net/publication/228340819_Multilayer_perceptron_and_neural_networks
10. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Best practice 10 – Deciding whether to rescale features.
11. IBM Article - "What is the k-nearest neighbors (KNN) algorithm?", 2018 - https://www.ibm.com/think/topics/knn
12. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Topic modeling using NMF, What is dimensionality reduction?
13. Research Gate Research - "Matrix Factorization Model in Collaborative Filtering Algorithms: A Survey", April 2015 - https://www.researchgate.net/publication/275645705_Matrix_Factorization_Model_in_Collaborative_Filtering_Algorithms_A_Survey
14. Research Artice - "Item-Based Collaborative Filtering Recommendation Algorithms", 2001 - https://files.grouplens.org/papers/www10_sarwar.pdf
15. Blog Post - "Item-Based Collaborative Filtering in Python", April 2021 - https://towardsdatascience.com/item-based-collaborative-filtering-in-python-91f747200fab/
16. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - NLP applications, Multi-label classification.
17. Research Gate Research - "Multi-Label Classification: An Overview", September 2009, https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview
18. "Python Machine Learning By Example, Third Edition, Yuxi (Hayden) Liu" - Chapter 7, Predicting Stock Prices with Regression Algorithms.
19. Blog Post - "BERT 101 🤗 State Of The Art NLP Model Explained", March 2022 - https://huggingface.co/blog/bert-101
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
20. Research Article - "A Comparative Study on Vectorization and Classification Techniques in Sentiment Analysis to Classify Student-Lecturer Comments, Ochilbek Rakhmanov", 2020, https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050920X00160/1-s2.0-S1877050920323954/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIAcQVpquzZ1U11Pt4Fg%2BWnbSeswnwFbh%2FDfg1EVuaRdWAiAnc7c1FCqyRc5jqY8tI5%2BUVClp8S6XxZIgA%2FWI%2FBcPRCqzBQhxEAUaDDA1OTAwMzU0Njg2NSIM0uNELdrq5ebPp5JgKpAFbHFF%2B%2FlB481bd6MaOqo%2BYrCvqy0pmw2IBWOd1L8cei5iVHTxrj9ip2Kr1zhG3PuN4mDeztEhluRp2ELfYOpdVXl9n9rYNggJcZ1f9JOYYJdJgCHtYuTcNqoOTzd9ig2%2Fixk%2F0RXCB644qExS%2Be2mV%2FBDQwwfAQm5JUEwrkYO1X5EOfb%2BtceskZHVCv3yoDKgs4iq26ra1x5Wl9nQBkEEuRTaF5mTZ6hi5zxHtzRfjIr83KZFV%2Bj1JinZyrqOdFbAPOUyh%2BbC9JMJZHf6bYX6xbM91K0IaDhe7brB%2B3ZcjufLSvSfI2QufAWzioScL5x6YYpj%2Bu58n4QEpJ256nzbSZi6S7wkW7s3imR304lHy%2FnbEQak2852bVCq9Do7AoVr2ICIePtOhUs%2BpG5u1xKWKhb5j92g51hsowDWmumGCHLOcf7m67pKY%2Fdkh12I%2BEmwRQJuG59RysaUBFw2UdIVTRIPyiaZjl0Bg4o1bIzIHjUdy0r0RS1fPMC3vaKUhHpJU%2F3UE2sJfT8rQrTYZ3j1uWquJHTqgs1TJV3QWReLTlOvR%2BE6fAu4HIqZ1ypt4s70aMOJRSVmhtlBDKPzmswTEZfTFSHdWboxvl49mM4Y4ufzzP7Fwr2Jebb%2BLr1oX4N%2BzVmx1AKF%2B0uIhlCF0NQYJAvzDkjkcNDXFXjjqTcgQTDlIf4uld0%2FMuJJ%2By61Jg%2BsfhG68PlWoPg3OyPiwRq6QichMrphLDPPx06%2FPSzwUHaNZI4QtHwM9pIgWV1VI0bg5jI0ZYss67WKmjdJiwPUQaSB5jgLY08z30Sq4xcngwaJPDq6DfjmzSajN4YqOhujB2l8QUHAsul%2Fo6h81EvWHr3zyBJ9HejNTI387gLS3dgw49qPwgY6sgH0nNpvebHgEtE5KzSoAbCGCh0SWI2uDu0fhx9vR%2F2kUdkF8leH01dslFm6w0CQOCS16L1Bp5HUNEoMuwTTVzY%2FfUuAZjnV3k0qxrjJ6JJdHLqVyHUdzwN95qPP6AEjoTazUXVtvTByS3%2FxagXVD5NfZcnFQxIgSKkN9RQHu%2F01UxDF30HDq51g2LW5ToMc0zC2BGcIcZeiiJczL8CvPiL93t0MAhRigmkbqtXJrgV1r9Wx&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250607T075834Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZ553RUS2%2F20250607%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=03a6fbf8e603fbbc88ea7d5dafe1c619fae54b66bf7304f570cde64a567fd8e8&hash=3fe7c4a8e7a6d2b46b9c8673182ff02c7a98984484b2ea2b19c788abfeae2d23&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050920323954&tid=spdf-bf95d451-7c89-49d3-bcf7-ea394a0c82a0&sid=47963271228bf6477d9a16a4622ce7f5906bgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0204565a0200580203&rr=94be9ae5b8f3959f&cc=ge
    
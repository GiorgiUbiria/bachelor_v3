# Security Analyzer ML System - Comprehensive Guide

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Database Models](#database-models)
3. [ML Process Flow](#ml-process-flow)
4. [Synthetic Data Generation](#synthetic-data-generation)
5. [Model Enhancements](#model-enhancements)
6. [Testing & Evaluation Framework](#testing--evaluation-framework)
7. [Visualization & Reporting](#visualization--reporting)
8. [Implementation TODO List](#implementation-todo-list)
9. [Success Criteria](#success-criteria)

---

## 🎯 System Overview

The Security Analyzer is a research-grade ML system designed to detect web application attacks in real-time using:

- **TF-IDF + multi-class classification**
- **Ensemble method (pattern matching + ML)**
- **Danger level scoring**
- **Real-time analysis with confidence**

### Key Features:
- Multi-class attack detection (XSS, SQLi, CSRF, Benign)
- Real-time HTTP request analysis
- Explainable AI with feature importance
- Comprehensive testing framework
- Production-ready deployment

---

## 📦 Database Models

### Core Models

```python
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, DECIMAL, ARRAY, JSON, UUID, ForeignKey, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "User"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    name = Column(String)
    region = Column(String)
    birth_year = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)

    comments = relationship("Comment", back_populates="user")
    favorites = relationship("Favorite", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")


class Category(Base):
    __tablename__ = "Category"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)

    products = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "Product"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text)
    price = Column(DECIMAL)
    tags = Column(ARRAY(Text))
    category_id = Column(UUID(as_uuid=True), ForeignKey("Category.id", ondelete="SET NULL"))
    curated_price = Column(DECIMAL)
    curated_tags = Column(ARRAY(Text))
    created_by = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category", back_populates="products")
    comments = relationship("Comment", back_populates="product")
    favorites = relationship("Favorite", back_populates="product")


class Recommendation(Base):
    __tablename__ = "Recommendation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="CASCADE"))
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    reason = Column(Text)
    model_version = Column(String)
    region_based = Column(Boolean, default=False)
    age_based = Column(Boolean, default=False)
    based_on_favorites = Column(Boolean, default=False)
    based_on_comments = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="recommendations")
```

### Favorites, Comments & Voting

```python
class Favorite(Base):
    __tablename__ = "Favorite"
    __table_args__ = ( 
        # Unique constraint on (user_id, product_id)
        {'sqlite_autoincrement': True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="CASCADE"))
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    favorited_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="favorites")
    product = relationship("Product", back_populates="favorites")


class Comment(Base):
    __tablename__ = "Comment"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="CASCADE"))
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    upvotes = Column(Integer, default=0)
    downvotes = Column(Integer, default=0)
    sentiment_label = Column(String, CheckConstraint("sentiment_label IN ('positive', 'neutral', 'negative')"))
    sentiment_score = Column(Float)

    user = relationship("User", back_populates="comments")
    product = relationship("Product", back_populates="comments")


class CommentVote(Base):
    __tablename__ = "CommentVote"
    __table_args__ = (
        # Unique vote per user per comment
        CheckConstraint("vote_type IN ('up', 'down')"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="CASCADE"))
    comment_id = Column(UUID(as_uuid=True), ForeignKey("Comment.id", ondelete="CASCADE"))
    vote_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Suggestions, Logs, and Vectors

```python
class ProductSuggestion(Base):
    __tablename__ = "ProductSuggestion"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    suggested_price_min = Column(DECIMAL)
    suggested_price_max = Column(DECIMAL)
    suggested_tags = Column(ARRAY(Text))
    model_version = Column(String)
    reason = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)


class ProductSimilarityData(Base):
    __tablename__ = "ProductSimilarityData"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    similar_product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    similarity_score = Column(Float)
    based_on = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProductFeatureVector(Base):
    __tablename__ = "ProductFeatureVector"

    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"), primary_key=True)
    embedding = Column(ARRAY(Float))
    updated_at = Column(DateTime, default=datetime.utcnow)
```

### Security & Logging Models

```python
class HttpRequestLog(Base):
    __tablename__ = "HttpRequestLog"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="SET NULL"))
    ip_address = Column(String)
    user_agent = Column(Text)
    path = Column(Text)
    method = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Integer)
    status_code = Column(Integer)
    referrer = Column(Text)
    suspected_attack_type = Column(String, CheckConstraint(
        "suspected_attack_type IN ('xss', 'csrf', 'sqli', 'unknown')"))
    attack_score = Column(Float)


class UserEvent(Base):
    __tablename__ = "UserEvent"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("User.id", ondelete="CASCADE"))
    event_type = Column(String, CheckConstraint(
        "event_type IN ('view', 'click', 'add_to_cart', 'comment', 'favorite')"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("Product.id", ondelete="CASCADE"))
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class MLAnalysisLog(Base):
    __tablename__ = "ml_analysis_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_type = Column(String(50), nullable=False)
    input_data = Column(Text)
    output_data = Column(Text)
    model_version = Column(String(20), default="2.0")
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer)
    confidence_score = Column(DECIMAL(5, 4))
```

---

## 🧠 ML Process Flow

### 1. Receive HTTP Request (Input Layer)

**Input Fields**:
- `method`, `headers`, `query params`, `body`, `path`, `ip_address`, `user_agent`, `cookies`, `referrer`

→ **Pass to preprocessing pipeline**

### 2. Preprocessing (Normalization Layer)

- **Extract & Flatten**: Query/body params, headers
- **Decode**: URL encoding, HTML entities
- **Normalize**: Lowercasing, whitespace removal, punctuation stripping
- **Concatenate**: Structured into a single raw input string

→ Output: normalized text string

### 3. Feature Engineering

#### A. TF-IDF Vectorization
- Convert preprocessed text into TF-IDF vector
- Fit model using labeled data with classes: `xss`, `sqli`, `csrf`, `benign`

#### B. Pattern Matching Features
- Use rule-based detectors to flag:
  - Regex matches for known XSS payloads
  - SQL keyword injection patterns
  - Missing/mismatched CSRF token usage
- Each rule becomes a binary feature

→ Output: Feature vector = `[TF-IDF + binary features]`

### 4. Ensemble Attack Classification (Inference Layer)

#### A. ML Classifier (multi-class)
- Classify request as one of: `xss`, `sqli`, `csrf`, `benign`, or `unknown`
- Model: Logistic Regression, Random Forest, or Gradient Boosted Trees

#### B. Ensemble Decision Logic
- Combine predictions from:
  - ML classifier (probability/confidence)
  - Pattern matcher (rule-based detection)

#### C. Conflict Resolution
- If ML and rules agree → accept
- If rules detect malicious but ML is uncertain → label as suspicious
- If ML is confident but no rule match → accept ML prediction with lower trust

### 5. Danger Score Assessment

For malicious requests, calculate:
- `attack_score ∈ [0.0, 1.0]` (danger level)
  - Factors:
    - ML classifier confidence
    - # of rules triggered
    - Type severity (SQLi > XSS > CSRF)
    - Request context (e.g., method = `POST`, referrer mismatch)

### 6. Response Generation

**Response Schema:**

```json
{
  "is_malicious": true,
  "attack_type": "sqli",
  "attack_score": 0.87,
  "confidence": 0.91,
  "suspected_attack_type": "sqli",
  "details": {
    "matched_patterns": ["' OR 1=1", "--"],
    "tfidf_top_terms": ["select", "from", "admin"],
    "request_path": "/login",
    "ip_address": "203.0.113.42"
  }
}
```

### 7. Real-Time Logging & Auditing

Persist to `HttpRequestLog` table:
- `ip_address`, `path`, `method`, `user_agent`
- `suspected_attack_type`, `attack_score`, `timestamp`, `referrer`
- Store `pattern match summary` and `model confidence` in metadata/logs

### ML Stack Overview

| Component           | Tool / Library                                                  |
| ------------------- | --------------------------------------------------------------- |
| Preprocessing       | `scikit-learn`, `re`, `html`, `urllib.parse`                    |
| TF-IDF Vectorizer   | `TfidfVectorizer` (`sklearn.feature_extraction.text`)           |
| Classifier          | `RandomForestClassifier`, `LogisticRegression`, `XGBoost`, etc. |
| Pattern Matching    | Regex, heuristic rules                                          |
| Ensemble            | Custom logic (Python class)                                     |
| Scoring & Inference | Confidence + severity fusion                                    |
| API Wrapper         | `FastAPI`                                                       |
| DB Logging          | PostgreSQL – `HttpRequestLog` table                             |

---

## 🛠 Synthetic Data Generation

To generate **synthetic data** for **Security Analyzer**, follow the steps below. You'll create **realistic HTTP request data** containing both **benign** and **malicious** examples (XSS, SQLi, CSRF) with labels.

### ⚙️ Define Your Schema

Match synthetic data fields with both your **ML model input** and **DB schema**:

| Field          | Use in Model              | DB Field?     | Example Value                   |
| -------------- | ------------------------- | ------------- | ------------------------------- |
| `method`       | Yes (important context)   | ✅             | `GET`, `POST`                   |
| `path`         | Yes                       | ✅             | `/login`, `/profile`            |
| `query_params` | Yes (attack vectors)      | ✅ (flattened) | `?id=1' OR 1=1--`               |
| `headers`      | Maybe                     | ✅             | `User-Agent`, `Referer`         |
| `body`         | Yes (XSS vectors)         | ✅             | `{ "username": "<script>..." }` |
| `cookies`      | Optional                  | ✅             | `csrftoken=...`                 |
| `ip_address`   | Optional (logging)        | ✅             | `198.51.100.12`                 |
| `user_agent`   | Optional (fingerprinting) | ✅             | `Mozilla/5.0...`                |
| `attack_type`  | **Label**                 | ✅             | `sqli`, `xss`, `csrf`, `benign` |
| `attack_score` | Generated by model        | ✅             | `0.00` (benign) to `1.00`       |

### 🛠 Create Attack Vector Templates

#### SQLi Examples:
```python
sqli_payloads = [
  "' OR '1'='1", "1 OR 1=1 --", "' UNION SELECT null, null--",
  "'; DROP TABLE users; --", "' OR EXISTS(SELECT * FROM users) --"
]
```

#### XSS Examples:
```python
xss_payloads = [
  "<script>alert('XSS')</script>", "<img src=x onerror=alert('XSS')>",
  "<svg/onload=alert('XSS')>", "javascript:alert('XSS')"
]
```

#### CSRF Simulation:
```python
csrf_examples = [
  {"path": "/transfer", "method": "POST", "headers": {"Referer": "https://attacker.com"}},
  {"path": "/update-email", "method": "POST", "headers": {}, "body": {"email": "attacker@example.com"}}
]
```

#### Benign Examples:
```python
benign_examples = [
  {"method": "GET", "path": "/home", "query": "q=product", "body": {}, "headers": {"User-Agent": "..."}},
  {"method": "POST", "path": "/login", "body": {"username": "john", "password": "safe123"}}
]
```

### 🧠 Generate Synthetic Entries

```python
import random
import json
import faker

faker = faker.Faker()

def generate_request(label: str):
    base = {
        "method": random.choice(["GET", "POST"]),
        "path": random.choice(["/login", "/search", "/comment", "/transfer"]),
        "headers": {
            "User-Agent": faker.user_agent(),
            "Referer": faker.url(),
        },
        "cookies": {"csrftoken": faker.sha1()},
        "ip_address": faker.ipv4_public(),
        "user_agent": faker.user_agent()
    }

    if label == "sqli":
        payload = random.choice(sqli_payloads)
        base["query_params"] = f"id={payload}"
        base["body"] = {}
    elif label == "xss":
        payload = random.choice(xss_payloads)
        base["body"] = {"comment": payload}
        base["query_params"] = ""
    elif label == "csrf":
        entry = random.choice(csrf_examples)
        base.update(entry)
    else:  # benign
        base.update(random.choice(benign_examples))

    base["attack_type"] = label
    base["attack_score"] = round(random.uniform(0.7, 1.0), 2) if label != "benign" else 0.0
    return base
```

### 🔁 Bulk Generate Dataset

```python
dataset = []

for _ in range(10000):  # Generate 10,000 samples
    label = random.choices(
        population=["sqli", "xss", "csrf", "benign"],
        weights=[0.25, 0.25, 0.2, 0.3],
        k=1
    )[0]
    dataset.append(generate_request(label))

# Save to JSON or CSV
with open("synthetic_http_requests.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

### 💡 Tips for Realism

- Vary payload placement: headers, cookies, body, query
- Add noise to benign requests (e.g. typos, long params)
- Use `faker` to generate user info, IPs, timestamps
- Encode some malicious payloads (`urlencode`, base64) to mimic evasion techniques
- Generate high volume data (10,000+ samples) for production-viable model training

---

## 🚀 Model Enhancements

### 🧠 Model Enhancements

| Area                    | Suggestion                                                                                               | Purpose                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------- |
| **Feature Engineering** | - Add custom features (e.g. URL length, presence of SQL keywords, suspicious headers)                    | Improve model accuracy    |
| **Embedding Upgrade**   | Consider switching from TF-IDF to transformer embeddings (e.g. BERT, FastText) for better generalization | Capture deeper semantics  |
| **Temporal Signals**    | Log-based features like request frequency spikes, time-of-day                                            | Detect DoS-like behaviors |
| **Online Learning**     | Use partial fit / streaming classifiers for real-time retraining                                         | Adaptive security system  |

### 🧩 Model Workflow Additions

| Area                            | Suggestion                                                                             | Purpose                                          |
| ------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Attack Explanation**          | Add explainability (e.g. SHAP or LIME)                                                 | Understand which tokens triggered classification |
| **Ensemble Weights Logging**    | Save how much weight pattern-matching vs. ML contributed                               | Useful for research validation                   |
| **Danger Level Categorization** | Use thresholds or clustering (e.g. k-means on score) to define "low", "medium", "high" | Structured reporting                             |
| **Attack Family Grouping**      | Classify into broader categories (Injection, Forgery, etc.)                            | Taxonomy-level analysis for research             |

### 🔐 Security-Oriented Design Improvements

| Area                          | Suggestion                                                              | Purpose                      |
| ----------------------------- | ----------------------------------------------------------------------- | ---------------------------- |
| **Sanitization Suggestions**  | Return mitigation advice with malicious responses                       | Defense in depth             |
| **Rule Injection Simulation** | Test adversarial attacks (e.g. obfuscated XSS) to improve robustness    | Attack resilience            |
| **False Positive Tracking**   | Allow user feedback or audit log review to confirm true/false positives | Continuous model improvement |

### 📊 Research Reporting & Paper-Worthy Features

| Area                                | Suggestion                                                                         | Purpose                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------ |
| **t-SNE/UMAP Projection**           | Project input samples into 2D space to visualize clusters of attack types          | Adds great value to research   |
| **Ablation Study**                  | Test performance with vs. without pattern matching, ML only, different vectorizers | Research rigor                 |
| **Error Analysis Module**           | Automatically collect & analyze high-confidence false positives/negatives          | Improve understanding          |
| **Real-World Dataset Benchmarking** | Evaluate on a public dataset (e.g., CSIC 2010, OWASP Juice Shop logs)              | Stronger validation            |

### 🧪 Testing & Deployment

| Area                         | Suggestion                                                     | Purpose                                |
| ---------------------------- | -------------------------------------------------------------- | -------------------------------------- |
| **End-to-End Pipeline Test** | Automate data → vectorization → prediction → report generation | Ensure reproducibility                 |
| **Stress/Spike Testing**     | Simulate traffic bursts                                        | Test reliability under real-world load |
| **Containerized Deployment** | Use Docker to wrap model + API + monitoring                    | Reproducible experiments               |
| **CI/CD Integration**        | Auto-run test suite on code/model updates                      | Engineering best practices             |

---

## 📊 Testing & Evaluation Framework

Creating a **comprehensive testing suite** for your Security Analyzer ML model involves systematically testing **performance**, **accuracy**, and **resource metrics**.

### 🧪 Prepare Your Testing Infrastructure

#### Requirements:
- **Test framework**: Pytest or unittest
- **Benchmarking**: `locust`, `ab`, or custom scripts with `asyncio + aiohttp`
- **Model metrics**: `scikit-learn`, `numpy`, `pandas`
- **Monitoring**: `psutil`, `tracemalloc`, or Docker metrics
- **Visualizations**: `matplotlib`, `seaborn`, `scikit-plot`

### 🧩 Test Dataset Preparation

- **Generate or collect a test set** with known `attack_type` and `attack_score`
- Ensure all classes (`benign`, `sqli`, `xss`, `csrf`) are **represented equally** or as per real-world distribution
- Split into:
  - ✅ **Accuracy test set**
  - ✅ **Load test set** (large volume)

### ⚙️ Functional and ML Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# y_true = actual attack_type
# y_pred = model-predicted class
# y_score = model-predicted probabilities

print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("ROC AUC (macro):", roc_auc_score(y_true, y_score, multi_class='ovr'))
```

**Metrics captured:**
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-score
- ✅ Confusion matrix
- ✅ False Positive Rate / False Negative Rate
- ✅ ROC-AUC

### 🚦 Performance Testing (Latency, Throughput, Response Time)

```python
import time, asyncio, aiohttp

async def send_request(session, data):
    start = time.time()
    async with session.post("http://localhost:8000/analyze", json=data) as resp:
        latency = time.time() - start
        return latency

async def run_test(data_batch):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, d) for d in data_batch]
        latencies = await asyncio.gather(*tasks)
        print("Avg Latency:", sum(latencies)/len(latencies))
```

Measure:
- ✅ **Throughput** = requests/sec
- ✅ **Latency** = time between request send and response received
- ✅ **Response Time** = time until the full response is available

### 📊 Resource Utilization Monitoring

```python
import psutil

proc = psutil.Process()
print("Memory (MB):", proc.memory_info().rss / 1024 / 1024)
print("CPU (%):", proc.cpu_percent(interval=1.0))
```

Log:
- Peak memory usage
- CPU usage during load test
- Optional: GPU utilization if using GPU-based models

### Model Folder Structure

security_analyzer/
│
├── __init__.py
├── config.py                  # Configurations (thresholds, model paths, etc.)
├── constants.py               # Regex patterns, labels, danger thresholds, etc.
├── routes.py                  # FastAPI routes for security analysis endpoints
├── run_comprehensive_tests.py # Comprehensive test suite runner
│
├── analysis/
│   ├── ablation_study.py      # Component importance analysis framework
│   └── error_analysis.py      # High-confidence error pattern analysis
│
├── benchmarks/
│   └── dataset_benchmark.py   # Real-world dataset benchmarking (CSIC 2010, etc.)
│
├── data/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning/tokenization
│   ├── synthetic_generator.py # Script to generate synthetic security data
│   └── datasets/              # Folder for generated/test datasets (created at runtime)
│
├── database/
│   ├── logger.py             # Database logging for requests and analysis
│   └── models.py             # SQLAlchemy models for security data storage
│
├── model/
│   ├── __init__.py
│   ├── classifier.py         # Main SecurityAnalyzer with ensemble logic
│   ├── enhanced_classifier.py # Advanced ensemble classifier implementation
│   ├── explainability.py     # LIME-based model explainers
│   └── scoring.py            # Danger level scoring logic
│
├── pattern_matching/
│   ├── __init__.py
│   ├── csrf.py              # Pattern rules for CSRF detection
│   ├── matcher.py           # Combined matcher interface
│   ├── sql_injection.py     # Pattern rules for SQLi detection
│   └── xss.py               # Regex or heuristic rules for XSS
│
├── security/
│   ├── feedback_system.py   # Human-in-the-loop feedback collection
│   └── mitigation_advisor.py # Context-aware mitigation advice generation
│
├── tests/
│   ├── __init__.py
│   ├── test_inference.py    # Unit tests for inference
│   ├── test_metrics.py      # Test metric calculations
│   ├── test_patterns.py     # Validate pattern detection
│   └── test_performance.py  # Throughput, latency, stress testing
│
├── train/
│   ├── __init__.py
│   ├── evaluate_model.py    # Comprehensive model evaluation and robustness testing
│   ├── evaluation.py        # Model evaluation utilities and metrics
│   ├── train_model.py       # Training script and model trainer
│   └── visualization.py     # Confusion matrix, ROC curve, model visualizations
│
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   ├── profiling.py        # CPU, memory, system resource monitoring
│   ├── schema.py           # Pydantic models for request/response
│   └── timer.py            # Latency / throughput tracking utilities
│
├── vectorizer/
│   ├── __init__.py
│   ├── embed_utils.py      # Advanced embedding utilities (BERT, FastText)
│   └── tfidf_vectorizer.py # TF-IDF loading & transformation
│
└── visualization/
    └── advanced_visualization.py # 2D clustering, ablation plots, attack timelines

### Key Components Overview

#### Core Analysis Pipeline
- **`model/classifier.py`**: Main SecurityAnalyzer class with ensemble decision logic
- **`model/enhanced_classifier.py`**: Advanced voting classifier with multiple algorithms
- **`pattern_matching/matcher.py`**: Rule-based detection system integration
- **`vectorizer/tfidf_vectorizer.py`**: Feature extraction and text vectorization

#### Research & Evaluation
- **`analysis/ablation_study.py`**: Systematic component importance analysis
- **`analysis/error_analysis.py`**: High-confidence error pattern clustering
- **`benchmarks/dataset_benchmark.py`**: Real-world dataset evaluation framework
- **`train/evaluate_model.py`**: Comprehensive robustness and cross-validation testing

#### Production Features
- **`database/logger.py`**: Complete request and analysis logging system
- **`security/feedback_system.py`**: Human feedback collection for model improvement
- **`security/mitigation_advisor.py`**: Context-aware security advice generation
- **`utils/profiling.py`**: Real-time system resource monitoring

#### Visualization & Analysis
- **`visualization/advanced_visualization.py`**: Research-grade 2D attack clustering and analysis
- **`train/visualization.py`**: Model performance visualization utilities

#### Testing Infrastructure
- **`tests/`**: Comprehensive unit and integration test suite
- **`run_comprehensive_tests.py`**: Automated full system validation

This structure supports:
- ✅ **Research-grade analysis** with ablation studies and error analysis
- ✅ **Production deployment** with comprehensive logging and monitoring
- ✅ **Real-world benchmarking** against established security datasets
- ✅ **Continuous improvement** through feedback systems and robustness testing
- ✅ **Academic rigor** with systematic evaluation and visualization capabilities

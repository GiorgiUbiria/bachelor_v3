Great! Let's **extend the database schema** to support:

1. **Regional and age-based recommendations**
2. **User behavioral signals like favorites, comments**
3. **Simplified abnormal request detection based on attack patterns**
4. **Smarter new product automation using content-based metadata**

---

## ✅ Finalized Schema Overview

### 🧍 User (extended)

```sql
User (
  id UUID PK,
  email VARCHAR UNIQUE NOT NULL,
  password_hash VARCHAR NOT NULL,
  name VARCHAR,
  region VARCHAR,       -- e.g., "EU", "NA", "Asia"
  birth_year INT,       -- used to calculate age
  created_at TIMESTAMP,
  is_admin BOOLEAN DEFAULT FALSE
)
```

---

### 📦 Product (unchanged except noted)

```sql
Product (
  id UUID PK,
  name VARCHAR NOT NULL,
  description TEXT,
  price DECIMAL,
  tags TEXT[],
  category_id UUID FK -> Category(id),
  curated_price DECIMAL,
  curated_tags TEXT[],
  created_by UUID FK -> User(id),
  created_at TIMESTAMP
)
```

---

### 💡 Recommendation (extended with source signals)

```sql
Recommendation (
  id UUID PK,
  user_id UUID FK -> User(id),
  product_id UUID FK -> Product(id),
  reason TEXT,
  model_version VARCHAR,
  region_based BOOLEAN DEFAULT FALSE,
  age_based BOOLEAN DEFAULT FALSE,
  based_on_favorites BOOLEAN DEFAULT FALSE,
  based_on_comments BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP
)
```

---

### ⭐ Favorites

```sql
Favorite (
  id UUID PK,
  user_id UUID FK -> User(id),
  product_id UUID FK -> Product(id),
  favorited_at TIMESTAMP
)
```

---

### 💬 Comments

* Allow users to upvote/downvote a comment
* Aggregate the votes (total, positive, negative)
* Use this to classify a comment as *negative*, *neutral*, or *positive*

```sql
Comment (
  id UUID PK,
  user_id UUID FK -> User(id),
  product_id UUID FK -> Product(id),
  body TEXT NOT NULL,
  created_at TIMESTAMP,
  
  upvotes INT DEFAULT 0,
  downvotes INT DEFAULT 0,

  sentiment_label VARCHAR,  -- "positive", "neutral", "negative"
  sentiment_score FLOAT     -- Optional: score from -1 to 1 for fine-grain control
)
```

> The `sentiment_label` and `sentiment_score` can be updated:
>
> * Periodically with a background job
> * Or triggered via upvote/downvote activity

### 👍👎 `CommentVote` Model

```sql
CommentVote (
  id UUID PK,
  user_id UUID FK -> User(id),
  comment_id UUID FK -> Comment(id),
  vote_type VARCHAR CHECK (vote_type IN ('up', 'down')),
  created_at TIMESTAMP,

  UNIQUE (user_id, comment_id) -- One vote per user per comment
)
```

> This enables:
>
> * Preventing multiple votes from same user
> * Recalculating vote counts accurately
> * Later analytics: e.g., abusive users, vote trends

### 🧠 Vote-Based Sentiment Rules (example)

* `upvotes - downvotes > 5` → "positive"
* `downvotes - upvotes > 5` → "negative"
* Otherwise → "neutral"

You can combine this with:

* NLP-based comment analysis (e.g., toxicity detection using a model)
* Reputation system (e.g., high-rep users’ votes weigh more)

### 📌 Recommendation Influence

You can also use `Comment.sentiment_label` to:

* Penalize products with many negative comments
* Use positive comments as signals in recommendations
* Cluster users by their sentiment patterns for collaborative filtering

---

### 🧠 ProductSuggestion (extended with reasoning metadata)

```sql
ProductSuggestion (
  id UUID PK,
  product_id UUID FK -> Product(id),
  suggested_price_min DECIMAL,
  suggested_price_max DECIMAL,
  suggested_tags TEXT[],
  model_version VARCHAR,
  reason TEXT,  -- e.g., "based on 42 similar products in category + NLP"
  generated_at TIMESTAMP
)
```

---

### 📁 ProductSimilarityData (supporting content-based ML)

```sql
ProductSimilarityData (
  id UUID PK,
  product_id UUID FK -> Product(id),
  similar_product_id UUID FK -> Product(id),
  similarity_score FLOAT, -- e.g., cosine similarity
  based_on TEXT,          -- e.g., "description", "features"
  created_at TIMESTAMP
)
```

---

### 🛡️ HttpRequestLog (extended with attack detection)

```sql
HttpRequestLog (
  id UUID PK,
  user_id UUID FK -> User(id),
  ip_address VARCHAR,
  user_agent TEXT,
  path TEXT,
  method VARCHAR,
  timestamp TIMESTAMP,
  duration_ms INT,
  status_code INT,
  referrer TEXT,
  suspected_attack_type TEXT, -- NULL | "xss" | "csrf" | "sqli" | "unknown"
  attack_score FLOAT          -- Optional ML or heuristic score
)
```

---

## 🔁 Example UserEvent Types for Recommendation System

```sql
UserEvent (
  id UUID PK,
  user_id UUID FK -> User(id),
  event_type VARCHAR, -- e.g. "view", "click", "add_to_cart", "comment", "favorite"
  product_id UUID FK -> Product(id),
  metadata JSONB,
  timestamp TIMESTAMP
)
```

These events will power both:

* **Personal recommendations (based on history, region, age group)**
* **Model features (for collaborative filtering or hybrid models)**

---

## 📌 Optional: Precomputed Similarity Matrix

If using ML offline to compute similarities:

```sql
ProductFeatureVector (
  product_id UUID PK,
  embedding FLOAT[], -- e.g., vector for similarity search
  updated_at TIMESTAMP
)
```

---

## 🧠 Summary of ML Support

| Purpose                    | Model(s) Used                                                                   |
| -------------------------- | ------------------------------------------------------------------------------- |
| **HTTP Request Analysis**  | `HttpRequestLog` (XSS, CSRF, SQLi detection flags)                              |
| **Personalized Recos**     | `UserEvent`, `Recommendation`, `Favorite`, `Comment`, `User (region/age)`       |
| **Favorites-based Recos**  | `Favorite`, `UserEvent`                                                         |
| **Comment-based Recos**    | `Comment`, `UserEvent`                                                          |
| **New Product Automation** | `ProductSuggestion`, `ProductSimilarityData`, `ProductFeatureVector`            |
| **Content-Based ML**       | `Product.description`, `tags`, `category_id`, `FeatureVector`, `SimilarityData` |

---
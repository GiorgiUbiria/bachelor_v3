# 🔄 Enhanced Application Flow (ML-Backend-Database)

## 1. HTTP Request Security Analysis Flow
1. **Request Ingress**:
   - User/attacker makes HTTP request to backend (Go Fiber)
   - Request logged in `HttpRequestLog` table with metadata (path, method, IP, etc.)

2. **Security Analysis**:
   - Request forwarded to ML service `/security/analyze` endpoint
   - ML service checks for:
     - XSS patterns (script tags, event handlers)
     - SQLi patterns (UNION, comments, quotes)
     - CSRF indicators (missing referrer, suspicious origin)
     - Anomalous patterns via Isolation Forest

3. **Response Handling**:
   - ML service returns attack score (0-1) and type
   - Backend logs analysis in `HttpRequestLog.suspected_attack_type`
   - Suspicious requests blocked/flagged

## 2. Enhanced Personalized Recommendations Flow
1. **Data Collection**:
   - User interactions logged in `UserEvent` (views, favorites, purchases)
   - Sentiment analysis on `Comment` content stored in `sentiment_score`
   - User metadata (region, age, purchase history) from `User` table
   - Product features from `ProductFeatureVector`

2. **Multi-Stage Recommendation Generation**:
   - **Stage 1**: Real-time scoring (every API call):
     - Lightweight k-NN for immediate recommendations
     - Checks `UserEvent` for recent activity
   - **Stage 2**: Batch processing (hourly):
     - Matrix factorization updates user-product affinity scores
     - Cluster users into segments for targeted deals
     - Generate dynamic pricing based on demand prediction
   - **Stage 3**: Hybrid enrichment:
     - Combine collaborative filtering with content-based scores
     - Apply business rules for deal eligibility
     - Store results in `Recommendation` with deal metadata

3. **Deal Generation Logic**:
   - Special offers generated for:
     - High-affinity products with low recent sales
     - Products similar to frequently viewed but not purchased
     - Seasonal/regional promotions
   - Deal attributes stored in `Recommendation`:
     - `discount_percentage`
     - `deal_expiry`
     - `deal_reason` ("Popular in your area", "Limited time offer", etc.)

4. **API Delivery**:
   - Frontend requests via `/api/recommendations?include_deals=true`
   - Backend returns:
     - Base recommendations (sorted by affinity score)
     - Personalized deals (with discount details)
     - "Why recommended" explanations

## 3. Enhanced Product Automation Flow
1. **On-Demand Suggestions**:
   - New endpoint `/api/products/suggest` accepts:
     - Product name/description (required)
     - Optional category/brand hints
   - ML service processes without DB storage:
     - Real-time TF-IDF vectorization
     - Similarity search against existing products
     - Price prediction from regression model
     - Tag suggestions from multi-label classifier

2. **Suggestion Response**:
   - Returns structured object with:
     - `suggested_price_range` (min/max)
     - `recommended_tags` (with confidence scores)
     - `similar_products` (IDs + similarity scores)
     - `model_version` (for traceability)

3. **Optional Persistence**:
   - Client can choose to save suggestions to:
     - `ProductSuggestion` (if product exists)
     - New `DraftProduct` table (if not yet created)

## Data Flow Diagram


package models

import (
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"gorm.io/gorm"
)

type User struct {
	ID           uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	Email        string    `json:"email" gorm:"unique;not null;index"`
	PasswordHash string    `json:"-" gorm:"not null"`
	Name         *string   `json:"name" gorm:"index"`
	Region       *string   `json:"region"`
	BirthYear    *int      `json:"birth_year"`
	CreatedAt    time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP;index"`
	IsAdmin      bool      `json:"is_admin" gorm:"default:false"`

	Recommendations []Recommendation `json:"recommendations,omitempty" gorm:"foreignKey:UserID"`
	Favorites       []Favorite       `json:"favorites,omitempty" gorm:"foreignKey:UserID"`
	Comments        []Comment        `json:"comments,omitempty" gorm:"foreignKey:UserID"`
	CommentVotes    []CommentVote    `json:"comment_votes,omitempty" gorm:"foreignKey:UserID"`
	UserEvents      []UserEvent      `json:"user_events,omitempty" gorm:"foreignKey:UserID"`
	HttpRequestLogs []HttpRequestLog `json:"http_request_logs,omitempty" gorm:"foreignKey:UserID"`
}

type Category struct {
	ID   uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	Name string    `json:"name" gorm:"not null"`

	Products []Product `json:"products,omitempty" gorm:"foreignKey:CategoryID"`
}

type Product struct {
	ID           uuid.UUID      `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	Name         string         `json:"name" gorm:"not null"`
	Description  *string        `json:"description"`
	Price        *float64       `json:"price" gorm:"type:decimal"`
	Tags         pq.StringArray `json:"tags" gorm:"type:text[]"`
	CategoryID   *uuid.UUID     `json:"category_id" gorm:"type:uuid"`
	CuratedPrice *float64       `json:"curated_price" gorm:"type:decimal"`
	CuratedTags  pq.StringArray `json:"curated_tags" gorm:"type:text[]"`
	CreatedBy    *uuid.UUID     `json:"created_by" gorm:"type:uuid"`
	CreatedAt    time.Time      `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`

	Category                *Category               `json:"category,omitempty" gorm:"foreignKey:CategoryID"`
	Creator                 *User                   `json:"creator,omitempty" gorm:"foreignKey:CreatedBy"`
	Recommendations         []Recommendation        `json:"recommendations,omitempty" gorm:"foreignKey:ProductID"`
	Favorites               []Favorite              `json:"favorites,omitempty" gorm:"foreignKey:ProductID"`
	Comments                []Comment               `json:"comments,omitempty" gorm:"foreignKey:ProductID"`
	UserEvents              []UserEvent             `json:"user_events,omitempty" gorm:"foreignKey:ProductID"`
	ProductSuggestions      []ProductSuggestion     `json:"product_suggestions,omitempty" gorm:"foreignKey:ProductID"`
	SimilarityDataAsProduct []ProductSimilarityData `json:"similarity_data_as_product,omitempty" gorm:"foreignKey:ProductID"`
	SimilarityDataAsSimilar []ProductSimilarityData `json:"similarity_data_as_similar,omitempty" gorm:"foreignKey:SimilarProductID"`
	FeatureVector           *ProductFeatureVector   `json:"feature_vector,omitempty" gorm:"foreignKey:ProductID"`
}

type Recommendation struct {
	ID               uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID           uuid.UUID `json:"user_id" gorm:"type:uuid;not null"`
	ProductID        uuid.UUID `json:"product_id" gorm:"type:uuid;not null"`
	Reason           *string   `json:"reason"`
	ModelVersion     *string   `json:"model_version"`
	RegionBased      bool      `json:"region_based" gorm:"default:false"`
	AgeBased         bool      `json:"age_based" gorm:"default:false"`
	BasedOnFavorites bool      `json:"based_on_favorites" gorm:"default:false"`
	BasedOnComments  bool      `json:"based_on_comments" gorm:"default:false"`
	CreatedAt        time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`

	User    User    `json:"user" gorm:"foreignKey:UserID"`
	Product Product `json:"product" gorm:"foreignKey:ProductID"`
}

type Favorite struct {
	ID          uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID      uuid.UUID `json:"user_id" gorm:"type:uuid;not null"`
	ProductID   uuid.UUID `json:"product_id" gorm:"type:uuid;not null"`
	FavoritedAt time.Time `json:"favorited_at" gorm:"default:CURRENT_TIMESTAMP"`

	User    User    `json:"user" gorm:"foreignKey:UserID"`
	Product Product `json:"product" gorm:"foreignKey:ProductID"`
}

type Comment struct {
	ID             uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID         uuid.UUID `json:"user_id" gorm:"type:uuid;not null"`
	ProductID      uuid.UUID `json:"product_id" gorm:"type:uuid;not null"`
	Body           string    `json:"body" gorm:"not null"`
	CreatedAt      time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`
	Upvotes        int       `json:"upvotes" gorm:"default:0"`
	Downvotes      int       `json:"downvotes" gorm:"default:0"`
	SentimentLabel *string   `json:"sentiment_label" gorm:"check:sentiment_label IN ('positive', 'neutral', 'negative')"`
	SentimentScore *float64  `json:"sentiment_score"`

	User         User          `json:"user" gorm:"foreignKey:UserID"`
	Product      Product       `json:"product" gorm:"foreignKey:ProductID"`
	CommentVotes []CommentVote `json:"comment_votes,omitempty" gorm:"foreignKey:CommentID"`
}

type CommentVote struct {
	ID        uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID    uuid.UUID `json:"user_id" gorm:"type:uuid;not null"`
	CommentID uuid.UUID `json:"comment_id" gorm:"type:uuid;not null"`
	VoteType  string    `json:"vote_type" gorm:"check:vote_type IN ('up', 'down')"`
	CreatedAt time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`

	User    User    `json:"user" gorm:"foreignKey:UserID"`
	Comment Comment `json:"comment" gorm:"foreignKey:CommentID"`
}

type ProductSuggestion struct {
	ID                uuid.UUID      `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	ProductID         uuid.UUID      `json:"product_id" gorm:"type:uuid;not null"`
	SuggestedPriceMin *float64       `json:"suggested_price_min" gorm:"type:decimal"`
	SuggestedPriceMax *float64       `json:"suggested_price_max" gorm:"type:decimal"`
	SuggestedTags     pq.StringArray `json:"suggested_tags" gorm:"type:text[]"`
	ModelVersion      *string        `json:"model_version"`
	Reason            *string        `json:"reason"`
	GeneratedAt       time.Time      `json:"generated_at" gorm:"default:CURRENT_TIMESTAMP"`

	Product Product `json:"product" gorm:"foreignKey:ProductID"`
}

type ProductSimilarityData struct {
	ID               uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	ProductID        uuid.UUID `json:"product_id" gorm:"type:uuid;not null"`
	SimilarProductID uuid.UUID `json:"similar_product_id" gorm:"type:uuid;not null"`
	SimilarityScore  float64   `json:"similarity_score"`
	BasedOn          *string   `json:"based_on"`
	CreatedAt        time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`

	Product        Product `json:"product" gorm:"foreignKey:ProductID"`
	SimilarProduct Product `json:"similar_product" gorm:"foreignKey:SimilarProductID"`
}

type HttpRequestLog struct {
	ID                  uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID              *uuid.UUID             `json:"user_id" gorm:"type:uuid"`
	IPAddress           *string                `json:"ip_address"`
	UserAgent           *string                `json:"user_agent"`
	Path                *string                `json:"path"`
	Method              *string                `json:"method"`
	Timestamp           time.Time              `json:"timestamp" gorm:"default:CURRENT_TIMESTAMP"`
	DurationMs          *int                   `json:"duration_ms"`
	StatusCode          *int                   `json:"status_code"`
	Referrer            *string                `json:"referrer"`
	SuspectedAttackType *string                `json:"suspected_attack_type" gorm:"check:suspected_attack_type IN ('xss', 'csrf', 'sqli', 'benign', 'unknown')"`
	AttackScore         *float64               `json:"attack_score"`
	// Enhanced security fields
	QueryParams     *string                `json:"query_params"`
	Headers         map[string]interface{} `json:"headers" gorm:"type:jsonb"`
	Body            map[string]interface{} `json:"body" gorm:"type:jsonb"`
	Cookies         map[string]interface{} `json:"cookies" gorm:"type:jsonb"`
	ConfidenceScore *float64               `json:"confidence_score"`
	IsMalicious     bool                   `json:"is_malicious" gorm:"default:false"`
	PatternMatches  map[string]interface{} `json:"pattern_matches" gorm:"type:jsonb"`
	MLPrediction    *string                `json:"ml_prediction"`
	EnsembleWeights map[string]interface{} `json:"ensemble_weights" gorm:"type:jsonb"`

	User           *User            `json:"user,omitempty" gorm:"foreignKey:UserID"`
	MLAnalysisLogs []MLAnalysisLog  `json:"ml_analysis_logs,omitempty" gorm:"foreignKey:RequestLogID"`
	SecurityFeedback []SecurityFeedback `json:"security_feedback,omitempty" gorm:"foreignKey:RequestLogID"`
}

type UserEvent struct {
	ID        uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID    uuid.UUID              `json:"user_id" gorm:"type:uuid;not null"`
	EventType string                 `json:"event_type" gorm:"not null;check:event_type IN ('view', 'click', 'add_to_cart', 'comment', 'favorite')"`
	ProductID *uuid.UUID             `json:"product_id" gorm:"type:uuid"`
	Metadata  map[string]interface{} `json:"metadata" gorm:"type:jsonb"`
	Timestamp time.Time              `json:"timestamp" gorm:"default:CURRENT_TIMESTAMP"`

	User    User     `json:"user" gorm:"foreignKey:UserID"`
	Product *Product `json:"product,omitempty" gorm:"foreignKey:ProductID"`
}

type ProductFeatureVector struct {
	ProductID uuid.UUID       `json:"product_id" gorm:"type:uuid;primary_key"`
	Embedding pq.Float64Array `json:"embedding" gorm:"type:float[]"`
	UpdatedAt time.Time       `json:"updated_at" gorm:"default:CURRENT_TIMESTAMP"`

	Product Product `json:"product" gorm:"foreignKey:ProductID"`
}

// New enhanced security models

type MLAnalysisLog struct {
	ID                uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	RequestLogID      *uuid.UUID             `json:"request_log_id" gorm:"type:uuid"`
	AnalysisType      string                 `json:"analysis_type" gorm:"not null"`
	ModelVersion      string                 `json:"model_version" gorm:"default:'2.0'"`
	InputFeatures     map[string]interface{} `json:"input_features" gorm:"type:jsonb"`
	MLProbabilities   map[string]interface{} `json:"ml_probabilities" gorm:"type:jsonb"`
	PatternResults    map[string]interface{} `json:"pattern_results" gorm:"type:jsonb"`
	EnsembleDecision  map[string]interface{} `json:"ensemble_decision" gorm:"type:jsonb"`
	FeatureImportance map[string]interface{} `json:"feature_importance" gorm:"type:jsonb"`
	ExplanationData   map[string]interface{} `json:"explanation_data" gorm:"type:jsonb"`
	ProcessingTimeMs  *int                   `json:"processing_time_ms"`
	Timestamp         time.Time              `json:"timestamp" gorm:"default:CURRENT_TIMESTAMP"`

	HttpRequestLog *HttpRequestLog `json:"http_request_log,omitempty" gorm:"foreignKey:RequestLogID"`
}

type SecurityMetrics struct {
	ID                     uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	Date                   time.Time              `json:"date" gorm:"type:date;default:CURRENT_DATE;uniqueIndex"`
	TotalRequests          int                    `json:"total_requests" gorm:"default:0"`
	MaliciousRequests      int                    `json:"malicious_requests" gorm:"default:0"`
	AttackTypeCounts       map[string]interface{} `json:"attack_type_counts" gorm:"type:jsonb"`
	FalsePositiveRate      *float64               `json:"false_positive_rate"`
	FalseNegativeRate      *float64               `json:"false_negative_rate"`
	AverageConfidence      *float64               `json:"average_confidence"`
	AverageProcessingTime  *float64               `json:"average_processing_time"`
	ModelAccuracy          *float64               `json:"model_accuracy"`
	TopAttackSources       map[string]interface{} `json:"top_attack_sources" gorm:"type:jsonb"`
	TopAttackPaths         map[string]interface{} `json:"top_attack_paths" gorm:"type:jsonb"`
}

type SecurityFeedback struct {
	ID                   uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	RequestLogID         uuid.UUID `json:"request_log_id" gorm:"type:uuid;not null"`
	FeedbackType         string    `json:"feedback_type" gorm:"check:feedback_type IN ('false_positive', 'false_negative', 'correct', 'unknown')"`
	OriginalPrediction   *string   `json:"original_prediction"`
	CorrectedLabel       *string   `json:"corrected_label"`
	FeedbackSource       *string   `json:"feedback_source"`
	FeedbackReason       *string   `json:"feedback_reason"`
	ConfidenceInFeedback float64   `json:"confidence_in_feedback" gorm:"default:1.0"`
	CreatedAt            time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`
	CreatedBy            *uuid.UUID `json:"created_by" gorm:"type:uuid"`

	HttpRequestLog *HttpRequestLog `json:"http_request_log" gorm:"foreignKey:RequestLogID"`
	Creator        *User           `json:"creator,omitempty" gorm:"foreignKey:CreatedBy"`
}

type AttackMitigation struct {
	ID                  uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	AttackType          string    `json:"attack_type" gorm:"not null"`
	AttackPattern       *string   `json:"attack_pattern"`
	MitigationStrategy  string    `json:"mitigation_strategy" gorm:"not null"`
	SanitizationCode    *string   `json:"sanitization_code"`
	PreventionTips      *string   `json:"prevention_tips"`
	SeverityLevel       string    `json:"severity_level" gorm:"check:severity_level IN ('low', 'medium', 'high', 'critical')"`
	CreatedAt           time.Time `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`
	UpdatedAt           time.Time `json:"updated_at" gorm:"default:CURRENT_TIMESTAMP"`
}

type ModelPerformanceLog struct {
	ID                  uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	ModelVersion        *string                `json:"model_version"`
	EvaluationType      *string                `json:"evaluation_type"`
	EvaluationData      map[string]interface{} `json:"evaluation_data" gorm:"type:jsonb"`
	Accuracy            *float64               `json:"accuracy"`
	PrecisionMacro      *float64               `json:"precision_macro"`
	RecallMacro         *float64               `json:"recall_macro"`
	F1Macro             *float64               `json:"f1_macro"`
	RocAuc              *float64               `json:"roc_auc"`
	EvaluationTimestamp time.Time              `json:"evaluation_timestamp" gorm:"default:CURRENT_TIMESTAMP"`
}

type AblationStudyResults struct {
	ID                 uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	StudyName          *string   `json:"study_name"`
	ComponentRemoved   *string   `json:"component_removed"`
	BaselineAccuracy   *float64  `json:"baseline_accuracy"`
	ReducedAccuracy    *float64  `json:"reduced_accuracy"`
	PerformanceImpact  *float64  `json:"performance_impact"`
	ComponentImportance *float64  `json:"component_importance"`
	TestSamples        *int      `json:"test_samples"`
	StudyTimestamp     time.Time `json:"study_timestamp" gorm:"default:CURRENT_TIMESTAMP"`
}

type VisualizationData struct {
	ID                 uuid.UUID              `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	VisualizationType  string                 `json:"visualization_type" gorm:"check:visualization_type IN ('tsne', 'umap', 'pca')"`
	DataPoints         map[string]interface{} `json:"data_points" gorm:"type:jsonb"`
	Parameters         map[string]interface{} `json:"parameters" gorm:"type:jsonb"`
	DatasetInfo        map[string]interface{} `json:"dataset_info" gorm:"type:jsonb"`
	CreatedAt          time.Time              `json:"created_at" gorm:"default:CURRENT_TIMESTAMP"`
}

func (u *User) BeforeCreate(tx *gorm.DB) error {
	if u.ID == uuid.Nil {
		u.ID = uuid.New()
	}
	return nil
}

func (c *Category) BeforeCreate(tx *gorm.DB) error {
	if c.ID == uuid.Nil {
		c.ID = uuid.New()
	}
	return nil
}

func (p *Product) BeforeCreate(tx *gorm.DB) error {
	if p.ID == uuid.Nil {
		p.ID = uuid.New()
	}
	return nil
}

func (r *Recommendation) BeforeCreate(tx *gorm.DB) error {
	if r.ID == uuid.Nil {
		r.ID = uuid.New()
	}
	return nil
}

func (f *Favorite) BeforeCreate(tx *gorm.DB) error {
	if f.ID == uuid.Nil {
		f.ID = uuid.New()
	}
	return nil
}

func (c *Comment) BeforeCreate(tx *gorm.DB) error {
	if c.ID == uuid.Nil {
		c.ID = uuid.New()
	}
	return nil
}

func (cv *CommentVote) BeforeCreate(tx *gorm.DB) error {
	if cv.ID == uuid.Nil {
		cv.ID = uuid.New()
	}
	return nil
}

func (ps *ProductSuggestion) BeforeCreate(tx *gorm.DB) error {
	if ps.ID == uuid.Nil {
		ps.ID = uuid.New()
	}
	return nil
}

func (psd *ProductSimilarityData) BeforeCreate(tx *gorm.DB) error {
	if psd.ID == uuid.Nil {
		psd.ID = uuid.New()
	}
	return nil
}

func (hrl *HttpRequestLog) BeforeCreate(tx *gorm.DB) error {
	if hrl.ID == uuid.Nil {
		hrl.ID = uuid.New()
	}
	return nil
}

func (ue *UserEvent) BeforeCreate(tx *gorm.DB) error {
	if ue.ID == uuid.Nil {
		ue.ID = uuid.New()
	}
	return nil
}

// Add BeforeCreate hooks for new models
func (ml *MLAnalysisLog) BeforeCreate(tx *gorm.DB) error {
	if ml.ID == uuid.Nil {
		ml.ID = uuid.New()
	}
	return nil
}

func (sm *SecurityMetrics) BeforeCreate(tx *gorm.DB) error {
	if sm.ID == uuid.Nil {
		sm.ID = uuid.New()
	}
	return nil
}

func (sf *SecurityFeedback) BeforeCreate(tx *gorm.DB) error {
	if sf.ID == uuid.Nil {
		sf.ID = uuid.New()
	}
	return nil
}

func (am *AttackMitigation) BeforeCreate(tx *gorm.DB) error {
	if am.ID == uuid.Nil {
		am.ID = uuid.New()
	}
	return nil
}

func (mpl *ModelPerformanceLog) BeforeCreate(tx *gorm.DB) error {
	if mpl.ID == uuid.Nil {
		mpl.ID = uuid.New()
	}
	return nil
}

func (asr *AblationStudyResults) BeforeCreate(tx *gorm.DB) error {
	if asr.ID == uuid.Nil {
		asr.ID = uuid.New()
	}
	return nil
}

func (vd *VisualizationData) BeforeCreate(tx *gorm.DB) error {
	if vd.ID == uuid.Nil {
		vd.ID = uuid.New()
	}
	return nil
}

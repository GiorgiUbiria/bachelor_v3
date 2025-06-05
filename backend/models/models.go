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
	ID                  uuid.UUID  `json:"id" gorm:"type:uuid;primary_key;default:uuid_generate_v4()"`
	UserID              *uuid.UUID `json:"user_id" gorm:"type:uuid"`
	IPAddress           *string    `json:"ip_address"`
	UserAgent           *string    `json:"user_agent"`
	Path                *string    `json:"path"`
	Method              *string    `json:"method"`
	Timestamp           time.Time  `json:"timestamp" gorm:"default:CURRENT_TIMESTAMP"`
	DurationMs          *int       `json:"duration_ms"`
	StatusCode          *int       `json:"status_code"`
	Referrer            *string    `json:"referrer"`
	SuspectedAttackType *string    `json:"suspected_attack_type" gorm:"check:suspected_attack_type IN ('xss', 'csrf', 'sqli', 'unknown')"`
	AttackScore         *float64   `json:"attack_score"`

	User *User `json:"user,omitempty" gorm:"foreignKey:UserID"`
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

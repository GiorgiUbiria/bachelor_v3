package database

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"bachelor_backend/models"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var DB *gorm.DB

func Connect() {
	var err error

	host := getEnv("DB_HOST", "localhost")
	port := getEnv("DB_PORT", "5432")
	user := getEnv("DB_USER", "postgres")

	password := os.Getenv("DB_PASSWORD")
	if password == "" {
		env := os.Getenv("GO_ENV")
		if env == "production" {
			log.Fatal("DB_PASSWORD environment variable is required in production")
		}
		log.Println("WARNING: Using default database password. Set DB_PASSWORD environment variable for production!")
		password = "postgres123"
	}

	dbname := getEnv("DB_NAME", "bachelor_db")

	if host == "" || port == "" || user == "" || dbname == "" {
		log.Fatal("Required database configuration missing: DB_HOST, DB_PORT, DB_USER, DB_NAME")
	}

	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable TimeZone=UTC connect_timeout=10",
		host, port, user, password, dbname)

	logLevel := logger.Warn
	if getEnv("DB_LOG_LEVEL", "warn") == "info" {
		logLevel = logger.Info
	} else if getEnv("DB_LOG_LEVEL", "warn") == "error" {
		logLevel = logger.Error
	} else if getEnv("DB_LOG_LEVEL", "warn") == "silent" {
		logLevel = logger.Silent
	}

	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logLevel),
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
		PrepareStmt:                              true,
		DisableForeignKeyConstraintWhenMigrating: false,
	})

	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	sqlDB, err := DB.DB()
	if err != nil {
		log.Fatal("Failed to get underlying sql.DB:", err)
	}

	maxOpenConns := getEnvInt("DB_MAX_OPEN_CONNS", 25)
	maxIdleConns := getEnvInt("DB_MAX_IDLE_CONNS", 5)
	maxLifetime := getEnvInt("DB_CONN_MAX_LIFETIME", 300)
	maxIdleTime := getEnvInt("DB_CONN_MAX_IDLE_TIME", 60)

	sqlDB.SetMaxOpenConns(maxOpenConns)
	sqlDB.SetMaxIdleConns(maxIdleConns)
	sqlDB.SetConnMaxLifetime(time.Duration(maxLifetime) * time.Second)
	sqlDB.SetConnMaxIdleTime(time.Duration(maxIdleTime) * time.Second)

	log.Printf("Database connected successfully with pool config: MaxOpen=%d, MaxIdle=%d, MaxLifetime=%ds, MaxIdleTime=%ds",
		maxOpenConns, maxIdleConns, maxLifetime, maxIdleTime)

	if err := sqlDB.Ping(); err != nil {
		log.Fatal("Failed to ping database:", err)
	}

	if err := AutoMigrate(); err != nil {
		log.Fatal("Failed to run auto-migration:", err)
	}

	log.Println("Database migration completed successfully")
}

func AutoMigrate() error {
	models := []any{
		&models.User{},
		&models.Category{},
		&models.Product{},
		&models.Recommendation{},
		&models.Favorite{},
		&models.Comment{},
		&models.CommentVote{},
		&models.ProductSuggestion{},
		&models.ProductSimilarityData{},
		&models.HttpRequestLog{},
		&models.UserEvent{},
		&models.ProductFeatureVector{},
		// Enhanced security models
		&models.MLAnalysisLog{},
		&models.SecurityMetrics{},
		&models.SecurityFeedback{},
		&models.AttackMitigation{},
		&models.ModelPerformanceLog{},
		&models.AblationStudyResults{},
		&models.VisualizationData{},
	}

	var migrationErrors []error

	if err := DB.Exec("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"").Error; err != nil {
		log.Printf("Warning: Failed to create uuid-ossp extension: %v", err)
		migrationErrors = append(migrationErrors, fmt.Errorf("failed to create uuid-ossp extension: %w", err))
	} else {
		log.Println("Successfully ensured uuid-ossp extension exists")
	}

	// Handle views that might depend on columns we're about to migrate
	if err := handleDependentViews(); err != nil {
		log.Printf("Warning: Failed to handle dependent views: %v", err)
		migrationErrors = append(migrationErrors, fmt.Errorf("failed to handle dependent views: %w", err))
	}

	for _, model := range models {
		if err := DB.AutoMigrate(model); err != nil {
			log.Printf("Warning: Failed to migrate model %T: %v", model, err)
			migrationErrors = append(migrationErrors, fmt.Errorf("failed to migrate model %T: %w", model, err))
		} else {
			log.Printf("Successfully migrated model %T", model)
		}
	}

	// Recreate views after migration
	if err := createViews(); err != nil {
		log.Printf("Warning: Failed to create views: %v", err)
		migrationErrors = append(migrationErrors, fmt.Errorf("failed to create views: %w", err))
	}

	if err := createCustomIndexes(); err != nil {
		log.Printf("Warning: Failed to create custom indexes: %v", err)
		migrationErrors = append(migrationErrors, fmt.Errorf("failed to create custom indexes: %w", err))
	}

	if err := createUniqueConstraints(); err != nil {
		log.Printf("Warning: Failed to create unique constraints: %v", err)
		migrationErrors = append(migrationErrors, fmt.Errorf("failed to create unique constraints: %w", err))
	}

	if len(migrationErrors) > 0 {
		log.Printf("Migration completed with %d warnings/errors", len(migrationErrors))
		for _, err := range migrationErrors {
			log.Printf("Migration error: %v", err)
		}
	} else {
		log.Println("All migrations completed successfully")
	}

	return nil
}

func handleDependentViews() error {
	// Check if ml_service_stats view exists and drop it temporarily
	var viewExists bool
	checkViewSQL := `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.views 
			WHERE table_name = 'ml_service_stats' AND table_schema = 'public'
		)`
	
	if err := DB.Raw(checkViewSQL).Scan(&viewExists).Error; err != nil {
		return fmt.Errorf("failed to check view existence: %w", err)
	}

	if viewExists {
		log.Println("Temporarily dropping ml_service_stats view for migration")
		if err := DB.Exec("DROP VIEW IF EXISTS ml_service_stats CASCADE").Error; err != nil {
			return fmt.Errorf("failed to drop ml_service_stats view: %w", err)
		}
	}

	return nil
}

func createViews() error {
	// Recreate the ml_service_stats view
	createViewSQL := `
		CREATE OR REPLACE VIEW ml_service_stats AS
		SELECT 
			DATE(timestamp) as analysis_date,
			analysis_type,
			model_version,
			COUNT(*) as total_analyses,
			AVG(processing_time_ms) as avg_processing_time_ms,
			COUNT(CASE WHEN processing_time_ms > 1000 THEN 1 END) as slow_analyses,
			MAX(processing_time_ms) as max_processing_time_ms,
			MIN(processing_time_ms) as min_processing_time_ms
		FROM ml_analysis_logs 
		WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
		GROUP BY DATE(timestamp), analysis_type, model_version
		ORDER BY analysis_date DESC, analysis_type, model_version`

	if err := DB.Exec(createViewSQL).Error; err != nil {
		return fmt.Errorf("failed to create ml_service_stats view: %w", err)
	}

	log.Println("Successfully created ml_service_stats view")
	return nil
}

func createCustomIndexes() error {
	indexes := []string{
		`CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)`,
		`CREATE INDEX IF NOT EXISTS idx_users_region ON users(region)`,
		`CREATE INDEX IF NOT EXISTS idx_users_birth_year ON users(birth_year)`,
		`CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)`,

		`CREATE INDEX IF NOT EXISTS idx_products_name ON products(name)`,
		`CREATE INDEX IF NOT EXISTS idx_products_category_id ON products(category_id)`,
		`CREATE INDEX IF NOT EXISTS idx_products_created_by ON products(created_by)`,
		`CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)`,

		`CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_recommendations_product_id ON recommendations(product_id)`,
		`CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_recommendations_model_version ON recommendations(model_version)`,

		`CREATE INDEX IF NOT EXISTS idx_comments_user_id ON comments(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_comments_product_id ON comments(product_id)`,
		`CREATE INDEX IF NOT EXISTS idx_comments_created_at ON comments(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_comments_sentiment_label ON comments(sentiment_label)`,

		`CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_user_events_product_id ON user_events(product_id)`,
		`CREATE INDEX IF NOT EXISTS idx_user_events_event_type ON user_events(event_type)`,
		`CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON user_events(timestamp)`,

		`CREATE INDEX IF NOT EXISTS idx_http_request_logs_user_id ON http_request_logs(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_http_request_logs_ip_address ON http_request_logs(ip_address)`,
		`CREATE INDEX IF NOT EXISTS idx_http_request_logs_timestamp ON http_request_logs(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_http_request_logs_suspected_attack_type ON http_request_logs(suspected_attack_type)`,

		`CREATE INDEX IF NOT EXISTS idx_product_similarity_product_id ON product_similarity_data(product_id)`,
		`CREATE INDEX IF NOT EXISTS idx_product_similarity_similar_product_id ON product_similarity_data(similar_product_id)`,
		`CREATE INDEX IF NOT EXISTS idx_product_similarity_score ON product_similarity_data(similarity_score)`,
	}

	for _, indexSQL := range indexes {
		if err := DB.Exec(indexSQL).Error; err != nil {
			log.Printf("Warning: Failed to create index: %s - %v", indexSQL, err)
		}
	}

	log.Println("Custom indexes creation completed")
	return nil
}

func createUniqueConstraints() error {
	// PostgreSQL doesn't support IF NOT EXISTS for constraints, so we check manually
	constraints := []struct {
		table      string
		constraint string
		sql        string
	}{
		{
			table:      "favorites",
			constraint: "unique_user_product_favorite",
			sql:        `ALTER TABLE favorites ADD CONSTRAINT unique_user_product_favorite UNIQUE (user_id, product_id)`,
		},
		{
			table:      "comment_votes",
			constraint: "unique_user_comment_vote",
			sql:        `ALTER TABLE comment_votes ADD CONSTRAINT unique_user_comment_vote UNIQUE (user_id, comment_id)`,
		},
		{
			table:      "product_feature_vectors",
			constraint: "unique_product_feature_vector",
			sql:        `ALTER TABLE product_feature_vectors ADD CONSTRAINT unique_product_feature_vector UNIQUE (product_id)`,
		},
	}

	for _, c := range constraints {
		// Check if constraint already exists
		var exists bool
		checkSQL := `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.table_constraints 
			WHERE table_name = $1 AND constraint_name = $2 AND table_schema = 'public'
		)`

		if err := DB.Raw(checkSQL, c.table, c.constraint).Scan(&exists).Error; err != nil {
			log.Printf("Warning: Failed to check constraint existence for %s: %v", c.constraint, err)
			continue
		}

		if !exists {
			if err := DB.Exec(c.sql).Error; err != nil {
				log.Printf("Warning: Failed to create constraint %s: %v", c.constraint, err)
			} else {
				log.Printf("Created constraint: %s", c.constraint)
			}
		} else {
			log.Printf("Constraint %s already exists, skipping", c.constraint)
		}
	}

	log.Println("Unique constraints creation completed")
	return nil
}

func GetDB() *gorm.DB {
	return DB
}

func getEnvInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
		log.Printf("Warning: Invalid integer value for %s: %s, using fallback: %d", key, value, fallback)
	}
	return fallback
}

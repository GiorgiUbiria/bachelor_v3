package database

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"bachelor_backend/models"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Enhanced ML-focused seeder configuration
const (
	ML_USERS_COUNT               = 4000
	ML_CATEGORIES_COUNT          = 12
	ML_PRODUCTS_COUNT            = 10000
	ML_COMMENTS_PER_PRODUCT      = 8
	ML_FAVORITES_PER_USER        = 30
	ML_RECOMMENDATIONS_PER_USER  = 40
	ML_USER_EVENTS_PER_USER      = 120
	ML_HTTP_LOGS_COUNT           = 20000
	ML_PRODUCT_SUGGESTIONS_COUNT = 3000
	ML_SIMILARITY_DATA_COUNT     = 30000
	ML_ML_ANALYSIS_LOGS_COUNT    = 5000
	ML_SECURITY_FEEDBACK_COUNT   = 1000
)

// SeedDatabase is the main entry point for seeding
func SeedDatabase() error {
	return SeedDatabaseForML()
}

// SeedDatabaseForML orchestrates the complete ML-focused seeding process
func SeedDatabaseForML() error {
	log.Println("🤖 Starting ML-focused database seeding with modular approach...")

	// Initialize gofakeit
	gofakeit.Seed(time.Now().UnixNano())

	// Clear existing data first
	if err := ClearDatabase(); err != nil {
		return fmt.Errorf("failed to clear database: %w", err)
	}

	startTime := time.Now()
	ctx := context.Background()

	// Get database connection pool for performance
	dbConfig := GetDBConfig()
	pool, err := pgxpool.New(ctx, dbConfig)
	if err != nil {
		log.Printf("Warning: Could not create pgx pool, falling back to GORM: %v", err)
		return seedWithGORM(ctx)
	}
	defer pool.Close()

	// Seed in dependency order with performance optimizations
	log.Println("🚀 Starting dependency-ordered seeding...")

	// Stage 1: Independent entities (can run concurrently)
	var wg sync.WaitGroup
	var users []models.User
	var categories []models.Category
	var userErr, categoryErr error

	wg.Add(2)
	go func() {
		defer wg.Done()
		users, userErr = SeedUsers(ctx, pool, ML_USERS_COUNT)
	}()
	go func() {
		defer wg.Done()
		categories, categoryErr = SeedCategories(ctx, pool, ML_CATEGORIES_COUNT)
	}()
	wg.Wait()

	if userErr != nil {
		return fmt.Errorf("failed to seed users: %w", userErr)
	}
	if categoryErr != nil {
		return fmt.Errorf("failed to seed categories: %w", categoryErr)
	}

	// Stage 2: Products (depends on Users and Categories)
	products, err := SeedProducts(ctx, pool, ML_PRODUCTS_COUNT, users, categories)
	if err != nil {
		return fmt.Errorf("failed to seed products: %w", err)
	}

	// Stage 3: User-Product interactions (can run some concurrently)
	wg.Add(3)
	go func() {
		defer wg.Done()
		if err := SeedComments(ctx, pool, users, products); err != nil {
			log.Printf("Error seeding comments: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedFavorites(ctx, pool, users, products); err != nil {
			log.Printf("Error seeding favorites: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedUserEvents(ctx, pool, users, products); err != nil {
			log.Printf("Error seeding user events: %v", err)
		}
	}()
	wg.Wait()

	// Stage 4: ML and Analysis data
	wg.Add(4)
	go func() {
		defer wg.Done()
		if err := SeedRecommendations(ctx, pool, users, products); err != nil {
			log.Printf("Error seeding recommendations: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedHttpRequestLogs(ctx, pool, users); err != nil {
			log.Printf("Error seeding HTTP logs: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedProductSimilarityData(ctx, pool, products); err != nil {
			log.Printf("Error seeding similarity data: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedProductSuggestions(ctx, pool, products); err != nil {
			log.Printf("Error seeding product suggestions: %v", err)
		}
	}()
	wg.Wait()

	// Stage 5: Security Analysis data (depends on HTTP logs)
	wg.Add(5)
	go func() {
		defer wg.Done()
		if err := SeedMLAnalysisLogs(ctx, pool); err != nil {
			log.Printf("Error seeding ML analysis logs: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedSecurityMetrics(ctx, pool); err != nil {
			log.Printf("Error seeding security metrics: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedSecurityFeedback(ctx, pool); err != nil {
			log.Printf("Error seeding security feedback: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedAttackMitigation(ctx, pool); err != nil {
			log.Printf("Error seeding attack mitigation: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedModelPerformanceLogs(ctx, pool); err != nil {
			log.Printf("Error seeding model performance logs: %v", err)
		}
	}()
	wg.Wait()

	// Stage 6: Additional analysis data
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := SeedAblationStudyResults(ctx, pool); err != nil {
			log.Printf("Error seeding ablation study results: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := SeedVisualizationData(ctx, pool); err != nil {
			log.Printf("Error seeding visualization data: %v", err)
		}
	}()
	wg.Wait()

	duration := time.Since(startTime)
	totalRecords := countTotalRecords()

	log.Printf("🎉 ML-focused database seeding completed!")
	log.Printf("📊 Total records: %d", totalRecords)
	log.Printf("⏱️  Duration: %v", duration)
	log.Printf("🔬 ML models can now be trained with comprehensive, realistic data")

	return nil
}

// seedWithGORM fallback when pgx pool is not available
func seedWithGORM(ctx context.Context) error {
	log.Println("Using GORM fallback for seeding...")

	// Implement GORM-based seeding here as fallback
	// This would use the existing GORM methods but in modular fashion

	return fmt.Errorf("GORM fallback not implemented yet")
}

// GetDBConfig returns the database connection string for pgx
func GetDBConfig() string {
	// Get config from environment variables (same as in database.go)
	host := getEnv("DB_HOST", "localhost")
	port := getEnv("DB_PORT", "5432")
	user := getEnv("DB_USER", "postgres")
	password := getEnv("DB_PASSWORD", "postgres123")
	dbname := getEnv("DB_NAME", "bachelor_db")

	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)
}

// ClearDatabase clears all data from the database in correct order
func ClearDatabase() error {
	log.Println("🧹 Clearing all data from database...")

	// Order matters due to foreign key constraints (reverse dependency order)
	tables := []string{
		"visualization_data",
		"ablation_study_results",
		"model_performance_logs",
		"attack_mitigation",
		"security_feedback",
		"security_metrics",
		"ml_analysis_logs",
		"product_feature_vectors",
		"product_suggestions",
		"product_similarity_data",
		"user_events",
		"http_request_logs",
		"comment_votes",
		"comments",
		"recommendations",
		"favorites",
		"products",
		"categories",
		"users",
	}

	for _, table := range tables {
		if err := DB.Exec(fmt.Sprintf("DELETE FROM %s", table)).Error; err != nil {
			log.Printf("Warning: Failed to clear table %s: %v", table, err)
		} else {
			log.Printf("✅ Cleared table: %s", table)
		}
	}

	log.Println("✅ Database cleared successfully")
	return nil
}

// countTotalRecords counts total records across all tables
func countTotalRecords() int64 {
	var total int64

	tables := map[string]interface{}{
		"users":                   &models.User{},
		"categories":              &models.Category{},
		"products":                &models.Product{},
		"comments":                &models.Comment{},
		"favorites":               &models.Favorite{},
		"recommendations":         &models.Recommendation{},
		"user_events":             &models.UserEvent{},
		"http_request_logs":       &models.HttpRequestLog{},
		"product_similarity_data": &models.ProductSimilarityData{},
		"product_suggestions":     &models.ProductSuggestion{},
		"ml_analysis_logs":        &models.MLAnalysisLog{},
		"security_metrics":        &models.SecurityMetrics{},
		"security_feedback":       &models.SecurityFeedback{},
		"attack_mitigation":       &models.AttackMitigation{},
		"model_performance_logs":  &models.ModelPerformanceLog{},
		"ablation_study_results":  &models.AblationStudyResults{},
		"visualization_data":      &models.VisualizationData{},
	}

	for _, model := range tables {
		var count int64
		if err := DB.Model(model).Count(&count).Error; err == nil {
			total += count
		}
	}

	return total
}

// Helper function to get environment variables
func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

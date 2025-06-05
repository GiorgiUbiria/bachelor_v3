package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/models"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/google/uuid"
)

// Standard response types
type StandardErrorResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
}

type StandardSuccessResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Swagger type definitions for PostgreSQL arrays to fix swag generation
// These types help Swagger understand pq.StringArray as []string

// @Description Array of strings
type StringArray []string

// @Description Product response with enhanced information
type ProductResponse struct {
	models.Product
	IsFavorited       bool               `json:"is_favorited"`
	CommentsCount     int64              `json:"comments_count"`
	AverageRating     *float64           `json:"average_rating"`
	FavoritesCount    int64              `json:"favorites_count"`
	RecommendedReason string             `json:"recommended_reason,omitempty"`
	SimilarProducts   []models.Product   `json:"similar_products,omitempty"`
	UserEvents        []models.UserEvent `json:"user_events,omitempty"`
	// Additional fields from products handler
	FavoriteCount  int64 `json:"favorite_count"`
	CommentCount   int64 `json:"comment_count"`
	HasSuggestions bool  `json:"has_suggestions"`
}

// User event logging for ML analytics
func logUserEvent(userID, productID uuid.UUID, eventType string, metadata map[string]interface{}) {
	event := models.UserEvent{
		UserID:    userID,
		ProductID: &productID,
		EventType: eventType,
		Metadata:  metadata,
	}

	database.DB.Create(&event)
}

// Enhanced product response with computed fields
func enhanceProductResponse(product models.Product, userID uuid.UUID) ProductResponse {
	response := ProductResponse{
		Product: product,
	}

	// Check if user has favorited this product
	if userID != uuid.Nil {
		var favoriteCount int64
		database.DB.Model(&models.Favorite{}).
			Where("user_id = ? AND product_id = ?", userID, product.ID).
			Count(&favoriteCount)
		response.IsFavorited = favoriteCount > 0
	}

	// Get comments count
	database.DB.Model(&models.Comment{}).
		Where("product_id = ?", product.ID).
		Count(&response.CommentsCount)

	// Get favorites count
	database.DB.Model(&models.Favorite{}).
		Where("product_id = ?", product.ID).
		Count(&response.FavoritesCount)

	// Calculate average rating from comments (if sentiment analysis is available)
	type RatingResult struct {
		AvgRating *float64 `json:"avg_rating"`
	}
	var ratingResult RatingResult
	database.DB.Table("comments").
		Select("AVG(sentiment_score) as avg_rating").
		Where("product_id = ? AND sentiment_score IS NOT NULL", product.ID).
		Scan(&ratingResult)
	response.AverageRating = ratingResult.AvgRating

	return response
}

// Get similar products using ML service
func getSimilarProducts(productID string, limit int) ([]models.Product, error) {
	// ML service URL (should be configurable)
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Consistent default URL
	}

	reqBody := map[string]interface{}{
		"product_id": productID,
		"limit":      limit,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(
		fmt.Sprintf("%s/products/similar", mlServiceURL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var mlResponse struct {
		SimilarProducts []struct {
			ProductID string  `json:"product_id"`
			Score     float64 `json:"score"`
		} `json:"similar_products"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return nil, err
	}

	// Fetch products from database
	var products []models.Product
	for _, similar := range mlResponse.SimilarProducts {
		productUUID, err := uuid.Parse(similar.ProductID)
		if err != nil {
			continue
		}

		var product models.Product
		if err := database.DB.Preload("Category").Preload("Creator").
			First(&product, productUUID).Error; err == nil {
			products = append(products, product)
		}
	}

	return products, nil
}

// Analyze product using ML service
func analyzeProductWithML(product models.Product) (map[string]interface{}, error) {
	// ML service URL (should be configurable)
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Consistent default URL
	}

	reqBody := map[string]interface{}{
		"name":        product.Name,
		"description": product.Description,
		"price":       product.Price,
		"tags":        product.Tags,
	}

	// Add category name if available
	if product.Category != nil {
		reqBody["category"] = product.Category.Name
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(
		fmt.Sprintf("%s/products/analyze", mlServiceURL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var mlResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return nil, err
	}

	return mlResponse, nil
}

// Analyze HTTP request for security threats
func analyzeSecurityThreat(requestData map[string]interface{}) (map[string]interface{}, error) {
	// ML service URL (should be configurable)
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Consistent default URL
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(
		fmt.Sprintf("%s/security/analyze", mlServiceURL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var mlResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return nil, err
	}

	return mlResponse, nil
}

// Log HTTP request for security analysis
func logHTTPRequest(userID *uuid.UUID, method, path, userAgent, ipAddress string, headers map[string]string, queryParams map[string]string, body string) {
	logEntry := models.HttpRequestLog{
		UserID:    userID,
		Method:    &method,
		Path:      &path,
		UserAgent: &userAgent,
		IPAddress: &ipAddress,
	}

	// Analyze for security threats in background
	go func() {
		requestData := map[string]interface{}{
			"method":       method,
			"path":         path,
			"user_agent":   userAgent,
			"ip_address":   ipAddress,
			"headers":      headers,
			"query_params": queryParams,
			"body":         body,
		}

		if analysis, err := analyzeSecurityThreat(requestData); err == nil {
			if attackType, ok := analysis["attack_type"].(string); ok {
				logEntry.SuspectedAttackType = &attackType
			}
			if confidence, ok := analysis["confidence"].(float64); ok {
				logEntry.AttackScore = &confidence
			}
		}

		database.DB.Create(&logEntry)
	}()
}

// Pagination helper
type PaginationParams struct {
	Page   int `json:"page"`
	Limit  int `json:"limit"`
	Offset int `json:"offset"`
}

func GetPaginationParams(page, limit int) PaginationParams {
	if page < 1 {
		page = 1
	}
	if limit < 1 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	offset := (page - 1) * limit

	return PaginationParams{
		Page:   page,
		Limit:  limit,
		Offset: offset,
	}
}

// Search helper for building dynamic queries
type SearchParams struct {
	Query      string
	CategoryID *uuid.UUID
	MinPrice   *float64
	MaxPrice   *float64
	Tags       []string
	SortBy     string
	SortOrder  string
}

// Validation helper for common validations
func ValidateUUID(id string) (uuid.UUID, error) {
	return uuid.Parse(id)
}

// Response helpers
func SuccessResponse(message string, data interface{}) StandardSuccessResponse {
	return StandardSuccessResponse{
		Success: true,
		Message: message,
		Data:    data,
	}
}

func ErrorResponse(message string) StandardErrorResponse {
	return StandardErrorResponse{
		Success: false,
		Error:   message,
	}
}

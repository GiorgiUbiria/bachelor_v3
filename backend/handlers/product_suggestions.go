package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"

	"time"

	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/lib/pq"
)

type CreateProductSuggestionRequest struct {
	ProductID         uuid.UUID      `json:"product_id" validate:"required" example:"123e4567-e89b-12d3-a456-426614174000"`
	SuggestedPriceMin *float64       `json:"suggested_price_min" validate:"omitempty,min=0" example:"99.99"`
	SuggestedPriceMax *float64       `json:"suggested_price_max" validate:"omitempty,min=0" example:"199.99"`
	SuggestedTags     pq.StringArray `json:"suggested_tags" validate:"omitempty" example:"[\"smartphone\", \"premium\", \"latest\"]"`
	ModelVersion      *string        `json:"model_version" validate:"omitempty,max=100" example:"v1.0"`
	Reason            *string        `json:"reason" validate:"omitempty,max=500" example:"Price optimization based on market analysis"`
}

type UpdateProductSuggestionRequest struct {
	SuggestedPriceMin *float64       `json:"suggested_price_min" validate:"omitempty,min=0" example:"99.99"`
	SuggestedPriceMax *float64       `json:"suggested_price_max" validate:"omitempty,min=0" example:"199.99"`
	SuggestedTags     pq.StringArray `json:"suggested_tags" validate:"omitempty" example:"[\"smartphone\", \"premium\", \"latest\"]"`
	ModelVersion      *string        `json:"model_version" validate:"omitempty,max=100" example:"v1.0"`
	Reason            *string        `json:"reason" validate:"omitempty,max=500" example:"Updated price optimization"`
}

type ProductSuggestionResponse struct {
	models.ProductSuggestion
	Product models.Product `json:"product"`
}

type ProductSuggestionsListResponse struct {
	Suggestions []ProductSuggestionResponse `json:"suggestions"`
	Total       int64                       `json:"total"`
	Page        int                         `json:"page"`
	Limit       int                         `json:"limit"`
	Filters     SuggestionFilters           `json:"filters"`
}

type SuggestionFilters struct {
	ProductID    *uuid.UUID `json:"product_id"`
	ModelVersion string     `json:"model_version"`
	HasPriceMin  bool       `json:"has_price_min"`
	HasPriceMax  bool       `json:"has_price_max"`
	HasTags      bool       `json:"has_tags"`
}

// GetProductSuggestions retrieves paginated product suggestions with optional filtering
// @Summary Get product suggestions
// @Description Retrieve a paginated list of ML-generated product suggestions with optional filtering
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Param product_id query string false "Filter by product ID"
// @Param model_version query string false "Filter by model version"
// @Param has_price_min query bool false "Filter suggestions with price minimum"
// @Param has_price_max query bool false "Filter suggestions with price maximum"
// @Param has_tags query bool false "Filter suggestions with tags"
// @Success 200 {object} ProductSuggestionsListResponse "Suggestions retrieved successfully"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions [get]
func GetProductSuggestions(c *fiber.Ctx) error {
	// Get pagination parameters
	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)
	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 20
	}
	offset := (page - 1) * limit

	// Get filter parameters
	productIDStr := c.Query("product_id")
	modelVersion := c.Query("model_version")
	hasPriceMin := c.QueryBool("has_price_min")
	hasPriceMax := c.QueryBool("has_price_max")
	hasTags := c.QueryBool("has_tags")

	// Build query
	query := database.DB.Model(&models.ProductSuggestion{}).
		Preload("Product").
		Preload("Product.Category").
		Preload("Product.Creator")

	// Apply filters
	if productIDStr != "" {
		productID, err := uuid.Parse(productIDStr)
		if err == nil {
			query = query.Where("product_id = ?", productID)
		}
	}

	if modelVersion != "" {
		query = query.Where("model_version = ?", modelVersion)
	}

	if hasPriceMin {
		query = query.Where("suggested_price_min IS NOT NULL")
	}

	if hasPriceMax {
		query = query.Where("suggested_price_max IS NOT NULL")
	}

	if hasTags {
		query = query.Where("suggested_tags IS NOT NULL AND array_length(suggested_tags, 1) > 0")
	}

	// Get total count
	var total int64
	if err := query.Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count suggestions",
		})
	}

	// Get suggestions with pagination
	var suggestions []models.ProductSuggestion
	if err := query.
		Order("generated_at DESC").
		Offset(offset).
		Limit(limit).
		Find(&suggestions).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve suggestions",
		})
	}

	// Convert to response format - ensure empty array instead of null
	var suggestionResponses []ProductSuggestionResponse = make([]ProductSuggestionResponse, 0)
	for _, suggestion := range suggestions {
		suggestionResponses = append(suggestionResponses, ProductSuggestionResponse{
			ProductSuggestion: suggestion,
			Product:           suggestion.Product,
		})
	}

	// Parse filters for response
	var productIDFilter *uuid.UUID
	if productIDStr != "" {
		if id, err := uuid.Parse(productIDStr); err == nil {
			productIDFilter = &id
		}
	}

	response := ProductSuggestionsListResponse{
		Suggestions: suggestionResponses,
		Total:       total,
		Page:        page,
		Limit:       limit,
		Filters: SuggestionFilters{
			ProductID:    productIDFilter,
			ModelVersion: modelVersion,
			HasPriceMin:  hasPriceMin,
			HasPriceMax:  hasPriceMax,
			HasTags:      hasTags,
		},
	}

	return c.JSON(response)
}

// GetProductSuggestion retrieves a specific ML-generated product suggestion
// @Summary Get product suggestion by ID
// @Description Retrieve a specific ML-generated product suggestion
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Param id path string true "Suggestion ID"
// @Success 200 {object} ProductSuggestionResponse "Suggestion retrieved successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid suggestion ID"
// @Failure 404 {object} StandardErrorResponse "Suggestion not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions/{id} [get]
func GetProductSuggestion(c *fiber.Ctx) error {
	suggestionID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid suggestion ID",
		})
	}

	var suggestion models.ProductSuggestion
	if err := database.DB.
		Preload("Product").
		Preload("Product.Category").
		Preload("Product.Creator").
		First(&suggestion, suggestionID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Suggestion not found",
		})
	}

	response := ProductSuggestionResponse{
		ProductSuggestion: suggestion,
		Product:           suggestion.Product,
	}

	return c.JSON(response)
}

// CreateProductSuggestion creates a new ML-generated suggestion (admin only)
// @Summary Create product suggestion
// @Description Create a new ML-generated suggestion for a product
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body CreateProductSuggestionRequest true "Suggestion creation data"
// @Success 201 {object} models.ProductSuggestion "Suggestion created successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions [post]
func CreateProductSuggestion(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req CreateProductSuggestionRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid request body",
		})
	}

	if err := middleware.ValidateStruct(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   err.Error(),
		})
	}

	// Check if product exists
	var product models.Product
	if err := database.DB.First(&product, req.ProductID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Create new suggestion
	suggestion := models.ProductSuggestion{
		ID:                uuid.New(),
		ProductID:         req.ProductID,
		SuggestedPriceMin: req.SuggestedPriceMin,
		SuggestedPriceMax: req.SuggestedPriceMax,
		SuggestedTags:     req.SuggestedTags,
		ModelVersion:      req.ModelVersion,
		Reason:            req.Reason,
		GeneratedAt:       time.Now(),
	}

	if err := database.DB.Create(&suggestion).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to create suggestion",
		})
	}

	// Log user event
	go logUserEvent(userID, req.ProductID, "create_suggestion", map[string]interface{}{
		"suggestion_id": suggestion.ID,
	})

	return c.Status(fiber.StatusCreated).JSON(suggestion)
}

// UpdateProductSuggestion updates an existing suggestion
// @Summary Update product suggestion
// @Description Update an existing ML-generated product suggestion (admin only)
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Suggestion ID"
// @Param request body UpdateProductSuggestionRequest true "Suggestion update data"
// @Success 200 {object} models.ProductSuggestion "Suggestion updated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "Admin access required"
// @Failure 404 {object} StandardErrorResponse "Suggestion not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions/{id} [put]
func UpdateProductSuggestion(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Check if user is admin
	var user models.User
	database.DB.First(&user, userID)
	if !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Admin access required",
		})
	}

	suggestionID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid suggestion ID",
		})
	}

	var req UpdateProductSuggestionRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
		})
	}

	if err := middleware.ValidateStruct(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   err.Error(),
		})
	}

	var suggestion models.ProductSuggestion
	if err := database.DB.First(&suggestion, suggestionID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Suggestion not found",
		})
	}

	// Update fields
	if req.SuggestedPriceMin != nil {
		suggestion.SuggestedPriceMin = req.SuggestedPriceMin
	}
	if req.SuggestedPriceMax != nil {
		suggestion.SuggestedPriceMax = req.SuggestedPriceMax
	}
	if req.SuggestedTags != nil {
		suggestion.SuggestedTags = req.SuggestedTags
	}
	if req.ModelVersion != nil {
		suggestion.ModelVersion = req.ModelVersion
	}
	if req.Reason != nil {
		suggestion.Reason = req.Reason
	}

	// Validate price range
	if suggestion.SuggestedPriceMin != nil && suggestion.SuggestedPriceMax != nil {
		if *suggestion.SuggestedPriceMin > *suggestion.SuggestedPriceMax {
			return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Suggested price minimum cannot be greater than maximum",
			})
		}
	}

	if err := database.DB.Save(&suggestion).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to update suggestion",
		})
	}

	// Load relations for response
	database.DB.Preload("Product").First(&suggestion, suggestion.ID)

	return c.JSON(suggestion)
}

// DeleteProductSuggestion deletes a suggestion
// @Summary Delete product suggestion
// @Description Delete an ML-generated product suggestion (admin only)
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Suggestion ID"
// @Success 204 "Suggestion deleted successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid suggestion ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "Admin access required"
// @Failure 404 {object} StandardErrorResponse "Suggestion not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions/{id} [delete]
func DeleteProductSuggestion(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Check if user is admin
	var user models.User
	database.DB.First(&user, userID)
	if !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Admin access required",
		})
	}

	suggestionID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid suggestion ID",
		})
	}

	var suggestion models.ProductSuggestion
	if err := database.DB.First(&suggestion, suggestionID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Suggestion not found",
		})
	}

	if err := database.DB.Delete(&suggestion).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to delete suggestion",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

// GetSuggestionStats gets suggestion statistics
// @Summary Get suggestion statistics
// @Description Get public statistics about ML-generated product suggestions
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{} "Suggestion statistics"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions/stats [get]
func GetSuggestionStats(c *fiber.Ctx) error {
	// Get total suggestions count
	var totalSuggestions int64
	database.DB.Model(&models.ProductSuggestion{}).Count(&totalSuggestions)

	// Get suggestions by model version
	type ModelVersionCount struct {
		ModelVersion string `json:"model_version"`
		Count        int64  `json:"count"`
	}

	var modelVersionStats []ModelVersionCount
	if totalSuggestions > 0 {
		database.DB.Table("product_suggestions").
			Select("COALESCE(model_version, 'unknown') as model_version, COUNT(*) as count").
			Group("model_version").
			Order("count DESC").
			Scan(&modelVersionStats)
	}

	// Ensure we always return an empty array instead of null
	if modelVersionStats == nil {
		modelVersionStats = []ModelVersionCount{}
	}

	// Get suggestions with price recommendations
	var priceMinCount int64
	database.DB.Model(&models.ProductSuggestion{}).
		Where("suggested_price_min IS NOT NULL").
		Count(&priceMinCount)

	var priceMaxCount int64
	database.DB.Model(&models.ProductSuggestion{}).
		Where("suggested_price_max IS NOT NULL").
		Count(&priceMaxCount)

	// Get suggestions with tag recommendations
	var tagSuggestionCount int64
	database.DB.Model(&models.ProductSuggestion{}).
		Where("suggested_tags IS NOT NULL AND array_length(suggested_tags, 1) > 0").
		Count(&tagSuggestionCount)

	// Get recent suggestions count (last 7 days)
	var recentSuggestions int64
	database.DB.Model(&models.ProductSuggestion{}).
		Where("generated_at > NOW() - INTERVAL '7 days'").
		Count(&recentSuggestions)

	// Get top products with suggestions
	type ProductSuggestionCount struct {
		ProductID   uuid.UUID `json:"product_id"`
		ProductName string    `json:"product_name"`
		Count       int64     `json:"count"`
	}

	var topProducts []ProductSuggestionCount
	if totalSuggestions > 0 {
		database.DB.Table("product_suggestions").
			Select("products.id as product_id, products.name as product_name, COUNT(*) as count").
			Joins("JOIN products ON product_suggestions.product_id = products.id").
			Group("products.id, products.name").
			Order("count DESC").
			Limit(10).
			Scan(&topProducts)
	}

	// Ensure we always return an empty array instead of null
	if topProducts == nil {
		topProducts = []ProductSuggestionCount{}
	}

	return c.JSON(map[string]interface{}{
		"total_suggestions":       totalSuggestions,
		"recent_suggestions":      recentSuggestions,
		"price_min_suggestions":   priceMinCount,
		"price_max_suggestions":   priceMaxCount,
		"tag_suggestions":         tagSuggestionCount,
		"model_version_breakdown": modelVersionStats,
		"top_products":            topProducts,
	})
}

// GenerateSuggestionsForProduct generates ML suggestions for a specific product using the ML service
// @Summary Generate suggestions for product
// @Description Generate ML suggestions for a specific product using the ML service
// @Tags Product Suggestions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Product ID"
// @Success 200 {object} map[string]interface{} "Suggestions generated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /product-suggestions/generate/{id} [post]
func GenerateSuggestionsForProduct(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	productID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid product ID",
		})
	}

	// Check if product exists
	var product models.Product
	if err := database.DB.Preload("Category").First(&product, productID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Call ML service to generate suggestions
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000"
	}

	requestData := map[string]interface{}{
		"product_id":  product.ID.String(),
		"name":        product.Name,
		"description": product.Description,
		"price":       product.Price,
		"tags":        product.Tags,
		"category":    "",
	}

	if product.Category != nil {
		requestData["category"] = product.Category.Name
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to prepare request data",
		})
	}

	resp, err := http.Post(
		fmt.Sprintf("%s/products/analyze", mlServiceURL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to connect to ML service: " + err.Error(),
		})
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "ML service returned error",
		})
	}

	var mlResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to parse ML response",
		})
	}

	// Create suggestion record in database
	suggestion := models.ProductSuggestion{
		ID:          uuid.New(),
		ProductID:   productID,
		GeneratedAt: time.Now(),
	}

	// Extract ML suggestions and populate the database record
	if priceRange, ok := mlResponse["suggested_price_range"].(map[string]interface{}); ok {
		if minPrice, ok := priceRange["min_price"].(float64); ok {
			suggestion.SuggestedPriceMin = &minPrice
		}
		if maxPrice, ok := priceRange["max_price"].(float64); ok {
			suggestion.SuggestedPriceMax = &maxPrice
		}
	}

	if tags, ok := mlResponse["recommended_tags"].([]interface{}); ok {
		var suggestedTags []string
		for _, tag := range tags {
			if tagMap, ok := tag.(map[string]interface{}); ok {
				if tagName, ok := tagMap["tag"].(string); ok {
					suggestedTags = append(suggestedTags, tagName)
				}
			}
		}
		if len(suggestedTags) > 0 {
			suggestion.SuggestedTags = suggestedTags
		}
	}

	if modelVersion, ok := mlResponse["model_version"].(string); ok {
		suggestion.ModelVersion = &modelVersion
	}

	if summary, ok := mlResponse["analysis_summary"].(string); ok {
		suggestion.Reason = &summary
	}

	// Save suggestion to database
	if err := database.DB.Create(&suggestion).Error; err != nil {
		log.Printf("Failed to save suggestion to database: %v", err)
		// Don't fail the request if we can't save to DB
	}

	// Log user event
	go logUserEvent(userID, productID, "generate_suggestions", map[string]interface{}{
		"suggestion_id": suggestion.ID,
	})

	// Return the ML response along with the database suggestion ID
	response := map[string]interface{}{
		"success":       true,
		"message":       "Suggestions generated successfully",
		"suggestion_id": suggestion.ID,
		"ml_analysis":   mlResponse,
	}

	return c.JSON(response)
}

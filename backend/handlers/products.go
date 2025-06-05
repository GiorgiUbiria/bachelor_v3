package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/lib/pq"
)

type CreateProductRequest struct {
	Name        string         `json:"name" validate:"required,min=2,max=255" example:"iPhone 15 Pro"`
	Description *string        `json:"description" validate:"omitempty,max=2000" example:"Latest iPhone with advanced features"`
	Price       *float64       `json:"price" validate:"omitempty,min=0" example:"999.99"`
	Tags        pq.StringArray `json:"tags" validate:"omitempty" example:"[\"smartphone\", \"apple\", \"premium\"]"`
	CategoryID  *uuid.UUID     `json:"category_id" validate:"omitempty" example:"123e4567-e89b-12d3-a456-426614174000"`
}

type UpdateProductRequest struct {
	Name         *string        `json:"name" validate:"omitempty,min=2,max=255" example:"iPhone 15 Pro"`
	Description  *string        `json:"description" validate:"omitempty,max=2000" example:"Latest iPhone with advanced features"`
	Price        *float64       `json:"price" validate:"omitempty,min=0" example:"999.99"`
	Tags         pq.StringArray `json:"tags" validate:"omitempty" example:"[\"smartphone\", \"apple\", \"premium\"]"`
	CategoryID   *uuid.UUID     `json:"category_id" validate:"omitempty" example:"123e4567-e89b-12d3-a456-426614174000"`
	CuratedPrice *float64       `json:"curated_price" validate:"omitempty,min=0" example:"899.99"`
	CuratedTags  pq.StringArray `json:"curated_tags" validate:"omitempty" example:"[\"smartphone\", \"premium\", \"bestseller\"]"`
}

type ProductsListResponse struct {
	Products []ProductResponse `json:"products"`
	Total    int64             `json:"total"`
	Page     int               `json:"page"`
	Limit    int               `json:"limit"`
	Filters  ProductFilters    `json:"filters"`
}

type ProductFilters struct {
	Search     string     `json:"search"`
	CategoryID *uuid.UUID `json:"category_id"`
	MinPrice   *float64   `json:"min_price"`
	MaxPrice   *float64   `json:"max_price"`
	Tags       []string   `json:"tags"`
	CreatedBy  *uuid.UUID `json:"created_by"`
}

type ProductSearchRequest struct {
	Query      string     `json:"query" validate:"required,min=1" example:"smartphone"`
	CategoryID *uuid.UUID `json:"category_id" validate:"omitempty" example:"123e4567-e89b-12d3-a456-426614174000"`
	MinPrice   *float64   `json:"min_price" validate:"omitempty,min=0" example:"100"`
	MaxPrice   *float64   `json:"max_price" validate:"omitempty,min=0" example:"1000"`
	Tags       []string   `json:"tags" validate:"omitempty" example:"[\"premium\", \"bestseller\"]"`
	Page       int        `json:"page" validate:"omitempty,min=1" example:"1"`
	Limit      int        `json:"limit" validate:"omitempty,min=1,max=100" example:"20"`
	SortBy     string     `json:"sort_by" validate:"omitempty,oneof=price created_at" example:"price"`
	SortOrder  string     `json:"sort_order" validate:"omitempty,oneof=asc desc" example:"asc"`
}

// GetProducts retrieves products with filtering and pagination
// @Summary Get products
// @Description Retrieve a paginated list of products with optional filtering
// @Tags Products
// @Accept json
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Param search query string false "Search in name and description"
// @Param category_id query string false "Filter by category ID"
// @Param min_price query number false "Minimum price filter"
// @Param max_price query number false "Maximum price filter"
// @Param tags query string false "Filter by tags (comma-separated)"
// @Param created_by query string false "Filter by creator user ID"
// @Success 200 {object} ProductsListResponse "Products retrieved successfully"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products [get]
func GetProducts(c *fiber.Ctx) error {
	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)
	offset := (page - 1) * limit

	// Parse filters
	filters := ProductFilters{
		Search: c.Query("search"),
	}

	if categoryIDStr := c.Query("category_id"); categoryIDStr != "" {
		if categoryID, err := uuid.Parse(categoryIDStr); err == nil {
			filters.CategoryID = &categoryID
		}
	}

	if minPriceStr := c.Query("min_price"); minPriceStr != "" {
		if minPrice := c.QueryFloat("min_price"); minPrice > 0 {
			filters.MinPrice = &minPrice
		}
	}

	if maxPriceStr := c.Query("max_price"); maxPriceStr != "" {
		if maxPrice := c.QueryFloat("max_price"); maxPrice > 0 {
			filters.MaxPrice = &maxPrice
		}
	}

	if tagsStr := c.Query("tags"); tagsStr != "" {
		filters.Tags = strings.Split(tagsStr, ",")
	}

	if createdByStr := c.Query("created_by"); createdByStr != "" {
		if createdBy, err := uuid.Parse(createdByStr); err == nil {
			filters.CreatedBy = &createdBy
		}
	}

	// Get current user ID for favorites check
	userID, _ := middleware.GetUserID(c)

	// Build query
	query := database.DB.Model(&models.Product{}).
		Preload("Category").
		Preload("Creator")

	// Apply filters
	if filters.Search != "" {
		searchTerm := "%" + strings.ToLower(filters.Search) + "%"
		query = query.Where("LOWER(name) LIKE ? OR LOWER(description) LIKE ?", searchTerm, searchTerm)
	}

	if filters.CategoryID != nil {
		query = query.Where("category_id = ?", *filters.CategoryID)
	}

	if filters.MinPrice != nil {
		query = query.Where("price >= ?", *filters.MinPrice)
	}

	if filters.MaxPrice != nil {
		query = query.Where("price <= ?", *filters.MaxPrice)
	}

	if len(filters.Tags) > 0 {
		query = query.Where("tags && ?", pq.StringArray(filters.Tags))
	}

	if filters.CreatedBy != nil {
		query = query.Where("created_by = ?", *filters.CreatedBy)
	}

	// Get total count
	var total int64
	if err := query.Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count products",
		})
	}

	// Get products
	var products []models.Product
	if err := query.
		Offset(offset).
		Limit(limit).
		Order("created_at DESC").
		Find(&products).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve products",
		})
	}

	// Enhance products with additional data
	var productResponses []ProductResponse
	for _, product := range products {
		response := enhanceProductResponse(product, userID)
		productResponses = append(productResponses, response)
	}

	return c.JSON(ProductsListResponse{
		Products: productResponses,
		Total:    total,
		Page:     page,
		Limit:    limit,
		Filters:  filters,
	})
}

// GetProduct retrieves a specific product by ID
// @Summary Get product by ID
// @Description Retrieve a specific product with detailed information
// @Tags Products
// @Accept json
// @Produce json
// @Param id path string true "Product ID"
// @Success 200 {object} ProductResponse "Product retrieved successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Router /products/{id} [get]
func GetProduct(c *fiber.Ctx) error {
	productID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid product ID",
		})
	}

	var product models.Product
	if err := database.DB.
		Preload("Category").
		Preload("Creator").
		First(&product, productID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Get current user ID for favorites check
	userID, _ := middleware.GetUserID(c)

	// Log user event if user is authenticated
	if userID != uuid.Nil {
		go logUserEvent(userID, productID, "view", nil)
	}

	response := enhanceProductResponse(product, userID)
	return c.JSON(response)
}

// CreateProduct creates a new product
// @Summary Create a new product
// @Description Create a new product
// @Tags Products
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body CreateProductRequest true "Product creation data"
// @Success 201 {object} models.Product "Product created successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products [post]
func CreateProduct(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req CreateProductRequest
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

	// Validate category exists if provided
	if req.CategoryID != nil {
		var category models.Category
		if err := database.DB.First(&category, *req.CategoryID).Error; err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Invalid category ID",
			})
		}
	}

	product := models.Product{
		Name:        req.Name,
		Description: req.Description,
		Price:       req.Price,
		Tags:        req.Tags,
		CategoryID:  req.CategoryID,
		CreatedBy:   &userID,
	}

	if err := database.DB.Create(&product).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to create product",
		})
	}

	// Load relations for response
	database.DB.Preload("Category").Preload("Creator").First(&product, product.ID)

	return c.Status(fiber.StatusCreated).JSON(product)
}

// UpdateProduct updates an existing product
// @Summary Update product
// @Description Update an existing product's information
// @Tags Products
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Product ID"
// @Param request body UpdateProductRequest true "Product update data"
// @Success 200 {object} models.Product "Product updated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "User not authorized to update this product"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products/{id} [put]
func UpdateProduct(c *fiber.Ctx) error {
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

	var req UpdateProductRequest
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

	var product models.Product
	if err := database.DB.First(&product, productID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Check if user is the creator or admin
	var user models.User
	database.DB.First(&user, userID)
	if product.CreatedBy != nil && *product.CreatedBy != userID && !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Not authorized to update this product",
		})
	}

	// Validate category exists if provided
	if req.CategoryID != nil {
		var category models.Category
		if err := database.DB.First(&category, *req.CategoryID).Error; err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Invalid category ID",
			})
		}
	}

	// Update fields
	if req.Name != nil {
		product.Name = *req.Name
	}
	if req.Description != nil {
		product.Description = req.Description
	}
	if req.Price != nil {
		product.Price = req.Price
	}
	if req.Tags != nil {
		product.Tags = req.Tags
	}
	if req.CategoryID != nil {
		product.CategoryID = req.CategoryID
	}
	if req.CuratedPrice != nil {
		product.CuratedPrice = req.CuratedPrice
	}
	if req.CuratedTags != nil {
		product.CuratedTags = req.CuratedTags
	}

	if err := database.DB.Save(&product).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to update product",
		})
	}

	// Load relations for response
	database.DB.Preload("Category").Preload("Creator").First(&product, product.ID)

	return c.JSON(product)
}

// DeleteProduct deletes a product
// @Summary Delete product
// @Description Delete a product (only by creator or admin)
// @Tags Products
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Product ID"
// @Success 204 "Product deleted successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "User not authorized to delete this product"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products/{id} [delete]
func DeleteProduct(c *fiber.Ctx) error {
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

	var product models.Product
	if err := database.DB.First(&product, productID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Check if user is the creator or admin
	var user models.User
	database.DB.First(&user, userID)
	if product.CreatedBy != nil && *product.CreatedBy != userID && !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Not authorized to delete this product",
		})
	}

	if err := database.DB.Delete(&product).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to delete product",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

// SearchProducts performs advanced product search
// @Summary Search products
// @Description Perform advanced search with multiple filters
// @Tags Products
// @Accept json
// @Produce json
// @Param request body ProductSearchRequest true "Search parameters"
// @Success 200 {object} ProductsListResponse "Search results"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products/search [post]
func SearchProducts(c *fiber.Ctx) error {
	var req ProductSearchRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
		})
	}

	// Set defaults for pagination
	if req.Page == 0 {
		req.Page = 1
	}
	if req.Limit == 0 {
		req.Limit = 20
	}

	// Clean up tags - remove malformed JSON strings
	var cleanedTags []string
	for _, tag := range req.Tags {
		// Remove JSON array brackets and quotes
		cleanTag := strings.Trim(tag, "[]\"")
		if cleanTag != "" && !strings.Contains(cleanTag, "[\"") {
			cleanedTags = append(cleanedTags, cleanTag)
		}
	}
	req.Tags = cleanedTags

	// Calculate pagination
	page := req.Page
	limit := req.Limit
	offset := (page - 1) * limit

	// Build query
	query := database.DB.Model(&models.Product{}).
		Preload("Category").
		Preload("Creator")

	// Apply filters
	if req.Query != "" {
		query = query.Where("LOWER(name) LIKE ? OR LOWER(description) LIKE ?",
			"%"+strings.ToLower(req.Query)+"%", "%"+strings.ToLower(req.Query)+"%")
	}

	if req.CategoryID != nil {
		query = query.Where("category_id = ?", *req.CategoryID)
	}

	if req.MinPrice != nil {
		query = query.Where("price >= ?", *req.MinPrice)
	}

	if req.MaxPrice != nil {
		query = query.Where("price <= ?", *req.MaxPrice)
	}

	if len(cleanedTags) > 0 {
		for _, tag := range cleanedTags {
			if tag != "" {
				query = query.Where("? = ANY(tags)", tag)
			}
		}
	}

	// Apply sorting
	switch req.SortBy {
	case "price":
		if req.SortOrder == "desc" {
			query = query.Order("price DESC")
		} else {
			query = query.Order("price ASC")
		}
	case "created_at":
		if req.SortOrder == "desc" {
			query = query.Order("created_at DESC")
		} else {
			query = query.Order("created_at ASC")
		}
	default:
		query = query.Order("created_at DESC")
	}

	// Execute query to get products
	var products []models.Product
	if err := query.
		Offset(offset).
		Limit(limit).
		Find(&products).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve products",
		})
	}

	// Get total count
	var total int64
	countQuery := database.DB.Model(&models.Product{}).
		Preload("Category").
		Preload("Creator")

	// Apply the same filters for counting
	if req.Query != "" {
		countQuery = countQuery.Where("LOWER(name) LIKE ? OR LOWER(description) LIKE ?",
			"%"+strings.ToLower(req.Query)+"%", "%"+strings.ToLower(req.Query)+"%")
	}

	if req.CategoryID != nil {
		countQuery = countQuery.Where("category_id = ?", *req.CategoryID)
	}

	if req.MinPrice != nil {
		countQuery = countQuery.Where("price >= ?", *req.MinPrice)
	}

	if req.MaxPrice != nil {
		countQuery = countQuery.Where("price <= ?", *req.MaxPrice)
	}

	// Apply cleaned tags filter for counting
	if len(cleanedTags) > 0 {
		for _, tag := range cleanedTags {
			if tag != "" {
				countQuery = countQuery.Where("? = ANY(tags)", tag)
			}
		}
	}

	if err := countQuery.Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count products",
		})
	}

	// Get current user ID for favorites check
	userID, _ := middleware.GetUserID(c)

	// Enhance product responses - ensure we initialize an empty slice
	var productResponses []ProductResponse = make([]ProductResponse, 0)
	for _, product := range products {
		productResponse := enhanceProductResponse(product, userID)
		productResponses = append(productResponses, productResponse)
	}

	response := ProductsListResponse{
		Products: productResponses,
		Total:    total,
		Page:     req.Page,
		Limit:    req.Limit,
		Filters: ProductFilters{
			Search:     req.Query,
			CategoryID: req.CategoryID,
			MinPrice:   req.MinPrice,
			MaxPrice:   req.MaxPrice,
			Tags:       cleanedTags, // Use cleaned tags in response
		},
	}

	return c.JSON(response)
}

// ProductSuggestionRequest represents a request for on-demand product suggestions
type ProductSuggestionRequest struct {
	Name        string  `json:"name" validate:"required,min=2,max=255" example:"iPhone 15 Pro Max"`
	Description string  `json:"description" validate:"required,min=10,max=2000" example:"Latest flagship smartphone with advanced camera system and A17 Pro chip"`
	Category    *string `json:"category" validate:"omitempty,max=100" example:"Electronics"`
	Brand       *string `json:"brand" validate:"omitempty,max=100" example:"Apple"`
}

// SuggestedPriceRange represents the suggested price range for a product
type SuggestedPriceRange struct {
	MinPrice   float64 `json:"min_price"`
	MaxPrice   float64 `json:"max_price"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning"`
}

// RecommendedTag represents a recommended tag with confidence score
type RecommendedTag struct {
	Tag        string  `json:"tag"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning"`
}

// SimilarProductInfo represents information about similar products
type SimilarProductInfo struct {
	ProductID        string  `json:"product_id"`
	Name             string  `json:"name"`
	Category         string  `json:"category"`
	Price            float64 `json:"price"`
	SimilarityScore  float64 `json:"similarity_score"`
	SimilarityReason string  `json:"similarity_reason"`
}

// OnDemandProductSuggestionResponse represents the response for on-demand product suggestions
type OnDemandProductSuggestionResponse struct {
	ModelVersion        string               `json:"model_version"`
	AnalysisSummary     string               `json:"analysis_summary"`
	Confidence          float64              `json:"confidence"`
	ProcessingTime      string               `json:"processing_time"`
	SuggestedPriceRange SuggestedPriceRange  `json:"suggested_price_range"`
	RecommendedTags     []RecommendedTag     `json:"recommended_tags"`
	SimilarProducts     []SimilarProductInfo `json:"similar_products"`
}

// SuggestProduct provides on-demand product suggestions without database storage
// @Summary Get on-demand product suggestions
// @Description Generate real-time product suggestions including price estimation, tag recommendations, and similar products without storing in database
// @Tags Products
// @Accept json
// @Produce json
// @Param request body ProductSuggestionRequest true "Product information for analysis"
// @Success 200 {object} OnDemandProductSuggestionResponse "Product suggestions generated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /products/suggest [post]
func SuggestProduct(c *fiber.Ctx) error {
	startTime := time.Now()

	var req ProductSuggestionRequest
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

	// Log user event if authenticated
	if userID, ok := middleware.GetUserID(c); ok {
		go logUserEvent(userID, uuid.Nil, "product_suggestion_request", map[string]interface{}{
			"product_name": req.Name,
			"category":     req.Category,
			"brand":        req.Brand,
		})
	}

	// Call ML service for real-time analysis
	mlAnalysis, err := getMLProductSuggestions(req)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to generate product suggestions: " + err.Error(),
		})
	}

	// Process ML response
	response := OnDemandProductSuggestionResponse{
		ModelVersion:    getStringValueFromMap(mlAnalysis, "model_version", "v1.0"),
		AnalysisSummary: getStringValueFromMap(mlAnalysis, "analysis_summary", "Product analysis completed"),
		Confidence:      getFloatValueFromMap(mlAnalysis, "confidence", 0.8),
		ProcessingTime:  time.Since(startTime).String(),
	}

	// Extract price range
	if priceData, ok := mlAnalysis["price_range"].(map[string]interface{}); ok {
		response.SuggestedPriceRange = SuggestedPriceRange{
			MinPrice:   getFloatValueFromMap(priceData, "min_price", 0),
			MaxPrice:   getFloatValueFromMap(priceData, "max_price", 0),
			Confidence: getFloatValueFromMap(priceData, "confidence", 0.8),
			Reasoning:  getStringValueFromMap(priceData, "reasoning", "Based on similar products analysis"),
		}
	}

	// Extract recommended tags
	if tagsData, ok := mlAnalysis["recommended_tags"].([]interface{}); ok {
		for _, tagData := range tagsData {
			if tagMap, ok := tagData.(map[string]interface{}); ok {
				tag := RecommendedTag{
					Tag:        getStringValueFromMap(tagMap, "tag", ""),
					Confidence: getFloatValueFromMap(tagMap, "confidence", 0.8),
					Reasoning:  getStringValueFromMap(tagMap, "reasoning", "ML-based recommendation"),
				}
				if tag.Tag != "" {
					response.RecommendedTags = append(response.RecommendedTags, tag)
				}
			}
		}
	}

	// Extract similar products
	if similarData, ok := mlAnalysis["similar_products"].([]interface{}); ok {
		for _, similarItem := range similarData {
			if similarMap, ok := similarItem.(map[string]interface{}); ok {
				similar := SimilarProductInfo{
					ProductID:        getStringValueFromMap(similarMap, "product_id", ""),
					Name:             getStringValueFromMap(similarMap, "name", ""),
					Category:         getStringValueFromMap(similarMap, "category", ""),
					Price:            getFloatValueFromMap(similarMap, "price", 0),
					SimilarityScore:  getFloatValueFromMap(similarMap, "similarity_score", 0),
					SimilarityReason: getStringValueFromMap(similarMap, "similarity_reason", "Content similarity"),
				}
				if similar.ProductID != "" {
					response.SimilarProducts = append(response.SimilarProducts, similar)
				}
			}
		}
	}

	return c.JSON(response)
}

// getMLProductSuggestions calls the ML service for real-time product suggestions
func getMLProductSuggestions(req ProductSuggestionRequest) (map[string]interface{}, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Default ML service URL
	}

	// Prepare request payload
	payload := map[string]interface{}{
		"name":        req.Name,
		"description": req.Description,
	}

	if req.Category != nil {
		payload["category"] = *req.Category
	}

	if req.Brand != nil {
		payload["brand"] = *req.Brand
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request data: %w", err)
	}

	// Send to ML service
	resp, err := http.Post(
		mlServiceURL+"/products/analyze",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode ML service response: %w", err)
	}

	return result, nil
}

// Helper functions for safe type conversion
func getStringValueFromMap(data map[string]interface{}, key, defaultValue string) string {
	if value, ok := data[key].(string); ok {
		return value
	}
	return defaultValue
}

func getFloatValueFromMap(data map[string]interface{}, key string, defaultValue float64) float64 {
	if value, ok := data[key].(float64); ok {
		return value
	}
	return defaultValue
}

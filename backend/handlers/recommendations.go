package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type RecommendationRequest struct {
	Strategy           string `json:"strategy" validate:"omitempty,oneof=collaborative content demographic hybrid" example:"hybrid"`
	NumRecommendations int    `json:"num_recommendations" validate:"omitempty,min=1,max=50" example:"10"`
	IncludeDeals       bool   `json:"include_deals" example:"true"`
}

type RecommendationResponse struct {
	models.Recommendation
	Product       ProductResponse `json:"product"`
	DealInfo      *DealInfo       `json:"deal_info,omitempty"`
	AffinityScore float64         `json:"affinity_score"`
	ReasonDetails []string        `json:"reason_details"`
}

type DealInfo struct {
	DiscountPercentage float64   `json:"discount_percentage"`
	OriginalPrice      float64   `json:"original_price"`
	DealPrice          float64   `json:"deal_price"`
	DealExpiry         time.Time `json:"deal_expiry"`
	DealReason         string    `json:"deal_reason"`
	IsLimitedTime      bool      `json:"is_limited_time"`
	IsRegionalDeal     bool      `json:"is_regional_deal"`
}

type RecommendationsListResponse struct {
	Recommendations []RecommendationResponse `json:"recommendations"`
	Deals           []RecommendationResponse `json:"deals,omitempty"`
	Total           int64                    `json:"total"`
	Page            int                      `json:"page"`
	Limit           int                      `json:"limit"`
	Strategy        string                   `json:"strategy"`
	UserProfile     map[string]interface{}   `json:"user_profile"`
	DealsSummary    *DealsSummary            `json:"deals_summary,omitempty"`
}

type DealsSummary struct {
	TotalDeals        int     `json:"total_deals"`
	AverageDiscount   float64 `json:"average_discount"`
	TotalSavings      float64 `json:"total_savings"`
	RegionalDeals     int     `json:"regional_deals"`
	PersonalizedDeals int     `json:"personalized_deals"`
}

type MLRecommendationRequest struct {
	UserID             string `json:"user_id"`
	NumRecommendations int    `json:"num_recommendations"`
	Strategy           string `json:"strategy"`
	IncludeDeals       bool   `json:"include_deals"`
}

type MLRecommendationResponse struct {
	UserID          string                 `json:"user_id"`
	Strategy        string                 `json:"strategy"`
	Recommendations []MLRecommendation     `json:"recommendations"`
	Deals           []MLRecommendation     `json:"deals,omitempty"`
	UserProfile     map[string]interface{} `json:"user_profile"`
	Timestamp       string                 `json:"timestamp"`
}

type MLRecommendation struct {
	ProductID          string  `json:"product_id"`
	Name               string  `json:"name"`
	Category           string  `json:"category"`
	Price              float64 `json:"price"`
	Score              float64 `json:"score"`
	Reason             string  `json:"reason"`
	DiscountPercentage float64 `json:"discount_percentage,omitempty"`
	DealExpiry         string  `json:"deal_expiry,omitempty"`
	DealReason         string  `json:"deal_reason,omitempty"`
}

// GetUserRecommendations retrieves personalized recommendations for a user with optional deals
// @Summary Get user recommendations with deals
// @Description Retrieve personalized product recommendations for the authenticated user with optional deal information
// @Tags Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(10)
// @Param strategy query string false "Recommendation strategy" Enums(collaborative,content,demographic,hybrid) default(hybrid)
// @Param include_deals query bool false "Include personalized deals" default(false)
// @Success 200 {object} RecommendationsListResponse "Recommendations retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations [get]
func GetUserRecommendations(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 10)
	offset := (page - 1) * limit
	strategy := c.Query("strategy", "hybrid")
	includeDeals := c.QueryBool("include_deals", false)

	// Log user event for recommendation request
	go logUserEvent(userID, uuid.Nil, "view", map[string]interface{}{
		"strategy":      strategy,
		"include_deals": includeDeals,
		"page":          page,
		"limit":         limit,
		"action":        "recommendation_request",
	})

	// Try ML service first
	mlRecommendations, err := getMLRecommendations(userID.String(), limit, strategy, includeDeals)
	if err == nil && mlRecommendations != nil && len(mlRecommendations.Recommendations) > 0 {
		// Store ML recommendations in database for future use
		go storeMLRecommendations(userID, mlRecommendations)

		// Convert ML recommendations to response format
		var recommendationResponses []RecommendationResponse = make([]RecommendationResponse, 0)
		var dealResponses []RecommendationResponse = make([]RecommendationResponse, 0)

		for _, mlRec := range mlRecommendations.Recommendations {
			if rec := convertMLRecommendationToResponse(mlRec, userID, strategy, false); rec != nil {
				recommendationResponses = append(recommendationResponses, *rec)
			}
		}

		for _, mlDeal := range mlRecommendations.Deals {
			if rec := convertMLRecommendationToResponse(mlDeal, userID, strategy, true); rec != nil {
				dealResponses = append(dealResponses, *rec)
			}
		}

		// Ensure user profile is not null
		userProfile := mlRecommendations.UserProfile
		if userProfile == nil {
			userProfile = map[string]interface{}{
				"birth_year": 1990,
				"region":     "US",
			}
		}

		response := RecommendationsListResponse{
			Recommendations: recommendationResponses,
			Deals:           dealResponses,
			Total:           int64(len(recommendationResponses)),
			Page:            page,
			Limit:           limit,
			Strategy:        strategy,
			UserProfile:     userProfile,
		}

		if includeDeals && len(dealResponses) > 0 {
			response.DealsSummary = &DealsSummary{
				TotalDeals:        len(dealResponses),
				AverageDiscount:   15.0,
				TotalSavings:      100.0,
				RegionalDeals:     len(dealResponses) / 2,
				PersonalizedDeals: len(dealResponses) - (len(dealResponses) / 2),
			}
		}

		return c.JSON(response)
	}

	// Fallback to database recommendations if ML service is unavailable
	log.Printf("ML service unavailable, falling back to database recommendations. Error: %v", err)

	// Try database recommendations first, if they exist return them
	if hasStoredRecommendations(userID) {
		return getDatabaseRecommendations(c, userID, page, limit, offset, strategy)
	}

	// Last resort: generate mock recommendations based on popular products
	return generateMockRecommendationsBasedOnPopularProducts(c, userID, page, limit, strategy, includeDeals)
}

// convertMLRecommendationToResponse converts ML recommendation to response format
func convertMLRecommendationToResponse(mlRec MLRecommendation, userID uuid.UUID, strategy string, isDeal bool) *RecommendationResponse {
	productID, err := uuid.Parse(mlRec.ProductID)
	if err != nil {
		return nil
	}

	var product models.Product
	if err := database.DB.Preload("Category").Preload("Creator").First(&product, productID).Error; err != nil {
		return nil
	}

	// Create recommendation record
	recommendation := models.Recommendation{
		UserID:           userID,
		ProductID:        productID,
		Reason:           &mlRec.Reason,
		ModelVersion:     &strategy,
		RegionBased:      strategy == "demographic",
		AgeBased:         strategy == "demographic",
		BasedOnFavorites: strategy == "collaborative",
		BasedOnComments:  strategy == "content",
	}

	productResponse := enhanceProductResponse(product, userID)

	response := RecommendationResponse{
		Recommendation: recommendation,
		Product:        productResponse,
		AffinityScore:  mlRec.Score,
		ReasonDetails:  []string{mlRec.Reason},
	}

	// Add deal information if this is a deal
	if isDeal && mlRec.DiscountPercentage > 0 {
		originalPrice := mlRec.Price
		dealPrice := originalPrice * (1 - mlRec.DiscountPercentage/100)

		var dealExpiry time.Time
		if mlRec.DealExpiry != "" {
			if parsed, err := time.Parse(time.RFC3339, mlRec.DealExpiry); err == nil {
				dealExpiry = parsed
			} else {
				// Default to 7 days from now
				dealExpiry = time.Now().Add(7 * 24 * time.Hour)
			}
		} else {
			dealExpiry = time.Now().Add(7 * 24 * time.Hour)
		}

		response.DealInfo = &DealInfo{
			DiscountPercentage: mlRec.DiscountPercentage,
			OriginalPrice:      originalPrice,
			DealPrice:          dealPrice,
			DealExpiry:         dealExpiry,
			DealReason:         mlRec.DealReason,
			IsLimitedTime:      true,
			IsRegionalDeal:     strategy == "demographic",
		}
	}

	return &response
}

// GenerateRecommendations generates new recommendations using ML service
// @Summary Generate new recommendations
// @Description Generate fresh recommendations using the ML service with optional deals
// @Tags Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body RecommendationRequest true "Recommendation parameters"
// @Success 200 {object} RecommendationsListResponse "Recommendations generated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/generate [post]
func GenerateRecommendations(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req RecommendationRequest
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

	// Set defaults
	if req.Strategy == "" {
		req.Strategy = "hybrid"
	}
	if req.NumRecommendations == 0 {
		req.NumRecommendations = 10
	}

	// Log user event for recommendation generation
	go logUserEvent(userID, uuid.Nil, "view", map[string]interface{}{
		"strategy":            req.Strategy,
		"num_recommendations": req.NumRecommendations,
		"include_deals":       req.IncludeDeals,
		"action":              "recommendation_generation",
	})

	// Try to generate recommendations using ML service
	mlRecommendations, err := getMLRecommendations(userID.String(), req.NumRecommendations, req.Strategy, req.IncludeDeals)
	if err != nil {
		log.Printf("ML service error: %v", err)
		// Return fallback recommendations instead of error
		return generateMockRecommendationsBasedOnPopularProducts(c, userID, 1, req.NumRecommendations, req.Strategy, req.IncludeDeals)
	}

	// Store fresh recommendations in database
	go storeMLRecommendations(userID, mlRecommendations)

	// Convert ML recommendations to response format
	var recommendationResponses []RecommendationResponse = make([]RecommendationResponse, 0)
	var dealResponses []RecommendationResponse = make([]RecommendationResponse, 0)

	for _, mlRec := range mlRecommendations.Recommendations {
		if rec := convertMLRecommendationToResponse(mlRec, userID, req.Strategy, false); rec != nil {
			recommendationResponses = append(recommendationResponses, *rec)
		}
	}

	for _, mlDeal := range mlRecommendations.Deals {
		if rec := convertMLRecommendationToResponse(mlDeal, userID, req.Strategy, true); rec != nil {
			dealResponses = append(dealResponses, *rec)
		}
	}

	// Ensure user profile is not null
	userProfile := mlRecommendations.UserProfile
	if userProfile == nil {
		userProfile = map[string]interface{}{
			"birth_year": 1990,
			"region":     "US",
		}
	}

	response := RecommendationsListResponse{
		Recommendations: recommendationResponses,
		Deals:           dealResponses,
		Total:           int64(len(recommendationResponses)),
		Page:            1,
		Limit:           req.NumRecommendations,
		Strategy:        req.Strategy,
		UserProfile:     userProfile,
	}

	if req.IncludeDeals && len(dealResponses) > 0 {
		response.DealsSummary = &DealsSummary{
			TotalDeals:        len(dealResponses),
			AverageDiscount:   15.0,
			TotalSavings:      100.0,
			RegionalDeals:     len(dealResponses) / 2,
			PersonalizedDeals: len(dealResponses) - (len(dealResponses) / 2),
		}
	}

	return c.JSON(response)
}

// GetStoredRecommendations retrieves stored recommendations from database
// @Summary Get stored recommendations
// @Description Retrieve previously generated recommendations from database
// @Tags Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(10)
// @Param strategy query string false "Filter by strategy"
// @Success 200 {object} RecommendationsListResponse "Stored recommendations retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/stored [get]
func GetStoredRecommendations(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 10)
	offset := (page - 1) * limit
	strategy := c.Query("strategy")

	return getDatabaseRecommendations(c, userID, page, limit, offset, strategy)
}

// GetRecommendationStats gets recommendation statistics
// @Summary Get recommendation statistics
// @Description Get statistics about user's recommendations
// @Tags Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} map[string]interface{} "Recommendation statistics"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/stats [get]
func GetRecommendationStats(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Get total recommendations count
	var totalRecommendations int64
	database.DB.Model(&models.Recommendation{}).Where("user_id = ?", userID).Count(&totalRecommendations)

	// Get recommendations by strategy
	type StrategyCount struct {
		Strategy string `json:"strategy"`
		Count    int64  `json:"count"`
	}

	var strategyStats []StrategyCount
	if totalRecommendations > 0 {
		database.DB.Table("recommendations").
			Select("COALESCE(model_version, 'unknown') as strategy, COUNT(*) as count").
			Where("user_id = ?", userID).
			Group("model_version").
			Scan(&strategyStats)
	}

	// Ensure we always return an empty array instead of null
	if strategyStats == nil {
		strategyStats = []StrategyCount{}
	}

	// Get recent recommendations count (last 7 days)
	var recentRecommendations int64
	database.DB.Model(&models.Recommendation{}).
		Where("user_id = ? AND created_at > NOW() - INTERVAL '7 days'", userID).
		Count(&recentRecommendations)

	// Get recommendations by category
	type CategoryCount struct {
		CategoryName string `json:"category_name"`
		Count        int64  `json:"count"`
	}

	var categoryStats []CategoryCount
	if totalRecommendations > 0 {
		database.DB.Table("recommendations").
			Select("COALESCE(categories.name, 'Uncategorized') as category_name, COUNT(*) as count").
			Joins("JOIN products ON recommendations.product_id = products.id").
			Joins("LEFT JOIN categories ON products.category_id = categories.id").
			Where("recommendations.user_id = ?", userID).
			Group("categories.name").
			Order("count DESC").
			Scan(&categoryStats)
	}

	// Ensure we always return an empty array instead of null
	if categoryStats == nil {
		categoryStats = []CategoryCount{}
	}

	// Get user interaction with recommendations
	var clickedRecommendations int64
	if totalRecommendations > 0 {
		database.DB.Table("user_events").
			Joins("JOIN recommendations ON user_events.product_id = recommendations.product_id").
			Where("user_events.user_id = ? AND recommendations.user_id = ? AND user_events.event_type IN ('view', 'click')", userID, userID).
			Count(&clickedRecommendations)
	}

	clickRate := float64(0)
	if totalRecommendations > 0 {
		clickRate = float64(clickedRecommendations) / float64(totalRecommendations) * 100
	}

	return c.JSON(map[string]interface{}{
		"total_recommendations":   totalRecommendations,
		"recent_recommendations":  recentRecommendations,
		"strategy_breakdown":      strategyStats,
		"category_breakdown":      categoryStats,
		"clicked_recommendations": clickedRecommendations,
		"click_rate_percentage":   clickRate,
	})
}

// Helper function to get recommendations from ML service
func getMLRecommendations(userID string, numRecommendations int, strategy string, includeDeals bool) (*MLRecommendationResponse, error) {
	// ML service URL (should be configurable)
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Updated default ML service URL
	}

	reqBody := MLRecommendationRequest{
		UserID:             userID,
		NumRecommendations: numRecommendations,
		Strategy:           strategy,
		IncludeDeals:       includeDeals,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	resp, err := client.Post(
		fmt.Sprintf("%s/recommendations/get", mlServiceURL),
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

	var mlResponse MLRecommendationResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return nil, err
	}

	return &mlResponse, nil
}

// Helper function to store ML recommendations in database
func storeMLRecommendations(userID uuid.UUID, mlRecommendations *MLRecommendationResponse) {
	for _, mlRec := range mlRecommendations.Recommendations {
		productID, err := uuid.Parse(mlRec.ProductID)
		if err != nil {
			continue
		}

		// Check if recommendation already exists
		var existingRec models.Recommendation
		if err := database.DB.Where("user_id = ? AND product_id = ?", userID, productID).
			First(&existingRec).Error; err != nil {
			// Create new recommendation
			recommendation := models.Recommendation{
				UserID:           userID,
				ProductID:        productID,
				Reason:           &mlRec.Reason,
				ModelVersion:     &mlRecommendations.Strategy,
				RegionBased:      mlRecommendations.Strategy == "demographic",
				AgeBased:         mlRecommendations.Strategy == "demographic",
				BasedOnFavorites: mlRecommendations.Strategy == "collaborative",
				BasedOnComments:  mlRecommendations.Strategy == "content",
			}

			database.DB.Create(&recommendation)
		}
	}
}

// Helper function to get recommendations from database
func getDatabaseRecommendations(c *fiber.Ctx, userID uuid.UUID, page, limit, offset int, strategy string) error {
	// Build query
	query := database.DB.Where("user_id = ?", userID).
		Preload("Product").
		Preload("Product.Category").
		Preload("Product.Creator")

	if strategy != "" {
		query = query.Where("model_version = ?", strategy)
	}

	// Get total count
	var total int64
	if err := query.Model(&models.Recommendation{}).Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count recommendations",
		})
	}

	// Get recommendations
	var recommendations []models.Recommendation
	if err := query.
		Offset(offset).
		Limit(limit).
		Order("created_at DESC").
		Find(&recommendations).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve recommendations",
		})
	}

	// Enhance recommendations with product data
	var recommendationResponses []RecommendationResponse = make([]RecommendationResponse, 0)
	for _, recommendation := range recommendations {
		productResponse := enhanceProductResponse(recommendation.Product, userID)
		recommendationResponses = append(recommendationResponses, RecommendationResponse{
			Recommendation: recommendation,
			Product:        productResponse,
		})
	}

	// Get user profile for response
	var user models.User
	database.DB.First(&user, userID)
	userProfile := map[string]interface{}{
		"region":     user.Region,
		"birth_year": user.BirthYear,
	}

	return c.JSON(RecommendationsListResponse{
		Recommendations: recommendationResponses,
		Total:           total,
		Page:            page,
		Limit:           limit,
		Strategy:        strategy,
		UserProfile:     userProfile,
	})
}

// GetEnhancedRecommendations retrieves recommendations using the enhanced multi-stage system
// @Summary Get enhanced recommendations with multi-stage processing
// @Description Get recommendations using the enhanced system with real-time scoring, batch processing, and dynamic deals
// @Tags Enhanced Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(10)
// @Param stage query string false "Recommendation stage" Enums(realtime,full) default(full)
// @Param include_deals query bool false "Include personalized deals" default(true)
// @Success 200 {object} RecommendationsListResponse "Enhanced recommendations retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/enhanced [get]
func GetEnhancedRecommendations(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	limit := c.QueryInt("limit", 10)
	stage := c.Query("stage", "full")
	includeDeals := c.QueryBool("include_deals", true)

	// Log user event for enhanced recommendation request
	go logUserEvent(userID, uuid.Nil, "view", map[string]interface{}{
		"stage":         stage,
		"include_deals": includeDeals,
		"limit":         limit,
		"action":        "enhanced_recommendation_request",
	})

	// Get recommendations from enhanced ML service
	var endpoint string
	if stage == "realtime" {
		endpoint = "/recommendations/enhanced/realtime"
	} else {
		endpoint = "/recommendations/enhanced/full"
	}

	enhancedRecommendations, err := getEnhancedMLRecommendations(userID.String(), limit, endpoint, includeDeals)
	if err != nil {
		log.Printf("Enhanced ML recommendations failed: %v", err)
		// Fallback to regular recommendations
		return GetUserRecommendations(c)
	}

	// Convert enhanced ML recommendations to response format
	var recommendationResponses []RecommendationResponse
	var dealResponses []RecommendationResponse
	var dealsSummary DealsSummary

	// Process regular recommendations
	for _, mlRec := range enhancedRecommendations.Recommendations {
		recResponse := convertMLRecommendationToResponse(mlRec, userID, "enhanced_"+stage, false)
		if recResponse != nil {
			recommendationResponses = append(recommendationResponses, *recResponse)
		}
	}

	// Process deals if included
	if includeDeals && len(enhancedRecommendations.Deals) > 0 {
		var totalSavings float64
		var totalDiscount float64
		regionalDeals := 0
		personalizedDeals := 0

		for _, mlDeal := range enhancedRecommendations.Deals {
			dealResponse := convertMLRecommendationToResponse(mlDeal, userID, "enhanced_deals", true)
			if dealResponse != nil {
				dealResponses = append(dealResponses, *dealResponse)

				if dealResponse.DealInfo != nil {
					totalSavings += dealResponse.DealInfo.OriginalPrice - dealResponse.DealInfo.DealPrice
					totalDiscount += dealResponse.DealInfo.DiscountPercentage

					if dealResponse.DealInfo.IsRegionalDeal {
						regionalDeals++
					} else {
						personalizedDeals++
					}
				}
			}
		}

		if len(dealResponses) > 0 {
			dealsSummary = DealsSummary{
				TotalDeals:        len(dealResponses),
				AverageDiscount:   totalDiscount / float64(len(dealResponses)),
				TotalSavings:      totalSavings,
				RegionalDeals:     regionalDeals,
				PersonalizedDeals: personalizedDeals,
			}
		}
	}

	response := RecommendationsListResponse{
		Recommendations: recommendationResponses,
		Deals:           dealResponses,
		Total:           int64(len(recommendationResponses)),
		Page:            1,
		Limit:           limit,
		Strategy:        "enhanced_" + stage,
		UserProfile:     enhancedRecommendations.UserProfile,
	}

	if includeDeals && len(dealResponses) > 0 {
		response.DealsSummary = &dealsSummary
	}

	return c.JSON(response)
}

// GetPersonalizedDeals retrieves personalized dynamic deals
// @Summary Get personalized dynamic deals
// @Description Get personalized deals with enhanced logic based on user preferences and behavior
// @Tags Enhanced Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param num_deals query int false "Number of deals" default(5)
// @Success 200 {object} EnhancedDealsResponse "Personalized deals retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/deals [get]
func GetPersonalizedDeals(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	numDeals := c.QueryInt("num_deals", 5)

	// Log user event for deal request
	go logUserEvent(userID, uuid.Nil, "view", map[string]interface{}{
		"num_deals": numDeals,
		"action":    "personalized_deals_request",
	})

	// Get personalized deals from enhanced ML service
	deals, err := getPersonalizedDealsFromML(userID.String(), numDeals)
	if err != nil {
		log.Printf("Personalized deals failed: %v", err)
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to generate personalized deals",
		})
	}

	return c.JSON(deals)
}

// TriggerBatchProcessing triggers the batch processing for recommendations
// @Summary Trigger batch processing for recommendations
// @Description Trigger Stage 2 batch processing for matrix factorization and user clustering
// @Tags Enhanced Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} map[string]interface{} "Batch processing triggered successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/batch-process [post]
func TriggerBatchProcessing(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Log admin action
	go logUserEvent(userID, uuid.Nil, "view", map[string]interface{}{
		"timestamp": time.Now(),
		"action":    "batch_processing_trigger",
	})

	// Trigger batch processing in ML service
	result, err := triggerMLBatchProcessing()
	if err != nil {
		log.Printf("Batch processing trigger failed: %v", err)
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to trigger batch processing",
		})
	}

	return c.JSON(result)
}

// GetUserSegments retrieves user segmentation analysis
// @Summary Get user segmentation analysis
// @Description Get detailed user clustering and segmentation results from the enhanced system
// @Tags Enhanced Recommendations
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} map[string]interface{} "User segments retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /recommendations/segments [get]
func GetUserSegments(c *fiber.Ctx) error {
	_, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Get user segments from enhanced ML service
	segments, err := getUserSegmentsFromML()
	if err != nil {
		log.Printf("User segments retrieval failed: %v", err)
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve user segments",
		})
	}

	return c.JSON(segments)
}

// Types for enhanced recommendations
type EnhancedMLRecommendationResponse struct {
	UserID          string                 `json:"user_id"`
	Recommendations []MLRecommendation     `json:"recommendations"`
	Deals           []MLRecommendation     `json:"deals,omitempty"`
	UserProfile     map[string]interface{} `json:"user_profile"`
	Strategy        string                 `json:"strategy"`
	Timestamp       string                 `json:"timestamp"`
	BatchUpdate     *string                `json:"batch_update,omitempty"`
	APIInfo         map[string]interface{} `json:"api_info,omitempty"`
}

type EnhancedDealsResponse struct {
	UserID             string                 `json:"user_id"`
	Deals              []MLRecommendation     `json:"deals"`
	UserProfile        map[string]interface{} `json:"user_profile"`
	DealGenerationInfo map[string]interface{} `json:"deal_generation_info"`
	Timestamp          string                 `json:"timestamp"`
}

// Helper functions for enhanced recommendations
func getEnhancedMLRecommendations(userID string, numRecommendations int, endpoint string, includeDeals bool) (*EnhancedMLRecommendationResponse, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000"
	}

	requestBody := MLRecommendationRequest{
		UserID:             userID,
		NumRecommendations: numRecommendations,
		Strategy:           "enhanced",
		IncludeDeals:       includeDeals,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(mlServiceURL+endpoint, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	var mlResponse EnhancedMLRecommendationResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		return nil, err
	}

	return &mlResponse, nil
}

func getPersonalizedDealsFromML(userID string, numDeals int) (*EnhancedDealsResponse, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000"
	}

	url := fmt.Sprintf("%s/recommendations/enhanced/deals/%s?num_deals=%d", mlServiceURL, userID, numDeals)

	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	var dealsResponse EnhancedDealsResponse
	if err := json.NewDecoder(resp.Body).Decode(&dealsResponse); err != nil {
		return nil, err
	}

	return &dealsResponse, nil
}

func triggerMLBatchProcessing() (map[string]interface{}, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000"
	}

	resp, err := http.Post(mlServiceURL+"/recommendations/enhanced/batch-process", "application/json", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func getUserSegmentsFromML() (map[string]interface{}, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000"
	}

	resp, err := http.Get(mlServiceURL + "/recommendations/enhanced/segments")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	var segments map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&segments); err != nil {
		return nil, err
	}

	return segments, nil
}

// Helper function to check if user has stored recommendations
func hasStoredRecommendations(userID uuid.UUID) bool {
	var count int64
	database.DB.Model(&models.Recommendation{}).Where("user_id = ?", userID).Count(&count)
	return count > 0
}

// generateMockRecommendationsBasedOnPopularProducts creates mock recommendations from popular products
func generateMockRecommendationsBasedOnPopularProducts(c *fiber.Ctx, userID uuid.UUID, page, limit int, strategy string, includeDeals bool) error {
	// Get popular products (products with most favorites or comments)
	var products []models.Product
	err := database.DB.Model(&models.Product{}).
		Preload("Category").
		Preload("Creator").
		Select("products.*, COALESCE(fav_count.count, 0) + COALESCE(comment_count.count, 0) as popularity_score").
		Joins("LEFT JOIN (SELECT product_id, COUNT(*) as count FROM favorites GROUP BY product_id) fav_count ON products.id = fav_count.product_id").
		Joins("LEFT JOIN (SELECT product_id, COUNT(*) as count FROM comments GROUP BY product_id) comment_count ON products.id = comment_count.product_id").
		Order("popularity_score DESC").
		Limit(limit).
		Find(&products).Error

	if err != nil {
		log.Printf("Error fetching popular products: %v", err)
		// Return empty response instead of error
		response := RecommendationsListResponse{
			Recommendations: make([]RecommendationResponse, 0),
			Deals:           make([]RecommendationResponse, 0),
			Total:           0,
			Page:            page,
			Limit:           limit,
			Strategy:        strategy,
			UserProfile: map[string]interface{}{
				"birth_year": 1990,
				"region":     "US",
			},
		}
		return c.JSON(response)
	}

	// Convert to recommendation responses
	var recommendationResponses []RecommendationResponse = make([]RecommendationResponse, 0)
	for i, product := range products {
		productResponse := enhanceProductResponse(product, userID)

		// Create mock recommendation
		recommendation := models.Recommendation{
			UserID:       userID,
			ProductID:    product.ID,
			ModelVersion: &strategy,
		}

		reason := fmt.Sprintf("Popular product recommended via %s strategy", strategy)
		recommendation.Reason = &reason

		response := RecommendationResponse{
			Recommendation: recommendation,
			Product:        productResponse,
			AffinityScore:  float64(limit-i) / float64(limit), // Decreasing score
			ReasonDetails:  []string{reason},
		}

		recommendationResponses = append(recommendationResponses, response)
	}

	// Generate mock deals if requested
	var dealResponses []RecommendationResponse = make([]RecommendationResponse, 0)
	if includeDeals && len(products) > 0 {
		// Use first few products as deals
		dealCount := len(products) / 2
		if dealCount > 3 {
			dealCount = 3
		}

		for i := 0; i < dealCount; i++ {
			product := products[i]
			productResponse := enhanceProductResponse(product, userID)

			recommendation := models.Recommendation{
				UserID:       userID,
				ProductID:    product.ID,
				ModelVersion: &strategy,
			}

			reason := fmt.Sprintf("Special deal on popular product")
			recommendation.Reason = &reason

			dealInfo := &DealInfo{
				DiscountPercentage: 15.0 + float64(i*5),
				OriginalPrice:      *product.Price,
				DealPrice:          *product.Price * (1.0 - (15.0+float64(i*5))/100.0),
				DealExpiry:         time.Now().Add(24 * time.Hour),
				DealReason:         "Limited time offer",
				IsLimitedTime:      true,
				IsRegionalDeal:     i%2 == 0,
			}

			response := RecommendationResponse{
				Recommendation: recommendation,
				Product:        productResponse,
				DealInfo:       dealInfo,
				AffinityScore:  0.8 - float64(i)*0.1,
				ReasonDetails:  []string{reason},
			}

			dealResponses = append(dealResponses, response)
		}
	}

	// Create response
	response := RecommendationsListResponse{
		Recommendations: recommendationResponses,
		Deals:           dealResponses,
		Total:           int64(len(recommendationResponses)),
		Page:            page,
		Limit:           limit,
		Strategy:        strategy,
		UserProfile: map[string]interface{}{
			"birth_year": 1990,
			"region":     "US",
		},
	}

	if includeDeals && len(dealResponses) > 0 {
		response.DealsSummary = &DealsSummary{
			TotalDeals:        len(dealResponses),
			AverageDiscount:   15.0,
			TotalSavings:      50.0,
			RegionalDeals:     len(dealResponses) / 2,
			PersonalizedDeals: len(dealResponses) - (len(dealResponses) / 2),
		}
	}

	return c.JSON(response)
}

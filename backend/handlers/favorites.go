package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type AddFavoriteRequest struct {
	ProductID uuid.UUID `json:"product_id" validate:"required" example:"123e4567-e89b-12d3-a456-426614174000"`
}

type FavoriteResponse struct {
	models.Favorite
	Product ProductResponse `json:"product"`
}

type FavoritesListResponse struct {
	Favorites []FavoriteResponse `json:"favorites"`
	Total     int64              `json:"total"`
	Page      int                `json:"page"`
	Limit     int                `json:"limit"`
}

// GetUserFavorites retrieves user's favorite products
// @Summary Get user's favorites
// @Description Retrieve a paginated list of user's favorite products
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Success 200 {object} FavoritesListResponse "Favorites retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites [get]
func GetUserFavorites(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)
	offset := (page - 1) * limit

	// Get total count
	var total int64
	if err := database.DB.Model(&models.Favorite{}).
		Where("user_id = ?", userID).
		Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count favorites",
		})
	}

	// Get favorites with products
	var favorites []models.Favorite
	if err := database.DB.
		Where("user_id = ?", userID).
		Preload("Product").
		Preload("Product.Category").
		Preload("Product.Creator").
		Offset(offset).
		Limit(limit).
		Order("favorited_at DESC").
		Find(&favorites).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve favorites",
		})
	}

	// Enhance favorites with additional product data
	var favoriteResponses []FavoriteResponse
	for _, favorite := range favorites {
		productResponse := enhanceProductResponse(favorite.Product, userID)
		favoriteResponses = append(favoriteResponses, FavoriteResponse{
			Favorite: favorite,
			Product:  productResponse,
		})
	}

	return c.JSON(FavoritesListResponse{
		Favorites: favoriteResponses,
		Total:     total,
		Page:      page,
		Limit:     limit,
	})
}

// AddFavorite adds a product to user's favorites
// @Summary Add product to favorites
// @Description Add a product to the authenticated user's favorites
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body AddFavoriteRequest true "Product to add to favorites"
// @Success 201 {object} models.Favorite "Product added to favorites successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 409 {object} StandardErrorResponse "Product already in favorites"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites [post]
func AddFavorite(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req AddFavoriteRequest
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

	// Check if product exists
	var product models.Product
	if err := database.DB.First(&product, req.ProductID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Check if already favorited
	var existingFavorite models.Favorite
	if err := database.DB.Where("user_id = ? AND product_id = ?", userID, req.ProductID).
		First(&existingFavorite).Error; err == nil {
		return c.Status(fiber.StatusConflict).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product already in favorites",
		})
	}

	// Create favorite
	favorite := models.Favorite{
		UserID:    userID,
		ProductID: req.ProductID,
	}

	if err := database.DB.Create(&favorite).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to add favorite",
		})
	}

	// Log user event
	go logUserEvent(userID, req.ProductID, "favorite", map[string]interface{}{
		"action": "add",
	})

	// Load relations for response
	database.DB.Preload("Product").Preload("User").First(&favorite, favorite.ID)

	return c.Status(fiber.StatusCreated).JSON(favorite)
}

// RemoveFavorite removes a product from user's favorites
// @Summary Remove product from favorites
// @Description Remove a product from the authenticated user's favorites
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param product_id path string true "Product ID"
// @Success 204 "Product removed from favorites successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Favorite not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites/{product_id} [delete]
func RemoveFavorite(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	productID, err := uuid.Parse(c.Params("product_id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid product ID",
		})
	}

	// Find and delete favorite
	var favorite models.Favorite
	if err := database.DB.Where("user_id = ? AND product_id = ?", userID, productID).
		First(&favorite).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Favorite not found",
		})
	}

	if err := database.DB.Delete(&favorite).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to remove favorite",
		})
	}

	// Log user event
	go logUserEvent(userID, productID, "favorite", map[string]interface{}{
		"action": "remove",
	})

	return c.SendStatus(fiber.StatusNoContent)
}

// CheckFavorite checks if a product is in user's favorites
// @Summary Check if product is favorited
// @Description Check if a specific product is in the authenticated user's favorites
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param product_id path string true "Product ID"
// @Success 200 {object} map[string]bool "Favorite status"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites/{product_id}/check [get]
func CheckFavorite(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	productID, err := uuid.Parse(c.Params("product_id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid product ID",
		})
	}

	// Check if favorite exists
	var count int64
	if err := database.DB.Model(&models.Favorite{}).
		Where("user_id = ? AND product_id = ?", userID, productID).
		Count(&count).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to check favorite status",
		})
	}

	return c.JSON(map[string]bool{
		"is_favorited": count > 0,
	})
}

// GetFavoriteStats gets user's favorite statistics
// @Summary Get favorite statistics
// @Description Get statistics about the authenticated user's favorites
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} map[string]interface{} "Favorite statistics"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites/stats [get]
func GetFavoriteStats(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Get total favorites count
	var totalFavorites int64
	database.DB.Model(&models.Favorite{}).Where("user_id = ?", userID).Count(&totalFavorites)

	// Get favorites by category
	type CategoryCount struct {
		CategoryName string `json:"category_name"`
		Count        int64  `json:"count"`
	}

	var categoryStats []CategoryCount
	if totalFavorites > 0 {
		database.DB.Table("favorites").
			Select("COALESCE(categories.name, 'Uncategorized') as category_name, COUNT(*) as count").
			Joins("JOIN products ON favorites.product_id = products.id").
			Joins("LEFT JOIN categories ON products.category_id = categories.id").
			Where("favorites.user_id = ?", userID).
			Group("categories.name").
			Order("count DESC").
			Scan(&categoryStats)
	}

	// Ensure we always return an empty array instead of null
	if categoryStats == nil {
		categoryStats = []CategoryCount{}
	}

	// Get recent favorites count (last 30 days)
	var recentFavorites int64
	database.DB.Model(&models.Favorite{}).
		Where("user_id = ? AND favorited_at > NOW() - INTERVAL '30 days'", userID).
		Count(&recentFavorites)

	// Get average price of favorited products
	type PriceStats struct {
		AvgPrice *float64 `json:"avg_price"`
		MinPrice *float64 `json:"min_price"`
		MaxPrice *float64 `json:"max_price"`
	}

	var priceStats PriceStats
	if totalFavorites > 0 {
		database.DB.Table("favorites").
			Select("AVG(products.price) as avg_price, MIN(products.price) as min_price, MAX(products.price) as max_price").
			Joins("JOIN products ON favorites.product_id = products.id").
			Where("favorites.user_id = ? AND products.price IS NOT NULL", userID).
			Scan(&priceStats)
	}

	// If no price stats available, set default values
	if priceStats.AvgPrice == nil && priceStats.MinPrice == nil && priceStats.MaxPrice == nil {
		priceStats = PriceStats{
			AvgPrice: nil,
			MinPrice: nil,
			MaxPrice: nil,
		}
	}

	return c.JSON(map[string]interface{}{
		"total_favorites":    totalFavorites,
		"recent_favorites":   recentFavorites,
		"category_breakdown": categoryStats,
		"price_stats":        priceStats,
	})
}

// ToggleFavorite toggles favorite status for a product
// @Summary Toggle favorite status
// @Description Add or remove a product from favorites based on current status
// @Tags Favorites
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param product_id path string true "Product ID"
// @Success 200 {object} map[string]interface{} "Toggle result"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /favorites/{product_id}/toggle [post]
func ToggleFavorite(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	productID, err := uuid.Parse(c.Params("product_id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid product ID",
		})
	}

	// Check if product exists
	var product models.Product
	if err := database.DB.First(&product, productID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Product not found",
		})
	}

	// Check if already favorited
	var existingFavorite models.Favorite
	err = database.DB.Where("user_id = ? AND product_id = ?", userID, productID).
		First(&existingFavorite).Error

	var action string
	var isFavorited bool

	if err == nil {
		// Remove favorite
		if err := database.DB.Delete(&existingFavorite).Error; err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Failed to remove favorite",
			})
		}
		action = "removed"
		isFavorited = false
	} else {
		// Add favorite
		favorite := models.Favorite{
			UserID:    userID,
			ProductID: productID,
		}

		if err := database.DB.Create(&favorite).Error; err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Failed to add favorite",
			})
		}
		action = "added"
		isFavorited = true
	}

	// Log user event
	go logUserEvent(userID, productID, "favorite", map[string]interface{}{
		"action": action,
	})

	return c.JSON(map[string]interface{}{
		"action":       action,
		"is_favorited": isFavorited,
		"product_id":   productID,
	})
}

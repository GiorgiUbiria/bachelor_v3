package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type CreateCommentRequest struct {
	ProductID uuid.UUID `json:"product_id" validate:"required" example:"123e4567-e89b-12d3-a456-426614174000"`
	Body      string    `json:"body" validate:"required,min=1,max=2000" example:"Great product! Highly recommended."`
}

type UpdateCommentRequest struct {
	Body string `json:"body" validate:"required,min=1,max=2000" example:"Updated comment text"`
}

type CommentResponse struct {
	models.Comment
	User     models.User    `json:"user"`
	Product  models.Product `json:"product"`
	UserVote *string        `json:"user_vote"` // "up", "down", or null
	NetVotes int            `json:"net_votes"` // upvotes - downvotes
}

type CommentsListResponse struct {
	Comments []CommentResponse `json:"comments"`
	Total    int64             `json:"total"`
	Page     int               `json:"page"`
	Limit    int               `json:"limit"`
}

type VoteCommentRequest struct {
	VoteType string `json:"vote_type" validate:"required,oneof=up down" example:"up"`
}

// GetProductComments retrieves comments for a specific product
// @Summary Get product comments
// @Description Get paginated comments for a specific product with optional user vote information
// @Tags Comments
// @Accept json
// @Produce json
// @Param product_id path string true "Product ID"
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Param sort_by query string false "Sort by (created_at, upvotes, downvotes)" default("created_at")
// @Param sort_order query string false "Sort order (asc, desc)" default("desc")
// @Success 200 {object} CommentsListResponse "Comments retrieved successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid product ID"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/product/{product_id} [get]
func GetProductComments(c *fiber.Ctx) error {
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

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)
	offset := (page - 1) * limit
	sortBy := c.Query("sort_by", "created_at")
	sortOrder := c.Query("sort_order", "desc")

	// Get current user ID for vote checking
	userID, _ := middleware.GetUserID(c)

	// Get total count
	var total int64
	if err := database.DB.Model(&models.Comment{}).
		Where("product_id = ?", productID).
		Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count comments",
		})
	}

	// Build query with sorting
	query := database.DB.
		Where("product_id = ?", productID).
		Preload("User").
		Preload("Product")

	switch sortBy {
	case "upvotes":
		query = query.Order("upvotes DESC")
	case "downvotes":
		query = query.Order("downvotes DESC")
	default:
		query = query.Order("created_at DESC")
	}

	if sortOrder == "asc" {
		query = query.Order("created_at ASC")
	}

	// Get comments
	var comments []models.Comment
	if err := query.
		Offset(offset).
		Limit(limit).
		Find(&comments).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve comments",
		})
	}

	// Enhance comments with user vote information
	var commentResponses []CommentResponse
	for _, comment := range comments {
		response := CommentResponse{
			Comment:  comment,
			User:     comment.User,
			Product:  comment.Product,
			NetVotes: comment.Upvotes - comment.Downvotes,
		}

		// Get user's vote if authenticated
		if userID != uuid.Nil {
			var vote models.CommentVote
			if err := database.DB.Where("user_id = ? AND comment_id = ?", userID, comment.ID).
				First(&vote).Error; err == nil {
				response.UserVote = &vote.VoteType
			}
		}

		commentResponses = append(commentResponses, response)
	}

	return c.JSON(CommentsListResponse{
		Comments: commentResponses,
		Total:    total,
		Page:     page,
		Limit:    limit,
	})
}

// CreateComment creates a new comment
// @Summary Create a comment
// @Description Create a new comment on a product
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body CreateCommentRequest true "Comment creation data"
// @Success 201 {object} models.Comment "Comment created successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Product not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments [post]
func CreateComment(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req CreateCommentRequest
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

	// Create comment
	comment := models.Comment{
		UserID:    userID,
		ProductID: req.ProductID,
		Body:      req.Body,
	}

	if err := database.DB.Create(&comment).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to create comment",
		})
	}

	// Log user event
	go logUserEvent(userID, req.ProductID, "comment", map[string]interface{}{
		"comment_id": comment.ID,
		"action":     "create",
	})

	// Load relations for response
	database.DB.Preload("User").Preload("Product").First(&comment, comment.ID)

	return c.Status(fiber.StatusCreated).JSON(comment)
}

// UpdateComment updates an existing comment
// @Summary Update comment
// @Description Update an existing comment (only by the author)
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Comment ID"
// @Param request body UpdateCommentRequest true "Comment update data"
// @Success 200 {object} models.Comment "Comment updated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "User not authorized to update this comment"
// @Failure 404 {object} StandardErrorResponse "Comment not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/{id} [put]
func UpdateComment(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	commentID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid comment ID",
		})
	}

	var req UpdateCommentRequest
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

	var comment models.Comment
	if err := database.DB.First(&comment, commentID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Comment not found",
		})
	}

	// Check if user is the author or admin
	var user models.User
	database.DB.First(&user, userID)
	if comment.UserID != userID && !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Not authorized to update this comment",
		})
	}

	// Update comment
	comment.Body = req.Body

	if err := database.DB.Save(&comment).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to update comment",
		})
	}

	// Load relations for response
	database.DB.Preload("User").Preload("Product").First(&comment, comment.ID)

	return c.JSON(comment)
}

// DeleteComment deletes a comment
// @Summary Delete comment
// @Description Delete a comment (only by the author or admin)
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Comment ID"
// @Success 204 "Comment deleted successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid comment ID"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 403 {object} StandardErrorResponse "User not authorized to delete this comment"
// @Failure 404 {object} StandardErrorResponse "Comment not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/{id} [delete]
func DeleteComment(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	commentID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid comment ID",
		})
	}

	var comment models.Comment
	if err := database.DB.First(&comment, commentID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Comment not found",
		})
	}

	// Check if user is the author or admin
	var user models.User
	database.DB.First(&user, userID)
	if comment.UserID != userID && !user.IsAdmin {
		return c.Status(fiber.StatusForbidden).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Not authorized to delete this comment",
		})
	}

	// Delete associated votes first
	database.DB.Where("comment_id = ?", commentID).Delete(&models.CommentVote{})

	// Delete comment
	if err := database.DB.Delete(&comment).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to delete comment",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

// VoteComment votes on a comment
// @Summary Vote on comment
// @Description Vote up or down on a comment
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Comment ID"
// @Param request body VoteCommentRequest true "Vote data"
// @Success 200 {object} map[string]interface{} "Vote recorded successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Comment not found"
// @Failure 409 {object} StandardErrorResponse "User already voted on this comment"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/{id}/vote [post]
func VoteComment(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	commentID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid comment ID",
		})
	}

	var req VoteCommentRequest
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

	// Check if comment exists
	var comment models.Comment
	if err := database.DB.First(&comment, commentID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Comment not found",
		})
	}

	// Check if user already voted
	var existingVote models.CommentVote
	err = database.DB.Where("user_id = ? AND comment_id = ?", userID, commentID).
		First(&existingVote).Error

	if err == nil {
		// User already voted, update or remove vote
		if existingVote.VoteType == req.VoteType {
			// Same vote type, remove vote
			database.DB.Delete(&existingVote)

			// Update comment vote counts
			if req.VoteType == "up" {
				comment.Upvotes--
			} else {
				comment.Downvotes--
			}

			database.DB.Save(&comment)

			return c.JSON(map[string]interface{}{
				"action":    "removed",
				"vote_type": req.VoteType,
				"upvotes":   comment.Upvotes,
				"downvotes": comment.Downvotes,
				"net_votes": comment.Upvotes - comment.Downvotes,
			})
		} else {
			// Different vote type, update vote
			oldVoteType := existingVote.VoteType
			existingVote.VoteType = req.VoteType
			database.DB.Save(&existingVote)

			// Update comment vote counts
			if oldVoteType == "up" {
				comment.Upvotes--
				comment.Downvotes++
			} else {
				comment.Downvotes--
				comment.Upvotes++
			}

			database.DB.Save(&comment)

			return c.JSON(map[string]interface{}{
				"action":    "updated",
				"vote_type": req.VoteType,
				"upvotes":   comment.Upvotes,
				"downvotes": comment.Downvotes,
				"net_votes": comment.Upvotes - comment.Downvotes,
			})
		}
	} else {
		// New vote
		vote := models.CommentVote{
			UserID:    userID,
			CommentID: commentID,
			VoteType:  req.VoteType,
		}

		if err := database.DB.Create(&vote).Error; err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
				Success: false,
				Error:   "Failed to record vote",
			})
		}

		// Update comment vote counts
		if req.VoteType == "up" {
			comment.Upvotes++
		} else {
			comment.Downvotes++
		}

		database.DB.Save(&comment)

		return c.JSON(map[string]interface{}{
			"action":    "added",
			"vote_type": req.VoteType,
			"upvotes":   comment.Upvotes,
			"downvotes": comment.Downvotes,
			"net_votes": comment.Upvotes - comment.Downvotes,
		})
	}
}

// GetUserComments retrieves comments by a specific user
// @Summary Get user's comments
// @Description Retrieve a paginated list of comments by a specific user
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Success 200 {object} CommentsListResponse "Comments retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/my [get]
func GetUserComments(c *fiber.Ctx) error {
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
	if err := database.DB.Model(&models.Comment{}).
		Where("user_id = ?", userID).
		Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count comments",
		})
	}

	// Get comments
	var comments []models.Comment
	if err := database.DB.
		Where("user_id = ?", userID).
		Preload("User").
		Preload("Product").
		Preload("Product.Category").
		Offset(offset).
		Limit(limit).
		Order("created_at DESC").
		Find(&comments).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve comments",
		})
	}

	// Enhance comments with vote information
	var commentResponses []CommentResponse
	for _, comment := range comments {
		response := CommentResponse{
			Comment:  comment,
			User:     comment.User,
			Product:  comment.Product,
			NetVotes: comment.Upvotes - comment.Downvotes,
		}

		// Get user's vote (should be null for own comments)
		var vote models.CommentVote
		if err := database.DB.Where("user_id = ? AND comment_id = ?", userID, comment.ID).
			First(&vote).Error; err == nil {
			response.UserVote = &vote.VoteType
		}

		commentResponses = append(commentResponses, response)
	}

	return c.JSON(CommentsListResponse{
		Comments: commentResponses,
		Total:    total,
		Page:     page,
		Limit:    limit,
	})
}

// GetCommentStats gets comment statistics
// @Summary Get comment statistics
// @Description Get statistics about comments (for admin or analytics)
// @Tags Comments
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} map[string]interface{} "Comment statistics"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /comments/stats [get]
func GetCommentStats(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Get total comments count
	var totalComments int64
	database.DB.Model(&models.Comment{}).Count(&totalComments)

	// Get user's comments count
	var userComments int64
	database.DB.Model(&models.Comment{}).Where("user_id = ?", userID).Count(&userComments)

	// Get recent comments count (last 7 days)
	var recentComments int64
	database.DB.Model(&models.Comment{}).
		Where("created_at > NOW() - INTERVAL '7 days'").
		Count(&recentComments)

	// Get top commented products
	type ProductCommentCount struct {
		ProductID   uuid.UUID `json:"product_id"`
		ProductName string    `json:"product_name"`
		Count       int64     `json:"count"`
	}

	var topProducts []ProductCommentCount
	database.DB.Table("comments").
		Select("products.id as product_id, products.name as product_name, COUNT(*) as count").
		Joins("JOIN products ON comments.product_id = products.id").
		Group("products.id, products.name").
		Order("count DESC").
		Limit(10).
		Scan(&topProducts)

	// Get sentiment distribution (if sentiment analysis is implemented)
	type SentimentCount struct {
		Sentiment string `json:"sentiment"`
		Count     int64  `json:"count"`
	}

	var sentimentStats []SentimentCount
	database.DB.Table("comments").
		Select("sentiment_label as sentiment, COUNT(*) as count").
		Where("sentiment_label IS NOT NULL").
		Group("sentiment_label").
		Scan(&sentimentStats)

	return c.JSON(map[string]interface{}{
		"total_comments":      totalComments,
		"user_comments":       userComments,
		"recent_comments":     recentComments,
		"top_products":        topProducts,
		"sentiment_breakdown": sentimentStats,
	})
}

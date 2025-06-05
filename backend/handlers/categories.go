package handlers

import (
	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type CreateCategoryRequest struct {
	Name string `json:"name" validate:"required,min=2,max=100" example:"Electronics"`
}

type UpdateCategoryRequest struct {
	Name string `json:"name" validate:"required,min=2,max=100" example:"Electronics"`
}

type CategoryResponse struct {
	models.Category
	ProductCount int64 `json:"product_count"`
}

type CategoriesListResponse struct {
	Categories []CategoryResponse `json:"categories"`
	Total      int64              `json:"total"`
	Page       int                `json:"page"`
	Limit      int                `json:"limit"`
}

// GetCategories retrieves all categories with pagination
// @Summary Get all categories
// @Description Retrieve a paginated list of all categories with product counts
// @Tags Categories
// @Accept json
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(10)
// @Success 200 {object} CategoriesListResponse "Categories retrieved successfully"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /categories [get]
func GetCategories(c *fiber.Ctx) error {
	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 10)
	offset := (page - 1) * limit

	var categories []models.Category
	var total int64

	// Get total count
	if err := database.DB.Model(&models.Category{}).Count(&total).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to count categories",
		})
	}

	// Get categories with pagination
	if err := database.DB.
		Offset(offset).
		Limit(limit).
		Order("name ASC").
		Find(&categories).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to retrieve categories",
		})
	}

	// Get product counts for each category
	var categoryResponses []CategoryResponse
	for _, category := range categories {
		var productCount int64
		database.DB.Model(&models.Product{}).Where("category_id = ?", category.ID).Count(&productCount)

		categoryResponses = append(categoryResponses, CategoryResponse{
			Category:     category,
			ProductCount: productCount,
		})
	}

	return c.JSON(CategoriesListResponse{
		Categories: categoryResponses,
		Total:      total,
		Page:       page,
		Limit:      limit,
	})
}

// GetCategory retrieves a specific category by ID
// @Summary Get category by ID
// @Description Retrieve a specific category with its product count
// @Tags Categories
// @Accept json
// @Produce json
// @Param id path string true "Category ID"
// @Success 200 {object} CategoryResponse "Category retrieved successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid category ID"
// @Failure 404 {object} StandardErrorResponse "Category not found"
// @Router /categories/{id} [get]
func GetCategory(c *fiber.Ctx) error {
	categoryID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid category ID",
		})
	}

	var category models.Category
	if err := database.DB.First(&category, categoryID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Category not found",
		})
	}

	// Get product count
	var productCount int64
	database.DB.Model(&models.Product{}).Where("category_id = ?", category.ID).Count(&productCount)

	return c.JSON(CategoryResponse{
		Category:     category,
		ProductCount: productCount,
	})
}

// CreateCategory creates a new category
// @Summary Create a new category
// @Description Create a new product category
// @Tags Categories
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body CreateCategoryRequest true "Category creation data"
// @Success 201 {object} models.Category "Category created successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 409 {object} StandardErrorResponse "Category with this name already exists"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /categories [post]
func CreateCategory(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	// Check if user is admin (optional - you might want to restrict category creation)
	var user models.User
	if err := database.DB.First(&user, userID).Error; err != nil {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not found",
		})
	}

	var req CreateCategoryRequest
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

	// Check if category with this name already exists
	var existingCategory models.Category
	if err := database.DB.Where("LOWER(name) = LOWER(?)", req.Name).First(&existingCategory).Error; err == nil {
		return c.Status(fiber.StatusConflict).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Category with this name already exists",
		})
	}

	category := models.Category{
		Name: req.Name,
	}

	if err := database.DB.Create(&category).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to create category",
		})
	}

	return c.Status(fiber.StatusCreated).JSON(category)
}

// UpdateCategory updates an existing category
// @Summary Update category
// @Description Update an existing category's information
// @Tags Categories
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Category ID"
// @Param request body UpdateCategoryRequest true "Category update data"
// @Success 200 {object} models.Category "Category updated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Category not found"
// @Failure 409 {object} StandardErrorResponse "Category with this name already exists"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /categories/{id} [put]
func UpdateCategory(c *fiber.Ctx) error {
	_, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	categoryID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid category ID",
		})
	}

	var req UpdateCategoryRequest
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

	var category models.Category
	if err := database.DB.First(&category, categoryID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Category not found",
		})
	}

	// Check if another category with this name already exists
	var existingCategory models.Category
	if err := database.DB.Where("LOWER(name) = LOWER(?) AND id != ?", req.Name, categoryID).First(&existingCategory).Error; err == nil {
		return c.Status(fiber.StatusConflict).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Category with this name already exists",
		})
	}

	category.Name = req.Name

	if err := database.DB.Save(&category).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to update category",
		})
	}

	return c.JSON(category)
}

// DeleteCategory deletes a category
// @Summary Delete category
// @Description Delete a category (only if it has no products)
// @Tags Categories
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Category ID"
// @Success 204 "Category deleted successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid category ID or category has products"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "Category not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /categories/{id} [delete]
func DeleteCategory(c *fiber.Ctx) error {
	_, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	categoryID, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid category ID",
		})
	}

	var category models.Category
	if err := database.DB.First(&category, categoryID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Category not found",
		})
	}

	// Check if category has products
	var productCount int64
	database.DB.Model(&models.Product{}).Where("category_id = ?", categoryID).Count(&productCount)
	if productCount > 0 {
		return c.Status(fiber.StatusBadRequest).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Cannot delete category with existing products",
		})
	}

	if err := database.DB.Delete(&category).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to delete category",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

package handlers

import (
	"time"

	"bachelor_backend/database"
	"bachelor_backend/middleware"
	"bachelor_backend/models"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type RegisterRequest struct {
	Email     string  `json:"email" validate:"required,email,max=255" example:"user@example.com"`
	Name      *string `json:"name" validate:"omitempty,min=2,max=100" example:"John Doe"`
	Password  string  `json:"password" validate:"required,min=8,max=128" example:"password123"`
	Region    *string `json:"region" validate:"omitempty,max=100" example:"US"`
	BirthYear *int    `json:"birth_year" validate:"omitempty,min=1900,max=2024" example:"1990"`
}

type LoginRequest struct {
	Email    string `json:"email" validate:"required,email,max=255" example:"user@example.com"`
	Password string `json:"password" validate:"required,min=1,max=128" example:"password123"`
}

type UpdateProfileRequest struct {
	Name      *string `json:"name" validate:"omitempty,min=2,max=100" example:"John Doe"`
	Region    *string `json:"region" validate:"omitempty,max=100" example:"US"`
	BirthYear *int    `json:"birth_year" validate:"omitempty,min=1900,max=2024" example:"1990"`
}

type AuthResponse struct {
	Token string      `json:"token" example:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."`
	User  models.User `json:"user"`
}

type UserProfileResponse struct {
	models.User
	Statistics UserProfileStatistics `json:"statistics"`
}

type UserProfileStatistics struct {
	TotalOrders             int64   `json:"total_orders"`
	TotalSpent              float64 `json:"total_spent"`
	AverageOrderValue       float64 `json:"average_order_value"`
	CartItemsCount          int64   `json:"cart_items_count"`
	CartTotalValue          float64 `json:"cart_total_value"`
	TotalInteractions       int64   `json:"total_interactions"`
	FavoriteCategory        string  `json:"favorite_category"`
	RecommendationsReceived int64   `json:"recommendations_received"`
	RecommendationsClicked  int64   `json:"recommendations_clicked"`
}

// Register handles user registration
// @Summary Register a new user
// @Description Create a new user account with email, name, and password
// @Tags Authentication
// @Accept json
// @Produce json
// @Param request body RegisterRequest true "Registration details"
// @Success 201 {object} AuthResponse "User registered successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 409 {object} StandardErrorResponse "User with this email already exists"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /auth/register [post]
func Register(c *fiber.Ctx) error {
	var req RegisterRequest

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

	var existingUser models.User
	if err := database.DB.Where("email = ?", req.Email).First(&existingUser).Error; err == nil {
		return c.Status(fiber.StatusConflict).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User with this email already exists",
		})
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to hash password",
		})
	}

	user := models.User{
		Email:        req.Email,
		Name:         req.Name,
		Region:       req.Region,
		BirthYear:    req.BirthYear,
		PasswordHash: string(hashedPassword),
	}

	if err := database.DB.Create(&user).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to create user",
		})
	}

	token, err := generateJWTToken(user)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to generate token",
		})
	}

	return c.Status(fiber.StatusCreated).JSON(AuthResponse{
		Token: token,
		User:  user,
	})
}

// Login handles user login
// @Summary User login
// @Description Authenticate user with email and password
// @Tags Authentication
// @Accept json
// @Produce json
// @Param request body LoginRequest true "Login credentials"
// @Success 200 {object} AuthResponse "Login successful"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "Invalid email or password"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /auth/login [post]
func Login(c *fiber.Ctx) error {
	var req LoginRequest

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

	var user models.User
	if err := database.DB.Where("email = ?", req.Email).First(&user).Error; err != nil {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid email or password",
		})
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Invalid email or password",
		})
	}

	token, err := generateJWTToken(user)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to generate token",
		})
	}

	return c.JSON(AuthResponse{
		Token: token,
		User:  user,
	})
}

// GetProfile returns the current user's profile
// @Summary Get user profile
// @Description Get the authenticated user's comprehensive profile information including statistics and recent activity
// @Tags Authentication
// @Accept json
// @Produce json
// @Security BearerAuth
// @Success 200 {object} UserProfileResponse "User profile retrieved successfully"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "User not found"
// @Router /auth/profile [get]
func GetProfile(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var user models.User
	if err := database.DB.
		First(&user, userID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not found",
		})
	}

	response := UserProfileResponse{
		User: user,
	}

	return c.JSON(response)
}

// UpdateProfile updates the current user's profile
// @Summary Update user profile
// @Description Update the authenticated user's profile information
// @Tags Authentication
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body UpdateProfileRequest true "Profile update data"
// @Success 200 {object} models.User "Profile updated successfully"
// @Failure 400 {object} StandardErrorResponse "Invalid request body or validation error"
// @Failure 401 {object} StandardErrorResponse "User not authenticated"
// @Failure 404 {object} StandardErrorResponse "User not found"
// @Failure 500 {object} StandardErrorResponse "Internal server error"
// @Router /auth/profile [put]
func UpdateProfile(c *fiber.Ctx) error {
	userID, ok := middleware.GetUserID(c)
	if !ok {
		return c.Status(fiber.StatusUnauthorized).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not authenticated",
		})
	}

	var req UpdateProfileRequest

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

	var user models.User
	if err := database.DB.First(&user, userID).Error; err != nil {
		return c.Status(fiber.StatusNotFound).JSON(StandardErrorResponse{
			Success: false,
			Error:   "User not found",
		})
	}

	if req.Name != nil {
		user.Name = req.Name
	}

	if req.Region != nil {
		user.Region = req.Region
	}

	if req.BirthYear != nil {
		user.BirthYear = req.BirthYear
	}

	if err := database.DB.Save(&user).Error; err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(StandardErrorResponse{
			Success: false,
			Error:   "Failed to update user",
		})
	}

	return c.JSON(user)
}

func generateJWTToken(user models.User) (string, error) {
	var userName string
	if user.Name != nil {
		userName = *user.Name
	}

	claims := middleware.JWTClaims{
		UserID: user.ID,
		Email:  user.Email,
		Name:   userName,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)

	secret := getJWTSecret()
	return token.SignedString([]byte(secret))
}

func getJWTSecret() string {
	return middleware.GetJWTSecret()
}

func init() {
}

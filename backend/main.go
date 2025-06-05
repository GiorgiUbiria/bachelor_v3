package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"bachelor_backend/database"
	_ "bachelor_backend/docs"
	"bachelor_backend/handlers"
	"bachelor_backend/middleware"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/helmet"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gofiber/fiber/v2/middleware/timeout"
	"github.com/gofiber/swagger"
)

// @title Bachelor E-commerce API
// @version 1.0
// @description ML-Powered E-commerce Platform API with advanced recommendation system, intelligent search, and comprehensive analytics
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.url http://www.swagger.io/support
// @contact.email support@bachelor-ecommerce.com

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host localhost:8080
// @BasePath /api/v1
// @schemes http https

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and JWT token.

// @tag.name Authentication
// @tag.description User authentication and profile management

// @tag.name Categories
// @tag.description Product category management

// @tag.name Products
// @tag.description Product management with advanced search and ML suggestions

// @tag.name Comments
// @tag.description Product reviews and comments with voting system

// @tag.name Favorites
// @tag.description User favorites management

// @tag.name Recommendations
// @tag.description ML-powered personalized product recommendations

// @tag.name Product Suggestions
// @tag.description ML-generated product optimization suggestions
func main() {
	// Parse command line flags
	var seedDB = flag.Bool("seed", false, "Seed the database with test data")
	var clearDB = flag.Bool("clear", false, "Clear all data from the database")
	flag.Parse()

	database.Connect()

	log.Println("Database migration completed successfully")

	// Handle database seeding/clearing
	if *clearDB {
		log.Println("🧹 Clearing database as requested...")
		if err := database.ClearDatabase(); err != nil {
			log.Fatal("Failed to clear database:", err)
		}
		log.Println("✅ Database cleared successfully")
		return
	}

	if *seedDB {
		log.Println("🌱 Seeding database as requested...")
		if err := database.SeedDatabase(); err != nil {
			log.Fatal("Failed to seed database:", err)
		}
		log.Println("✅ Database seeded successfully")
		return
	}

	app := fiber.New(fiber.Config{
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}
			return c.Status(code).JSON(fiber.Map{
				"success": false,
				"error":   err.Error(),
			})
		},
		BodyLimit:             4 * 1024 * 1024,
		ReadTimeout:           10 * time.Second,
		WriteTimeout:          10 * time.Second,
		IdleTimeout:           120 * time.Second,
		DisableStartupMessage: false,
		ServerHeader:          "",
		AppName:               "Bachelor Backend API v1.0",
	})

	app.Get("/swagger/*", swagger.New(swagger.Config{
		URL:          "/swagger/doc.json",
		DeepLinking:  true,
		DocExpansion: "list",
		OAuth: &swagger.OAuthConfig{
			AppName:  "Bachelor E-commerce API",
			ClientId: "bachelor-api-client",
		},
		OAuth2RedirectUrl: "http://localhost:8080/swagger/oauth2-redirect.html",
	}))

	app.Use(helmet.New(helmet.Config{
		XSSProtection:         "1; mode=block",
		ContentTypeNosniff:    "nosniff",
		XFrameOptions:         "DENY",
		HSTSMaxAge:            31536000,
		HSTSExcludeSubdomains: false,
		HSTSPreloadEnabled:    true,
		ContentSecurityPolicy: "default-src 'self'",
		ReferrerPolicy:        "strict-origin-when-cross-origin",
	}))

	app.Use(recover.New())

	app.Use(logger.New(logger.Config{
		Format:     "[${time}] ${status} - ${method} ${path} - ${ip} - ${latency}\n",
		TimeFormat: "2006-01-02 15:04:05",
		TimeZone:   "UTC",
	}))

	app.Use(limiter.New(limiter.Config{
		Max:        100,
		Expiration: 1 * time.Minute,
		KeyGenerator: func(c *fiber.Ctx) string {
			return c.IP()
		},
		LimitReached: func(c *fiber.Ctx) error {
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{
				"success": false,
				"error":   "Rate limit exceeded. Please try again later.",
			})
		},
	}))

	app.Use(cors.New(cors.Config{
		AllowMethods:     "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders:     "Origin,Content-Type,Accept,Authorization,X-Session-ID,X-Requested-With",
		AllowCredentials: true,
		MaxAge:           86400,
		AllowOriginsFunc: func(origin string) bool {
			log.Printf("CORS: Checking origin: %s", origin)

			allowedPatterns := []string{
				"http://localhost:",
				"http://127.0.0.1:",
				"https://localhost:",
				"https://127.0.0.1:",
			}

			for _, pattern := range allowedPatterns {
				if len(origin) > len(pattern) && origin[:len(pattern)] == pattern {
					log.Printf("CORS: Allowed origin (pattern match): %s", origin)
					return true
				}
			}

			allowedOrigins := getEnv("ALLOWED_ORIGINS", "")
			if allowedOrigins != "" {
				origins := strings.Split(allowedOrigins, ",")
				for _, allowedOrigin := range origins {
					if strings.TrimSpace(allowedOrigin) == origin {
						log.Printf("CORS: Allowed origin (env match): %s", origin)
						return true
					}
				}
			}

			log.Printf("CORS: Rejected origin: %s", origin)
			return false
		},
	}))

	app.Use("/api", timeout.New(func(c *fiber.Ctx) error {
		return c.Next()
	}, 30*time.Second))

	// Add security analysis middleware for all API requests
	app.Use("/api", middleware.SecurityAnalysisMiddleware())

	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status":    "ok",
			"service":   "bachelor_backend",
			"version":   "1.0.0",
			"timestamp": time.Now().UTC(),
		})
	})

	app.All("/cors-test", func(c *fiber.Ctx) error {
		origin := c.Get("Origin")
		method := c.Method()
		log.Printf("CORS Test: Method=%s, Origin=%s", method, origin)

		return c.JSON(fiber.Map{
			"message": "CORS test successful",
			"origin":  origin,
			"method":  method,
			"headers": c.GetReqHeaders(),
		})
	})

	api := app.Group("/api/v1")

	// Authentication routes
	auth := api.Group("/auth")
	auth.Use(limiter.New(limiter.Config{
		Max:        10,
		Expiration: 1 * time.Minute,
		KeyGenerator: func(c *fiber.Ctx) string {
			return c.IP()
		},
		LimitReached: func(c *fiber.Ctx) error {
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{
				"success": false,
				"error":   "Too many authentication attempts. Please try again later.",
			})
		},
	}))

	auth.Post("/register", handlers.Register)
	auth.Post("/login", handlers.Login)
	auth.Get("/profile", middleware.AuthRequired(), handlers.GetProfile)
	auth.Put("/profile", middleware.AuthRequired(), handlers.UpdateProfile)

	// Categories routes
	categories := api.Group("/categories")
	categories.Get("/", handlers.GetCategories)
	categories.Get("/:id", handlers.GetCategory)
	categories.Post("/", middleware.AuthRequired(), handlers.CreateCategory)
	categories.Put("/:id", middleware.AuthRequired(), handlers.UpdateCategory)
	categories.Delete("/:id", middleware.AuthRequired(), handlers.DeleteCategory)

	// Products routes
	products := api.Group("/products")
	products.Get("/", handlers.GetProducts)
	products.Get("/:id", middleware.OptionalAuth(), handlers.GetProduct)
	products.Post("/", middleware.AuthRequired(), handlers.CreateProduct)
	products.Put("/:id", middleware.AuthRequired(), handlers.UpdateProduct)
	products.Delete("/:id", middleware.AuthRequired(), handlers.DeleteProduct)
	products.Post("/search", handlers.SearchProducts)
	products.Post("/suggest", handlers.SuggestProduct) // On-demand product automation endpoint

	// Comments routes
	comments := api.Group("/comments")
	comments.Get("/product/:product_id", middleware.OptionalAuth(), handlers.GetProductComments)
	comments.Post("/", middleware.AuthRequired(), handlers.CreateComment)
	comments.Put("/:id", middleware.AuthRequired(), handlers.UpdateComment)
	comments.Delete("/:id", middleware.AuthRequired(), handlers.DeleteComment)
	comments.Post("/:id/vote", middleware.AuthRequired(), handlers.VoteComment)
	comments.Get("/my", middleware.AuthRequired(), handlers.GetUserComments)
	comments.Get("/stats", middleware.AuthRequired(), handlers.GetCommentStats)

	// Favorites routes
	favorites := api.Group("/favorites")
	favorites.Get("/", middleware.AuthRequired(), handlers.GetUserFavorites)
	favorites.Post("/", middleware.AuthRequired(), handlers.AddFavorite)
	favorites.Delete("/:product_id", middleware.AuthRequired(), handlers.RemoveFavorite)
	favorites.Get("/:product_id/check", middleware.AuthRequired(), handlers.CheckFavorite)
	favorites.Get("/stats", middleware.AuthRequired(), handlers.GetFavoriteStats)
	favorites.Post("/:product_id/toggle", middleware.AuthRequired(), handlers.ToggleFavorite)

	// Recommendations routes (Enhanced with deals)
	recommendations := api.Group("/recommendations")
	recommendations.Get("/", middleware.AuthRequired(), handlers.GetUserRecommendations)
	recommendations.Post("/generate", middleware.AuthRequired(), handlers.GenerateRecommendations)
	recommendations.Get("/stored", middleware.AuthRequired(), handlers.GetStoredRecommendations)
	recommendations.Get("/stats", middleware.AuthRequired(), handlers.GetRecommendationStats)

	// Enhanced recommendation routes (Multi-stage flow from Description-3.md)
	recommendations.Get("/enhanced", middleware.AuthRequired(), handlers.GetEnhancedRecommendations)
	recommendations.Get("/deals", middleware.AuthRequired(), handlers.GetPersonalizedDeals)
	recommendations.Post("/batch-process", middleware.AuthRequired(), handlers.TriggerBatchProcessing)
	recommendations.Get("/segments", middleware.AuthRequired(), handlers.GetUserSegments)

	// Product suggestions routes
	suggestions := api.Group("/product-suggestions")
	suggestions.Get("/", handlers.GetProductSuggestions)
	suggestions.Get("/:id", handlers.GetProductSuggestion)
	suggestions.Post("/", middleware.AuthRequired(), handlers.CreateProductSuggestion)
	suggestions.Put("/:id", middleware.AuthRequired(), handlers.UpdateProductSuggestion)
	suggestions.Delete("/:id", middleware.AuthRequired(), handlers.DeleteProductSuggestion)
	suggestions.Get("/stats", handlers.GetSuggestionStats)
	suggestions.Post("/generate/:id", middleware.AuthRequired(), handlers.GenerateSuggestionsForProduct)

	app.Use(func(c *fiber.Ctx) error {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"success": false,
			"error":   "Route not found",
		})
	})

	port := getEnv("PORT", "8080")

	log.Printf("🚀 Server starting on port %s", port)
	log.Printf("🔒 Security features enabled: Helmet, Rate Limiting, CORS, Security Analysis")
	log.Printf("📊 Request timeout: 30s, Body limit: 4MB")
	log.Printf("🤖 ML-powered features: Security Analysis, Recommendations, Product Automation")
	log.Fatal(app.Listen(":" + port))
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

package middleware

import (
	"bachelor_backend/database"
	"bachelor_backend/models"
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

type JWTClaims struct {
	UserID uuid.UUID `json:"user_id"`
	Email  string    `json:"email"`
	Name   string    `json:"name"`
	jwt.RegisteredClaims
}

func SecurityAnalysisMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		startTime := time.Now()

		// Skip analysis for health checks and static files
		path := c.Path()
		if strings.Contains(path, "/health") || strings.Contains(path, "/swagger") || strings.Contains(path, "/cors-test") {
			return c.Next()
		}

		// Get user ID if authenticated
		var userID *uuid.UUID
		if id, ok := GetUserID(c); ok {
			userID = &id
		}

		// Capture request data
		method := c.Method()
		userAgent := c.Get("User-Agent")
		ipAddress := c.IP()
		referrer := c.Get("Referer")

		// Get headers
		headers := make(map[string]string)
		c.Request().Header.VisitAll(func(key, value []byte) {
			headers[string(key)] = string(value)
		})

		// Get query parameters
		queryParams := make(map[string]string)
		c.Request().URI().QueryArgs().VisitAll(func(key, value []byte) {
			queryParams[string(key)] = string(value)
		})

		// Get body content (for POST/PUT requests)
		var body string
		if method == "POST" || method == "PUT" || method == "PATCH" {
			bodyBytes := c.Body()
			if len(bodyBytes) > 0 && len(bodyBytes) < 10000 { // Limit body size for analysis
				body = string(bodyBytes)
			}
		}

		// Continue with request processing
		err := c.Next()

		// Calculate duration and get status code
		duration := time.Since(startTime)
		statusCode := c.Response().StatusCode()

		// Log request to database
		go logHTTPRequestWithAnalysis(userID, method, path, userAgent, ipAddress, referrer, headers, queryParams, body, int(duration.Milliseconds()), statusCode)

		return err
	}
}

func logHTTPRequestWithAnalysis(userID *uuid.UUID, method, path, userAgent, ipAddress, referrer string, headers, queryParams map[string]string, body string, durationMs, statusCode int) {
	// Create HTTP request log entry
	requestLog := models.HttpRequestLog{
		UserID:     userID,
		IPAddress:  &ipAddress,
		UserAgent:  &userAgent,
		Path:       &path,
		Method:     &method,
		DurationMs: &durationMs,
		StatusCode: &statusCode,
		Referrer:   &referrer,
	}

	// Prepare comprehensive data for ML analysis
	requestData := map[string]interface{}{
		"path":         path,
		"method":       method,
		"user_agent":   userAgent,
		"ip_address":   ipAddress,
		"headers":      headers,
		"query_params": queryParams,
		"body":         body,
		"referrer":     referrer,
	}

	// Perform security analysis via ML service
	analysis, err := analyzeSecurityThreat(requestData)
	if err != nil {
		log.Printf("⚠️  Security analysis failed for %s %s: %v", method, path, err)
		// Continue with basic logging even if analysis fails
	} else {
		// Extract analysis results with proper type checking
		if attackScore, ok := analysis["attack_score"].(float64); ok {
			requestLog.AttackScore = &attackScore
		}

		if attackType, ok := analysis["suspected_attack_type"].(string); ok && attackType != "" {
			requestLog.SuspectedAttackType = &attackType
		}

		// Enhanced security alerting with request ID
		requestID := ""
		if id, ok := analysis["request_id"].(string); ok {
			requestID = id
		}

		if requestLog.AttackScore != nil && *requestLog.AttackScore > 0.7 {
			log.Printf("🚨 HIGH-RISK SECURITY THREAT DETECTED [%s]", requestID)
			log.Printf("   Score: %.3f | Type: %s | Path: %s | IP: %s",
				*requestLog.AttackScore,
				getStringValue(requestLog.SuspectedAttackType),
				path,
				ipAddress)

			// Log recommendations if available
			if recommendations, ok := analysis["recommendations"].([]interface{}); ok && len(recommendations) > 0 {
				log.Printf("   Recommendations: %v", recommendations[0])
			}
		} else if requestLog.AttackScore != nil && *requestLog.AttackScore > 0.4 {
			log.Printf("⚠️  MEDIUM-RISK SECURITY ALERT [%s]: Score=%.3f, Type=%s, Path=%s, IP=%s",
				requestID,
				*requestLog.AttackScore,
				getStringValue(requestLog.SuspectedAttackType),
				path,
				ipAddress)
		}

		// Log detailed analysis for debugging (only in development)
		if os.Getenv("ENVIRONMENT") == "development" && requestLog.AttackScore != nil && *requestLog.AttackScore > 0.3 {
			if details, ok := analysis["details"].(map[string]interface{}); ok {
				if classification, ok := details["classification"].(map[string]interface{}); ok {
					log.Printf("   Analysis Details: %+v", classification)
				}
			}
		}
	}

	// Save to database with error handling
	if err := database.DB.Create(&requestLog).Error; err != nil {
		log.Printf("❌ Failed to log HTTP request to database: %v", err)
	}
}

func analyzeSecurityThreat(requestData map[string]interface{}) (map[string]interface{}, error) {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://localhost:8000" // Default ML service URL
	}

	// Prepare request payload with comprehensive data
	payload := map[string]interface{}{
		"path":       requestData["path"],
		"method":     requestData["method"],
		"user_agent": requestData["user_agent"],
		"ip_address": requestData["ip_address"],
		"headers":    requestData["headers"],
		"body":       requestData["body"],
		"referrer":   requestData["referrer"],
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request data: %w", err)
	}

	// Send to ML service with timeout
	client := &http.Client{
		Timeout: 5 * time.Second, // 5 second timeout for security analysis
	}

	resp, err := client.Post(
		mlServiceURL+"/security/analyze?log_to_db=false", // Don't double-log
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode ML service response: %w", err)
	}

	// Extract relevant fields for database logging
	analysisResult := map[string]interface{}{
		"attack_score":          result["attack_score"],
		"suspected_attack_type": result["suspected_attack_type"],
		"confidence":            result["confidence"],
		"request_id":            result["request_id"],
		"recommendations":       result["recommendations"],
	}

	// Add detailed analysis if available
	if details, ok := result["details"].(map[string]interface{}); ok {
		analysisResult["details"] = details
	}

	return analysisResult, nil
}

// getStringValue safely gets string value from pointer
func getStringValue(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func AuthRequired() fiber.Handler {
	return func(c *fiber.Ctx) error {
		authHeader := c.Get("Authorization")
		if authHeader == "" {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"success": false,
				"error":   "Authorization header required",
			})
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString == authHeader {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"success": false,
				"error":   "Bearer token required",
			})
		}

		token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(getJWTSecret()), nil
		})

		if err != nil {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"success": false,
				"error":   "Invalid token",
			})
		}

		if claims, ok := token.Claims.(*JWTClaims); ok && token.Valid {
			c.Locals("user_id", claims.UserID)
			c.Locals("user_email", claims.Email)
			c.Locals("user_name", claims.Name)
			return c.Next()
		}

		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"success": false,
			"error":   "Invalid token claims",
		})
	}
}

func OptionalAuth() fiber.Handler {
	return func(c *fiber.Ctx) error {
		authHeader := c.Get("Authorization")
		if authHeader == "" {
			return c.Next()
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString == authHeader {
			return c.Next()
		}

		token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(getJWTSecret()), nil
		})

		if err == nil {
			if claims, ok := token.Claims.(*JWTClaims); ok && token.Valid {
				c.Locals("user_id", claims.UserID)
				c.Locals("user_email", claims.Email)
				c.Locals("user_name", claims.Name)
			}
		}

		return c.Next()
	}
}

func GetUserID(c *fiber.Ctx) (uuid.UUID, bool) {
	userID := c.Locals("user_id")
	if userID == nil {
		return uuid.Nil, false
	}

	if id, ok := userID.(uuid.UUID); ok {
		return id, true
	}

	return uuid.Nil, false
}

func GetUserEmail(c *fiber.Ctx) (string, bool) {
	email := c.Locals("user_email")
	if email == nil {
		return "", false
	}

	if emailStr, ok := email.(string); ok {
		return emailStr, true
	}

	return "", false
}

func GetUserName(c *fiber.Ctx) (string, bool) {
	name := c.Locals("user_name")
	if name == nil {
		return "", false
	}

	if nameStr, ok := name.(string); ok {
		return nameStr, true
	}

	return "", false
}

func GetUserNameSafe(c *fiber.Ctx, defaultName string) string {
	if name, ok := GetUserName(c); ok {
		return name
	}
	return defaultName
}

func getJWTSecret() string {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		log.Println("WARNING: JWT_SECRET not set, using default secret. This is insecure for production!")
		return "your-super-secret-jwt-key-change-this-in-production"
	}
	return secret
}

func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	for i := range b {
		b[i] = charset[b[i]%byte(len(charset))]
	}
	return string(b)
}

func GetJWTSecret() string {
	return getJWTSecret()
}

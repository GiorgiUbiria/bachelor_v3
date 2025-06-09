package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"bachelor_backend/models"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Attack patterns for security ML training
var xssAttacks = []string{
	"<script>alert('XSS')</script>",
	"<img src=x onerror=alert('XSS')>",
	"javascript:alert(document.cookie)",
	"<iframe src=javascript:alert('XSS')></iframe>",
	"<svg onload=alert('XSS')>",
}

var sqlInjectionAttacks = []string{
	"' OR '1'='1",
	"'; DROP TABLE users; --",
	"' UNION SELECT username, password FROM admin_users --",
	"admin'/**/--",
	"' OR 1=1 /*",
}

var csrfAttacks = []string{
	"<form action='http://bank.com/transfer' method='post'>",
	"<img src='http://admin.site.com/delete_user?id=123'>",
	"<iframe src='http://site.com/admin/change_password'>",
}

// SeedHttpRequestLogs creates realistic HTTP logs with security attack patterns
func SeedHttpRequestLogs(ctx context.Context, pool *pgxpool.Pool, users []models.User) error {
	log.Printf("🔒 Seeding %d HTTP request logs for security analysis...", ML_HTTP_LOGS_COUNT)

	logs := make([][]interface{}, 0, ML_HTTP_LOGS_COUNT)

	normalPaths := []string{
		"/api/v1/products", "/api/v1/products/search", "/api/v1/categories",
		"/api/v1/users/profile", "/api/v1/auth/login", "/api/v1/auth/register",
		"/api/v1/favorites", "/api/v1/comments", "/health", "/api/v1/cart",
	}

	methods := []string{"GET", "POST", "PUT", "DELETE", "PATCH"}
	userAgents := []string{
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
	}

	for i := 0; i < ML_HTTP_LOGS_COUNT; i++ {
		id := uuid.New()
		var userID *uuid.UUID
		var ipAddress string
		var path string
		var method string
		var suspectedAttackType string
		var isMalicious bool
		var confidenceScore *float64
		var attackScore *float64

		// 85% normal traffic, 15% malicious
		if gofakeit.Float64() < 0.85 {
			// Normal traffic
			if gofakeit.Bool() && len(users) > 0 {
				selectedUser := users[gofakeit.IntRange(0, len(users)-1)]
				userID = &selectedUser.ID
			}
			ipAddress = gofakeit.IPv4Address()
			path = normalPaths[gofakeit.IntRange(0, len(normalPaths)-1)]
			method = methods[gofakeit.IntRange(0, len(methods)-1)]
			suspectedAttackType = "benign"
			isMalicious = false
			score := gofakeit.Float64Range(0.1, 0.3)
			confidenceScore = &score
		} else {
			// Malicious traffic
			ipAddress = gofakeit.IPv4Address()

			// Choose attack type
			attackRoll := gofakeit.Float64()
			switch {
			case attackRoll < 0.4: // 40% XSS
				suspectedAttackType = "xss"
				attack := xssAttacks[gofakeit.IntRange(0, len(xssAttacks)-1)]
				path = fmt.Sprintf("/search?q=%s", attack)
				method = "GET"
			case attackRoll < 0.7: // 30% SQL Injection
				suspectedAttackType = "sqli"
				path = "/api/v1/auth/login"
				method = "POST"
			default: // 30% CSRF
				suspectedAttackType = "csrf"
				path = "/api/v1/transfer"
				method = "POST"
			}
			isMalicious = true
			score := gofakeit.Float64Range(0.7, 0.95)
			confidenceScore = &score
			attackScore = &score
		}

		userAgent := userAgents[gofakeit.IntRange(0, len(userAgents)-1)]
		timestamp := gofakeit.DateRange(time.Now().AddDate(0, 0, -30), time.Now())
		statusCode := 200
		durationMs := gofakeit.IntRange(10, 500)

		// Create realistic headers
		headers := map[string]interface{}{
			"User-Agent":    userAgent,
			"Accept":        "application/json",
			"Content-Type":  "application/json",
			"Authorization": "Bearer " + gofakeit.UUID(),
		}

		headersJSON, _ := json.Marshal(headers)

		logs = append(logs, []interface{}{
			id,
			userID,
			ipAddress,
			userAgent,
			path,
			method,
			timestamp,
			durationMs,
			statusCode,
			nil, // referrer
			suspectedAttackType,
			attackScore,
			nil,         // query_params
			headersJSON, // headers
			nil,         // body
			nil,         // cookies
			confidenceScore,
			isMalicious,
			nil, // pattern_matches
			nil, // ml_prediction
			nil, // ensemble_weights
		})
	}

	// Use pgx.CopyFrom for maximum performance
	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"http_request_logs"},
		[]string{"id", "user_id", "ip_address", "user_agent", "path", "method", "timestamp",
			"duration_ms", "status_code", "referrer", "suspected_attack_type", "attack_score",
			"query_params", "headers", "body", "cookies", "confidence_score", "is_malicious",
			"pattern_matches", "ml_prediction", "ensemble_weights"},
		pgx.CopyFromRows(logs),
	)

	if err != nil {
		return fmt.Errorf("failed to copy HTTP logs: %w", err)
	}

	log.Printf("✅ Successfully created %d HTTP logs (85%% normal, 15%% attacks)", ML_HTTP_LOGS_COUNT)
	return nil
}

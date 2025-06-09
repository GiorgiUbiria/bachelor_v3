package database

import (
	"context"
	"fmt"
	"log"
	"time"

	"bachelor_backend/models"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Enhanced user demographics for ML training
var enhancedRegions = []string{
	"United States", "Canada", "United Kingdom", "Germany", "France", "Japan",
	"China", "India", "Brazil", "Australia", "Mexico", "Italy", "Spain", "Russia",
	"Netherlands", "Sweden", "Norway", "Denmark", "South Korea", "Thailand",
}

// SeedUsers creates realistic user data with diverse demographics
func SeedUsers(ctx context.Context, pool *pgxpool.Pool, count int) ([]models.User, error) {
	log.Printf("👥 Seeding %d users with realistic demographics...", count)

	users := make([][]interface{}, 0, count)
	userModels := make([]models.User, 0, count)

	for i := 0; i < count; i++ {
		id := uuid.New()

		// Realistic age distribution
		var birthYear int
		ageCategory := gofakeit.Float64()
		switch {
		case ageCategory < 0.15: // 15% teens
			birthYear = gofakeit.IntRange(2006, 2011)
		case ageCategory < 0.35: // 20% young adults
			birthYear = gofakeit.IntRange(1999, 2005)
		case ageCategory < 0.65: // 30% adults
			birthYear = gofakeit.IntRange(1989, 1998)
		case ageCategory < 0.85: // 20% middle-aged
			birthYear = gofakeit.IntRange(1974, 1988)
		default: // 15% seniors
			birthYear = gofakeit.IntRange(1944, 1973)
		}

		name := gofakeit.Name()
		email := gofakeit.Email()
		region := enhancedRegions[gofakeit.IntRange(0, len(enhancedRegions)-1)]
		passwordHash := "$2a$10$" + gofakeit.LetterN(53) // Realistic bcrypt hash length
		isAdmin := i < 20                                // First 20 users are admins
		createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -400), time.Now())

		userModel := models.User{
			ID:           id,
			Email:        email,
			PasswordHash: passwordHash,
			Name:         &name,
			Region:       &region,
			BirthYear:    &birthYear,
			CreatedAt:    createdAt,
			IsAdmin:      isAdmin,
		}

		userModels = append(userModels, userModel)

		users = append(users, []interface{}{
			id,
			email,
			passwordHash,
			name,
			region,
			birthYear,
			createdAt,
			isAdmin,
		})
	}

	// Use pgx.CopyFrom for maximum performance
	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"users"},
		[]string{"id", "email", "password_hash", "name", "region", "birth_year", "created_at", "is_admin"},
		pgx.CopyFromRows(users),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to copy users: %w", err)
	}

	log.Printf("✅ Successfully created %d users with diverse demographics", count)
	return userModels, nil
}

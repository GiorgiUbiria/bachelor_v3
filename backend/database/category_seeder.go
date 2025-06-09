package database

import (
	"context"
	"fmt"
	"log"

	"bachelor_backend/models"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Enhanced category data for realistic ML training
var categoryNames = []string{
	"Electronics",
	"Gaming & Computers",
	"Sports & Fitness",
	"Fashion & Clothing",
	"Beauty & Cosmetics",
	"Books & Media",
	"Home & Kitchen",
	"Health & Wellness",
	"Tools & Hardware",
	"Automotive",
	"Outdoor Recreation",
	"Garden & Outdoor",
}

// SeedCategories creates comprehensive product categories
func SeedCategories(ctx context.Context, pool *pgxpool.Pool, count int) ([]models.Category, error) {
	log.Printf("📂 Seeding %d categories...", len(categoryNames))

	categories := make([][]interface{}, 0, len(categoryNames))
	categoryModels := make([]models.Category, 0, len(categoryNames))

	for i, name := range categoryNames {
		id := uuid.New()

		categoryModel := models.Category{
			ID:   id,
			Name: name,
		}

		categoryModels = append(categoryModels, categoryModel)
		categories = append(categories, []interface{}{
			id,
			name,
		})

		if i >= count-1 {
			break
		}
	}

	// Use pgx.CopyFrom for performance
	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"categories"},
		[]string{"id", "name"},
		pgx.CopyFromRows(categories),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to copy categories: %w", err)
	}

	log.Printf("✅ Successfully created %d categories", len(categoryModels))
	return categoryModels, nil
}

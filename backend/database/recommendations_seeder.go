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

// SeedRecommendations creates ML-generated recommendation data
func SeedRecommendations(ctx context.Context, pool *pgxpool.Pool, users []models.User, products []models.Product) error {
	log.Printf("💡 Seeding recommendations for ML training...")

	recommendations := make([][]interface{}, 0)
	totalRecommendations := 0

	modelVersions := []string{"v1.0", "v1.1", "v2.0", "v2.1"}
	reasons := []string{
		"Based on your purchase history",
		"Popular in your region",
		"Frequently bought together",
		"Similar to your favorites",
		"Trending in your age group",
		"Recommended by users like you",
	}

	for _, user := range users {
		// Each user gets 20-60 recommendations
		numRecommendations := gofakeit.IntRange(20, ML_RECOMMENDATIONS_PER_USER*2)
		recommendedProducts := make(map[uuid.UUID]bool)

		for i := 0; i < numRecommendations && i < len(products); i++ {
			// Select random product, avoiding duplicates
			var selectedProduct models.Product
			attempts := 0
			for attempts < 10 {
				selectedProduct = products[gofakeit.IntRange(0, len(products)-1)]
				if !recommendedProducts[selectedProduct.ID] {
					break
				}
				attempts++
			}

			if recommendedProducts[selectedProduct.ID] {
				continue
			}
			recommendedProducts[selectedProduct.ID] = true

			id := uuid.New()
			modelVersion := modelVersions[gofakeit.IntRange(0, len(modelVersions)-1)]
			reason := reasons[gofakeit.IntRange(0, len(reasons)-1)]

			// Generate recommendation flags based on reason
			regionBased := reason == "Popular in your region"
			ageBased := reason == "Trending in your age group"
			basedOnFavorites := reason == "Similar to your favorites"
			basedOnComments := reason == "Recommended by users like you"

			createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -90), time.Now())

			recommendations = append(recommendations, []interface{}{
				id,
				user.ID,
				selectedProduct.ID,
				reason,
				modelVersion,
				regionBased,
				ageBased,
				basedOnFavorites,
				basedOnComments,
				createdAt,
			})

			totalRecommendations++
		}

		// Insert in batches of 1000
		if len(recommendations) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"recommendations"},
				[]string{"id", "user_id", "product_id", "reason", "model_version",
					"region_based", "age_based", "based_on_favorites", "based_on_comments", "created_at"},
				pgx.CopyFromRows(recommendations),
			)
			if err != nil {
				return fmt.Errorf("failed to copy recommendations batch: %w", err)
			}
			recommendations = recommendations[:0]
		}
	}

	// Insert remaining recommendations
	if len(recommendations) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"recommendations"},
			[]string{"id", "user_id", "product_id", "reason", "model_version",
				"region_based", "age_based", "based_on_favorites", "based_on_comments", "created_at"},
			pgx.CopyFromRows(recommendations),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final recommendations batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d ML recommendations", totalRecommendations)
	return nil
}

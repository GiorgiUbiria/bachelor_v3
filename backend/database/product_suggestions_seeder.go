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
	"github.com/lib/pq"
)

// SeedProductSuggestions creates ML-generated product improvement suggestions
func SeedProductSuggestions(ctx context.Context, pool *pgxpool.Pool, products []models.Product) error {
	log.Printf("🧠 Seeding product suggestions...")

	suggestions := make([][]interface{}, 0)
	totalSuggestions := 0

	modelVersions := []string{"v1.0", "v1.5", "v2.0", "v2.1"}
	reasons := []string{
		"Market analysis suggests price optimization",
		"User feedback indicates missing features",
		"Trending tags in category",
		"Competitor analysis recommendations",
		"Seasonal demand patterns",
	}

	suggestedTags := []string{
		"eco-friendly", "premium", "bestseller", "limited-edition",
		"professional", "budget-friendly", "innovative", "trending",
		"handcrafted", "organic", "smart", "wireless",
	}

	// Select random subset of products for suggestions
	selectedProducts := make([]models.Product, 0, ML_PRODUCT_SUGGESTIONS_COUNT)
	for i := 0; i < ML_PRODUCT_SUGGESTIONS_COUNT && i < len(products); i++ {
		selectedProducts = append(selectedProducts, products[gofakeit.IntRange(0, len(products)-1)])
	}

	for _, product := range selectedProducts {
		id := uuid.New()
		modelVersion := modelVersions[gofakeit.IntRange(0, len(modelVersions)-1)]
		reason := reasons[gofakeit.IntRange(0, len(reasons)-1)]

		// Generate price suggestions
		var suggestedPriceMin, suggestedPriceMax *float64
		if product.Price != nil {
			currentPrice := *product.Price
			minPrice := currentPrice * gofakeit.Float64Range(0.8, 0.95)
			maxPrice := currentPrice * gofakeit.Float64Range(1.05, 1.3)
			suggestedPriceMin = &minPrice
			suggestedPriceMax = &maxPrice
		}

		// Generate tag suggestions
		numTagSuggestions := gofakeit.IntRange(1, 4)
		selectedTags := make([]string, 0, numTagSuggestions)
		for i := 0; i < numTagSuggestions; i++ {
			tag := suggestedTags[gofakeit.IntRange(0, len(suggestedTags)-1)]
			// Avoid duplicates
			exists := false
			for _, existingTag := range selectedTags {
				if existingTag == tag {
					exists = true
					break
				}
			}
			if !exists {
				selectedTags = append(selectedTags, tag)
			}
		}

		generatedAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -60), time.Now())

		suggestions = append(suggestions, []interface{}{
			id,
			product.ID,
			suggestedPriceMin,
			suggestedPriceMax,
			pq.Array(selectedTags),
			modelVersion,
			reason,
			generatedAt,
		})

		totalSuggestions++

		// Insert in batches of 1000
		if len(suggestions) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"product_suggestions"},
				[]string{"id", "product_id", "suggested_price_min", "suggested_price_max",
					"suggested_tags", "model_version", "reason", "generated_at"},
				pgx.CopyFromRows(suggestions),
			)
			if err != nil {
				return fmt.Errorf("failed to copy suggestions batch: %w", err)
			}
			suggestions = suggestions[:0]
		}
	}

	// Insert remaining suggestions
	if len(suggestions) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"product_suggestions"},
			[]string{"id", "product_id", "suggested_price_min", "suggested_price_max",
				"suggested_tags", "model_version", "reason", "generated_at"},
			pgx.CopyFromRows(suggestions),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final suggestions batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d product suggestions", totalSuggestions)
	return nil
}

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

// SeedProductSimilarityData creates product similarity relationships
func SeedProductSimilarityData(ctx context.Context, pool *pgxpool.Pool, products []models.Product) error {
	log.Printf("🔗 Seeding product similarity data...")

	similarities := make([][]interface{}, 0)
	similarityMap := make(map[string]bool)
	totalSimilarities := 0

	basedOnOptions := []string{"tags", "category", "price_range", "user_behavior", "ml_features"}

	for _, product1 := range products {
		if totalSimilarities >= ML_SIMILARITY_DATA_COUNT {
			break
		}

		// Each product gets 3-10 similar products
		numSimilar := gofakeit.IntRange(3, 10)

		for j := 0; j < numSimilar && totalSimilarities < ML_SIMILARITY_DATA_COUNT; j++ {
			// Select random product (not the same one)
			var product2 models.Product
			attempts := 0
			for attempts < 20 {
				product2 = products[gofakeit.IntRange(0, len(products)-1)]
				if product2.ID != product1.ID {
					break
				}
				attempts++
			}

			if product2.ID == product1.ID {
				continue
			}

			// Create unique key for this pair to avoid duplicates
			key1 := fmt.Sprintf("%s-%s", product1.ID.String(), product2.ID.String())
			key2 := fmt.Sprintf("%s-%s", product2.ID.String(), product1.ID.String())

			if similarityMap[key1] || similarityMap[key2] {
				continue
			}

			// Generate realistic similarity score
			var score float64

			// Higher similarity if same category
			if product1.CategoryID != nil && product2.CategoryID != nil && *product1.CategoryID == *product2.CategoryID {
				score = gofakeit.Float64Range(0.6, 0.95)
			} else {
				score = gofakeit.Float64Range(0.1, 0.7)
			}

			id := uuid.New()
			basedOn := basedOnOptions[gofakeit.IntRange(0, len(basedOnOptions)-1)]
			createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -180), time.Now())

			similarities = append(similarities, []interface{}{
				id,
				product1.ID,
				product2.ID,
				score,
				basedOn,
				createdAt,
			})

			similarityMap[key1] = true
			totalSimilarities++
		}

		// Insert in batches of 1000
		if len(similarities) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"product_similarity_data"},
				[]string{"id", "product_id", "similar_product_id", "similarity_score", "based_on", "created_at"},
				pgx.CopyFromRows(similarities),
			)
			if err != nil {
				return fmt.Errorf("failed to copy similarity batch: %w", err)
			}
			similarities = similarities[:0]
		}
	}

	// Insert remaining similarities
	if len(similarities) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"product_similarity_data"},
			[]string{"id", "product_id", "similar_product_id", "similarity_score", "based_on", "created_at"},
			pgx.CopyFromRows(similarities),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final similarity batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d product similarity relationships", totalSimilarities)
	return nil
}

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

// SeedFavorites creates realistic favorite relationships based on user demographics
func SeedFavorites(ctx context.Context, pool *pgxpool.Pool, users []models.User, products []models.Product) error {
	log.Printf("❤️ Seeding favorites for recommendation training...")

	favorites := make([][]interface{}, 0)
	totalFavorites := 0

	for _, user := range users {
		// Each user gets 10-50 favorites
		numFavorites := gofakeit.IntRange(10, ML_FAVORITES_PER_USER*2)
		favoriteProducts := make(map[uuid.UUID]bool)

		for i := 0; i < numFavorites && i < len(products); i++ {
			// Select random product, avoiding duplicates
			var selectedProduct models.Product
			attempts := 0
			for attempts < 10 {
				selectedProduct = products[gofakeit.IntRange(0, len(products)-1)]
				if !favoriteProducts[selectedProduct.ID] {
					break
				}
				attempts++
			}

			if favoriteProducts[selectedProduct.ID] {
				continue
			}
			favoriteProducts[selectedProduct.ID] = true

			id := uuid.New()
			favoritedAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -365), time.Now())

			favorites = append(favorites, []interface{}{
				id,
				user.ID,
				selectedProduct.ID,
				favoritedAt,
			})

			totalFavorites++
		}

		// Insert in batches of 1000
		if len(favorites) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"favorites"},
				[]string{"id", "user_id", "product_id", "favorited_at"},
				pgx.CopyFromRows(favorites),
			)
			if err != nil {
				return fmt.Errorf("failed to copy favorites batch: %w", err)
			}
			favorites = favorites[:0]
		}
	}

	// Insert remaining favorites
	if len(favorites) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"favorites"},
			[]string{"id", "user_id", "product_id", "favorited_at"},
			pgx.CopyFromRows(favorites),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final favorites batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d favorite relationships", totalFavorites)
	return nil
}

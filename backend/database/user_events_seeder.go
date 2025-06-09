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

// SeedUserEvents creates realistic user interaction events
func SeedUserEvents(ctx context.Context, pool *pgxpool.Pool, users []models.User, products []models.Product) error {
	log.Printf("📊 Seeding user events for interaction analysis...")

	events := make([][]interface{}, 0)
	totalEvents := 0

	eventTypes := []string{"view", "click", "add_to_cart", "comment", "favorite"}
	eventWeights := []float64{0.5, 0.2, 0.15, 0.1, 0.05} // View most common, favorite least

	for _, user := range users {
		// Each user generates 50-200 events
		numEvents := gofakeit.IntRange(50, ML_USER_EVENTS_PER_USER*2)

		for i := 0; i < numEvents; i++ {
			id := uuid.New()
			selectedProduct := products[gofakeit.IntRange(0, len(products)-1)]

			// Choose event type based on weights
			eventType := weightedChoice(eventTypes, eventWeights)

			// Create metadata based on event type
			var metadata map[string]interface{}
			switch eventType {
			case "view":
				metadata = map[string]interface{}{
					"duration_seconds": gofakeit.IntRange(5, 300),
					"source":           gofakeit.RandomString([]string{"search", "category", "recommendation", "featured"}),
				}
			case "click":
				metadata = map[string]interface{}{
					"element": gofakeit.RandomString([]string{"image", "title", "price", "button"}),
					"page":    gofakeit.RandomString([]string{"home", "search", "category"}),
				}
			case "add_to_cart":
				metadata = map[string]interface{}{
					"quantity": gofakeit.IntRange(1, 5),
					"price":    gofakeit.Float64Range(10.0, 1000.0),
				}
			case "comment":
				metadata = map[string]interface{}{
					"rating": gofakeit.IntRange(1, 5),
					"length": gofakeit.IntRange(10, 500),
				}
			case "favorite":
				metadata = map[string]interface{}{
					"source": gofakeit.RandomString([]string{"product_page", "search_results", "recommendations"}),
				}
			}

			metadataJSON, _ := json.Marshal(metadata)
			timestamp := gofakeit.DateRange(time.Now().AddDate(0, 0, -100), time.Now())

			events = append(events, []interface{}{
				id,
				user.ID,
				eventType,
				selectedProduct.ID,
				metadataJSON,
				timestamp,
			})

			totalEvents++
		}

		// Insert in batches of 1000
		if len(events) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"user_events"},
				[]string{"id", "user_id", "event_type", "product_id", "metadata", "timestamp"},
				pgx.CopyFromRows(events),
			)
			if err != nil {
				return fmt.Errorf("failed to copy user events batch: %w", err)
			}
			events = events[:0]
		}
	}

	// Insert remaining events
	if len(events) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"user_events"},
			[]string{"id", "user_id", "event_type", "product_id", "metadata", "timestamp"},
			pgx.CopyFromRows(events),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final user events batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d user interaction events", totalEvents)
	return nil
}

// weightedChoice selects an item based on weights
func weightedChoice(items []string, weights []float64) string {
	r := gofakeit.Float64()
	cumulative := 0.0

	for i, weight := range weights {
		cumulative += weight
		if r <= cumulative {
			return items[i]
		}
	}

	return items[len(items)-1]
}

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

// Sentiment-based comments for ML training
var sentimentComments = map[string][]string{
	"positive": {
		"Excellent product! Really satisfied with the quality.",
		"Great value for money. Works perfectly.",
		"Outstanding quality and fast shipping.",
		"Highly recommend this product to everyone.",
		"Perfect in every aspect. Love it!",
	},
	"neutral": {
		"Good product overall. Does what it should.",
		"Decent quality for the price.",
		"It's okay. Works as expected.",
		"Average product. Nothing special.",
		"Acceptable quality.",
	},
	"negative": {
		"Disappointed with the quality.",
		"Product doesn't work as advertised.",
		"Poor build quality.",
		"Not worth the money.",
		"Unsatisfied with purchase.",
	},
}

// SeedComments creates realistic comments with sentiment labels
func SeedComments(ctx context.Context, pool *pgxpool.Pool, users []models.User, products []models.Product) error {
	log.Printf("💬 Seeding comments with sentiment analysis data...")

	comments := make([][]interface{}, 0)
	totalComments := 0

	for _, product := range products {
		// Each product gets 1-8 comments
		numComments := gofakeit.IntRange(1, ML_COMMENTS_PER_PRODUCT)

		for i := 0; i < numComments; i++ {
			id := uuid.New()
			user := users[gofakeit.IntRange(0, len(users)-1)]

			// Choose sentiment: 60% positive, 25% neutral, 15% negative
			var sentiment string
			var commentBody string

			sentimentRoll := gofakeit.Float64()
			switch {
			case sentimentRoll < 0.15:
				sentiment = "negative"
				commentBody = sentimentComments["negative"][gofakeit.IntRange(0, len(sentimentComments["negative"])-1)]
			case sentimentRoll < 0.40:
				sentiment = "neutral"
				commentBody = sentimentComments["neutral"][gofakeit.IntRange(0, len(sentimentComments["neutral"])-1)]
			default:
				sentiment = "positive"
				commentBody = sentimentComments["positive"][gofakeit.IntRange(0, len(sentimentComments["positive"])-1)]
			}

			createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -200), time.Now())
			upvotes := gofakeit.IntRange(0, 50)
			downvotes := gofakeit.IntRange(0, 10)

			// Generate sentiment score based on label
			var sentimentScore float64
			switch sentiment {
			case "negative":
				sentimentScore = gofakeit.Float64Range(-1.0, -0.1)
			case "neutral":
				sentimentScore = gofakeit.Float64Range(-0.1, 0.1)
			case "positive":
				sentimentScore = gofakeit.Float64Range(0.1, 1.0)
			}

			comments = append(comments, []interface{}{
				id,
				user.ID,
				product.ID,
				commentBody,
				createdAt,
				upvotes,
				downvotes,
				sentiment,
				sentimentScore,
			})

			totalComments++
		}

		// Insert in batches of 1000
		if len(comments) >= 1000 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"comments"},
				[]string{"id", "user_id", "product_id", "body", "created_at", "upvotes", "downvotes", "sentiment_label", "sentiment_score"},
				pgx.CopyFromRows(comments),
			)
			if err != nil {
				return fmt.Errorf("failed to copy comment batch: %w", err)
			}
			comments = comments[:0] // Reset slice
		}
	}

	// Insert remaining comments
	if len(comments) > 0 {
		_, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"comments"},
			[]string{"id", "user_id", "product_id", "body", "created_at", "upvotes", "downvotes", "sentiment_label", "sentiment_score"},
			pgx.CopyFromRows(comments),
		)
		if err != nil {
			return fmt.Errorf("failed to copy final comment batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d sentiment-labeled comments", totalComments)
	return nil
}

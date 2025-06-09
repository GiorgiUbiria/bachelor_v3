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

// Product templates for realistic data generation
type ProductTemplate struct {
	Name       string
	Brands     []string
	BasePrices []float64
	Tags       []string
}

// Enhanced product data organized by category
var productTemplates = map[string][]ProductTemplate{
	"Electronics": {
		{Name: "Wireless Bluetooth Headphones", Brands: []string{"SoundMax", "AudioPro", "BeatWave"}, BasePrices: []float64{49.99, 99.99, 199.99}, Tags: []string{"wireless", "bluetooth", "audio"}},
		{Name: "4K Smart TV", Brands: []string{"ViewTech", "ScreenMaster", "DisplayPro"}, BasePrices: []float64{299.99, 599.99, 999.99}, Tags: []string{"4k", "smart", "tv"}},
		{Name: "Smartphone", Brands: []string{"TechPhone", "SmartDevice", "MobileMax"}, BasePrices: []float64{199.99, 499.99, 799.99}, Tags: []string{"mobile", "phone", "smart"}},
	},
	"Gaming & Computers": {
		{Name: "Gaming Laptop", Brands: []string{"GameForce", "ProGamer", "EliteBook"}, BasePrices: []float64{799.99, 1299.99, 1999.99}, Tags: []string{"gaming", "laptop", "performance"}},
		{Name: "Mechanical Keyboard", Brands: []string{"KeyMaster", "GameKeys", "TypePro"}, BasePrices: []float64{69.99, 129.99, 199.99}, Tags: []string{"mechanical", "gaming", "keyboard"}},
		{Name: "Gaming Mouse", Brands: []string{"ClickPro", "GameMouse", "PrecisionMax"}, BasePrices: []float64{34.99, 69.99, 119.99}, Tags: []string{"gaming", "mouse", "precision"}},
	},
	"Sports & Fitness": {
		{Name: "Adjustable Dumbbells", Brands: []string{"FitMax", "StrengthPro", "MuscleForce"}, BasePrices: []float64{99.99, 199.99, 299.99}, Tags: []string{"fitness", "dumbbells", "strength"}},
		{Name: "Yoga Mat", Brands: []string{"YogaMax", "FlexPro", "ZenMat"}, BasePrices: []float64{24.99, 49.99, 79.99}, Tags: []string{"yoga", "mat", "fitness"}},
		{Name: "Running Shoes", Brands: []string{"RunMax", "SpeedFit", "RacePro"}, BasePrices: []float64{79.99, 129.99, 189.99}, Tags: []string{"running", "shoes", "athletic"}},
	},
}

// SeedProducts creates realistic product data with variations
func SeedProducts(ctx context.Context, pool *pgxpool.Pool, count int, users []models.User, categories []models.Category) ([]models.Product, error) {
	log.Printf("📦 Seeding %d products with realistic variations...", count)

	products := make([][]interface{}, 0, count)
	productModels := make([]models.Product, 0, count)

	// Create category map for quick lookup
	categoryMap := make(map[string]uuid.UUID)
	for _, cat := range categories {
		categoryMap[cat.Name] = cat.ID
	}

	productsPerCategory := count / len(categories)
	productCounter := 0

	for categoryName, templates := range productTemplates {
		categoryID, exists := categoryMap[categoryName]
		if !exists {
			continue
		}

		productsPerTemplate := productsPerCategory / len(templates)
		if productsPerTemplate < 1 {
			productsPerTemplate = 1
		}

		for _, template := range templates {
			for i := 0; i < productsPerTemplate && productCounter < count; i++ {
				id := uuid.New()

				// Generate realistic variations
				brand := template.Brands[gofakeit.IntRange(0, len(template.Brands)-1)]
				basePrice := template.BasePrices[gofakeit.IntRange(0, len(template.BasePrices)-1)]

				// Add price variation ±20%
				priceVariation := gofakeit.Float64Range(0.8, 1.2)
				finalPrice := basePrice * priceVariation

				productName := fmt.Sprintf("%s %s", brand, template.Name)
				if i > 0 {
					productName = fmt.Sprintf("%s %s v%d", brand, template.Name, i+1)
				}

				description := gofakeit.ProductDescription()
				createdBy := users[gofakeit.IntRange(0, len(users)-1)].ID
				createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -365), time.Now())

				// Combine template tags with random additional tags
				allTags := make([]string, len(template.Tags))
				copy(allTags, template.Tags)

				randomTags := []string{"bestseller", "new", "premium", "eco-friendly"}
				if gofakeit.Bool() {
					allTags = append(allTags, randomTags[gofakeit.IntRange(0, len(randomTags)-1)])
				}

				productModel := models.Product{
					ID:          id,
					Name:        productName,
					Description: &description,
					Price:       &finalPrice,
					CategoryID:  &categoryID,
					Tags:        pq.StringArray(allTags),
					CreatedBy:   &createdBy,
					CreatedAt:   createdAt,
				}

				productModels = append(productModels, productModel)

				products = append(products, []interface{}{
					id,
					productName,
					description,
					finalPrice,
					pq.Array(allTags),
					categoryID,
					nil, // curated_price
					nil, // curated_tags
					createdBy,
					createdAt,
				})

				productCounter++
			}
		}
	}

	// Fill remaining products with random data if needed
	for productCounter < count {
		id := uuid.New()
		categoryID := categories[gofakeit.IntRange(0, len(categories)-1)].ID

		productName := gofakeit.ProductName()
		description := gofakeit.ProductDescription()
		price := gofakeit.Float64Range(10.0, 1000.0)
		createdBy := users[gofakeit.IntRange(0, len(users)-1)].ID
		createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -365), time.Now())

		tags := []string{gofakeit.Word(), gofakeit.Word()}

		productModel := models.Product{
			ID:          id,
			Name:        productName,
			Description: &description,
			Price:       &price,
			CategoryID:  &categoryID,
			Tags:        pq.StringArray(tags),
			CreatedBy:   &createdBy,
			CreatedAt:   createdAt,
		}

		productModels = append(productModels, productModel)

		products = append(products, []interface{}{
			id,
			productName,
			description,
			price,
			pq.Array(tags),
			categoryID,
			nil, // curated_price
			nil, // curated_tags
			createdBy,
			createdAt,
		})

		productCounter++
	}

	// Use pgx.CopyFrom for maximum performance
	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"products"},
		[]string{"id", "name", "description", "price", "tags", "category_id", "curated_price", "curated_tags", "created_by", "created_at"},
		pgx.CopyFromRows(products),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to copy products: %w", err)
	}

	log.Printf("✅ Successfully created %d realistic products", count)
	return productModels, nil
}

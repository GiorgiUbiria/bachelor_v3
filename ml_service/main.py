from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from security_analyzer.routes import router as security_router
    from security_analyzer.utils.profiling import system_profiler
    SECURITY_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Security analyzer not available: {e}")
    SECURITY_ANALYZER_AVAILABLE = False
    security_router = None
    system_profiler = None

try:
    from recommendation_engine.routes import router as recommendation_router
    RECOMMENDATION_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Recommendation engine not available: {e}")
    RECOMMENDATION_ENGINE_AVAILABLE = False
    recommendation_router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    if system_profiler:
        system_profiler.start_monitoring(interval=5.0)
    yield
    if system_profiler:
        system_profiler.stop_monitoring()

app = FastAPI(
    title="Enhanced ML Service",
    description="Research-Grade ML Service with Security Analysis and Recommendations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SECURITY_ANALYZER_AVAILABLE and security_router:
    app.include_router(security_router)

if RECOMMENDATION_ENGINE_AVAILABLE and recommendation_router:
    app.include_router(recommendation_router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "security_analyzer": SECURITY_ANALYZER_AVAILABLE,
            "recommendation_engine": RECOMMENDATION_ENGINE_AVAILABLE
        }
    }

@app.get("/")
async def root():
    features = ["Real-time health monitoring"]
    
    if SECURITY_ANALYZER_AVAILABLE:
        features.extend([
            "Real-time security analysis",
            "Ensemble ML classification", 
            "Pattern matching integration",
            "Explainable AI (LIME)"
        ])
        
    if RECOMMENDATION_ENGINE_AVAILABLE:
        features.extend([
            "Personalized recommendations",
            "Dynamic pricing",
            "User segmentation",
            "Hybrid recommendation models"
        ])
    
    return {
        "message": "Enhanced ML Service",
        "version": "2.0.0",
        "documentation": "/docs",
        "services_available": {
            "security_analyzer": SECURITY_ANALYZER_AVAILABLE,
            "recommendation_engine": RECOMMENDATION_ENGINE_AVAILABLE
        },
        "features": features
    } 
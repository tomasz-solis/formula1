"""
FastAPI application for F1 qualifying predictions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from .models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from .predictor import QualifyingPredictor
from .config import API_TITLE, API_VERSION, API_DESCRIPTION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        predictor = QualifyingPredictor()
        logger.info("✅ Predictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "🏎️ F1 Qualifying Predictor API",
        "version": API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        features_count=len(predictor.features) if predictor else 0
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_qualifying(request: PredictionRequest):
    """
    Predict qualifying position.
    
    **Example request:**
```json
    {
      "driver": "VER",
      "circuit": "Monza",
      "year": 2025,
      "avg_rainfall": 0.0,
      "avg_track_temp": 35.0
    }
```
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract manual features
        manual_features = {
            'avg_rainfall': request.avg_rainfall,
            'avg_track_temp': request.avg_track_temp,
            'avg_air_temp': request.avg_air_temp,
            'tyre_age': request.tyre_age,
            'is_fresh_tyre': float(request.is_fresh_tyre) if request.is_fresh_tyre is not None else None
        }
        
        # Predict
        pred, ci_lower, ci_upper, features = predictor.predict(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            manual_features=manual_features
        )
        
        return PredictionResponse(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            predicted_position=float(pred),
            predicted_position_rounded=int(round(pred)),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            model_name=predictor.metadata['model_name'],
            model_mae=predictor.metadata['mae'],
            features_used=features
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/drivers", tags=["Reference"])
async def list_drivers():
    """List all available drivers."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    drivers = sorted(predictor.historical_data['driver'].unique().tolist())
    return {"drivers": drivers, "count": len(drivers)}


@app.get("/circuits", tags=["Reference"])
async def list_circuits():
    """List all available circuits."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    circuits = sorted(predictor.historical_data['event'].unique().tolist())
    return {"circuits": circuits, "count": len(circuits)}
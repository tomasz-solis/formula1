"""
FastAPI application for F1 qualifying classifications.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Use absolute imports instead of relative imports
from models import (
    PredictionRequest,
    Q3PredictionResponse,
    Top3PredictionResponse,
    Q2PredictionResponse,
    CombinedPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from predictor import QualifyingPredictor
from config import API_TITLE, API_VERSION, API_DESCRIPTION

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
    """Load models on startup."""
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
        "message": "🏎️ F1 Qualifying Classifier API",
        "version": API_VERSION,
        "models": ["Q3 Binary", "Top 3 Binary", "Q2 Multi-class"],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    if not predictor:
        return HealthResponse(
            status="unhealthy",
            models_loaded={
                "q3": False,
                "top3": False,
                "q2": False
            },
            features_count=0
        )
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "q3": predictor.model_q3 is not None,
            "top3": predictor.model_top3 is not None,
            "q2": predictor.model_q2 is not None
        },
        features_count=len(predictor.features)
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict/q3", response_model=Q3PredictionResponse, tags=["Prediction"])
async def predict_q3(request: PredictionRequest):
    """
    Predict Q3 qualification (top 10).
    
    Returns binary classification: Will driver make Q3?
    
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
            'avg_air_temp': request.avg_air_temp
        }
        
        # Predict
        will_make_q3, probability, features = predictor.predict_q3(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            manual_features=manual_features
        )
        
        # Get top 5 features
        top_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return Q3PredictionResponse(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            will_make_q3=will_make_q3,
            probability=probability,
            confidence=predictor._get_confidence(probability),
            model_accuracy=predictor.metadata['models']['q3_binary']['accuracy'],
            features_used=top_features
        )
        
    except Exception as e:
        logger.error(f"Q3 prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/top3", response_model=Top3PredictionResponse, tags=["Prediction"])
async def predict_top3(request: PredictionRequest):
    """
    Predict Top 3 finish in qualifying.
    
    Returns binary classification: Will driver podium?
    
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
            'avg_air_temp': request.avg_air_temp
        }
        
        # Predict
        will_make_top3, probability, features = predictor.predict_top3(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            manual_features=manual_features
        )
        
        # Get top 5 features
        top_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return Top3PredictionResponse(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            will_make_top3=will_make_top3,
            probability=probability,
            confidence=predictor._get_confidence(probability),
            model_accuracy=predictor.metadata['models']['top3_binary']['accuracy'],
            features_used=top_features
        )
        
    except Exception as e:
        logger.error(f"Top 3 prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/round", response_model=Q2PredictionResponse, tags=["Prediction"])
async def predict_round(request: PredictionRequest):
    """
    Predict qualifying round (Q1/Q2/Q3).
    
    Returns multi-class classification: Which round will driver reach?
    
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
            'avg_air_temp': request.avg_air_temp
        }
        
        # Predict
        predicted_round, probabilities, features = predictor.predict_round(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            manual_features=manual_features
        )
        
        # Get top 5 features
        top_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return Q2PredictionResponse(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            predicted_round=predicted_round,
            probabilities=probabilities,
            confidence=predictor._get_confidence(max(probabilities.values())),
            model_accuracy=predictor.metadata['models']['q2_multiclass']['accuracy'],
            features_used=top_features
        )
        
    except Exception as e:
        logger.error(f"Round prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/all", tags=["Prediction"])
async def predict_all(request: PredictionRequest):
    """
    Get all predictions at once (Q3, Top 3, Round).
    
    Returns combined predictions from all models.
    
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
            'avg_air_temp': request.avg_air_temp
        }
        
        # Get all predictions
        results = predictor.predict_all(
            driver=request.driver,
            circuit=request.circuit,
            year=request.year,
            manual_features=manual_features
        )
        
        # Format response
        return {
            "driver": request.driver,
            "circuit": request.circuit,
            "year": request.year,
            "q3": {
                "will_make_q3": results['q3']['prediction'],
                "probability": results['q3']['probability'],
                "confidence": results['q3']['confidence']
            },
            "top3": {
                "will_make_top3": results['top3']['prediction'],
                "probability": results['top3']['probability'],
                "confidence": results['top3']['confidence']
            },
            "round": {
                "predicted_round": results['round']['prediction'],
                "probabilities": results['round']['probabilities'],
                "confidence": results['round']['confidence']
            },
            "model_accuracies": {
                "q3": predictor.metadata['models']['q3_binary']['accuracy'],
                "top3": predictor.metadata['models']['top3_binary']['accuracy'],
                "round": predictor.metadata['models']['q2_multiclass']['accuracy']
            }
        }
        
    except Exception as e:
        logger.error(f"Combined prediction failed: {e}")
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
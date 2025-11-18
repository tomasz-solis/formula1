"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Request schema for qualifying classification.
    """
    driver: str = Field(..., description="Driver abbreviation (e.g., 'VER', 'HAM')")
    circuit: str = Field(..., description="Circuit/Event name (e.g., 'Monza', 'Silverstone')")
    year: int = Field(..., description="Year of the session", ge=2022, le=2030)
    
    # Optional: Manual feature overrides
    avg_rainfall: Optional[float] = Field(None, description="Average rainfall (mm/h)", ge=0)
    avg_track_temp: Optional[float] = Field(None, description="Track temperature (°C)")
    avg_air_temp: Optional[float] = Field(None, description="Air temperature (°C)")
    
    @validator('driver')
    def validate_driver(cls, v):
        return v.upper().strip()
    
    @validator('circuit')
    def validate_circuit(cls, v):
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "driver": "VER",
                "circuit": "Monza",
                "year": 2025,
                "avg_rainfall": 0.0,
                "avg_track_temp": 35.0,
                "avg_air_temp": 28.0
            }
        }


class Q3PredictionResponse(BaseModel):
    """
    Response schema for Q3 qualification prediction.
    """
    driver: str
    circuit: str
    year: int
    will_make_q3: bool = Field(..., description="Predicted to make Q3 (top 10)")
    probability: float = Field(..., description="Probability of making Q3", ge=0, le=1)
    confidence: str = Field(..., description="Confidence level: high/medium/low")
    model_accuracy: float = Field(..., description="Model's test accuracy")
    features_used: Dict[str, float] = Field(..., description="Key features used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "driver": "VER",
                "circuit": "Monza",
                "year": 2025,
                "will_make_q3": True,
                "probability": 0.92,
                "confidence": "high",
                "model_accuracy": 0.788,
                "features_used": {
                    "dry_avg_position": 1.8,
                    "wet_avg_position": 2.3,
                    "team_recent_avg": 1.5
                },
                "timestamp": "2025-11-17T10:30:00"
            }
        }


class Top3PredictionResponse(BaseModel):
    """
    Response schema for Top 3 prediction.
    """
    driver: str
    circuit: str
    year: int
    will_make_top3: bool = Field(..., description="Predicted to finish top 3")
    probability: float = Field(..., description="Probability of top 3", ge=0, le=1)
    confidence: str = Field(..., description="Confidence level: high/medium/low")
    model_accuracy: float = Field(..., description="Model's test accuracy")
    features_used: Dict[str, float] = Field(..., description="Key features used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "driver": "VER",
                "circuit": "Monza",
                "year": 2025,
                "will_make_top3": True,
                "probability": 0.78,
                "confidence": "high",
                "model_accuracy": 0.891,
                "features_used": {
                    "dry_avg_position": 1.8,
                    "wet_avg_position": 2.3,
                    "team_recent_avg": 1.5
                },
                "timestamp": "2025-11-17T10:30:00"
            }
        }


class Q2PredictionResponse(BaseModel):
    """
    Response schema for qualifying round prediction (Q1/Q2/Q3).
    """
    driver: str
    circuit: str
    year: int
    predicted_round: str = Field(..., description="Predicted round: Q1, Q2, or Q3")
    probabilities: Dict[str, float] = Field(..., description="Probability for each round")
    confidence: str = Field(..., description="Confidence level: high/medium/low")
    model_accuracy: float = Field(..., description="Model's test accuracy")
    features_used: Dict[str, float] = Field(..., description="Key features used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "driver": "VER",
                "circuit": "Monza",
                "year": 2025,
                "predicted_round": "Q3",
                "probabilities": {
                    "Q1": 0.05,
                    "Q2": 0.15,
                    "Q3": 0.80
                },
                "confidence": "high",
                "model_accuracy": 0.70,
                "features_used": {
                    "dry_avg_position": 1.8,
                    "wet_avg_position": 2.3,
                    "team_recent_avg": 1.5
                },
                "timestamp": "2025-11-17T10:30:00"
            }
        }


class CombinedPredictionResponse(BaseModel):
    """
    Combined response with all predictions.
    """
    driver: str
    circuit: str
    year: int
    q3: Q3PredictionResponse
    top3: Top3PredictionResponse
    round: Q2PredictionResponse
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str
    models_loaded: Dict[str, bool]
    features_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """
    Model information response.
    """
    models: Dict[str, Dict[str, float]] = Field(..., description="Info for each model")
    features: List[str]
    feature_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "q3_binary": {
                        "accuracy": 0.788,
                        "precision": 0.813,
                        "recall": 0.752,
                        "auc": 0.874
                    },
                    "top3_binary": {
                        "accuracy": 0.891,
                        "precision": 0.723,
                        "recall": 0.444,
                        "auc": 0.921
                    },
                    "q2_multiclass": {
                        "accuracy": 0.70
                    }
                },
                "features": ["dry_avg_position", "wet_avg_position", "..."],
                "feature_count": 47
            }
        }

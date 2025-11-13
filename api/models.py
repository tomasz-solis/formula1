"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Request schema for qualifying prediction.
    """
    driver: str = Field(..., description="Driver name (e.g., 'VER', 'HAM')")
    circuit: str = Field(..., description="Circuit name (e.g., 'Monza', 'Silverstone')")
    year: int = Field(..., description="Year of the session", ge=2022, le=2030)
    
    # Optional: Manual feature overrides
    avg_rainfall: Optional[float] = Field(None, description="Average rainfall (mm/h)", ge=0)
    avg_track_temp: Optional[float] = Field(None, description="Track temperature (°C)")
    avg_air_temp: Optional[float] = Field(None, description="Air temperature (°C)")
    tyre_age: Optional[int] = Field(None, description="Tire age (laps)", ge=0)
    is_fresh_tyre: Optional[bool] = Field(None, description="Is using fresh tires?")
    
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


class PredictionResponse(BaseModel):
    """
    Response schema for qualifying prediction.
    """
    driver: str
    circuit: str
    year: int
    predicted_position: float = Field(..., description="Predicted qualifying position")
    predicted_position_rounded: int = Field(..., description="Rounded position")
    confidence_interval_lower: float = Field(..., description="95% CI lower bound")
    confidence_interval_upper: float = Field(..., description="95% CI upper bound")
    model_name: str
    model_mae: float = Field(..., description="Model's mean absolute error")
    features_used: Dict[str, float] = Field(..., description="Feature values used in prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "driver": "VER",
                "circuit": "Monza",
                "year": 2025,
                "predicted_position": 1.8,
                "predicted_position_rounded": 2,
                "confidence_interval_lower": 1.2,
                "confidence_interval_upper": 3.4,
                "model_name": "Random Forest",
                "model_mae": 2.35,
                "features_used": {
                    "circuit_avg_position": 1.5,
                    "recent_avg_position": 2.1,
                    "team_circuit_avg_position": 1.8
                },
                "timestamp": "2025-11-11T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str
    model_loaded: bool
    features_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """
    Model information response.
    """
    model_name: str
    mae: float
    r2: float
    features: List[str]
    feature_count: int
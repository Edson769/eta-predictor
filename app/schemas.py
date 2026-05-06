from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Literal
import math

class ETARequest(BaseModel):
    """Validation schema for ETA request."""
    origin_lat: float = Field(..., ge=-90, le=90, description=" origin latitude")
    origin_lon: float = Field(..., ge=-180, le=180, description="origin longitude")
    dest_lat: float = Field(..., ge=-90, le=90, description="destination latitude")
    dest_lon: float = Field(..., ge=-180, le=180, description="destination longitude")

    cargo_weight_kg: float = Field(..., gt=0, le=20000, description="cargo weight in kilograms")
    hour_of_day: int = Field(..., ge=0, le=23, description="hour of departure (0-23)")
    num_stops: int = Field(1, ge=1, le=20, description="number of delivery stops (default: 1)")
    vehicle_type: Literal['truck', 'van', 'motorcycle'] = 'truck'
    traffic_index: float = Field(1.0, ge=0.5, le=5.0)

    # @field_validator runs AFTER the type check 
    # It validates a single field 
    @field_validator('cargo_weight_kg') 
    @classmethod 
    def weight_must_make_sense_for_vehicle(cls, v: float, info) -> float: 
        """Motorcycles can carry at most 100kg.""" 
        # Note: at this point we can't access other fields easily in field_validator 
        # For cross-field validation, use model_validator (see below) 
        if v <= 0: 
            raise ValueError('Cargo weight must be positive') 
        return round(v, 2)   # Round to 2 decimal places
    

    # @model_validator runs AFTER all fields are validated 
    # It can check relationships BETWEEN fields 
    @model_validator(mode='after') 
    def check_origin_and_dest_differ(self) -> 'ETARequest': 
        """Origin and destination cannot be the same point.""" 
        if (abs(self.origin_lat - self.dest_lat) < 0.001 and 
                abs(self.origin_lon - self.dest_lon) < 0.001): 
            raise ValueError('Origin and destination appear to be the same location') 
        return self
    

    @computed_field 
    @property 
    def distance_km(self) -> float: 
        """Haversine distance from origin to destination in km.""" 
        # The Haversine formula calculates straight-line distance 
        # between two GPS coordinates on the Earth's surface 
        R = 6371  # Earth's radius in km 
        lat1 = math.radians(self.origin_lat) 
        lat2 = math.radians(self.dest_lat) 
        dlat = lat2 - lat1 
        dlon = math.radians(self.dest_lon - self.origin_lon) 
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2 
        return round(R * 2 * math.asin(math.sqrt(a)), 3) 
    

    @computed_field 
    @property 
    def is_rush_hour(self) -> bool: 
        """True if departure is during morning or evening rush hour.""" 
        morning_rush = list(range(7, 10))    # 7am - 9am 
        evening_rush = list(range(17, 20))   # 5pm - 7pm 
        return self.hour_of_day in morning_rush + evening_rush 
    
    def to_feature_vector(self) -> list[float]: 
        """Convert to a flat list of numbers for the ML model.""" 
        return [ 
            self.distance_km, 
            self.cargo_weight_kg, 
            float(self.is_rush_hour),   # True -> 1.0, False -> 0.0 
            float(self.hour_of_day), 
            float(self.num_stops), 
            self.traffic_index, 
            # Encode vehicle type as numbers 
            1.0 if self.vehicle_type == 'truck'      else 0.0, 
            1.0 if self.vehicle_type == 'van'        else 0.0, 
            1.0 if self.vehicle_type == 'motorcycle' else 0.0, 
        ]
    
class ETAResponse(BaseModel): 
    """Structured output from the ETA prediction endpoint.""" 
    eta_minutes:          float 
    eta_hours_minutes:    str     # Human-readable: '2 hours 15 minutes' 
    distance_km:          float 
    confidence_low_min:   float 
    confidence_high_min:  float 
    model_version:        str 
    is_rush_hour:         bool 
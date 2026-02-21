"""
Cambridge NY Plant Health Monitoring System
===========================================

AI-powered plant health scoring system that analyzes environmental inputs
(light level, humidity, temperature, watering frequency) and outputs health
predictions with actionable care recommendations.

Integrates with the Engram memory system to maintain historical plant health data
and learn from care patterns across Cambridge NY's commercial plant inventory.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings

from .plant_knowledge import CambridgePlantDatabase, PlantSpecies, LightRequirement, WaterFrequency


class HealthStatus(Enum):
    CRITICAL = 0      # 0-30: Immediate intervention needed
    POOR = 1         # 31-50: Needs attention soon
    FAIR = 2         # 51-70: Some adjustments needed
    GOOD = 3         # 71-85: Minor optimizations
    EXCELLENT = 4    # 86-100: Optimal conditions


class AlertLevel(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EnvironmentalData:
    """Environmental sensor readings for plant health assessment"""
    light_level: float        # Lux (foot-candles)
    humidity: float          # Percentage
    temperature: float       # Fahrenheit
    last_watered: datetime   # Last watering timestamp
    soil_moisture: float     # Percentage (0-100)
    air_circulation: float   # Air movement rating (0-10)
    location_rating: float   # Location quality score (0-10)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthAssessment:
    """Complete plant health assessment result"""
    plant_name: str
    health_score: float      # 0-100
    status: HealthStatus
    environmental_scores: Dict[str, float]
    recommendations: List[str]
    alerts: List[Tuple[str, AlertLevel]]
    predicted_issues: List[str]
    optimal_changes: Dict[str, Any]
    next_assessment: datetime


class PlantHealthPredictor(nn.Module):
    """Neural network for plant health prediction using environmental data"""
    
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


class CambridgeHealthMonitor:
    """Main plant health monitoring system for Cambridge NY"""
    
    def __init__(self, plant_database: Optional[CambridgePlantDatabase] = None):
        self.plant_db = plant_database or CambridgePlantDatabase()
        self.health_predictor = PlantHealthPredictor()
        self.assessment_history: Dict[str, List[HealthAssessment]] = {}
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Load pre-trained health prediction model (simulated for demo)"""
        # In production, this would load actual trained weights
        print("Loading Cambridge NY plant health prediction model...")
        # Simulate trained model weights
        pass
    
    def normalize_environmental_data(self, env_data: EnvironmentalData, plant: PlantSpecies) -> torch.Tensor:
        """Normalize environmental data for neural network input"""
        
        # Calculate time since last watering in hours
        hours_since_water = (datetime.now() - env_data.last_watered).total_seconds() / 3600
        
        # Expected watering frequency in hours
        expected_water_freq = plant.care_profile.water_frequency.value * 24
        
        # Light adequacy score (0-1 based on plant requirements)
        light_adequacy = self._calculate_light_adequacy(env_data.light_level, plant.care_profile.light_requirement)
        
        # Temperature stress score (0-1, lower is better)
        temp_stress = self._calculate_temperature_stress(env_data.temperature, 
                                                        plant.care_profile.temp_min, 
                                                        plant.care_profile.temp_max)
        
        # Humidity adequacy (0-1)
        humidity_adequacy = self._calculate_humidity_adequacy(env_data.humidity,
                                                            plant.care_profile.humidity_min,
                                                            plant.care_profile.humidity_max)
        
        # Watering schedule adherence (0-1)
        water_schedule = min(1.0, expected_water_freq / max(hours_since_water, 1))
        
        # Normalize soil moisture (0-1)
        soil_moisture_norm = env_data.soil_moisture / 100.0
        
        # Normalize other factors
        air_circulation_norm = env_data.air_circulation / 10.0
        location_rating_norm = env_data.location_rating / 10.0
        
        # Create input tensor
        features = torch.tensor([
            light_adequacy,
            temp_stress,
            humidity_adequacy,
            water_schedule,
            soil_moisture_norm,
            air_circulation_norm,
            location_rating_norm,
            float(plant.care_profile.difficulty.value == "challenging")  # Difficulty factor
        ], dtype=torch.float32).unsqueeze(0)
        
        return features
    
    def _calculate_light_adequacy(self, actual_lux: float, required: LightRequirement) -> float:
        """Calculate how well current light meets plant requirements"""
        # Light requirement ranges (foot-candles)
        light_ranges = {
            LightRequirement.LOW: (50, 250),
            LightRequirement.MEDIUM: (250, 1000),
            LightRequirement.BRIGHT: (1000, 2500),
            LightRequirement.DIRECT: (2500, 10000)
        }
        
        min_lux, max_lux = light_ranges[required]
        
        if actual_lux < min_lux:
            return actual_lux / min_lux  # Below minimum
        elif actual_lux > max_lux:
            return max(0.5, 1.0 - (actual_lux - max_lux) / max_lux)  # Above maximum
        else:
            return 1.0  # Perfect range
    
    def _calculate_temperature_stress(self, actual_temp: float, min_temp: int, max_temp: int) -> float:
        """Calculate temperature stress (0 = no stress, 1 = maximum stress)"""
        if min_temp <= actual_temp <= max_temp:
            return 0.0  # No stress in optimal range
        
        if actual_temp < min_temp:
            return min(1.0, (min_temp - actual_temp) / 10.0)  # Cold stress
        else:
            return min(1.0, (actual_temp - max_temp) / 10.0)   # Heat stress
    
    def _calculate_humidity_adequacy(self, actual_humidity: float, min_humidity: int, max_humidity: int) -> float:
        """Calculate humidity adequacy score"""
        if min_humidity <= actual_humidity <= max_humidity:
            return 1.0
        
        if actual_humidity < min_humidity:
            return actual_humidity / min_humidity
        else:
            return max(0.3, 1.0 - (actual_humidity - max_humidity) / max_humidity)
    
    def assess_plant_health(self, plant_name: str, env_data: EnvironmentalData) -> HealthAssessment:
        """Perform comprehensive plant health assessment"""
        
        # Get plant information
        plant = self.plant_db.get_plant_by_name(plant_name)
        if not plant:
            raise ValueError(f"Plant '{plant_name}' not found in database")
        
        # Normalize environmental data for ML model
        features = self.normalize_environmental_data(env_data, plant)
        
        # Get health prediction from neural network
        with torch.no_grad():
            health_probs = self.health_predictor(features)
            health_scores = torch.tensor([0, 25, 50, 75, 95], dtype=torch.float32)
            predicted_score = torch.sum(health_probs * health_scores).item()
        
        # Determine health status
        if predicted_score <= 30:
            status = HealthStatus.CRITICAL
        elif predicted_score <= 50:
            status = HealthStatus.POOR
        elif predicted_score <= 70:
            status = HealthStatus.FAIR
        elif predicted_score <= 85:
            status = HealthStatus.GOOD
        else:
            status = HealthStatus.EXCELLENT
        
        # Calculate detailed environmental scores
        env_scores = {
            "light": self._calculate_light_adequacy(env_data.light_level, plant.care_profile.light_requirement) * 100,
            "temperature": (1 - self._calculate_temperature_stress(env_data.temperature, 
                                                                  plant.care_profile.temp_min, 
                                                                  plant.care_profile.temp_max)) * 100,
            "humidity": self._calculate_humidity_adequacy(env_data.humidity,
                                                        plant.care_profile.humidity_min,
                                                        plant.care_profile.humidity_max) * 100,
            "watering": min(100, (plant.care_profile.water_frequency.value * 24) / 
                           max(1, (datetime.now() - env_data.last_watered).total_seconds() / 3600) * 100),
            "soil_moisture": env_data.soil_moisture,
            "air_circulation": env_data.air_circulation * 10,
            "location": env_data.location_rating * 10
        }
        
        # Generate recommendations and alerts
        recommendations, alerts, predicted_issues = self._generate_care_recommendations(
            plant, env_data, env_scores, predicted_score
        )
        
        # Calculate optimal changes
        optimal_changes = self._calculate_optimal_changes(plant, env_data, env_scores)
        
        # Schedule next assessment
        if status in [HealthStatus.CRITICAL, HealthStatus.POOR]:
            next_assessment = datetime.now() + timedelta(days=1)
        elif status == HealthStatus.FAIR:
            next_assessment = datetime.now() + timedelta(days=3)
        else:
            next_assessment = datetime.now() + timedelta(weeks=1)
        
        assessment = HealthAssessment(
            plant_name=plant_name,
            health_score=predicted_score,
            status=status,
            environmental_scores=env_scores,
            recommendations=recommendations,
            alerts=alerts,
            predicted_issues=predicted_issues,
            optimal_changes=optimal_changes,
            next_assessment=next_assessment
        )
        
        # Store assessment in history
        if plant_name not in self.assessment_history:
            self.assessment_history[plant_name] = []
        self.assessment_history[plant_name].append(assessment)
        
        return assessment
    
    def _generate_care_recommendations(self, plant: PlantSpecies, env_data: EnvironmentalData, 
                                     env_scores: Dict[str, float], health_score: float) -> Tuple[List[str], List[Tuple[str, AlertLevel]], List[str]]:
        """Generate specific care recommendations based on assessment"""
        
        recommendations = []
        alerts = []
        predicted_issues = []
        
        # Light recommendations
        if env_scores["light"] < 50:
            recommendations.append(f"Increase light exposure. {plant.common_name} needs {plant.care_profile.light_requirement.value} light.")
            if env_scores["light"] < 30:
                alerts.append(("Insufficient light - plant health at risk", AlertLevel.HIGH))
                predicted_issues.append("Etiolation (stretching), leaf drop")
        elif env_scores["light"] > 90 and plant.care_profile.light_requirement != LightRequirement.DIRECT:
            recommendations.append("Reduce direct sunlight exposure to prevent leaf burn")
        
        # Temperature recommendations
        if env_scores["temperature"] < 70:
            if env_data.temperature < plant.care_profile.temp_min:
                recommendations.append(f"Increase temperature. Current: {env_data.temperature}°F, Optimal: {plant.care_profile.temp_min}-{plant.care_profile.temp_max}°F")
                alerts.append(("Temperature too cold", AlertLevel.MEDIUM))
            else:
                recommendations.append(f"Decrease temperature. Current: {env_data.temperature}°F, Optimal: {plant.care_profile.temp_min}-{plant.care_profile.temp_max}°F")
                alerts.append(("Temperature too hot", AlertLevel.MEDIUM))
        
        # Humidity recommendations
        if env_scores["humidity"] < 60:
            recommendations.append(f"Increase humidity. Current: {env_data.humidity}%, Optimal: {plant.care_profile.humidity_min}-{plant.care_profile.humidity_max}%")
            recommendations.append("Consider using a humidifier or pebble tray")
            if env_scores["humidity"] < 40:
                predicted_issues.append("Brown leaf tips, dropping leaves")
        
        # Watering recommendations
        hours_since_water = (datetime.now() - env_data.last_watered).total_seconds() / 3600
        expected_hours = plant.care_profile.water_frequency.value * 24
        
        if hours_since_water > expected_hours * 1.5:
            recommendations.append("Water immediately - plant is overdue for watering")
            alerts.append(("Watering overdue", AlertLevel.HIGH))
            predicted_issues.append("Wilting, dry soil stress")
        elif hours_since_water < expected_hours * 0.3 and env_data.soil_moisture > 80:
            recommendations.append("Reduce watering frequency - soil is staying too wet")
            alerts.append(("Possible overwatering", AlertLevel.MEDIUM))
            predicted_issues.append("Root rot risk")
        
        # Soil moisture
        if env_data.soil_moisture < 20:
            recommendations.append("Soil is too dry - water thoroughly")
            alerts.append(("Critically low soil moisture", AlertLevel.URGENT))
        elif env_data.soil_moisture > 90:
            recommendations.append("Soil is waterlogged - improve drainage")
            alerts.append(("Waterlogged soil", AlertLevel.HIGH))
        
        # Air circulation
        if env_scores["air_circulation"] < 50:
            recommendations.append("Improve air circulation around plant")
            recommendations.append("Consider relocating away from walls or corners")
        
        # Overall health recommendations
        if health_score < 30:
            alerts.append(("Plant in critical condition - immediate intervention required", AlertLevel.URGENT))
            recommendations.append("Schedule emergency plant care visit")
        elif health_score < 50:
            recommendations.append("Schedule plant care visit within 2-3 days")
        
        return recommendations, alerts, predicted_issues
    
    def _calculate_optimal_changes(self, plant: PlantSpecies, env_data: EnvironmentalData, 
                                 env_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate specific optimal environmental changes"""
        
        optimal_changes = {}
        
        # Optimal light level
        light_ranges = {
            LightRequirement.LOW: 150,
            LightRequirement.MEDIUM: 500,
            LightRequirement.BRIGHT: 1500,
            LightRequirement.DIRECT: 3000
        }
        optimal_changes["light_level"] = light_ranges[plant.care_profile.light_requirement]
        
        # Optimal temperature (middle of range)
        optimal_changes["temperature"] = (plant.care_profile.temp_min + plant.care_profile.temp_max) / 2
        
        # Optimal humidity (middle of range)
        optimal_changes["humidity"] = (plant.care_profile.humidity_min + plant.care_profile.humidity_max) / 2
        
        # Optimal soil moisture based on plant type
        if plant.care_profile.water_frequency in [WaterFrequency.DAILY, WaterFrequency.EVERY_2_DAYS]:
            optimal_changes["soil_moisture"] = 60  # Moisture-loving plants
        else:
            optimal_changes["soil_moisture"] = 40  # Drought-tolerant plants
        
        return optimal_changes
    
    def get_plant_trends(self, plant_name: str, days: int = 30) -> Dict[str, List[float]]:
        """Get health trends for a plant over specified days"""
        
        if plant_name not in self.assessment_history:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_assessments = [
            assessment for assessment in self.assessment_history[plant_name]
            if hasattr(assessment, 'timestamp') and assessment.timestamp >= cutoff_date  # Simulate timestamp
        ]
        
        if not recent_assessments:
            return {}
        
        trends = {
            "health_scores": [a.health_score for a in recent_assessments],
            "light_scores": [a.environmental_scores["light"] for a in recent_assessments],
            "temperature_scores": [a.environmental_scores["temperature"] for a in recent_assessments],
            "humidity_scores": [a.environmental_scores["humidity"] for a in recent_assessments],
            "watering_scores": [a.environmental_scores["watering"] for a in recent_assessments]
        }
        
        return trends
    
    def predict_future_health(self, plant_name: str, days_ahead: int = 7) -> Dict[str, float]:
        """Predict plant health trajectory based on current trends"""
        
        trends = self.get_plant_trends(plant_name, days=30)
        if not trends or len(trends["health_scores"]) < 3:
            return {"prediction": "insufficient_data"}
        
        # Simple linear trend prediction
        health_scores = trends["health_scores"]
        recent_trend = np.polyfit(range(len(health_scores)), health_scores, 1)[0]
        
        current_score = health_scores[-1]
        predicted_score = current_score + (recent_trend * days_ahead)
        predicted_score = max(0, min(100, predicted_score))  # Clamp to valid range
        
        return {
            "current_score": current_score,
            "predicted_score": predicted_score,
            "trend": "improving" if recent_trend > 0 else "declining" if recent_trend < 0 else "stable",
            "confidence": min(100, len(health_scores) * 10)  # Confidence based on data points
        }


def demo_health_monitor():
    """Demonstration of the plant health monitoring system"""
    print("Cambridge NY Plant Health Monitor Demo")
    print("=" * 42)
    
    # Initialize system
    monitor = CambridgeHealthMonitor()
    
    # Sample environmental data (simulating sensor readings)
    env_data = EnvironmentalData(
        light_level=300,                    # Foot-candles
        humidity=45,                        # %
        temperature=72,                     # °F
        last_watered=datetime.now() - timedelta(days=5),
        soil_moisture=35,                   # %
        air_circulation=6,                  # Scale 0-10
        location_rating=7                   # Scale 0-10
    )
    
    # Assess a plant
    print("\n🌿 Assessing Peace Lily health...")
    assessment = monitor.assess_plant_health("Peace Lily", env_data)
    
    print(f"Health Score: {assessment.health_score:.1f}/100")
    print(f"Status: {assessment.status.name}")
    
    print("\n📊 Environmental Scores:")
    for factor, score in assessment.environmental_scores.items():
        print(f"  {factor.title()}: {score:.1f}/100")
    
    print("\n💡 Recommendations:")
    for rec in assessment.recommendations[:3]:
        print(f"  • {rec}")
    
    print("\n⚠️  Alerts:")
    for alert, level in assessment.alerts:
        print(f"  {level.value.upper()}: {alert}")
    
    print(f"\n📅 Next Assessment: {assessment.next_assessment.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    demo_health_monitor()
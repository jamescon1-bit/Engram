"""
Cambridge NY Weekly Lobby Flower Arrangement Planner
===================================================

AI-powered weekly lobby flower arrangement planning system for Cambridge NY's
commercial floral operations. Optimizes seasonal flower availability, color schemes,
budget constraints, and delivery scheduling.

Uses Engram's memory architecture to learn from arrangement success patterns
and client preferences over time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import random
from collections import defaultdict

from .rental_tracker import ClientLocation, LocationType


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class ColorScheme(Enum):
    MONOCHROMATIC = "monochromatic"
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    NEUTRAL = "neutral"
    BOLD = "bold"


class ArrangementStyle(Enum):
    MODERN = "modern"
    TRADITIONAL = "traditional"
    CONTEMPORARY = "contemporary"
    RUSTIC = "rustic"
    MINIMALIST = "minimalist"
    LUXURY = "luxury"


class FlowerAvailability(Enum):
    ABUNDANT = "abundant"
    AVAILABLE = "available"
    LIMITED = "limited"
    SPECIAL_ORDER = "special_order"
    UNAVAILABLE = "unavailable"


@dataclass
class FlowerSpecies:
    """Individual flower species with seasonal and cost information"""
    name: str
    scientific_name: str
    colors: List[str]
    seasonal_availability: Dict[Season, FlowerAvailability]
    cost_per_stem: float
    longevity_days: int
    fragrance: bool
    size_category: str  # "small", "medium", "large"
    care_difficulty: str  # "easy", "moderate", "difficult"
    style_suitability: List[ArrangementStyle]
    peak_seasons: List[Season]


@dataclass
class ArrangementSpec:
    """Specification for a flower arrangement"""
    arrangement_id: str
    client_location_id: str
    week_start_date: datetime
    budget: float
    style_preference: ArrangementStyle
    color_scheme: ColorScheme
    size_category: str  # "small", "medium", "large", "extra_large"
    special_requirements: List[str]
    delivery_date: datetime
    arrangement_duration: int = 7  # days
    fragrance_preference: bool = True
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class ArrangementDesign:
    """Complete flower arrangement design with specific flowers and quantities"""
    design_id: str
    arrangement_spec: ArrangementSpec
    flowers: Dict[str, int]  # flower_name -> quantity
    total_cost: float
    estimated_longevity: float
    color_palette: List[str]
    design_notes: str
    difficulty_score: float  # 1-10 scale
    visual_impact_score: float  # 1-10 scale
    seasonal_appropriateness: float  # 1-10 scale
    created_date: datetime = field(default_factory=datetime.now)


@dataclass 
class DeliverySchedule:
    """Delivery scheduling for flower arrangements"""
    schedule_id: str
    delivery_date: datetime
    delivery_time_window: Tuple[str, str]  # ("09:00", "11:00")
    arrangements: List[str]  # arrangement_ids
    delivery_route: List[str]  # location_ids in optimal order
    driver_assigned: str
    total_delivery_cost: float
    estimated_duration: float  # hours
    special_instructions: List[str]
    status: str = "scheduled"  # "scheduled", "in_transit", "completed", "failed"


class CambridgeFlowerDatabase:
    """Comprehensive database of flowers available for arrangements"""
    
    def __init__(self):
        self.flowers = self._initialize_flower_database()
        self.flower_index = {flower.name.lower(): flower for flower in self.flowers}
    
    def _initialize_flower_database(self) -> List[FlowerSpecies]:
        """Initialize database with seasonal flowers and their characteristics"""
        
        flowers = []
        
        # Classic roses
        flowers.append(FlowerSpecies(
            name="Red Rose",
            scientific_name="Rosa rubiginosa",
            colors=["red", "deep red", "burgundy"],
            seasonal_availability={
                Season.SPRING: FlowerAvailability.AVAILABLE,
                Season.SUMMER: FlowerAvailability.ABUNDANT,
                Season.FALL: FlowerAvailability.AVAILABLE,
                Season.WINTER: FlowerAvailability.LIMITED
            },
            cost_per_stem=4.50,
            longevity_days=7,
            fragrance=True,
            size_category="medium",
            care_difficulty="easy",
            style_suitability=[ArrangementStyle.TRADITIONAL, ArrangementStyle.LUXURY],
            peak_seasons=[Season.SUMMER, Season.FALL]
        ))
        
        # Spring flowers
        flowers.append(FlowerSpecies(
            name="Tulip",
            scientific_name="Tulipa gesneriana",
            colors=["red", "pink", "yellow", "white", "purple", "orange"],
            seasonal_availability={
                Season.SPRING: FlowerAvailability.ABUNDANT,
                Season.SUMMER: FlowerAvailability.UNAVAILABLE,
                Season.FALL: FlowerAvailability.UNAVAILABLE,
                Season.WINTER: FlowerAvailability.SPECIAL_ORDER
            },
            cost_per_stem=2.75,
            longevity_days=5,
            fragrance=False,
            size_category="medium",
            care_difficulty="easy",
            style_suitability=[ArrangementStyle.MODERN, ArrangementStyle.CONTEMPORARY],
            peak_seasons=[Season.SPRING]
        ))
        
        flowers.append(FlowerSpecies(
            name="Daffodil",
            scientific_name="Narcissus pseudonarcissus",
            colors=["yellow", "white", "orange"],
            seasonal_availability={
                Season.SPRING: FlowerAvailability.ABUNDANT,
                Season.SUMMER: FlowerAvailability.UNAVAILABLE,
                Season.FALL: FlowerAvailability.UNAVAILABLE,
                Season.WINTER: FlowerAvailability.LIMITED
            },
            cost_per_stem=2.25,
            longevity_days=6,
            fragrance=True,
            size_category="medium",
            care_difficulty="easy",
            style_suitability=[ArrangementStyle.TRADITIONAL, ArrangementStyle.RUSTIC],
            peak_seasons=[Season.SPRING]
        ))
        
        # Summer flowers
        flowers.append(FlowerSpecies(
            name="Sunflower",
            scientific_name="Helianthus annuus",
            colors=["yellow", "orange", "red"],
            seasonal_availability={
                Season.SPRING: FlowerAvailability.LIMITED,
                Season.SUMMER: FlowerAvailability.ABUNDANT,
                Season.FALL: FlowerAvailability.AVAILABLE,
                Season.WINTER: FlowerAvailability.UNAVAILABLE
            },
            cost_per_stem=3.50,
            longevity_days=8,
            fragrance=False,
            size_category="large",
            care_difficulty="easy",
            style_suitability=[ArrangementStyle.RUSTIC, ArrangementStyle.CONTEMPORARY],
            peak_seasons=[Season.SUMMER]
        ))
        
        # Year-round availability
        flowers.append(FlowerSpecies(
            name="White Lily",
            scientific_name="Lilium candidum",
            colors=["white", "cream"],
            seasonal_availability={
                Season.SPRING: FlowerAvailability.AVAILABLE,
                Season.SUMMER: FlowerAvailability.ABUNDANT,
                Season.FALL: FlowerAvailability.AVAILABLE,
                Season.WINTER: FlowerAvailability.AVAILABLE
            },
            cost_per_stem=5.25,
            longevity_days=9,
            fragrance=True,
            size_category="large",
            care_difficulty="moderate",
            style_suitability=[ArrangementStyle.LUXURY, ArrangementStyle.TRADITIONAL, ArrangementStyle.MINIMALIST],
            peak_seasons=[Season.SUMMER, Season.SPRING]
        ))
        
        # Add more flowers...
        # [Continue with chrysanthemums, carnations, peonies, etc.]
        
        return flowers
    
    def get_flowers_by_season(self, season: Season, availability_threshold: FlowerAvailability = FlowerAvailability.LIMITED) -> List[FlowerSpecies]:
        """Get flowers available in specified season"""
        available_flowers = []
        threshold_values = {
            FlowerAvailability.ABUNDANT: 4,
            FlowerAvailability.AVAILABLE: 3,
            FlowerAvailability.LIMITED: 2,
            FlowerAvailability.SPECIAL_ORDER: 1,
            FlowerAvailability.UNAVAILABLE: 0
        }
        
        min_threshold = threshold_values[availability_threshold]
        
        for flower in self.flowers:
            flower_availability = threshold_values[flower.seasonal_availability[season]]
            if flower_availability >= min_threshold:
                available_flowers.append(flower)
        
        return available_flowers
    
    def get_flowers_by_budget(self, max_cost_per_stem: float) -> List[FlowerSpecies]:
        """Filter flowers by cost per stem"""
        return [flower for flower in self.flowers if flower.cost_per_stem <= max_cost_per_stem]
    
    def get_flowers_by_color(self, color: str) -> List[FlowerSpecies]:
        """Get flowers available in specified color"""
        return [flower for flower in self.flowers if color.lower() in [c.lower() for c in flower.colors]]


class CambridgeLobbyPlanner:
    """Main lobby flower arrangement planning system"""
    
    def __init__(self, flower_database: Optional[CambridgeFlowerDatabase] = None):
        self.flower_db = flower_database or CambridgeFlowerDatabase()
        self.arrangement_specs: Dict[str, ArrangementSpec] = {}
        self.arrangement_designs: Dict[str, ArrangementDesign] = {}
        self.delivery_schedules: Dict[str, DeliverySchedule] = {}
        self.client_preferences: Dict[str, Dict] = {}  # location_id -> preferences
        self.seasonal_trends: Dict[Season, Dict] = {}
        
        self._initialize_seasonal_trends()
    
    def _initialize_seasonal_trends(self):
        """Initialize seasonal color and style trends"""
        self.seasonal_trends = {
            Season.SPRING: {
                "popular_colors": ["pink", "yellow", "white", "light purple"],
                "color_schemes": [ColorScheme.ANALOGOUS, ColorScheme.NEUTRAL],
                "styles": [ArrangementStyle.MODERN, ArrangementStyle.CONTEMPORARY]
            },
            Season.SUMMER: {
                "popular_colors": ["bright yellow", "orange", "red", "white"],
                "color_schemes": [ColorScheme.BOLD, ColorScheme.COMPLEMENTARY],
                "styles": [ArrangementStyle.CONTEMPORARY, ArrangementStyle.RUSTIC]
            },
            Season.FALL: {
                "popular_colors": ["orange", "burgundy", "gold", "brown"],
                "color_schemes": [ColorScheme.MONOCHROMATIC, ColorScheme.ANALOGOUS],
                "styles": [ArrangementStyle.TRADITIONAL, ArrangementStyle.RUSTIC]
            },
            Season.WINTER: {
                "popular_colors": ["red", "white", "green", "silver"],
                "color_schemes": [ColorScheme.COMPLEMENTARY, ColorScheme.NEUTRAL],
                "styles": [ArrangementStyle.LUXURY, ArrangementStyle.TRADITIONAL]
            }
        }
    
    def _get_current_season(self, date: datetime = None) -> Season:
        """Determine current season based on date"""
        if date is None:
            date = datetime.now()
        
        month = date.month
        
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER
    
    def create_arrangement_spec(self, client_location_id: str, week_start_date: datetime,
                              budget: float, style_preference: ArrangementStyle = None,
                              color_scheme: ColorScheme = None, size_category: str = "medium",
                              **kwargs) -> str:
        """Create arrangement specification for client location"""
        
        arrangement_id = f"ARR-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        
        # Determine seasonal defaults if not specified
        season = self._get_current_season(week_start_date)
        seasonal_trends = self.seasonal_trends[season]
        
        if style_preference is None:
            style_preference = random.choice(seasonal_trends["styles"])
        
        if color_scheme is None:
            color_scheme = random.choice(seasonal_trends["color_schemes"])
        
        # Set delivery date (typically Monday for weekly arrangements)
        delivery_date = week_start_date
        
        spec = ArrangementSpec(
            arrangement_id=arrangement_id,
            client_location_id=client_location_id,
            week_start_date=week_start_date,
            budget=budget,
            style_preference=style_preference,
            color_scheme=color_scheme,
            size_category=size_category,
            delivery_date=delivery_date,
            special_requirements=kwargs.get('special_requirements', []),
            fragrance_preference=kwargs.get('fragrance_preference', True)
        )
        
        self.arrangement_specs[arrangement_id] = spec
        return arrangement_id
    
    def design_arrangement(self, arrangement_id: str) -> ArrangementDesign:
        """Create optimized flower arrangement design"""
        
        if arrangement_id not in self.arrangement_specs:
            raise ValueError(f"Arrangement specification {arrangement_id} not found")
        
        spec = self.arrangement_specs[arrangement_id]
        season = self._get_current_season(spec.week_start_date)
        
        # Get available flowers for the season
        available_flowers = self.flower_db.get_flowers_by_season(season, FlowerAvailability.LIMITED)
        
        # Filter by style suitability
        suitable_flowers = [f for f in available_flowers if spec.style_preference in f.style_suitability]
        
        if not suitable_flowers:
            suitable_flowers = available_flowers  # Fallback to all available
        
        # Optimize flower selection based on color scheme and budget
        selected_flowers = self._optimize_flower_selection(suitable_flowers, spec)
        
        # Calculate costs and characteristics
        total_cost = sum(flower.cost_per_stem * quantity for flower, quantity in selected_flowers)
        avg_longevity = np.mean([flower.longevity_days for flower, _ in selected_flowers])
        
        # Generate color palette
        color_palette = []
        for flower, _ in selected_flowers:
            color_palette.extend(flower.colors)
        color_palette = list(set(color_palette))  # Remove duplicates
        
        # Calculate scores
        difficulty_score = self._calculate_difficulty_score(selected_flowers)
        visual_impact_score = self._calculate_visual_impact_score(selected_flowers, spec)
        seasonal_appropriateness = self._calculate_seasonal_score(selected_flowers, season)
        
        design = ArrangementDesign(
            design_id=f"DES-{arrangement_id}",
            arrangement_spec=spec,
            flowers={flower.name: quantity for flower, quantity in selected_flowers},
            total_cost=total_cost,
            estimated_longevity=avg_longevity,
            color_palette=color_palette,
            design_notes=self._generate_design_notes(selected_flowers, spec),
            difficulty_score=difficulty_score,
            visual_impact_score=visual_impact_score,
            seasonal_appropriateness=seasonal_appropriateness
        )
        
        self.arrangement_designs[arrangement_id] = design
        return design
    
    def _optimize_flower_selection(self, available_flowers: List[FlowerSpecies], 
                                 spec: ArrangementSpec) -> List[Tuple[FlowerSpecies, int]]:
        """Optimize flower selection using budget and aesthetic constraints"""
        
        # Size-based stem count targets
        size_targets = {
            "small": (8, 15),
            "medium": (15, 25),
            "large": (25, 40),
            "extra_large": (40, 60)
        }
        
        min_stems, max_stems = size_targets.get(spec.size_category, (15, 25))
        
        # Color scheme preferences
        seasonal_colors = self.seasonal_trends[self._get_current_season(spec.week_start_date)]["popular_colors"]
        
        # Filter flowers by color preferences
        preferred_flowers = []
        for flower in available_flowers:
            flower_colors = [c.lower() for c in flower.colors]
            seasonal_colors_lower = [c.lower() for c in seasonal_colors]
            
            if any(color in flower_colors for color in seasonal_colors_lower):
                preferred_flowers.append(flower)
        
        if not preferred_flowers:
            preferred_flowers = available_flowers
        
        # Budget-conscious selection
        budget_per_stem = spec.budget / max_stems
        affordable_flowers = [f for f in preferred_flowers if f.cost_per_stem <= budget_per_stem * 1.2]
        
        if not affordable_flowers:
            affordable_flowers = sorted(preferred_flowers, key=lambda x: x.cost_per_stem)[:5]
        
        # Select 2-4 flower types for variety
        num_types = min(4, max(2, len(affordable_flowers)))
        selected_flower_types = np.random.choice(affordable_flowers, num_types, replace=False)
        
        # Distribute stems among selected flowers
        selected_flowers = []
        remaining_budget = spec.budget
        remaining_stems = random.randint(min_stems, max_stems)
        
        for i, flower in enumerate(selected_flower_types):
            if i == len(selected_flower_types) - 1:
                # Last flower gets remaining stems (within budget)
                max_affordable_stems = int(remaining_budget // flower.cost_per_stem)
                stems = min(remaining_stems, max_affordable_stems)
            else:
                # Distribute stems roughly equally, with some randomization
                target_stems = remaining_stems // (len(selected_flower_types) - i)
                variation = max(1, target_stems // 3)
                stems = random.randint(max(1, target_stems - variation), target_stems + variation)
                
                # Ensure we don't exceed budget
                max_affordable_stems = int(remaining_budget // flower.cost_per_stem)
                stems = min(stems, max_affordable_stems)
            
            if stems > 0:
                selected_flowers.append((flower, stems))
                remaining_budget -= flower.cost_per_stem * stems
                remaining_stems -= stems
        
        return selected_flowers
    
    def _calculate_difficulty_score(self, selected_flowers: List[Tuple[FlowerSpecies, int]]) -> float:
        """Calculate arrangement difficulty based on flower care requirements"""
        difficulty_values = {"easy": 1, "moderate": 2, "difficult": 3}
        
        weighted_difficulty = sum(
            difficulty_values[flower.care_difficulty] * quantity
            for flower, quantity in selected_flowers
        )
        total_stems = sum(quantity for _, quantity in selected_flowers)
        
        avg_difficulty = weighted_difficulty / total_stems if total_stems > 0 else 1
        return (avg_difficulty / 3) * 10  # Scale to 1-10
    
    def _calculate_visual_impact_score(self, selected_flowers: List[Tuple[FlowerSpecies, int]], 
                                     spec: ArrangementSpec) -> float:
        """Calculate visual impact score based on flower combination"""
        
        # Size diversity bonus
        sizes = set(flower.size_category for flower, _ in selected_flowers)
        size_diversity = len(sizes) / 3  # Max 3 sizes (small, medium, large)
        
        # Color diversity
        all_colors = []
        for flower, _ in selected_flowers:
            all_colors.extend(flower.colors)
        color_diversity = len(set(all_colors)) / 6  # Normalized to typical color range
        
        # Fragrance appeal
        fragrant_stems = sum(quantity for flower, quantity in selected_flowers if flower.fragrance)
        total_stems = sum(quantity for _, quantity in selected_flowers)
        fragrance_factor = (fragrant_stems / total_stems) if spec.fragrance_preference else 0.5
        
        # Style appropriateness
        style_matches = sum(
            quantity for flower, quantity in selected_flowers
            if spec.style_preference in flower.style_suitability
        )
        style_factor = style_matches / total_stems if total_stems > 0 else 0
        
        # Combine factors
        impact_score = (size_diversity * 2 + color_diversity * 3 + fragrance_factor * 2 + style_factor * 3) / 10
        return min(10, max(1, impact_score * 10))
    
    def _calculate_seasonal_score(self, selected_flowers: List[Tuple[FlowerSpecies, int]], 
                                season: Season) -> float:
        """Calculate seasonal appropriateness score"""
        
        seasonal_stems = sum(
            quantity for flower, quantity in selected_flowers
            if season in flower.peak_seasons
        )
        total_stems = sum(quantity for _, quantity in selected_flowers)
        
        if total_stems == 0:
            return 5.0
        
        seasonal_ratio = seasonal_stems / total_stems
        return min(10, max(1, seasonal_ratio * 10 + 3))  # Base score of 3, up to 10
    
    def _generate_design_notes(self, selected_flowers: List[Tuple[FlowerSpecies, int]], 
                             spec: ArrangementSpec) -> str:
        """Generate descriptive notes for the arrangement design"""
        
        flower_names = [flower.name for flower, _ in selected_flowers]
        primary_colors = []
        for flower, _ in selected_flowers:
            primary_colors.extend(flower.colors[:2])  # Take first 2 colors
        
        unique_colors = list(set(primary_colors))
        
        notes = f"Elegant {spec.style_preference.value} arrangement featuring {', '.join(flower_names)}. "
        notes += f"Color palette: {', '.join(unique_colors)}. "
        notes += f"Designed for {spec.size_category} display with {spec.color_scheme.value} color scheme."
        
        return notes
    
    def generate_weekly_schedule(self, week_start: datetime, location_ids: List[str]) -> Dict[str, List[str]]:
        """Generate weekly arrangement schedule for multiple locations"""
        
        weekly_arrangements = {}
        
        for location_id in location_ids:
            # Get client preferences (simulated)
            client_prefs = self.client_preferences.get(location_id, {})
            budget = client_prefs.get('weekly_budget', 150.00)
            style = client_prefs.get('preferred_style', None)
            
            # Create arrangement specification
            arrangement_id = self.create_arrangement_spec(
                client_location_id=location_id,
                week_start_date=week_start,
                budget=budget,
                style_preference=style
            )
            
            # Design the arrangement
            design = self.design_arrangement(arrangement_id)
            
            if location_id not in weekly_arrangements:
                weekly_arrangements[location_id] = []
            weekly_arrangements[location_id].append(arrangement_id)
        
        return weekly_arrangements
    
    def optimize_delivery_route(self, arrangement_ids: List[str], delivery_date: datetime) -> DeliverySchedule:
        """Optimize delivery route for multiple arrangements"""
        
        # Group arrangements by location
        location_groups = defaultdict(list)
        for arr_id in arrangement_ids:
            if arr_id in self.arrangement_specs:
                spec = self.arrangement_specs[arr_id]
                location_groups[spec.client_location_id].append(arr_id)
        
        # Simulate route optimization (in production, would use actual addresses)
        location_ids = list(location_groups.keys())
        random.shuffle(location_ids)  # Simple randomization for demo
        
        # Calculate delivery costs
        base_cost = 25.00  # Base delivery fee
        per_location_cost = 15.00
        total_cost = base_cost + (per_location_cost * len(location_ids))
        
        schedule_id = f"DEL-{delivery_date.strftime('%Y%m%d')}-{random.randint(100, 999)}"
        
        schedule = DeliverySchedule(
            schedule_id=schedule_id,
            delivery_date=delivery_date,
            delivery_time_window=("08:00", "12:00"),
            arrangements=arrangement_ids,
            delivery_route=location_ids,
            driver_assigned="TBD",
            total_delivery_cost=total_cost,
            estimated_duration=len(location_ids) * 0.5 + 1,  # 30 min per location + 1 hour base
            special_instructions=[]
        )
        
        self.delivery_schedules[schedule_id] = schedule
        return schedule
    
    def set_client_preferences(self, location_id: str, preferences: Dict[str, Any]):
        """Set arrangement preferences for client location"""
        self.client_preferences[location_id] = preferences
    
    def get_arrangement_cost_analysis(self, arrangement_id: str) -> Dict[str, Any]:
        """Get detailed cost analysis for arrangement"""
        
        if arrangement_id not in self.arrangement_designs:
            raise ValueError(f"Arrangement design {arrangement_id} not found")
        
        design = self.arrangement_designs[arrangement_id]
        spec = design.arrangement_spec
        
        flower_costs = []
        for flower_name, quantity in design.flowers.items():
            flower = self.flower_db.flower_index.get(flower_name.lower())
            if flower:
                cost = flower.cost_per_stem * quantity
                flower_costs.append({
                    "flower": flower_name,
                    "quantity": quantity,
                    "cost_per_stem": flower.cost_per_stem,
                    "total_cost": cost,
                    "percentage_of_budget": (cost / spec.budget) * 100
                })
        
        total_flower_cost = sum(item["total_cost"] for item in flower_costs)
        remaining_budget = spec.budget - total_flower_cost
        
        return {
            "arrangement_id": arrangement_id,
            "total_budget": spec.budget,
            "total_flower_cost": total_flower_cost,
            "remaining_budget": remaining_budget,
            "budget_utilization": (total_flower_cost / spec.budget) * 100,
            "flower_breakdown": flower_costs,
            "estimated_longevity": design.estimated_longevity,
            "cost_per_day": total_flower_cost / design.estimated_longevity if design.estimated_longevity > 0 else 0
        }


def demo_lobby_planner():
    """Demonstration of the lobby flower arrangement planner"""
    print("Cambridge NY Lobby Flower Arrangement Planner Demo")
    print("=" * 52)
    
    # Initialize planner
    planner = CambridgeLobbyPlanner()
    
    print(f"🌸 Flower Database: {len(planner.flower_db.flowers)} species loaded")
    
    # Set client preferences
    planner.set_client_preferences("LOC001", {
        "weekly_budget": 200.00,
        "preferred_style": ArrangementStyle.MODERN,
        "fragrance_preference": True
    })
    
    # Create arrangement for next week
    next_monday = datetime.now() + timedelta(days=(7 - datetime.now().weekday()))
    
    arrangement_id = planner.create_arrangement_spec(
        client_location_id="LOC001",
        week_start_date=next_monday,
        budget=200.00,
        style_preference=ArrangementStyle.MODERN,
        color_scheme=ColorScheme.ANALOGOUS
    )
    
    print(f"\n🎨 Created Arrangement Spec: {arrangement_id}")
    
    # Design the arrangement
    design = planner.design_arrangement(arrangement_id)
    
    print(f"\n💐 Arrangement Design:")
    print(f"  Flowers: {', '.join(design.flowers.keys())}")
    print(f"  Total Cost: ${design.total_cost:.2f}")
    print(f"  Visual Impact: {design.visual_impact_score:.1f}/10")
    print(f"  Estimated Longevity: {design.estimated_longevity:.1f} days")
    
    # Cost analysis
    cost_analysis = planner.get_arrangement_cost_analysis(arrangement_id)
    print(f"\n💰 Cost Analysis:")
    print(f"  Budget Utilization: {cost_analysis['budget_utilization']:.1f}%")
    print(f"  Cost per Day: ${cost_analysis['cost_per_day']:.2f}")
    
    # Generate weekly schedule
    weekly_schedule = planner.generate_weekly_schedule(next_monday, ["LOC001", "LOC002"])
    print(f"\n📅 Weekly Schedule: {len(weekly_schedule)} locations")
    
    # Optimize delivery
    all_arrangements = []
    for arrangements in weekly_schedule.values():
        all_arrangements.extend(arrangements)
    
    delivery_schedule = planner.optimize_delivery_route(all_arrangements, next_monday)
    print(f"\n🚚 Delivery Schedule: {delivery_schedule.schedule_id}")
    print(f"  Route: {len(delivery_schedule.delivery_route)} stops")
    print(f"  Total Cost: ${delivery_schedule.total_delivery_cost:.2f}")
    print(f"  Estimated Duration: {delivery_schedule.estimated_duration:.1f} hours")


if __name__ == "__main__":
    demo_lobby_planner()
"""
Cambridge NY Plant Rental Inventory Management System
====================================================

Comprehensive plant rental tracking system for Cambridge NY's commercial operations.
Tracks plant locations, rotation schedules, replacement triggers, and cost analysis.

Integrates with Engram's memory system to maintain historical rental data and optimize
plant placement decisions based on success patterns.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np

from .plant_knowledge import CambridgePlantDatabase, PlantSpecies
from .plant_health_monitor import CambridgeHealthMonitor, HealthStatus, EnvironmentalData


class RentalStatus(Enum):
    ACTIVE = "active"
    SCHEDULED = "scheduled"
    MAINTENANCE = "maintenance"
    REPLACEMENT_DUE = "replacement_due"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlantCondition(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    RETIRED = "retired"


class LocationType(Enum):
    LOBBY = "lobby"
    OFFICE = "office"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    CONFERENCE_ROOM = "conference_room"
    RECEPTION = "reception"
    HALLWAY = "hallway"


@dataclass
class ClientLocation:
    """Commercial client location information"""
    location_id: str
    client_name: str
    address: str
    contact_person: str
    contact_email: str
    contact_phone: str
    location_type: LocationType
    square_footage: Optional[int] = None
    foot_traffic: str = "medium"  # low, medium, high
    lighting_conditions: str = "mixed"  # low, medium, bright, mixed
    climate_controlled: bool = True
    notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class PlantInventoryItem:
    """Individual plant in Cambridge's rental inventory"""
    plant_id: str
    species_name: str
    purchase_date: datetime
    purchase_cost: float
    current_condition: PlantCondition
    pot_size: str
    pot_style: str
    height: float  # inches
    last_maintenance: datetime
    maintenance_notes: str = ""
    total_rental_days: int = 0
    times_rented: int = 0
    retirement_date: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class RentalContract:
    """Plant rental contract with client"""
    contract_id: str
    client_location: ClientLocation
    start_date: datetime
    end_date: Optional[datetime]
    contract_type: str  # "monthly", "weekly", "seasonal", "event"
    monthly_rate: float
    plants_included: List[str]  # plant_ids
    service_frequency: int  # days between maintenance visits
    contract_status: RentalStatus
    special_requirements: List[str] = field(default_factory=list)
    auto_renew: bool = True
    payment_terms: str = "monthly"
    discount_applied: float = 0.0
    notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class PlantPlacement:
    """Specific plant placement at client location"""
    placement_id: str
    plant_id: str
    contract_id: str
    location_id: str
    placement_date: datetime
    specific_location: str  # "Front lobby corner", "Reception desk", etc.
    environmental_conditions: Optional[EnvironmentalData] = None
    rotation_schedule: int = 30  # days
    next_rotation: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    performance_rating: float = 5.0  # 1-10 scale
    client_satisfaction: float = 5.0  # 1-10 scale
    maintenance_history: List[str] = field(default_factory=list)
    issue_history: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class MaintenanceVisit:
    """Record of maintenance visit to client location"""
    visit_id: str
    contract_id: str
    location_id: str
    visit_date: datetime
    technician_name: str
    plants_serviced: List[str]  # plant_ids
    services_performed: List[str]
    plants_replaced: List[Tuple[str, str]]  # (old_plant_id, new_plant_id)
    issues_found: List[str]
    client_feedback: str = ""
    visit_duration: float = 0.0  # hours
    cost: float = 0.0
    next_visit_scheduled: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    photos_taken: List[str] = field(default_factory=list)  # file paths
    created_date: datetime = field(default_factory=datetime.now)


class CambridgeRentalTracker:
    """Main plant rental tracking and management system"""
    
    def __init__(self, plant_database: Optional[CambridgePlantDatabase] = None,
                 health_monitor: Optional[CambridgeHealthMonitor] = None):
        self.plant_db = plant_database or CambridgePlantDatabase()
        self.health_monitor = health_monitor or CambridgeHealthMonitor(self.plant_db)
        
        # Core data storage
        self.client_locations: Dict[str, ClientLocation] = {}
        self.plant_inventory: Dict[str, PlantInventoryItem] = {}
        self.rental_contracts: Dict[str, RentalContract] = {}
        self.plant_placements: Dict[str, PlantPlacement] = {}
        self.maintenance_visits: Dict[str, MaintenanceVisit] = {}
        
        # Initialize with demo data
        self._initialize_demo_data()
    
    def _initialize_demo_data(self):
        """Initialize system with sample data for demonstration"""
        
        # Sample client locations
        self.add_client_location(
            client_name="Manhattan Corporate Plaza",
            address="123 Broadway, New York, NY 10001",
            contact_person="Sarah Johnson",
            contact_email="sarah.johnson@manplaza.com",
            contact_phone="(212) 555-0123",
            location_type=LocationType.LOBBY,
            square_footage=2500,
            foot_traffic="high",
            lighting_conditions="bright"
        )
        
        self.add_client_location(
            client_name="Tech Startup Hub",
            address="456 Park Ave, New York, NY 10022",
            contact_person="Mike Chen",
            contact_email="mike@techstartup.com",
            contact_phone="(212) 555-0456",
            location_type=LocationType.OFFICE,
            square_footage=1200,
            foot_traffic="medium",
            lighting_conditions="mixed"
        )
        
        # Sample plant inventory
        self.add_plant_to_inventory(
            species_name="Snake Plant",
            purchase_cost=45.00,
            pot_size="10 inch",
            pot_style="Modern ceramic",
            height=36
        )
        
        self.add_plant_to_inventory(
            species_name="Fiddle Leaf Fig",
            purchase_cost=85.00,
            pot_size="12 inch",
            pot_style="Woven basket",
            height=60
        )
    
    def add_client_location(self, client_name: str, address: str, contact_person: str,
                          contact_email: str, contact_phone: str, location_type: LocationType,
                          **kwargs) -> str:
        """Add new client location to system"""
        
        location_id = str(uuid.uuid4())
        location = ClientLocation(
            location_id=location_id,
            client_name=client_name,
            address=address,
            contact_person=contact_person,
            contact_email=contact_email,
            contact_phone=contact_phone,
            location_type=location_type,
            **kwargs
        )
        
        self.client_locations[location_id] = location
        return location_id
    
    def add_plant_to_inventory(self, species_name: str, purchase_cost: float,
                             pot_size: str, pot_style: str, height: float) -> str:
        """Add new plant to rental inventory"""
        
        # Validate species exists
        if not self.plant_db.get_plant_by_name(species_name):
            raise ValueError(f"Unknown plant species: {species_name}")
        
        plant_id = f"CPL-{uuid.uuid4().hex[:8].upper()}"  # Cambridge Plant Library ID
        plant = PlantInventoryItem(
            plant_id=plant_id,
            species_name=species_name,
            purchase_date=datetime.now(),
            purchase_cost=purchase_cost,
            current_condition=PlantCondition.EXCELLENT,
            pot_size=pot_size,
            pot_style=pot_style,
            height=height,
            last_maintenance=datetime.now()
        )
        
        self.plant_inventory[plant_id] = plant
        return plant_id
    
    def create_rental_contract(self, location_id: str, start_date: datetime,
                             contract_type: str, monthly_rate: float,
                             service_frequency: int = 30, **kwargs) -> str:
        """Create new plant rental contract"""
        
        if location_id not in self.client_locations:
            raise ValueError(f"Location {location_id} not found")
        
        contract_id = f"RC-{uuid.uuid4().hex[:8].upper()}"
        contract = RentalContract(
            contract_id=contract_id,
            client_location=self.client_locations[location_id],
            start_date=start_date,
            contract_type=contract_type,
            monthly_rate=monthly_rate,
            service_frequency=service_frequency,
            contract_status=RentalStatus.SCHEDULED,
            plants_included=[],
            **kwargs
        )
        
        self.rental_contracts[contract_id] = contract
        return contract_id
    
    def add_plant_to_contract(self, contract_id: str, plant_id: str, specific_location: str,
                            rotation_schedule: int = 30) -> str:
        """Add plant to existing rental contract"""
        
        if contract_id not in self.rental_contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        if plant_id not in self.plant_inventory:
            raise ValueError(f"Plant {plant_id} not found in inventory")
        
        # Check if plant is available
        if self.is_plant_currently_rented(plant_id):
            raise ValueError(f"Plant {plant_id} is currently rented to another client")
        
        placement_id = str(uuid.uuid4())
        placement = PlantPlacement(
            placement_id=placement_id,
            plant_id=plant_id,
            contract_id=contract_id,
            location_id=self.rental_contracts[contract_id].client_location.location_id,
            placement_date=datetime.now(),
            specific_location=specific_location,
            rotation_schedule=rotation_schedule
        )
        
        self.plant_placements[placement_id] = placement
        
        # Update contract and plant records
        self.rental_contracts[contract_id].plants_included.append(plant_id)
        self.plant_inventory[plant_id].times_rented += 1
        
        return placement_id
    
    def is_plant_currently_rented(self, plant_id: str) -> bool:
        """Check if plant is currently rented to a client"""
        for placement in self.plant_placements.values():
            if placement.plant_id == plant_id:
                contract = self.rental_contracts.get(placement.contract_id)
                if contract and contract.contract_status == RentalStatus.ACTIVE:
                    return True
        return False
    
    def schedule_maintenance_visit(self, contract_id: str, visit_date: datetime,
                                 technician_name: str) -> str:
        """Schedule maintenance visit for contract location"""
        
        if contract_id not in self.rental_contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        visit_id = str(uuid.uuid4())
        visit = MaintenanceVisit(
            visit_id=visit_id,
            contract_id=contract_id,
            location_id=self.rental_contracts[contract_id].client_location.location_id,
            visit_date=visit_date,
            technician_name=technician_name,
            plants_serviced=self.rental_contracts[contract_id].plants_included.copy()
        )
        
        self.maintenance_visits[visit_id] = visit
        return visit_id
    
    def record_maintenance_completion(self, visit_id: str, services_performed: List[str],
                                    plants_replaced: List[Tuple[str, str]] = None,
                                    issues_found: List[str] = None,
                                    visit_duration: float = 0.0,
                                    client_feedback: str = "") -> None:
        """Record completion of maintenance visit"""
        
        if visit_id not in self.maintenance_visits:
            raise ValueError(f"Visit {visit_id} not found")
        
        visit = self.maintenance_visits[visit_id]
        visit.services_performed = services_performed
        visit.plants_replaced = plants_replaced or []
        visit.issues_found = issues_found or []
        visit.visit_duration = visit_duration
        visit.client_feedback = client_feedback
        
        # Update plant maintenance records
        for plant_id in visit.plants_serviced:
            if plant_id in self.plant_inventory:
                self.plant_inventory[plant_id].last_maintenance = visit.visit_date
                self.plant_inventory[plant_id].total_rental_days += visit.service_frequency
        
        # Handle plant replacements
        for old_plant_id, new_plant_id in visit.plants_replaced:
            self._process_plant_replacement(old_plant_id, new_plant_id, visit.contract_id)
    
    def _process_plant_replacement(self, old_plant_id: str, new_plant_id: str, contract_id: str):
        """Process plant replacement in contract"""
        
        # Find placement to update
        for placement in self.plant_placements.values():
            if placement.plant_id == old_plant_id and placement.contract_id == contract_id:
                placement.plant_id = new_plant_id
                break
        
        # Update contract plant list
        contract = self.rental_contracts[contract_id]
        if old_plant_id in contract.plants_included:
            contract.plants_included.remove(old_plant_id)
            contract.plants_included.append(new_plant_id)
        
        # Update plant inventory records
        if old_plant_id in self.plant_inventory:
            old_plant = self.plant_inventory[old_plant_id]
            if old_plant.current_condition == PlantCondition.POOR:
                old_plant.retirement_date = datetime.now()
        
        if new_plant_id in self.plant_inventory:
            self.plant_inventory[new_plant_id].times_rented += 1
    
    def get_plants_due_for_rotation(self, days_ahead: int = 7) -> List[PlantPlacement]:
        """Get plants that need rotation within specified days"""
        
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        due_for_rotation = []
        
        for placement in self.plant_placements.values():
            if placement.next_rotation <= cutoff_date:
                # Check if contract is still active
                contract = self.rental_contracts.get(placement.contract_id)
                if contract and contract.contract_status == RentalStatus.ACTIVE:
                    due_for_rotation.append(placement)
        
        return due_for_rotation
    
    def get_plants_needing_replacement(self) -> List[PlantInventoryItem]:
        """Identify plants that need replacement based on condition and age"""
        
        replacement_candidates = []
        
        for plant in self.plant_inventory.values():
            # Skip already retired plants
            if plant.retirement_date:
                continue
            
            # Check condition
            if plant.current_condition in [PlantCondition.POOR, PlantCondition.RETIRED]:
                replacement_candidates.append(plant)
                continue
            
            # Check age and usage
            plant_age_months = (datetime.now() - plant.purchase_date).days / 30.44
            species_info = self.plant_db.get_plant_by_name(plant.species_name)
            
            if species_info and plant_age_months >= species_info.care_profile.replacement_cycle:
                replacement_candidates.append(plant)
                continue
            
            # Check rental intensity
            if plant.times_rented > 10 and plant.total_rental_days > 365:
                replacement_candidates.append(plant)
        
        return replacement_candidates
    
    def generate_rotation_schedule(self, weeks_ahead: int = 8) -> Dict[str, List[PlantPlacement]]:
        """Generate plant rotation schedule for upcoming weeks"""
        
        schedule = {}
        start_date = datetime.now()
        
        for week in range(weeks_ahead):
            week_start = start_date + timedelta(weeks=week)
            week_end = week_start + timedelta(days=7)
            week_key = week_start.strftime("%Y-W%U")
            
            week_rotations = []
            for placement in self.plant_placements.values():
                if week_start <= placement.next_rotation <= week_end:
                    contract = self.rental_contracts.get(placement.contract_id)
                    if contract and contract.contract_status == RentalStatus.ACTIVE:
                        week_rotations.append(placement)
            
            if week_rotations:
                schedule[week_key] = week_rotations
        
        return schedule
    
    def calculate_contract_profitability(self, contract_id: str) -> Dict[str, float]:
        """Calculate profitability metrics for a rental contract"""
        
        if contract_id not in self.rental_contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        contract = self.rental_contracts[contract_id]
        
        # Calculate contract duration
        if contract.end_date:
            duration_days = (contract.end_date - contract.start_date).days
        else:
            duration_days = (datetime.now() - contract.start_date).days
        
        # Calculate revenue
        monthly_revenue = contract.monthly_rate * (1 - contract.discount_applied / 100)
        total_revenue = monthly_revenue * (duration_days / 30.44)
        
        # Calculate costs
        plant_costs = sum(self.plant_inventory[pid].purchase_cost 
                         for pid in contract.plants_included 
                         if pid in self.plant_inventory)
        
        # Estimate maintenance costs (average $25 per plant per visit)
        maintenance_visits_count = len([v for v in self.maintenance_visits.values() 
                                      if v.contract_id == contract_id])
        maintenance_costs = maintenance_visits_count * len(contract.plants_included) * 25
        
        # Calculate profit
        total_costs = plant_costs + maintenance_costs
        total_profit = total_revenue - total_costs
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        return {
            "total_revenue": total_revenue,
            "total_costs": total_costs,
            "total_profit": total_profit,
            "profit_margin": profit_margin,
            "monthly_revenue": monthly_revenue,
            "plant_costs": plant_costs,
            "maintenance_costs": maintenance_costs,
            "duration_days": duration_days
        }
    
    def get_client_satisfaction_report(self, location_id: str) -> Dict[str, Any]:
        """Generate client satisfaction report for location"""
        
        if location_id not in self.client_locations:
            raise ValueError(f"Location {location_id} not found")
        
        location = self.client_locations[location_id]
        
        # Get all placements for this location
        location_placements = [p for p in self.plant_placements.values() 
                             if p.location_id == location_id]
        
        if not location_placements:
            return {"error": "No plants currently at location"}
        
        # Calculate satisfaction metrics
        avg_performance = np.mean([p.performance_rating for p in location_placements])
        avg_satisfaction = np.mean([p.client_satisfaction for p in location_placements])
        
        # Get maintenance visit feedback
        location_visits = [v for v in self.maintenance_visits.values() 
                          if v.location_id == location_id and v.client_feedback]
        
        total_issues = sum(len(p.issue_history) for p in location_placements)
        
        return {
            "client_name": location.client_name,
            "location_type": location.location_type.value,
            "plants_count": len(location_placements),
            "avg_performance_rating": round(avg_performance, 2),
            "avg_satisfaction_rating": round(avg_satisfaction, 2),
            "total_maintenance_visits": len(location_visits),
            "total_issues_reported": total_issues,
            "recent_feedback": [v.client_feedback for v in location_visits[-3:]]
        }
    
    def export_rental_data(self, filepath: str):
        """Export rental tracking data to JSON"""
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "client_locations": {k: self._serialize_dataclass(v) for k, v in self.client_locations.items()},
            "plant_inventory": {k: self._serialize_dataclass(v) for k, v in self.plant_inventory.items()},
            "rental_contracts": {k: self._serialize_dataclass(v) for k, v in self.rental_contracts.items()},
            "plant_placements": {k: self._serialize_dataclass(v) for k, v in self.plant_placements.items()},
            "maintenance_visits": {k: self._serialize_dataclass(v) for k, v in self.maintenance_visits.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _serialize_dataclass(self, obj) -> Dict:
        """Convert dataclass to dictionary for JSON serialization"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, Enum):
                    result[key] = value.value
                elif hasattr(value, '__dict__'):
                    result[key] = self._serialize_dataclass(value)
                else:
                    result[key] = value
            return result
        return obj


def demo_rental_tracker():
    """Demonstration of the rental tracking system"""
    print("Cambridge NY Plant Rental Tracker Demo")
    print("=" * 40)
    
    # Initialize system
    tracker = CambridgeRentalTracker()
    
    print(f"📍 Client Locations: {len(tracker.client_locations)}")
    print(f"🌿 Plant Inventory: {len(tracker.plant_inventory)}")
    
    # Create a rental contract
    location_id = list(tracker.client_locations.keys())[0]
    contract_id = tracker.create_rental_contract(
        location_id=location_id,
        start_date=datetime.now(),
        contract_type="monthly",
        monthly_rate=350.00,
        service_frequency=14
    )
    
    print(f"\n📋 Created Contract: {contract_id}")
    
    # Add plants to contract
    plant_ids = list(tracker.plant_inventory.keys())
    placement_id = tracker.add_plant_to_contract(
        contract_id=contract_id,
        plant_id=plant_ids[0],
        specific_location="Main lobby entrance",
        rotation_schedule=30
    )
    
    print(f"🌱 Added Plant Placement: {placement_id}")
    
    # Check rotation schedule
    rotation_schedule = tracker.generate_rotation_schedule(weeks_ahead=4)
    print(f"\n📅 Upcoming Rotations: {len(rotation_schedule)} weeks scheduled")
    
    # Calculate profitability
    profitability = tracker.calculate_contract_profitability(contract_id)
    print(f"\n💰 Contract Profitability:")
    print(f"  Monthly Revenue: ${profitability['monthly_revenue']:.2f}")
    print(f"  Profit Margin: {profitability['profit_margin']:.1f}%")
    
    # Client satisfaction
    satisfaction = tracker.get_client_satisfaction_report(location_id)
    print(f"\n⭐ Client Satisfaction:")
    print(f"  Performance Rating: {satisfaction['avg_performance_rating']}/10")
    print(f"  Plants Count: {satisfaction['plants_count']}")


if __name__ == "__main__":
    demo_rental_tracker()
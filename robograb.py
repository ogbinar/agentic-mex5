# robograb.py

import random
import time
from dataclasses import dataclass

# Use dataclasses for clean input/output objects

@dataclass
class DetectionResult:
    object_name: str
    position: dict

@dataclass
class TrajectoryPlan:
    target_position: dict
    waypoints: list

@dataclass
class ExecutionResult:
    success: bool
    trajectory: TrajectoryPlan


# Step 1: Detect Object
def detect_object(object_name: str) -> DetectionResult:
    print(f"Detecting object: {object_name}...")
    time.sleep(0.5)
    position = {
        'x': round(random.uniform(0.5, 2.0), 2),
        'y': round(random.uniform(-1.0, 1.0), 2),
        'z': 0.0
    }
    print(f"Object '{object_name}' detected at position: {position}")
    return DetectionResult(object_name=object_name, position=position)

# Step 2: Plan Pick Trajectory
def plan_pick(detection_result: DetectionResult) -> TrajectoryPlan:
    print(f"Planning trajectory for object '{detection_result.object_name}' at {detection_result.position}...")
    time.sleep(0.5)
    waypoints = [
        {'x': detection_result.position['x'] - 0.2, 'y': detection_result.position['y'], 'z': 0.3},
        {'x': detection_result.position['x'], 'y': detection_result.position['y'], 'z': detection_result.position['z']}
    ]
    print("Trajectory planned.")
    return TrajectoryPlan(target_position=detection_result.position, waypoints=waypoints)

# Step 3: Execute Trajectory
def execute_trajectory(trajectory_plan: TrajectoryPlan) -> ExecutionResult:
    print(f"Executing trajectory to position {trajectory_plan.target_position}...")
    time.sleep(1)
    success = random.choice([True, False, True])
    if success:
        print("✅ Movement succeeded.")
    else:
        print("❌ Movement failed.")
    return ExecutionResult(success=success, trajectory=trajectory_plan)

# Optional full chain function
def pick_object(object_name: str):
    detection = detect_object(object_name)
    trajectory = plan_pick(detection)
    execution = execute_trajectory(trajectory)
    return execution


# Test run
if __name__ == "__main__":
    obj = "remote control"
    for attempt in range(3):
        print(f"\nAttempt {attempt + 1}:")
        result = pick_object(obj)
        if result.success:
            print(f"Task complete. {obj} picked successfully.")
            break
        else:
            print("Replanning and retrying...")

# mcp_server.py

from fastmcp import FastMCP
from robograb import detect_object, plan_pick, execute_trajectory, DetectionResult, TrajectoryPlan, ExecutionResult
import json
from dataclasses import asdict

mcp = FastMCP()

# Utility function to convert dataclasses to dict for serialization
def dataclass_to_dict(obj):
    return asdict(obj)

@mcp.tool()
def detect_tool(object_name: str) -> dict:
    detection = detect_object(object_name)
    return dataclass_to_dict(detection)

@mcp.tool()
def plan_tool(detection_result: dict) -> dict:
    # Convert dict back to DetectionResult dataclass
    detection = DetectionResult(
        object_name=detection_result["object_name"],
        position=detection_result["position"]
    )
    trajectory = plan_pick(detection)
    return dataclass_to_dict(trajectory)

@mcp.tool()
def execute_tool(trajectory_plan: dict) -> dict:
    # Convert dict back to TrajectoryPlan dataclass
    trajectory = TrajectoryPlan(
        target_position=trajectory_plan["target_position"],
        waypoints=trajectory_plan["waypoints"]
    )
    result = execute_trajectory(trajectory)
    return dataclass_to_dict(result)

if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )

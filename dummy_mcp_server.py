# dummy_mcp_server.py

from fastmcp import FastMCP
from dataclasses import dataclass, asdict

# Initialize FastMCP server
enabled_transport = "streamable-http"
mcp = FastMCP()

# --------------------
# Data classes for tool outputs
# --------------------

@dataclass
class CaptureFrameResult:
    image_path: str
    depth_path: str

@dataclass
class LoadInputsResult:
    img_shape: list
    depth_shape: list

@dataclass
class DetectObjectResult:
    bbox: list
    cls: str

@dataclass
class SegmentObjectResult:
    mask_shape: list

@dataclass
class GraspGeometryResult:
    center: list
    angle: float
    width: float

@dataclass
class ContainerResult:
    container: list

@dataclass
class MidpointResult:
    pickup_depth: float
    drop_depth: float
    mid_depth: float

@dataclass
class WorldXYResult:
    world_xy: list

@dataclass
class TrajectoryResult:
    trajectory: list

@dataclass
class ExecutionResult:
    success: bool
    trajectory: list = None

@dataclass
class FinalAnswerResult:
    message: str

# --------------------
# mex5 tools
# --------------------

@mcp.tool()
def capture_frame() -> dict:
    """Dummy capture: fixed test image paths."""
    result = CaptureFrameResult(
        image_path="rgb_pic.png",
        depth_path="depth_map.npy"
    )
    return asdict(result)

@mcp.tool()
def load_inputs(image_path: str, depth_path: str) -> dict:
    """Dummy loader: returns image and depth shapes."""
    result = LoadInputsResult(
        img_shape=[480, 640, 3],
        depth_shape=[480, 640]
    )
    return asdict(result)

@mcp.tool()
def detect_object(img_shape: list, target_class: str) -> dict:
    """Dummy YOLO: returns fixed bbox and class."""
    result = DetectObjectResult(
        bbox=[100, 100, 200, 200],
        cls=target_class
    )
    return asdict(result)

@mcp.tool()
def segment_object(mask_shape: list, bbox: list) -> dict:
    """Dummy segmentation: returns full-mask shape."""
    result = SegmentObjectResult(mask_shape=[480, 640])
    return asdict(result)

@mcp.tool()
def compute_grasp_geometry(mask_shape: list) -> dict:
    """Dummy grasp geometry: center at mask center, zero angle."""
    h, w = mask_shape
    result = GraspGeometryResult(
        center=[w//2, h//2],
        angle=0.0,
        width=float(min(w, h) / 4)
    )
    return asdict(result)

@mcp.tool()
def detect_container(img_shape: list) -> dict:
    """Dummy container detection: fixed centroid."""
    result = ContainerResult(container=[300, 300])
    return asdict(result)

@mcp.tool()
def compute_midpoint(depth_shape: list, coords: dict) -> dict:
    """Dummy midpoint depths: fixed values."""
    result = MidpointResult(
        pickup_depth=0.1,
        drop_depth=0.2,
        mid_depth=0.15
    )
    return asdict(result)

@mcp.tool()
def pixel_to_world(pixel: list, y_limits: list, x_limits: list, img_shape: list) -> dict:
    """Dummy pixel->world mapping: scale by 0.001."""
    x_pix, y_pix = pixel
    result = WorldXYResult(world_xy=[x_pix * 0.001, y_pix * 0.001])
    return asdict(result)

@mcp.tool()
def plan_pick(world_start: list, world_target: list, mid_depth: float, angle: float) -> dict:
    """Dummy planning: create simple two-point trajectory."""
    waypoints = [
        {"x": world_start[0], "y": world_start[1], "z": mid_depth + 0.1, "angle": 0.0},
        {"x": world_target[0], "y": world_target[1], "z": mid_depth,       "angle": angle}
    ]
    result = TrajectoryResult(trajectory=waypoints)
    return asdict(result)

@mcp.tool()
def execute_motion(trajectory: list) -> dict:
    """Dummy execution: always succeeds."""
    result = ExecutionResult(success=True, trajectory=trajectory)
    return asdict(result)

@mcp.tool()
def final_answer(message: str) -> dict:
    """Return the final natural-language answer."""
    result = FinalAnswerResult(message=message)
    return asdict(result)

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000, transport=enabled_transport)

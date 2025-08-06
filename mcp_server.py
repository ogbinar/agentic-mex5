# mcp_server.py
#!/usr/bin/env python3
"""
Object Detection and Manipulation Server (MCP)
Provides tools for robotic vision and pick-and-place via FastMCP.
"""

# === Standard library imports ===
import os
import json
from typing import Any, Dict, List, Optional, Tuple

# === Third-party imports ===
import cv2
import numpy as np
import requests
import pyrealsense2 as rs
from pydantic import Field
from fastmcp import FastMCP
from ultralytics import YOLOWorld, SAM

# === Configuration ===
TEST_MODE = True  # Set to False when connected to a real robot server

# Predefined world coordinate limits for pixel-to-world mapping
Y_LIMITS = [-0.33, 0.33]
X_LIMITS = [0.31, 0.57]

# === REST Client for Robot Commands ===
class RestClient:
    """
    Sends float-array commands to the robot server via HTTP POST to /api/floats.
    """
    def __init__(self, ip: str = "192.168.2.1", port: int = 34568):
        self.ip = ip
        self.port = port
        self.url = f"http://{self.ip}:{self.port}/api/floats"

    def send_floats(self, command: str, parameters: List[float]) -> requests.Response:
        payload = {command: parameters}
        return requests.post(self.url, json=payload)

    def move_to_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        rotation: Optional[Tuple[float, float, float]] = None
    ) -> requests.Response:
        params = [x, y, z, t]
        if rotation:
            params.extend(rotation)
        return self.send_floats("moveToCartesian", params)

    def close_gripper(self) -> requests.Response:
        return self.send_floats("closeGripper", [0.03])

    def open_gripper(self) -> requests.Response:
        return self.send_floats("openGripper", [0.1])

# === Instantiate MCP server ===
mcp = FastMCP("Object Detection and Manipulation Server")

# === Tool Definitions ===

@mcp.tool(title="Echo Tool", description="Echos the input text back to the user.")
def echo_tool(
    text: str = Field(..., description="The text to echo back to the user.")
) -> str:
    return text


@mcp.tool(
    title="Capture Frame",
    description="Returns paths to the current RGB image and depth map for processing."
)
def capture_frame() -> Dict[str, str]:
    """
    Captures aligned RGB and depth at 640×480, saving 'rgb_pic.png' and 'depth_map.npy'.
    """
    if TEST_MODE:
        return {
            "image_path": "/projects/agentic-mex5/rgb_pic.png",
            "depth_path": "/projects/agentic-mex5/depth_map.npy"
        }

    TARGET_WIDTH, TARGET_HEIGHT = 640, 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, TARGET_WIDTH, TARGET_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, TARGET_WIDTH, TARGET_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not (color_frame and depth_frame):
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            # Resize if needed
            if color_img.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                color_img = cv2.resize(color_img, (TARGET_WIDTH, TARGET_HEIGHT))
            if depth_data.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                depth_data = cv2.resize(depth_data, (TARGET_WIDTH, TARGET_HEIGHT))

            cv2.imwrite("rgb_pic.png", color_img)
            np.save("depth_map.npy", depth_data)
            break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return {"image_path": "rgb_pic.png", "depth_path": "depth_map.npy"}


@mcp.tool(
    title="Detect Object",
    description="Detects a specified object class in an image using YOLOWorld and returns bbox & label."
)
def detect_object(
    target_class: str = Field(..., description="The object class label (e.g., 'scissor')."),
    img: Any = Field(None, description="Image array or file path.")
) -> Dict[str, Any]:
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    model = YOLOWorld("/projects/agentic-mex5/models/yolov8x-world.pt")
    model.set_classes([target_class])
    result = model.predict(img, imgsz=640, conf=0.20)[0]

    if not result.boxes:
        return {"bbox": None, "cls": None}

    box = result.boxes[0].xyxy[0].cpu().numpy().astype(int)
    cls_idx = int(result.boxes[0].cls[0])
    return {"bbox": box.tolist(), "cls": result.names[cls_idx]}


@mcp.tool(
    title="Segment Object",
    description="Segments a detected object in an image using SAM and saves mask & metadata."
)
def segment_object(
    bbox: List[int] = Field(..., description="Bounding box [x1,y1,x2,y2]."),
    img: Any = Field(None, description="Image array or file path.")
) -> Dict[str, Any]:
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    if len(bbox) != 4:
        return {"mask_path": None, "meta_path": None, "message": "Invalid bbox"}

    mask_path = "mask.png"
    meta_path = "mask_meta.json"

    sam = SAM("/projects/agentic-mex5/models/sam2.1_l.pt")
    seg = sam.predict(img, bboxes=[bbox])[0]
    mask = (seg.masks.data[0].cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite(mask_path, mask)
    metadata = {
        "bbox": bbox,
        "mask": seg.masks.data[0].cpu().numpy().tolist(),
        "image_dimensions": img.shape[:2],
        "model_used": "SAM 2.1 Large",
        "original_image": "/projects/agentic-mex5/rgb_pic.png"
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {"mask_path": mask_path, "meta_path": meta_path, "message": "Mask saved"}


@mcp.tool(
    title="Compute Grasp Geometry",
    description="Computes the grasp center, angle, and width from saved mask metadata."
)
def compute_grasp_geometry() -> Dict[str, Any]:
    data = json.load(open("mask_meta.json"))
    mask = np.array(data.get("mask", []), dtype=np.uint8)
    if mask.size == 0:
        return {"center": None, "angle": None, "width": None}

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"center": None, "angle": None, "width": None}

    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    angle = ang if rw > rh else -ang
    angle = 90 - angle if angle > 0 else angle
    width = min(rw, rh)

    return {"center": [int(cx), int(cy)], "angle": float(angle), "width": float(width)}


@mcp.tool(
    title="Detect Container",
    description="Detects a container via HSV color segmentation and returns its centroid."
)
def detect_container(
    img: Any = Field(None, description="Image array or path.")
) -> Dict[str, Any]:
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 30), (25, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 10000:
        return {"container": None}

    M = cv2.moments(max(cnts, key=cv2.contourArea))
    if M["m00"] == 0:
        return {"container": None}

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return {"container": [cx, cy]} 


@mcp.tool(
    title="Compute Midpoint Depth",
    description="Returns a fixed midpoint depth between target and container."
)
def compute_midpoint() -> Dict[str, float]:
    return {"mid_depth": 0.016}


@mcp.tool(
    title="Map Pixels to World",
    description="Converts image pixel coords to world-frame x,y using fixed table limits."
)
def map_pixels_to_world(
    target_pixel: List[int] = Field(..., description="[col, row] in image coords."),
    img_path: Optional[str] = Field(None, description="Path to base RGB image.")
) -> Dict[str, Any]:
    if not img_path:
        raise ValueError("img_path is required for pixel-to-world mapping.")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    px, py = target_pixel
    target_y = -(Y_LIMITS[1] - (px / w) * (Y_LIMITS[1] - Y_LIMITS[0]))
    target_x = X_LIMITS[0] + ((h - py) / h) * (X_LIMITS[1] - X_LIMITS[0])

    center_x = X_LIMITS[0] + ((h - h//2) / h) * (X_LIMITS[1] - X_LIMITS[0])
    center_y = -(Y_LIMITS[1] - ((w//2) / w) * (Y_LIMITS[1] - Y_LIMITS[0]))

    return {
        "target_world": [round(target_x, 4), round(target_y, 4)],
        "center_world": [round(center_x, 4), round(center_y, 4)],
        "movement_vector": [
            round(target_x - center_x, 4),
            round(target_y - center_y, 4)
        ],
        "image_dimensions": [h, w],
        "limits": {"x": X_LIMITS, "y": Y_LIMITS}
    }


@mcp.tool(
    title="Plan Pick Trajectory",
    description="Generates a three-waypoint approach→align→descend pick trajectory."
)
def plan_pick(
    world_start: List[float] = Field(..., description="[x,y] at image center in world."),
    world_target: List[float] = Field(..., description="[x,y] mapped from target pixel."),
    mid_depth: float = Field(..., description="Approach depth in meters."),
    angle: float = Field(..., description="Roll angle for gripper in degrees.")
) -> Dict[str, Any]:
    sx, sy = world_start
    tx, ty = world_target
    approach_z = mid_depth + 0.1
    trajectory = [
        {"x": sx, "y": sy, "z": approach_z, "angle": 0.0},
        {"x": tx, "y": ty, "z": approach_z, "angle": angle},
        {"x": tx, "y": ty, "z": mid_depth, "angle": angle},
    ]
    return {"trajectory": trajectory}


@mcp.tool(
    title="Execute Motion",
    description="Executes a pick trajectory on the robot (or no-op in TEST_MODE)."
)
def execute_motion(
    trajectory: List[Dict[str, float]] = Field(..., description="Waypoints list: x,y,z,angle")
) -> Dict[str, bool]:
    if TEST_MODE:
        return {"success": True}
    client = RestClient(ip="192.168.1.130", port=34568)
    for wp in trajectory:
        resp = client.move_to_cartesian(
            wp["x"], wp["y"], wp["z"], 5.0, (wp.get("angle", 0.0), 0, 0)
        )
        if not resp.ok:
            return {"success": False}
    return {"success": True}

# === Visualization Helpers ===
def draw_dotted_arrow(
    img: np.ndarray,
    start: Tuple[int, int], end: Tuple[int, int],
    thickness: int = 2, gap: int = 20, head_len: int = 20
) -> None:
    vec = np.subtract(end, start)
    length = int(np.hypot(*vec))
    if length < 1:
        return
    unit = vec / length
    # Shaft
    for d in range(0, length, gap):
        p0 = (start + unit * d).astype(int)
        p1 = (start + unit * min(d + gap // 2, length)).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (255, 0, 255), thickness)
    # Arrowhead
    ang = np.arctan2(vec[1], vec[0])
    for s in (-1, 1):
        arr = (end - head_len * np.array([np.cos(ang + s * np.pi/6), np.sin(ang + s * np.pi/6)])).astype(int)
        cv2.line(img, tuple(end), tuple(arr), (255, 0, 255), thickness)


def illustrate_trajectory(
    img: np.ndarray,
    start_pt: Tuple[int, int], target_pt: Tuple[int, int],
    container_pt: Optional[Tuple[int, int]] = None,
    waypoints: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    overlay = img.copy()
    h, w = img.shape[:2]
    # Draw points and arrows
    cv2.circle(overlay, start_pt, 8, (255, 0, 255), -1)
    cv2.circle(overlay, target_pt, 8, (0, 255, 255), -1)
    if container_pt:
        cv2.circle(overlay, container_pt, 8, (0, 255, 0), -1)
        draw_dotted_arrow(overlay, target_pt, container_pt)
    draw_dotted_arrow(overlay, start_pt, target_pt)
    if waypoints:
        for wp in waypoints:
            cv2.circle(overlay, (int(wp[0]), int(wp[1])), 5, (255, 255, 0), -1)
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

@mcp.tool(
    title="Visualize Trajectory",
    description="Overlays pick trajectory on image and saves to file."
)
def visualize_trajectory(
    img_path: str = Field(..., description="Path to base image."),
    start_pt: List[int] = Field(..., description="[x,y] in pixels."),
    target_pt: List[int] = Field(..., description="[x,y] in pixels."),
    container_pt: Optional[List[int]] = Field(None, description="[x,y] in pixels."),
    waypoints: Optional[List[List[float]]] = Field(None, description="List of waypoint coords."),
    output_path: str = Field("trajectory_overlay.png", description="Filename for overlay.")
) -> Dict[str, str]:
    img = cv2.imread(img_path)
    cont = tuple(container_pt) if container_pt else None
    wp_tups = [tuple(w) for w in waypoints] if waypoints else None
    overlaid = illustrate_trajectory(img, tuple(start_pt), tuple(target_pt), cont, wp_tups)
    cv2.imwrite(output_path, overlaid)
    return {"overlay_path": output_path}

# === Server Run ===
if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000, transport="streamable-http")

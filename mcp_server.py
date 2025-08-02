# mcp_server.py

from fastmcp import FastMCP
#from robograb import detect_object, plan_pick, execute_trajectory, DetectionResult, TrajectoryPlan, ExecutionResult
import json
from dataclasses import asdict
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time

from ultralytics import YOLO, SAM
from ultralytics import YOLOWorld   # swap in the YOLO-World class
from typing import Any
import requests
import argparse

TEST_MODE = True  # Set to True for testing without a real robot server

class RestClient:
    """
    A simple REST client to send float-array commands to a robot server
    via HTTP POST to /api/floats.
    """

    def __init__(self, ip: str = "192.168.2.1", port: int = 34568):
        """
        Initialize the client with the server IP and port.
        """
        self.ip = ip
        self.port = port
        self.url = f"http://{self.ip}:{self.port}/api/floats"

    def send_floats(self, command: str, parameters: list[float]) -> requests.Response:
        """
        Send a JSON payload of the form { command: parameters } to the server.

        Args:
            command (str): The command key (e.g., "moveToCartesian", "closeGripper").
            parameters (list[float]): The array of floats associated with this command.

        Returns:
            requests.Response: The HTTP response object.
        """
        payload = {command: parameters}
        response = requests.post(self.url, json=payload)
        return response

    def move_to_cartesian(self, x: float, y: float, z: float, t: float, rotation: tuple[float, float, float] | None = None) -> requests.Response:
        """
        Convenience method for sending a 'moveToCartesian' command.

        Args:
            x, y, z (float): Cartesian coordinates.
            t (float): Time in seconds over which to execute the move.
            rotation (tuple[float, float, float], optional): Delta rotations (rx, ry, rz) 
                for the gripper (in degrees). If provided, they are appended to the parameter list.

        Returns:
            requests.Response: The HTTP response object from the server.
        """
        params = [x, y, z, t]
        if rotation is not None:
            rx, ry, rz = rotation
            params.extend([rx, ry, rz])
        return self.send_floats("moveToCartesian", params)

    def close_gripper(self) -> requests.Response:
        """
        Convenience method for sending a 'closeGripper' command.
        Many servers expect a single float parameter (e.g., 0.0) for gripper commands.
        """
        return self.send_floats("closeGripper", [0.03])

    def open_gripper(self) -> requests.Response:
        """
        Convenience method for sending an 'openGripper' command.
        """
        return self.send_floats("openGripper", [0.1])


mcp = FastMCP()

# Utility function to convert dataclasses to dict for serialization
def dataclass_to_dict(obj):
    return asdict(obj)


########## mex 5 tools ##########

@mcp.tool()
def capture_frame_realsense() -> dict:
    """
    RealSense capture tool: captures one aligned RGB and depth snapshot at 640×480,
    saves 'rgb_pic.png' and 'depth_map.npy', and returns their paths.
    """
    if TEST_MODE:
        print("Running in test mode, skipping RealSense capture.")
        return {"image_path": "rgb_pic.png", "depth_path": "depth_map.npy"}

    # Setup RealSense pipeline
    TARGET_WIDTH, TARGET_HEIGHT = 640, 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, TARGET_WIDTH, TARGET_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, TARGET_WIDTH, TARGET_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            # Ensure correct size
            if color_image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                color_image = cv2.resize(color_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
            if depth_data.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                depth_data = cv2.resize(depth_data, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Save outputs
            rgb_filename = "rgb_pic.png"
            depth_map_filename = "depth_map.npy"
            cv2.imwrite(rgb_filename, color_image)
            np.save(depth_map_filename, depth_data)
            break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return {"image_path": rgb_filename, "depth_path": depth_map_filename}


@mcp.tool()
def load_inputs(image_path: str, depth_path: str) -> dict:
    """
    Loads and returns image and depth arrays from disk paths.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    depth = np.load(depth_path)
    # Convert numpy arrays to lists for JSON serialization
    return {
        "img": img.tolist(),
        "depth": depth.tolist()
    }


@mcp.tool()
def detect_object(img: Any, target_class: str) -> dict:
    # if img came in as a JSON list, turn it back into an ndarray
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
    """
    Uses YOLO to detect the target object. Returns its bounding box and class.
    """
    model = YOLOWorld("/projects/agentic-mex5/models/yolov8x-world.pt")
    model.set_classes([target_class])
    result = model.predict(img, imgsz=640,conf=0.20)[0]
    if not result.boxes:
        return {"bbox": None, "cls": None}
    box = result.boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box.tolist()
    cls_idx = int(result.boxes[0].cls[0])
    return {"bbox": [x1, y1, x2, y2], "cls": result.names[cls_idx]}


@mcp.tool()
def detect_container(img: Any) -> dict:
    """
    Uses an OpenCV HSV-based color segmentation to approximate the container centroid.
    Returns its center pixel coordinates or None if not found.
    """
    # If we got JSON, img will be a nested list—convert back to ndarray
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    # HSV thresholds (tunable for container color)
    hsv_lo = (0, 30, 30)
    hsv_hi = (25, 255, 255)
    min_area = 10000

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"container": None}
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return {"container": None}
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return {"container": None}
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return {"container": [cx, cy]}


@mcp.tool()
def detect_container_YOLO(img: Any, container_class: str) -> dict:
    """
    Uses YOLO to detect the container. Returns its center pixel coordinates.
    """
    model = YOLOWorld("/projects/agentic-mex5/models/yolov8x-worldv2.pt")
    model.set_classes([container_class])
    result = model.predict(img, imgsz=640,conf=0.20)[0]
    if not result.boxes:
        return {"container": None}
    box = result.boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box.tolist()
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return {"container": [cx, cy]}


@mcp.tool()
def segment_object(img: Any, bbox: list) -> dict:
    """
    Uses SAM to segment the detected object region.
    """
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    if not bbox:
        return {"mask": None}
    x1, y1, x2, y2 = bbox
    sam = SAM("/projects/agentic-mex5/models/sam2.1_l.pt")
    seg = sam.predict(img, bboxes=[(x1, y1, x2, y2)])[0]
    mask = (seg.masks.data[0].cpu().numpy() * 255).astype(np.uint8)
    return {"mask": mask.tolist()}


@mcp.tool()
def compute_grasp_geometry(mask: Any) -> dict:
    """
    Computes center, angle, and width of the grasp region from the mask.
    """
    # If we got a JSON list, turn it back into a proper ndarray
    if isinstance(mask, list):
        mask = np.array(mask, dtype=np.uint8)

    if mask is None:
        return {"center": None, "angle": None, "width": None}
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return {"center": None, "angle": None, "width": None}
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    angle = ang if rw > rh else -ang
    angle = 90 - angle if angle > 0 else angle
    width = float(min(rw, rh))
    return {"center": [int(cx), int(cy)], "angle": float(angle), "width": width}


@mcp.tool()
def compute_midpoint(depth: Any, coords: dict) -> dict:
    """
    Calculates pickup, drop, and mid depths using target and container pixels.
    """
    tgt = coords.get("target")
    ctr = coords.get("container")
    if not tgt or not ctr:
        return {"pickup_depth": None, "drop_depth": None, "mid_depth": None}
    px, py = tgt; cx, cy = ctr
    pd = float(depth[py][px])
    dd = float(depth[cy][cx])
    md = max((dd - pd) / 1000.0, 0.016)
    return {"pickup_depth": pd, "drop_depth": dd, "mid_depth": md}


@mcp.tool()
def pixel_to_world(pixel: list, y_limits: list, x_limits: list, img_shape: list) -> dict:
    """
    Maps a pixel to real-world coordinates.
    """
    px, py = pixel
    h, w = img_shape[0], img_shape[1]
    y_max, y_min = y_limits; x_min, x_max = x_limits
    yw = -(y_max - (px / w) * (y_max - y_min))
    xw = x_min + ((h - py) / h) * (x_max - x_min)
    return {"world_xy": [xw, yw]}


@mcp.tool()
def plan_pick(world_start: list, world_target: list, mid_depth: float, angle: float) -> dict:
    """
    Creates a pick trajectory of three waypoints.
    """
    sx, sy = world_start; tx, ty = world_target
    sz = mid_depth + 0.1
    traj = [
        {"x": sx, "y": sy, "z": sz, "angle": 0.0},
        {"x": tx, "y": ty, "z": sz, "angle": angle},
        {"x": tx, "y": ty, "z": mid_depth, "angle": angle}
    ]
    return {"trajectory": traj}


@mcp.tool()
def execute_motion(trajectory: list) -> dict:
    """
    Sends moveToCartesian commands for each waypoint.
    """
    if TEST_MODE:
        print("Running in test mode, skipping actual motion execution.")
        return {"success": True}
    
    client = RestClient(ip="192.168.1.130", port=34568)
    for wp in trajectory:
        resp = client.move_to_cartesian(
            wp["x"], wp["y"], wp["z"], 5.0,
            (wp.get("angle", 0.0), 0, 0)
        )
        if not resp.ok:
            return {"success": False}
    return {"success": True}


if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )

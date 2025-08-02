from fastmcp import FastMCP
import json
import pyrealsense2 as rs
import numpy as np
import cv2
import tempfile
import uuid
import os
import time

from ultralytics import YOLO, SAM, YOLOWorld
from typing import Any, Tuple, List, Dict
import requests

TEST_MODE = True  # for offline testing

mcp = FastMCP()

def _write_temp_image(img: np.ndarray, prefix: str = "img") -> str:
    fn = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}.png")
    cv2.imwrite(fn, img)
    return fn


@mcp.tool()
def capture_frame_realsense() -> Dict[str, str]:
    """
    Capture one aligned RGB+depth and write files.
    Returns:
      rgb_path: str, depth_path: str
    """
    if TEST_MODE:
        return {"rgb_path": "rgb_pic.png", "depth_path": "depth_map.npy"}

    # RealSense setup...
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            img = np.asanyarray(color.get_data())
            dmap = np.asanyarray(depth.get_data())

            rgb_fn = _write_temp_image(img, "rgb")
            depth_fn = os.path.join(tempfile.gettempdir(), f"depth_{uuid.uuid4().hex}.npy")
            np.save(depth_fn, dmap)
            break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return {"rgb_path": rgb_fn, "depth_path": depth_fn}


@mcp.tool()
def detect_object(image_path: str, target_class: str) -> Dict[str, Any]:
    """Load image from path, detect target_class, return bbox & cls"""
    img = cv2.imread(image_path)
    model = YOLOWorld("/projects/agentic-mex5/models/yolov8x-world.pt")
    model.set_classes([target_class])
    res = model.predict(img, imgsz=640, conf=0.2)[0]
    if not res.boxes:
        return {"bbox": None, "cls": None}
    x1, y1, x2, y2 = res.boxes[0].xyxy[0].cpu().numpy().astype(int).tolist()
    cls_idx = int(res.boxes[0].cls[0])
    return {"bbox": [x1, y1, x2, y2], "cls": res.names[cls_idx]}


@mcp.tool()
def segment_object(image_path: str, bbox: List[int]) -> Dict[str, str]:
    """Segment the ROI, write a mask png, return its path"""
    img = cv2.imread(image_path)
    if not bbox:
        return {"mask_path": None}
    x1, y1, x2, y2 = bbox
    sam = SAM("/projects/agentic-mex5/models/sam2.1_l.pt")
    seg = sam.predict(img, bboxes=[(x1, y1, x2, y2)])[0]
    mask = (seg.masks.data[0].cpu().numpy() * 255).astype(np.uint8)
    mask_fn = _write_temp_image(mask, "mask")
    return {"mask_path": mask_fn}


@mcp.tool()
def compute_grasp_geometry(mask_path: str) -> Dict[str, Any]:
    """Load mask from path, compute center, angle, width"""
    if not mask_path or not os.path.exists(mask_path):
        return {"center": None, "angle": None, "width": None}
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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
def compute_midpoint(depth_path: str, coords: Dict[str, List[int]]) -> Dict[str, float]:
    """Load depth map, calculate pickup, drop, mid depths"""
    depth = np.load(depth_path)
    tgt = coords.get("target")
    ctr = coords.get("container")
    if not tgt or not ctr:
        return {"pickup_depth": None, "drop_depth": None, "mid_depth": None}
    px, py = tgt; cx, cy = ctr
    pd = float(depth[py, px])
    dd = float(depth[cy, cx])
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

#mcp_server.py

from fastmcp import FastMCP
from pydantic import Field
import numpy as np
import cv2
import os
import json
from ultralytics import YOLOWorld, SAM
from typing import Any, Dict,Optional
import requests
import pyrealsense2 as rs

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


# Create server
mcp = FastMCP("Object Detection and Manipulation Server")

@mcp.tool(
    title="Echo Tool",
    description="Echos the input text back to the user."
)
def echo_tool(
    text: str = Field(..., description="The text to echo back to the user.")
) -> str:
    """Echo the input text"""
    return text

@mcp.tool(
    title="Capture Frame",
    description="Returns paths to the current RGB image and depth map for processing."
)
def capture_frame() -> dict:
    """
    RealSense capture tool: captures one aligned RGB and depth snapshot at 640×480,
    saves 'rgb_pic.png' and 'depth_map.npy', and returns their paths.
    """
    if TEST_MODE:
        print("Running in test mode, skipping RealSense capture.")
        return {
        "image_path": "/projects/agentic-mex5/rgb_pic.png",
        "depth_path": "/projects/agentic-mex5/depth_map.npy"
            }

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


@mcp.tool(
    title="Detect Object",
    description="Detects a specified object class in an image using YOLOWorld and returns bounding box and class label."
)
def detect_object(
    target_class: str = Field(..., description="The object class label to detect (e.g., 'scissor')."),
    img: Any = Field(None, description="An image array or file path to run detection on.")
) -> dict:
    """
    Uses YOLOWorld to detect the target object in the image.
    Returns a dict with 'bbox' as [x1, y1, x2, y2] or None if not found, and 'cls' as class name.
    """
    # Load image from path if necessary
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
    x1, y1, x2, y2 = box.tolist()
    cls_idx = int(result.boxes[0].cls[0])
    return {"bbox": [x1, y1, x2, y2], "cls": result.names[cls_idx]}

@mcp.tool(
    title="Segment Object",
    description="Segments a detected object in an image using SAM and saves mask image and metadata JSON."
)
def segment_object(
    bbox: list = Field(..., description="Bounding box [x1, y1, x2, y2] of the object to segment."),
    img: Any = Field(None, description="An image array or file path to segment.")
) -> Dict[str, Any]:
    """
    Uses SAM to segment the specified bounding box in the image.
    Saves a binary mask as 'mask.png' and metadata as 'mask_meta.json'.
    Returns file paths and a success message.
    """
    # Load image
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    # Validate bbox
    if not bbox or len(bbox) != 4:
        return {"mask_path": None, "meta_path": None, "message": "Invalid bounding box provided"}

    mask_path = "mask.png"
    meta_path = "mask_meta.json"

    # Perform segmentation
    sam = SAM("/projects/agentic-mex5/models/sam2.1_l.pt")
    seg = sam.predict(img, bboxes=[bbox])[0]
    mask = (seg.masks.data[0].cpu().numpy() * 255).astype(np.uint8)

    # Save mask image
    cv2.imwrite(mask_path, mask)

    # Prepare metadata
    metadata = {
        "bbox": bbox,
        "mask": seg.masks.data[0].cpu().numpy().tolist(),
        "image_dimensions": img.shape[:2],
        "model_used": "SAM 2.1 Large",
        "original_image": "/projects/agentic-mex5/rgb_pic.png"
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {"mask_path": mask_path, "meta_path": meta_path, "message": "Mask and metadata saved successfully"}

@mcp.tool(
    title="Compute Grasp Geometry",
    description="Computes the grasp region's center coordinates, angle, and width from the last saved mask metadata."
)
def compute_grasp_geometry() -> dict:
    """
    Loads 'mask_meta.json', extracts the mask, computes the minimum-area rectangle,
    and returns the grasp center, angle, and width.
    """
    metadata = json.load(open("mask_meta.json"))
    mask = metadata.get("mask")
    if mask is None:
        return {"center": None, "angle": None, "width": None}

    mask = np.array(mask, dtype=np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"center": None, "angle": None, "width": None}

    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    angle = ang if rw > rh else -ang
    angle = 90 - angle if angle > 0 else angle
    width = float(min(rw, rh))
    return {"center": [int(cx), int(cy)], "angle": float(angle), "width": width}

@mcp.tool(
    title="Detect Container",
    description="Detects a container object via HSV-based color segmentation and returns its centroid coordinates."
)
def detect_container(
    img: Any = Field(
        None,
        description="An image array or file path to detect the container in."
    )
) -> dict:
    """
    Uses HSV thresholds and morphology to find the largest colored contour
    matching the container, then returns its centroid or None.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    hsv_lo, hsv_hi = (0, 30, 30), (25, 255, 255)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 10000:
        return {"container": None}

    M = cv2.moments(max(cnts, key=cv2.contourArea))
    if M["m00"] == 0:
        return {"container": None}
    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
    return {"container": [cx, cy]}

@mcp.tool(
    title="Compute Midpoint Depth",
    description="Calculates and returns a fixed midpoint depth between target and container in the scene."
)
def compute_midpoint() -> dict:
    """
    Returns a predefined midpoint depth value (in meters).
    """
    return {"mid_depth": 0.016}


from pydantic import Field
import cv2
import numpy as np

# Predefined world coordinate limits (MCP side)
Y_LIMITS = [-0.33, 0.33]
X_LIMITS = [0.31, 0.57]
from typing import List, Dict, Any

@mcp.tool(title="Map Pixels to World", description="Maps a target pixel to world-frame coordinates using fixed table limits; also returns image-center mapping and movement vector.")
def map_pixels_to_world(
    target_pixel: List[int] = Field(..., description="Target pixel [x,y] in image coordinates (cols, rows)."),
    img_path: Optional[str] = Field(None, description="Path to the RGB image (640x480 expected). Provide either img_path or img_array."),
   ) -> Dict[str, Any]:
    """
    Returns:
      - target_world: [x,y] in meters
      - center_world: [x,y] in meters (world mapping of the image center)
      - movement_vector: [dx,dy] in meters (target - center)
      - dims: [H,W] image dimensions
    """
    # Load image from either path or array input
    img = None
    if img_path:
        img = cv2.imread(img_path)
    elif isinstance(img_array, list):
        img = np.array(img_array, dtype=np.uint8)
    elif img_array is not None:
        img = img_array
        
    if img is None:
        raise ValueError("Could not load image from provided inputs")
    
    h, w = img.shape[:2]
    
    # Target pixel mapping
    px, py = target_pixel
    target_y = -(Y_LIMITS[1] - (px / w) * (Y_LIMITS[1] - Y_LIMITS[0]))
    target_x = X_LIMITS[0] + ((h - py) / h) * (X_LIMITS[1] - X_LIMITS[0])
    
    # Center pixel mapping
    center_x = X_LIMITS[0] + ((h - h//2) / h) * (X_LIMITS[1] - X_LIMITS[0])
    center_y = -(Y_LIMITS[1] - (w//2 / w) * (Y_LIMITS[1] - Y_LIMITS[0]))
    
    return {
        "target_world": [round(target_x, 4), round(target_y, 4)],
        "center_world": [round(center_x, 4), round(center_y, 4)],
        "movement_vector": [
            round(target_x - center_x, 4), 
            round(target_y - center_y, 4)
        ],
        "image_dimensions": [h, w],
        "limits": {
            "x": X_LIMITS,
            "y": Y_LIMITS
        }
    }

@mcp.tool(title="Plan Pick Trajectory", description="Generates a three-waypoint pick trajectory (approach, align, descend).")
def plan_pick(
    world_start: List[float] = Field(..., description="Start [x,y] in meters (image center in world)."),
    world_target: List[float] = Field(..., description="Target [x,y] in meters (mapped from target pixel)."),
    mid_depth: float = Field(..., description="Approach depth in meters (z for final waypoint)."),
    angle: float = Field(..., description="Tool/gripper roll angle in degrees.")
) -> Dict[str, Any]:
    """
    Returns:
      - trajectory: list of 3 waypoints, each {x,y,z,angle}
        * wp0: move above start at z=mid_depth+0.1
        * wp1: move above target at z=mid_depth+0.1 with angle
        * wp2: descend to z=mid_depth with angle
    """
    sx, sy = world_start; tx, ty = world_target
    sz = mid_depth + 0.1
    traj = [
        {"x": sx, "y": sy, "z": sz, "angle": 0.0},
        {"x": tx, "y": ty, "z": sz, "angle": angle},
        {"x": tx, "y": ty, "z": mid_depth, "angle": angle}
    ]
    return {"trajectory": traj}


@mcp.tool(title="Execute Motion", description="Executes a pick trajectory on the robot (or no-op in test mode).")
def execute_motion(
    trajectory: List[Dict[str, float]] = Field(..., description="Waypoints array; each item has numeric keys: x, y, z (meters) and angle (degrees).")
) -> Dict[str, bool]:
    """
    Returns:
      - success: bool
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



# === Visualization helpers ===
from typing import Any, Dict, List, Optional, Tuple
def draw_dotted_arrow(img: np.ndarray, 
                      start: Tuple[int,int], end: Tuple[int,int],
                      color: Tuple[int,int,int]=(255,0,255), thickness:int=2,
                      gap:int=20, head_len:int=20):
    vec = np.subtract(end, start); length=int(np.hypot(*vec))
    if length<1: return
    unit = vec/length
    # shaft
    for d in range(0,length,gap):
        p0=(start + unit*d).astype(int)
        p1=(start + unit*min(d+gap//2,length)).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), color, thickness)
    # arrowhead
    ang=np.arctan2(vec[1],vec[0])
    for s in (-1,1):
        arr=(end - head_len*np.array([np.cos(ang+s*np.pi/6),np.sin(ang+s*np.pi/6)])).astype(int)
        cv2.line(img, tuple(end), tuple(arr), color, thickness)


def illustrate_trajectory(
    img: np.ndarray,
    start_pt: Tuple[int,int], target_pt: Tuple[int,int],
    container_pt: Optional[Tuple[int,int]] = None,
    waypoints: Optional[List[Tuple[float,float]]] = None
) -> np.ndarray:
    overlay=img.copy(); h,w=img.shape[:2]
    cv2.circle(overlay, start_pt, 8, (255,0,255), -1)
    cv2.circle(overlay, target_pt, 8, (0,255,255), -1)
    if container_pt: cv2.circle(overlay, container_pt, 8, (0,255,0), -1)
    draw_dotted_arrow(overlay, start_pt, target_pt)
    if container_pt: draw_dotted_arrow(overlay, target_pt, container_pt, color=(0,255,0))
    if waypoints:
        for wp in waypoints:
            cv2.circle(overlay, (int(wp[0]),int(wp[1])),5,(255,255,0),-1)
    return cv2.addWeighted(img,0.7,overlay,0.3,0)

@mcp.tool(title="Visualize Trajectory", description="Overlays the pick trajectory on the input image and saves it.")
def visualize_trajectory(
    img_path: str = Field(..., description="Path to base image file."),
    start_pt: List[int] = Field(..., description="Start point [x,y] in pixels."),
    target_pt: List[int] = Field(..., description="Target point [x,y] in pixels."),
    container_pt: Optional[List[int]] = Field(None, description="Container point [x,y] in pixels."),
    waypoints: Optional[List[List[float]]] = Field(None, description="Intermediate waypoints [[x,y],...]."),
    output_path: str = Field("trajectory_overlay.png", description="Output image path.")
) -> Dict[str, str]:
    img=cv2.imread(img_path)
    container=tuple(container_pt) if container_pt else None
    wp_tuples=[tuple(wp) for wp in waypoints] if waypoints else None
    overlaid=illustrate_trajectory(img, tuple(start_pt), tuple(target_pt), container, wp_tuples)
    cv2.imwrite(output_path, overlaid)
    return {"overlay_path": output_path}

# Run the server
if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )

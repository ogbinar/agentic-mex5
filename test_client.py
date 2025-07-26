import requests
import cv2
import numpy as np
import json
from typing import Dict, Any

# Configuration
TARGET_CLASS = "remote"
Y_LIMITS = [-0.33, 0.33]
X_LIMITS = [0.31, 0.57]
MCP_SERVER_URL = "http://localhost:8000/"

def call_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous tool caller with proper JSON-RPC formatting"""
    payload = {
        "jsonrpc": "2.0",
        "method": tool_name,
        "params": params,
        "id": 1  # Simple static ID for synchronous calls
    }
    
    try:
        response = requests.post(
            MCP_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json().get("result", {})
    except requests.exceptions.RequestException as e:
        print(f"Error calling {tool_name}: {str(e)}")
        raise

def run_pipeline():
    try:
        # 1. Capture frame
        print("Capturing frame...")
        frame_data = call_tool("capture_frame_realsense", {})
        
        # 2. Load inputs
        print("Loading image data...")
        loaded_data = call_tool("load_inputs", {
            "image_path": frame_data.get("image_path"),
            "depth_path": frame_data.get("depth_path")
        })
        img = loaded_data.get("img")
        depth = loaded_data.get("depth")
        img_shape = img.shape if img is not None else (480, 640, 3)  # Default shape

        # 3. Detect object
        print("Detecting target object...")
        detection = call_tool("detect_object", {
            "img": img.tolist() if img is not None else [],  # Convert numpy to list for JSON
            "target_class": TARGET_CLASS
        })
        bbox = detection.get("bbox")

        # 4. Segment object
        print("Segmenting object...")
        segmentation = call_tool("segment_object", {
            "img": img.tolist(),
            "bbox": bbox
        })
        mask = np.array(segmentation.get("mask")) if segmentation.get("mask") else None

        # 5. Compute grasp geometry
        print("Calculating grasp...")
        grasp = call_tool("compute_grasp_geometry", {
            "mask": mask.tolist() if mask is not None else []
        })
        coords = {
            "target": grasp.get("center"),
            "angle": grasp.get("angle")
        }

        # 6. Detect container
        print("Finding container...")
        container = call_tool("detect_container", {"img": img.tolist()})
        coords["container"] = container.get("container")

        # 7. Compute depths
        print("Calculating depths...")
        depths = call_tool("compute_midpoint", {
            "depth": depth.tolist() if depth is not None else [],
            "coords": coords
        })

        # 8. Convert pixels to world coordinates
        print("Mapping to world coordinates...")
        wt = call_tool("pixel_to_world", {
            "pixel": coords["target"],
            "y_limits": Y_LIMITS,
            "x_limits": X_LIMITS,
            "img_shape": list(img_shape)
        })
        wc = call_tool("pixel_to_world", {
            "pixel": coords["container"],
            "y_limits": Y_LIMITS,
            "x_limits": X_LIMITS,
            "img_shape": list(img_shape)
        })
        
        # Start position (image center)
        start_px = [img_shape[1]//2, img_shape[0]//2]
        ws = call_tool("pixel_to_world", {
            "pixel": start_px,
            "y_limits": Y_LIMITS,
            "x_limits": X_LIMITS,
            "img_shape": list(img_shape)
        })

        # 9. Plan trajectory
        print("Planning trajectory...")
        plan = call_tool("plan_pick", {
            "world_start": ws.get("world_xy"),
            "world_target": wt.get("world_xy"),
            "mid_depth": depths.get("mid_depth"),
            "angle": coords.get("angle")
        })
        trajectory = plan.get("trajectory")

        # 10. Execute motion
        print("Executing motion...")
        result = call_tool("execute_motion", {"trajectory": trajectory})
        print("Pipeline completed successfully!")
        print("Final result:", result)

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Simple test first
    print("Testing frame capture...")
    test_data = call_tool("capture_frame_realsense", {})
    print("Test frame paths:", test_data)
    
    # Full pipeline
    #if input("Run full pipeline? (y/n): ").lower() == "y":
    #    run_pipeline()
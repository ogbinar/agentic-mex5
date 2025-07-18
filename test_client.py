import requests

# URL of your running MCP server
device_url = "http://localhost:8000"

def call_tool(tool_name: str, arguments: dict) -> dict:
    url = f"{device_url}/mcp/tool/{tool_name}"
    headers = {
        "Content-Type": "application/json",
        # 👇 add text/event-stream to satisfy the spec
        "Accept": "application/json, text/event-stream",
    }
    resp = requests.post(url, json=arguments, headers=headers)
    resp.raise_for_status()
    return resp.json()

def run_pipeline():
    # Configuration
    TARGET_CLASS = "remote"
    Y_LIMITS = [-0.33, 0.33]
    X_LIMITS = [0.31, 0.57]

    # 1. Capture frame
    paths = call_tool("capture_frame", {})

    # 2. Load inputs
    data = call_tool("load_inputs", paths)
    img = data["img"] if "img" in data else None
    depth = data["depth"] if "depth" in data else None

    # 3. Detect object
    det = call_tool("detect_object", {"img": img, "target_class": TARGET_CLASS})
    bbox = det.get("bbox")

    # 4. Segment object
    seg = call_tool("segment_object", {"img": img, "bbox": bbox})
    mask = seg.get("mask")

    # 5. Compute grasp geometry
    grasp = call_tool("compute_grasp_geometry", {"mask": mask})
    coords = {"target": grasp.get("center"), "angle": grasp.get("angle")}

    # 6. Detect container
    cont = call_tool("detect_container", {"img": img})
    coords["container"] = cont.get("container")

    # 7. Compute depths
    depths = call_tool("compute_midpoint", {"depth": depth, "coords": coords})

    # 8. Pixel to world for target and container
    wt = call_tool("pixel_to_world", {"pixel": coords["target"], "y_limits": Y_LIMITS, "x_limits": X_LIMITS, "img_shape": data["img_shape"]})
    wc = call_tool("pixel_to_world", {"pixel": coords["container"], "y_limits": Y_LIMITS, "x_limits": X_LIMITS, "img_shape": data["img_shape"]})
    # Start at image center
    img_shape = data.get("img_shape", img.shape)
    start_px = [img_shape[1]//2, img_shape[0]//2]
    ws = call_tool("pixel_to_world", {"pixel": start_px, "y_limits": Y_LIMITS, "x_limits": X_LIMITS, "img_shape": img_shape})

    # 9. Plan trajectory
    plan = call_tool("plan_pick", {"world_start": ws["world_xy"], "world_target": wt["world_xy"], "mid_depth": depths.get("mid_depth"), "angle": coords.get("angle")})
    trajectory = plan.get("trajectory")

    # 10. Execute motion
    result = call_tool("execute_motion", {"trajectory": trajectory})
    print("Pipeline result:", result)


if __name__ == "__main__":
    run_pipeline()

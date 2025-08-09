#test_client.py

import asyncio
from fastmcp import Client
from mcp_server import mcp   # ensure mcp_server.py doesn't auto-run the server on import
import numpy as np

# Choose the class you want to pick
# TARGET_CLASS = "remote control"
# TARGET_CLASS = "scissor"
TARGET_CLASS = "marker pen"


async def run_pipeline():
    async with Client(mcp) as client:
        # 1) Capture an RGB + Depth snapshot (server returns file paths)
        print("1) Capturing frame…")
        cap_res = await client.call_tool("capture_frame", {})
        paths = cap_res.data
        image_path = paths["image_path"]
        depth_path = paths["depth_path"]
        print(f"   → image: {image_path}")
        print(f"   → depth: {depth_path}")

        # 2) Detect the target object (pass the image path to avoid huge payloads)
        print("2) Detecting target object…")
        det_res = await client.call_tool("detect_object", {
            "target_class": TARGET_CLASS,
            "img": image_path
        })
        bbox = det_res.data["bbox"]
        cls_name = det_res.data["cls"]
        print(f"   → detected: {cls_name}, bbox={bbox}")
        if bbox is None:
            print("   ! No target detected, aborting.")
            return

        # 3) Segment the detected object (saves mask + metadata server-side)
        print("3) Segmenting object…")
        seg_res = await client.call_tool("segment_object", {
            "bbox": bbox,
            "img": image_path
        })
        print(f"   → {seg_res.data.get('message')} | mask={seg_res.data.get('mask_path')} | meta={seg_res.data.get('meta_path')}")

        # 4) Compute grasp geometry (reads mask_meta.json on server)
        print("4) Computing grasp geometry…")
        grasp_res = await client.call_tool("compute_grasp_geometry", {})
        center_px = grasp_res.data["center"]            # [x, y] in pixels
        angle_deg = grasp_res.data["angle"]             # gripper/tool roll
        width_px  = grasp_res.data["width"]
        print(f"   → center(px)={center_px}, angle(deg)={angle_deg:.2f}, width(px)={width_px:.2f}")
        if center_px is None:
            print("   ! No grasp center computed, aborting.")
            return

        # 5) Detect the container (drop target) from the same image
        print("5) Detecting container…")
        cont_res = await client.call_tool("detect_container", {
            "img": image_path
        })
        container_px = cont_res.data.get("container")   # [x, y] or None
        print(f"   → container(px)={container_px}")

        # 6) Get an approach depth (server returns a fixed mid-depth for now)
        print("6) Getting approach depth…")
        depth_res = await client.call_tool("compute_midpoint", {})
        mid_depth = depth_res.data["mid_depth"]
        print(f"   → mid_depth(m)={mid_depth}")

        # 7) Map pixels → world (meters) using the server’s fixed limits
        #    - returns target_world, center_world, and image_dimensions [H, W]
        print("7) Mapping pixels to world coords…")
        map_res = await client.call_tool("map_pixels_to_world", {
            "target_pixel": center_px,      # [x, y] pixels of grasp center
            "img_path": image_path
        })
        target_world = map_res.data["target_world"]      # [x, y] meters
        center_world = map_res.data["center_world"]      # [x, y] meters (image center)
        dims = map_res.data["image_dimensions"]          # [H, W]
        H, W = dims
        start_px = [W // 2, H // 2]                      # image center in pixels
        print(f"   → target_world(m)={target_world}, center_world(m)={center_world}, dims={dims}")

        # 8) Plan a 3-waypoint pick trajectory in world frame
        print("8) Planning pick trajectory…")
        plan_res = await client.call_tool("plan_pick", {
            "world_start": center_world,
            "world_target": target_world,
            "mid_depth": mid_depth,
            "angle": float(angle_deg) if angle_deg is not None else 0.0
        })
        trajectory = plan_res.data["trajectory"]
        print(f"   → trajectory: {trajectory}")

        # 8.1) Map container pixel to world (if detected); else bail
        if container_px is None:
            print("   ! No container detected, aborting place.")
            return

        map_drop = await client.call_tool("map_pixels_to_world", {
            "target_pixel": container_px,
            "img_path": image_path
        })
        drop_world = map_drop.data["target_world"]   # [x,y] in meters

        # 8.2) Get drop height (0.245)
        drop_h = await client.call_tool("compute_drop_height", {})
        drop_mid = drop_h.data["drop_mid"]

        # 8.3) Plan place trajectory
        place_res = await client.call_tool("plan_place", {
            "world_drop": drop_world,
            "drop_mid": drop_mid
        })
        place_traj = place_res.data["trajectory"]
        print(f"   → place trajectory: {place_traj}")


        # 9) Execute full pick & place
        print("9) Executing pick & place…")
        exec_res = await client.call_tool("execute_pick_and_place", {
            "pick_traj": trajectory,
            "place_traj": place_traj
        })
        print(f"   → executed: {exec_res.data['success']}")


        # 10) Visualize (server overlays arrows/points on the image)
        print("10) Visualizing trajectory overlay…")
        vis_args = {
            "img_path": image_path,
            "start_pt": start_px,            # pixel center
            "target_pt": center_px          # grasp center (pixels)
        }
        if container_px is not None:
            vis_args["container_pt"] = container_px

        vis_res = await client.call_tool("visualize_trajectory", vis_args)
        print(f"   → overlay saved to: {vis_res.data['overlay_path']}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())

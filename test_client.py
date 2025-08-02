# test_client.py

import asyncio
import numpy as np
from fastmcp import Client
from mcp_server import mcp   # your FastMCP app

# Pipeline configuration
TARGET_CLASS = "Permanent marker"
TARGET_CLASS = "remote control"
TARGET_CLASS = "scissor"
Y_LIMITS = [-0.33, 0.33]
X_LIMITS = [0.31, 0.57]

import cv2
import numpy as np

def draw_dotted_arrow(img, start, end, color=(255, 0, 255), thickness=2, gap=20, head_len=20):
    """
    Draws a dotted line with arrowhead from `start` to `end`.
    """
    vec = np.subtract(end, start)
    length = int(np.hypot(*vec))
    if length < 1:
        return
    unit = vec / length
    # dotted shaft
    for d in range(0, length, gap):
        pt0 = (start + unit * d).astype(int)
        pt1 = (start + unit * min(d + gap//2, length)).astype(int)
        cv2.line(img, tuple(pt0), tuple(pt1), color, thickness)
    # arrowhead
    angle = np.arctan2(vec[1], vec[0])
    for sign in (-1, 1):
        arrow = (
            end - head_len * np.array([
                np.cos(angle + sign * np.pi/6),
                np.sin(angle + sign * np.pi/6)
            ])
        ).astype(int)
        cv2.line(img, tuple(end), tuple(arrow), color, thickness)

def illustrate_trajectory(
    img: np.ndarray,
    start_pt: tuple[int,int],
    target_pt: tuple[int,int],
    container_pt: tuple[int,int] | None = None,
    waypoints: list[tuple[float,float]] | None = None,
    output_path: str = "trajectory_overlay.png"
):
    """
    Draws start/target/container and arrows onto `img`, then saves to `output_path`.

    - start_pt:    (x,y) of image-center or robot-home.
    - target_pt:   (x,y) of detected object.
    - container_pt:(x,y) of drop location, or None.
    - waypoints:   optional list of (x,y) stages you want to visualize.
    """
    overlay = img.copy()
    h, w = img.shape[:2]

    # 1) draw start
    cv2.circle(overlay, start_pt, 8, (255,0,255), -1)

    # 2) draw target
    cv2.circle(overlay, target_pt, 8, (0,255,255), -1)

    # 3) draw container if present
    if container_pt is not None:
        cv2.circle(overlay, container_pt, 8, (0,255,0), -1)

    # 4) arrows: start → target, target → container
    draw_dotted_arrow(overlay, start_pt, target_pt)
    if container_pt is not None:
        draw_dotted_arrow(overlay, target_pt, container_pt, color=(0,255,0))

    # 5) waypoints if any
    if waypoints:
        for wp in waypoints:
            pt = (int(wp[0]), int(wp[1]))
            cv2.circle(overlay, pt, 5, (255,255,0), -1)

    # blend with original
    output = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite(output_path, output)
    print(f"Saved overlay illustration to {output_path}")
    return output_path


async def run_pipeline():
    async with Client(mcp) as client:
        # 1. Capture frame
        print("Capturing frame…")
        frame_res = await client.call_tool("capture_frame_realsense", {})
        paths = frame_res.data
        image_path = paths["image_path"]
        depth_path = paths["depth_path"]



        # 2. Load inputs
        print("Loading image data…")
        load_res = await client.call_tool("load_inputs", {
            "image_path": image_path,
            "depth_path": depth_path
        })
        # Convert back into numpy arrays
        img = np.array(load_res.data["img"], dtype=np.uint8)
        depth = np.array(load_res.data["depth"], dtype=float)
        img_shape = img.shape
                # After loading the image:
        h, w = img.shape[:2]
        start_px = (w // 2, h // 2)

        coords = {}


        # 3. Detect object
        print("Detecting target object…")
        print("  → target class:", TARGET_CLASS)
        det_res = await client.call_tool("detect_object", {
            "img": img.tolist(),
            "target_class": TARGET_CLASS
        })
        bbox = det_res.data["bbox"]
        print("  → bbox:", bbox)

        # 4. Segment object
        print("Segmenting object…")
        seg_res = await client.call_tool("segment_object", {
            "img": img.tolist(),
            "bbox": bbox
        })
        mask_list = seg_res.data["mask"]
        mask = np.array(mask_list, dtype=np.uint8) if mask_list is not None else None

        # 5. Compute grasp geometry
        print("Calculating grasp geometry…")
        grasp_res = await client.call_tool("compute_grasp_geometry", {
            "mask": mask.tolist() if mask is not None else []
        })
        center = grasp_res.data["center"]
        angle = grasp_res.data["angle"]
        print(f"  → center={center}, angle={angle}")
        coords['target'] = center



        # 6. Detect container
        print("Finding container…")
        cont_res = await client.call_tool("detect_container", {
            "img": img.tolist()
        })
        container_xy = cont_res.data["container"]
        coords['container'] = container_xy

        # 7. Compute depths
        print("Computing depths…")
        depths_res = await client.call_tool("compute_midpoint", {
            "depth": depth.tolist(),
            "coords": {"target": center, "container": container_xy}
        })
        mid_depth = depths_res.data["mid_depth"]

        # 8. Map pixels → world coords
        print("Mapping to world coordinates…")
        wt_res = await client.call_tool("pixel_to_world", {
            "pixel": center,
            "y_limits": Y_LIMITS,
            "x_limits": X_LIMITS,
            "img_shape": list(img_shape)
        })
        world_target = wt_res.data["world_xy"]

        ws_res = await client.call_tool("pixel_to_world", {
            "pixel": [img_shape[1] // 2, img_shape[0] // 2],
            "y_limits": Y_LIMITS,
            "x_limits": X_LIMITS,
            "img_shape": list(img_shape)
        })
        world_start = ws_res.data["world_xy"]

        # 9. Plan trajectory
        print("Planning trajectory…")
        plan_res = await client.call_tool("plan_pick", {
            "world_start": world_start,
            "world_target": world_target,
            "mid_depth": mid_depth,
            "angle": angle
        })
        trajectory = plan_res.data["trajectory"]
        

        # 10. Execute motion
        print("Executing motion…")
        exec_res = await client.call_tool("execute_motion", {
            "trajectory": trajectory
        })
        print("Done:", exec_res.data)

        # after you've got:
        #   img:        original numpy image
        #   start_px:   (w//2, h//2)
        #   coords['target'], coords['container']
        # optional: trajectory list of pixel (x, y) coords

        illustrate_trajectory(
            img=img,
            start_pt=start_px,
            target_pt=coords['target'],
            container_pt=coords.get('container'),
            waypoints=[(int(wp["x"]), int(wp["y"])) for wp in trajectory]  # if you want to overlay the three pick waypoints
        )



if __name__ == "__main__":
    asyncio.run(run_pipeline())

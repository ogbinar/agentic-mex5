import time
import json
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import subprocess
import sys

if len(sys.argv) > 1:
    input_string = sys.argv[1]
    print(f"You entered: {input_string}")
else:
    print("No input string provided.")
    

import requests
import argparse

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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    #"input_image": "rgb_1748081113.png",
    #"depth_map": "depth_map_1748081113.npy",
    "input_image": "rgb_pic.png",
    "depth_map": "depth_map.npy",
    "yolo_model": "yolov8x-worldv2.pt",
    "sam_model": "sam2.1_l.pt",
#    "target_class": "marker pen",
#    "target_class": "remote",
    "target_class": input_string,
    "container_class": "container",
    # Real-world coordinate mapping
    "y_limits": (-0.33, 0.33),  # image right → left
    "x_limits": (0.31, 0.57),   # image back  → front
}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_depth(path):
    return np.load(path)


def draw_dotted_arrow(img, start, end, color=(255, 0, 255), thickness=2, gap=20, head_len=20):
    vec = np.subtract(end, start)
    length = int(np.hypot(*vec))
    if length == 0:
        return
    unit = vec / length
    for d in range(0, length, gap):
        pt0 = (start + unit * d).astype(int)
        pt1 = (start + unit * min(d + gap//2, length)).astype(int)
        cv2.line(img, tuple(pt0), tuple(pt1), color, thickness)
    angle = np.arctan2(vec[1], vec[0])
    for sign in (-1, 1):
        arrow = (
            end - head_len * np.array([np.cos(angle + sign * np.pi/6),
                                       np.sin(angle + sign * np.pi/6)])
        ).astype(int)
        cv2.line(img, tuple(end), tuple(arrow), color, thickness)


def estimate_box_centroid(img_bgr, hsv_lo=(0,30,30), hsv_hi=(25,255,255), min_area=10_000):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))


def pixel_to_world(px, py, width, height, y_lim, x_lim):
    y_max, y_min = y_lim
    x_min, x_max = x_lim
    #y_world = y_max - (px / width) * (y_max - y_min)
    # y_world = y_min + (px / width) * (y_max - y_min)
    

    y_world = -( y_max - (px / width) * (y_max - y_min) )
    x_world = x_min + ((height - py) / height) * (x_max - x_min)

    y_world1 = -((px-320) * 0.125 )/100  #13125
    x_world1 = (((480-py) * 0.125) + 6.5)/100 #5

    print(f"pixel x{px:.3f}, y{py:.3f}.")
    print(f"img x{height}, y{width}.")
    # print(f"est. world x{x_world:.3f}, y{y_world:.3f}.")
    print(f"ted est. world x{x_world1:.3f}, y{y_world1:.3f}.")
    

    return x_world1, y_world1

def _estimate_y(y):
    # Calibration for y-coordinate
    return 1.0 * y + 0.008


def print_world_point(label, pt, img_shape, y_limits, x_limits, offset=(0.03, 0)):
    xw, yw = pixel_to_world(pt[0], pt[1], img_shape[1], img_shape[0], y_limits, x_limits)
    est_x = xw #+ offset[0]
    est_y = yw #_estimate_y(yw)
    print(f"{label}: x={xw:.3f}, y={yw:.3f}")
    print(f"Estimated: x={est_x:.3f}, y={est_y:.3f}\n")
    return est_x, est_y



# ──────────────────────────────────────────────────────────────────────────────
# DETECTION & SEGMENTATION CLASS
# ──────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, config):
        self.yolo = YOLO(config['yolo_model'])
        self.yolo.set_classes([config['container_class'], config['target_class']])
        self.sam = SAM(config['sam_model'])
        self.config = config

    def detect(self, img):
        # Run YOLO detection
        result = self.yolo.predict(img, imgsz=640)[0]
        if not result.boxes:
            return None
        # Extract first bounding box tensor and convert to numpy ints
        xyxy_tensor = result.boxes[0].xyxy[0]
        box_np = xyxy_tensor.cpu().numpy().astype(int)
        x1, y1, x2, y2 = box_np.tolist()
        cls = result.names[int(result.boxes[0].cls[0])]
        return x1, y1, x2, y2, cls

    def segment(self, img, box):
        x1, y1, x2, y2, _ = box
        seg = self.sam(img, bboxes=[(x1, y1, x2, y2)])[0]
        mask = (seg.masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        return mask

# ──────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────
def main():
    cmd_pic = (
        f"/usr/bin/python3 /home/myk/project/ME5/take_rspic.py"
    )
    try:
        print("Capturing frame...")
        subprocess.run(cmd_pic, shell=True, check=True)
        time.sleep(3)
    except subprocess.CalledProcessError as err:
        print(f"❌ A REST command failed (exit code {err.returncode}):\n  {err.cmd}")

    cfg = CONFIG
    img = load_image(cfg['input_image'])
    depth_map = load_depth(cfg['depth_map'])
    h, w = img.shape[:2]
    # print("img dimension: ", img.shape)

    # Draw start point
    start_pt = (w // 2, h // 2)
    cv2.circle(img, start_pt, 10, (255,0,255), -1)

    detector = Detector(cfg)
    timing = {'yolo': 0, 'sam': 0, 'process': 0}
    coords = {}

    # 1) Detection
    t0 = time.time()
    det = detector.detect(img)
    timing['yolo'] = time.time() - t0

    if det:
        x1, y1, x2, y2, cls = det
        # 2) Segmentation
        t1 = time.time()
        mask = detector.segment(img, det)
        timing['sam'] = time.time() - t1

        # Draw mask
        overlay = np.zeros_like(img); overlay[:,:,2] = mask
        img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)

        # Find contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
        rect = cv2.minAreaRect(cnt)
        
        
        center = cv2.boxPoints(rect)
        center = np.intp(center)
        center = tuple(map(int, rect[0]))
        center = (center[0], center[1])
        (cx, cy) = center
        print("center : ",center)
        coords['target'] = center
        # Compute grasp geometry
        grasp_angle = angle if rw > rh else -angle
        grasp_angle = 90 - grasp_angle if grasp_angle > 0 else grasp_angle
        coords['grasp_angle'] = grasp_angle
        coords['grasp_width'] = min(rw, rh)
        # Draw grasp line
        rad = np.deg2rad(grasp_angle)
        ux, uy = np.cos(rad), np.sin(rad)
        wx, wy = -uy, ux
        half_len = (coords['grasp_width'] * 0.8) / 2
        p0 = (int(cx - wx*half_len), int(cy - wy*half_len))
        p1 = (int(cx + wx*half_len), int(cy + wy*half_len))
        cv2.line(img, p0, p1, (0,255,255), 3)
        timing['process'] = time.time() - t1
    else:
        print("No objects detected!")
        return

    # 3) Fallback centroid for container
    container_pt = estimate_box_centroid(img)
    if container_pt:
        coords['container'] = container_pt
        cv2.circle(img, container_pt, 10, (0,255,0), -1)

    # 4) Draw arrows
    if 'target' in coords:
        draw_dotted_arrow(img, start_pt, coords['target'])
    if 'target' in coords and 'container' in coords:
        draw_dotted_arrow(img, coords['target'], coords['container'])

    # 5) Depth extraction & compute midpoint
    if 'container' in coords:
        dz = depth_map[coords['container'][1], coords['container'][0]]
        print(f"Drop depth:   {dz:.3f} m")
    drop_mid = 0.49/2
    if 'target' in coords:
        dp = depth_map[coords['target'][1], coords['target'][0]]
        print(f"Pickup depth: {dp:.3f} m")
    
    print("dz: ",dz)
    print("dp: ",dp)
    mid = max((dz - dp)/1000, 0.016)
    mid = 0.009
    #if(mid >  0.16):
    #    mid=0.016 #0.18
    print(f"Midpoint depth: {mid:.3f} m\n")

    # 6) World coordinates
    print_world_point("Start pos (world)", start_pt, img.shape, cfg['y_limits'], cfg['x_limits'])
    ex, ey = print_world_point("Grasp pos (world)", coords['target'], img.shape, cfg['y_limits'], cfg['x_limits'])
    print(f"Grasp angle (deg): {coords['grasp_angle']:.1f}\n")
    if 'container' in coords:
        dx,dy = print_world_point("Drop pos (world)", coords['container'], img.shape, cfg['y_limits'], cfg['x_limits'])

    # 7) Timing & save
    fps = 1 / (timing['yolo'] + timing['sam'] + timing['process'])
    print(f"Timing: YOLO={timing['yolo']:.3f}s | SAM={timing['sam']:.3f}s | Process={timing['process']:.3f}s | FPS={fps:.1f}\n")
    cv2.imwrite("grasp_angle_analysis.jpg", img)
    print(f"Saved estimated trajectory image as grasp_angle_analysis.jpg")

    # 8) REST command
    cmd_xy = (
        f"/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129 --parameters "
        f"{ex:.3f} {ey:.3f} 0.49 6.0 0 0 0"
    )

    # 8) REST command
    cmd_z = (
        f"/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129 --parameters "
        f"{ex:.3f} {ey:.3f} {mid:.3f} 6.0 {coords['grasp_angle']:.1f} 0 0"
    )

    cmd_close = (
        "/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129  --command closeGripper --parameters 0.01"
    )

    # 8) REST command
    cmd_up = (
        f"/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129 --parameters "
        f"{ex:.3f} {ey:.3f} 0.49 6.0 {coords['grasp_angle']:.1f} 0 0"
    )

    cmd_drop = (
        f"/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129 --parameters "
        f"{dx:.3f} {dy:.3f} {drop_mid:.3f} 6.0 0 0 0"
    )

    cmd_open = (
        f"/usr/bin/python3 /home/myk/project/ME5/autonomous-robots/franka/python/rest.py --ip 192.168.1.129  --command openGripper --parameters 0.1"
    )

    print("MOVE command:", cmd_xy)
    print("GRASP command:", cmd_z)
    print("CLOSE command:", cmd_close)
    print("UP command:", cmd_up)
    print("MOVE command:", cmd_drop)
    print("OPEN command:", cmd_open)


    # Execute sequentially: subprocess.run() waits for each to finish before proceeding
    # call_rest.py (somewhere under /home/myk/project/ME5/)
    client = RestClient(ip="192.168.1.130", port=34568)

    # 2) Example: move the robot arm to (0.5, 0, 0.3) in 5 seconds
    
     
    resp_move = client.move_to_cartesian(ex, ey, 0.49, 3.0, (0.0, 0.0, 0.0))
    resp_move = client.move_to_cartesian(ex, ey, mid, 5.0, (coords['grasp_angle'], 0.0, 0.0))
    #time.sleep(3)
    resp_close = client.close_gripper()
    resp_move = client.move_to_cartesian(ex, ey, 0.49, 15.0, (0.0, 0.0, 0.0))
    resp_move = client.move_to_cartesian(dx, dy, drop_mid, 10.0, (0.0, 0.0, 0.0))
    resp_open = client.open_gripper()

    

if __name__ == "__main__":
    main()

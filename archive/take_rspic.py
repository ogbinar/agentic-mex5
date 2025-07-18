#!/usr/bin/env python3
"""
Intel RealSense Single Snapshot RGB & Depth Capture
- Captures one aligned RGB and depth frame (forced to 640×480)
- Saves RGB image, depth colormap, and raw depth map then exits
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time

def main():
    # Desired output dimensions
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480

    # Configure RealSense pipeline
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("No RealSense device detected")
        sys.exit(1)

    config = rs.config()
    # Request 640×480 streams explicitly
    config.enable_stream(rs.stream.color, TARGET_WIDTH, TARGET_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, TARGET_WIDTH, TARGET_HEIGHT, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        print("Capturing one frame...")
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            # Force resize to 640×480 if not already
            if (color_image.shape[1], color_image.shape[0]) != (TARGET_WIDTH, TARGET_HEIGHT):
                color_image = cv2.resize(color_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
            if (depth_data.shape[1], depth_data.shape[0]) != (TARGET_WIDTH, TARGET_HEIGHT):
                # Use nearest interpolation for depth to avoid smoothing
                depth_data = cv2.resize(depth_data, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Apply colormap for visualization (on resized depth_data)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_data, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Save images
            timestamp = int(time.time())
            rgb_filename = "rgb_pic.png"  # or f"rgb_{timestamp}.png" if you want timestamped files
            depth_colormap_filename = "depth_colormap_pic.png"  # or f"depth_colormap_{timestamp}.png"
            depth_map_filename = "depth_map.npy"  # or f"depth_map_{timestamp}.npy"

            cv2.imwrite(rgb_filename, color_image)
            cv2.imwrite(depth_colormap_filename, depth_colormap)
            np.save(depth_map_filename, depth_data)

            print(f"Saved RGB image as {rgb_filename}")
            print(f"Saved depth colormap as {depth_colormap_filename}")
            print(f"Saved raw depth map as {depth_map_filename}")
            break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Done. Exiting.")

if __name__ == "__main__":
    main()

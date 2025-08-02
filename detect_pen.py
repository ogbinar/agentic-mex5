#!/usr/bin/env python3
import sys
import argparse
import cv2
from ultralytics import YOLOWorld   # swap in the YOLO-World class

def parse_args():
    p = argparse.ArgumentParser(
        description="Open-vocabulary detection with YOLO-World"
    )
    p.add_argument("image", help="Path to the input image")
    p.add_argument("prompt", help="Text prompt describing the object to detect")
    p.add_argument(
        "--model",
        default="/projects/agentic-mex5/models/yolov8x-worldv2.pt",
        help="Path to a YOLO-World weights file"
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size (pixels)"
    )
    p.add_argument(
        "--output",
        default="detected.jpg",
        help="Where to save the annotated image"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Load a YOLO-World model
    model = YOLOWorld(args.model)

    # Tell it exactly what you want to find:
    model.set_classes([args.prompt])

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read '{args.image}'")
        sys.exit(1)

    # Run open-vocabulary inference
    results = model.predict(img, imgsz=args.imgsz)[0]

        
    # If there are any detections at all:
    if len(results.boxes) > 0:
        # get confidences as a NumPy array
        confs = results.boxes.conf.cpu().numpy()
        # pick the index of the highest confidence
        best_idx = confs.argmax()

        # extract the box coordinates, class and confidence
        x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        cls_id   = int(results.boxes.cls[best_idx].cpu().numpy())
        label    = results.names[cls_id]
        best_conf = confs[best_idx]

        print(f"Best detection: {label} ({best_conf:.2f}) at [{x1},{y1},{x2},{y2}]")

        # draw just that box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{label} {best_conf:.2f}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )
        cv2.imwrite(args.output, img)
        print(f"Detected “{args.prompt}” → {args.output}")
    else:
        print("No detections found.")




if __name__ == "__main__":
    main()

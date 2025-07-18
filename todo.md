

## 1. Current monolithic workflow

1. **Frame capture**

   ```python
   subprocess.run("/usr/bin/python3 take_rspic.py", …)
   ```
2. **Load inputs**

   ```python
   img       = cv2.imread("rgb_pic.png")
   depth_map = np.load("depth_map.npy")
   ```
3. **Detection**

   ```python
   x1,y1,x2,y2,cls = YOLO.detect(img)
   ```
4. **Segmentation**

   ```python
   mask = SAM.segment(img, bbox=(x1,y1,x2,y2))
   ```
5. **Geometric analysis**

   * extract the largest contour
   * compute minimum‑area rect → grasp angle, width
   * draw lines/arrows for visualization
6. **Depth lookup & mid‑point calc**

   ```python
   dp = depth_map[cy, cx]
   dz = depth_map[container_y, container_x]
   mid = some function of (dz – dp)
   ```
7. **Pixel → world mapping**

   ```python
   ex, ey = pixel_to_world(cx, cy, …)
   dx, dy = pixel_to_world(container_x, container_y, …)
   ```
8. **Build REST commands**

   ```python
   client.move_to_cartesian(ex,ey,z,…)
   client.close_gripper()
   client.open_gripper()
   ```
9. **Execute** the move → grasp → lift → move → drop → open sequence.

---

## 2. Refactoring into MCP tools

Break out each logical phase into its own RPC‑style tool. For example:

| Tool name                | Input                                         | Output                                                               |
| ------------------------ | --------------------------------------------- | -------------------------------------------------------------------- |
| `capture_frame`          | (none)                                        | `{ "image_path": str, "depth_path": str }`                           |
| `load_inputs`            | `image_path, depth_path`                      | `{ "img": np.array, "depth": np.array }`                             |
| `detect_object`          | `img, target_class`                           | `{ "bbox": [x1,y1,x2,y2], "cls": str }`                              |
| `segment_object`         | `img, bbox`                                   | `{ "mask": np.array }`                                               |
| `compute_grasp_geometry` | `mask`                                        | `{ "center": [cx,cy], "angle": float, "width": float }`              |
| `compute_midpoint`       | `depth_map, coords`                           | `{ "pickup_depth": float, "drop_depth": float, "mid_depth": float }` |
| `pixel_to_world`         | `pixel_coords, y_limits, x_limits, img_shape` | `{ "world_xy": [x,y] }`                                              |
| `plan_pick`              | `world_start, world_target, mid_depth, angle` | `{ "trajectory": […] }`                                              |
| `execute_motion`         | `trajectory`                                  | `{ "success": bool }`                                                |

You’ll register each in your `mcp_server.py`:

```python
from fastmcp import FastMCP
mcp = FastMCP()

@mcp.tool()
def capture_frame() -> dict:
    # run take_rspic.py, return paths
    …

@mcp.tool()
def load_inputs(image_path: str, depth_path: str) -> dict:
    img = cv2.imread(image_path)
    depth = np.load(depth_path)
    return {"img": img, "depth": depth}

@mcp.tool()
def detect_object(img: np.ndarray, target_class: str) -> dict:
    detection = YOLO(config["yolo_model"]).predict(img, …)[0]
    # extract bbox + cls
    return {"bbox": [x1,y1,x2,y2], "cls": cls}

# …and so on for each phase…

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000)
```

---

## 3. Agent‑centric pick‑and‑place prompt

In your agent client (`agent.py`), define a system prompt that tells the LLM to invoke these tools in sequence:

````python
system_prompt = """
You are an autonomous pick‑and‑place agent.  You have these tools:

1. capture_frame() -> {image_path, depth_path}
2. load_inputs(image_path, depth_path) -> {img, depth}
3. detect_object(img, target_class) -> {bbox, cls}
4. segment_object(img, bbox) -> {mask}
5. compute_grasp_geometry(mask) -> {center, angle, width}
6. compute_midpoint(depth, center, container) -> {mid_depth}
7. pixel_to_world(center, y_limits, x_limits, img_shape) -> {world_xy}
8. plan_pick(start, target, mid_depth, angle) -> {trajectory}
9. execute_motion(trajectory) -> {success}

**Rules**  
- Always start with `capture_frame()`.  
- Pass outputs verbatim into the next tool.  
- Do not invent new data—only call tools or output final status.  
- When motion succeeds, stop; otherwise, you may retry planning.

User: “Pick up the remote control and place it in the container.”

Respond with a JSON array of tool calls in order:
```json
[
  { "name": "capture_frame", "arguments": {} },
  { "name": "load_inputs", "arguments": { ... } },
  …
]
````

"""

````

Your client code then runs:

```python
agent = ToolCallingAgent(tools=tc.tools, model=model)
agent.prompt_templates["system_prompt"] = system_prompt
print(agent.run("Pick up the remote control and place it in the container"))
````

---

## 4. Next steps

1. **Implement each MCP tool** in `mcp_server.py`, using your existing helper functions (`pixel_to_world`, `estimate_box_centroid`, etc.).
2. **Adapt your helper code** so it works purely on inputs/outputs (no printing or `cv2.imshow`).
3. **Wire up** the REST client as its own tool, e.g. `@mcp.tool() def execute_motion(trajectory): …`.
4. **Test** each tool in isolation via `curl http://localhost:8000/tools/<tool_name>`.
5. **Iterate** on your system prompt so the agent reliably sequences detection → planning → execution, with sensible retries.


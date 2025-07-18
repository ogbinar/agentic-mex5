
# agent.py

from smolagents import ToolCollection, ToolCallingAgent, LiteLLMModel
from smolagents.agents import PromptTemplates

server = {"url": "http://localhost:8000/mcp", "transport": "streamable-http"}

with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
    model = LiteLLMModel(
        #model_id="ollama/qwen3:4b",
        #api_base="http://localhost:11434",
        model_id="ollama/qwen3:8b",
        api_base="http://202.92.159.241:11434",
        temperature=0.2,
    )

    system_prompt = """
You are an autonomous pick‑and‑place agent. You have these tools:

1. capture_frame() → { "image_path", "depth_path" }  
2. load_inputs(image_path, depth_path) → { "img", "depth" }  
3. detect_object(img, target_class) → { "bbox", "cls" }  
4. segment_object(img, bbox) → { "mask" }  
5. compute_grasp_geometry(mask) → { "center": [px,py], "angle", "width" }  
6. detect_container(img) → { "container": [px,py] }  
7. compute_midpoint(depth, { "target": center, "container": container })  
   → { "pickup_depth", "drop_depth", "mid_depth" }  
8. pixel_to_world(center, y_limits, x_limits, img_shape)  
   → { "world_xy": [x,y] }  
9. pixel_to_world(container, y_limits, x_limits, img_shape)  
   → { "world_xy": [x,y] }  
10. plan_pick(world_start, world_target, mid_depth, angle)  
    → { "trajectory": […] }  
11. execute_motion(trajectory) → { "success" }

**Flow rules**  
- Always begin with **(1) capture_frame** and **(2) load_inputs**.  
- Then:  
  1. **detect_object** → **segment_object** → **compute_grasp_geometry**  
  2. **detect_container**  
  3. **compute_midpoint**  
  4. Two calls to **pixel_to_world** (for object **center** and for **container**)  
  5. Derive **world_start** by pixel_to_world of image center  
  6. **plan_pick** → **execute_motion**  
- Pass outputs verbatim from one call to the next.  
- Do not invent any data.  
- If **execute_motion** returns `success: false`, you may retry **plan_pick** + **execute_motion**.  
- End as soon as you achieve `success: true`.

User: “Pick up the remote control and place it into the container.”

Respond with a JSON array of tool calls, e.g.:

```json
[
  { "name": "capture_frame", "arguments": {} },
  { "name": "load_inputs",  "arguments": { "image_path": "...", "depth_path": "..." } },
  { "name": "detect_object", "arguments": { "img": <img>, "target_class": "remote" } },
  { "name": "segment_object","arguments": { "img": <img>, "bbox": <bbox> } },
  { "name": "compute_grasp_geometry", "arguments": { "mask": <mask> } },
  { "name": "detect_container", "arguments": { "img": <img> } },
  { "name": "compute_midpoint", "arguments": { "depth": <depth>, "coords": { "target": <center>, "container": <container> } } },
  { "name": "pixel_to_world","arguments": { "pixel": <center>, "y_limits": [...], "x_limits": [...], "img_shape": [...] } },
  { "name": "pixel_to_world","arguments": { "pixel": <container>, "y_limits": [...], "x_limits": [...], "img_shape": [...] } },
  { "name": "plan_pick", "arguments": { "world_start": [...], "world_target": [...], "mid_depth": <mid_depth>, "angle": <angle> } },
  { "name": "execute_motion","arguments": { "trajectory": <trajectory> } }
]

    """

  

    # ----------------- 1. align the prompt with the real tool API -----------------
    system_prompt = """
    You are an autonomous pick‑and‑place agent.

    When you reply you must output **exactly one** JSON object with the keys:
      • "name" – the tool you are calling
      • "arguments" – a JSON object with the arguments for that tool

    📌 Never output anything else. No arrays, no prose, no additional keys.

    Available tools:

    1. capture_frame() → {"image_path", "depth_path"}
    2. load_inputs(image_path, depth_path) → {"img_shape", "depth_shape"}
    3. detect_object(img_shape, target_class) → {"bbox", "cls"}
    4. segment_object(mask_shape, bbox) → {"mask_shape"}
    5. compute_grasp_geometry(mask_shape) → {"center", "angle", "width"}
    6. detect_container(img_shape) → {"container"}
    7. compute_midpoint(depth_shape, coords) → {"pickup_depth", "drop_depth", "mid_depth"}
    8. pixel_to_world(pixel, y_limits, x_limits, img_shape) → {"world_xy"}
    9. plan_pick(world_start, world_target, mid_depth, angle) → {"trajectory"}
    10. execute_motion(trajectory) → {"success", "trajectory"}
    11. final_answer(message) → {"message"}   <-- **include it!**

    **Flow**
    - Start with capture_frame → load_inputs …
    - Stop after execute_motion returns `"success": true`, then call final_answer with a short success message.
    """
    
 

    agent = ToolCallingAgent(tools=[*tc.tools], model=model)
    agent.prompt_templates["system_prompt"] = system_prompt

    print(agent.run("Pick up the remote control and place it in the container"))

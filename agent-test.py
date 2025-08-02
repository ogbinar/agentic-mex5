import os
from mcpadapt import MCPAdapt
from smolagents import ToolCallingAgent, LiteLLMModel
from mcpadapt.adapters import SmolAgentsAdapter

# MCP server configuration
SERVER = {
    "url": "http://localhost:8000/mcp/",
    "transport": "streamable-http"
}

# Initialize the agent with MCP tools via mcpadapt
with MCPAdapt(SERVER, adapter=SmolAgentsAdapter()) as tools:
    model = LiteLLMModel(
        model_id="ollama/qwen3:8b",
        api_base="http://202.92.159.241:11434",
        temperature=0.2,
    )

    system_prompt = """
You are an autonomous pick-and-place agent. You have exactly these tools:

1. capture_frame_realsense() → {"rgb_path": str, "depth_path": str}
2. detect_object(image_path: str, target_class: str) → {"bbox": [x1,y1,x2,y2] | null, "cls": str | null}
3. segment_object(image_path: str, bbox: [x1,y1,x2,y2]) → {"mask_path": str | null}
4. compute_grasp_geometry(mask_path: str) → {"center": [x,y] | null, "angle": float, "width": float}
5. compute_midpoint(depth_path: str, coords: {"target":[x,y], "container":[x,y]})  
   → {"pickup_depth": float, "drop_depth": float, "mid_depth": float}
6. pixel_to_world(pixel: [x,y], y_limits: [f,f], x_limits: [f,f], img_shape: [h,w])  
   → {"world_xy": [x,y]}
7. plan_pick(world_start: [x,y], world_target: [x,y], mid_depth: float, angle: float)  
   → {"trajectory": [{"x":f,"y":f,"z":f,"angle":f},…]}
8. execute_motion(trajectory: list) → {"success": bool}

**FLOW**  
1) capture_frame_realsense  
2) detect_object → segment_object → compute_grasp_geometry  
3) compute_midpoint  
4) pixel_to_world for object center  
5) pixel_to_world for image center (world_start)  
6) plan_pick → execute_motion  

Stop as soon as execute_motion returns `{"success": true}`.

When you reply, output **exactly one** JSON object with keys:
- `"tool"`: the tool name  
- `"inputs"`: the arguments object  

Do **not** output anything else (no arrays, no prose, no extra fields).
"""

    agent = ToolCallingAgent(
        tools=tools,
        model=model,
    )
    agent.prompt_templates["system_prompt"] = system_prompt

    # Run with user instruction
    print(agent.run("Pick up the remote control and place it in the container."))
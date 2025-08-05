#agent.py

from mcp import StdioServerParameters
from smolagents import ToolCollection, LiteLLMModel, ToolCallingAgent

# Server connection parameters
server = {"url": "http://localhost:8000/mcp", "transport": "streamable-http"}

# Load tools from MCP server
with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
    # Initialize LLM model
    model = LiteLLMModel(
        model_id="ollama/qwen3:8b",
        api_base="http://202.92.159.241:11434",
        temperature=0.2,
    )

    # Print available tools
    import pprint
    print("Available tools:")
    pprint.pprint([tool.name for tool in tc.tools])

    # Create ToolCallingAgent with custom system prompt
    agent = ToolCallingAgent(
        tools=[*tc.tools],
        model=model,

    )

    # Define detailed system prompt guiding the multi-step workflow
    system_prompt = '''
RULE 0: EVERY tool invocation MUST be exactly this JSON, and only this:
{"name":"<tool_name>","arguments":{…}} 
RULE 1: **NO tool** may be called more than once.

RULE 2: You **must** call tools in **this exact** sequence—no reordering, no early exit:
────────────────────────────────────────────────────────────────────────

Tool: capture_frame()  
Example:
{"name":"capture_frame","arguments":{}}

Tool: detect_object(img: str|array, target_class: str)  
Example:
{"name":"detect_object","arguments":{"img":IMG_PATH,"target_class":"scissor"}}

Tool: segment_object(img: str|array, bbox:[int,int,int,int])  
Example:
{"name":"segment_object","arguments":{"img":IMG_PATH,"bbox":[338,62,453,176]}}

Tool: compute_grasp_geometry()  
Example:
{"name":"compute_grasp_geometry","arguments":{}}

Tool: detect_container(img: str|array)  
Example:
{"name":"detect_container","arguments":{"img":IMG_PATH}}

Tool: compute_midpoint()  
Example:
{"name":"compute_midpoint","arguments":{}}

Tool: map_pixels_to_world(target_pixel:[int,int],img_path:str)  
Example:
{"name":"map_pixels_to_world","arguments":{"target_pixel":[393,109],"img_path":IMG_PATH}}

Tool: plan_pick(world_start:[float,float],world_target:[float,float],mid_depth:float,angle:float)  
Example:
{"name":"plan_pick","arguments":{"world_start":[0.44,0.0],"world_target":[0.511,0.0753],"mid_depth":0.016,"angle":-33.16}}

Tool: execute_motion(trajectory:[{"x":float,"y":float,"z":float,"angle":float},…])  
Example:
{"name":"execute_motion","arguments":{"trajectory":[{"x":0.44,"y":0.0,"z":0.116,"angle":0.0},{"x":0.511,"y":0.0753,"z":0.116,"angle":-33.16},{"x":0.511,"y":0.0753,"z":0.016,"angle":-33.16}]}}

NOTE: All arrays must be native JSON arrays (no quotes).

Tool: visualize_trajectory(
    img_path:str,
    start_pt:[int,int],
    target_pt:[int,int],
    container_pt?:[int,int],
    output_path:str
)  
Example:
{"name":"visualize_trajectory","arguments":{"img_path":IMG_PATH,"start_pt":[320,240],"target_pt":[393,109],"container_pt":[125,135],"output_path":"trajectory_overlay.png"}}

Tool: final_answer(answer:str)  
Example:
{"name":"final_answer","arguments":{"answer":"Picked successfully."}}

NOTE: the "answer" field is REQUIRED and must be non-empty.
──────────────────────────────────────────────────────────────────────── 

'''  
    agent.prompt_templates["system_prompt"] = system_prompt

    # Launch the agent with an initial user query
    agent.run("Can you locate, grasp, and pick the marker pen from the scene?")

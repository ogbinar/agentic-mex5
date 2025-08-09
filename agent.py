# agent.py
import os, sys, argparse, yaml
from smolagents import ToolCollection, LiteLLMModel, ToolCallingAgent

def load_system_prompt(path: str, key: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            prompts = yaml.safe_load(f) or {}
        v = prompts.get("versions", {}).get(key)
        if isinstance(v, str) and v.strip():
            return v
    # Fallback minimal safety/system constraints
    return (
        "You are a tool-using robotic assistant. "
        "Rules: (1) Call tools with JSON only. (2) Do not invent numbers—use tool outputs. "
        "(3) Prefer deterministic behavior. (4) Keep answers concise."
    )

def main():
    parser = argparse.ArgumentParser(description="Run the MCP pick/place agent with a prompt.")
    parser.add_argument("prompt", nargs="?", default=None, help="User prompt to run.")
    parser.add_argument("--prompt-file", help="Read the prompt from a file.")
    parser.add_argument("--prompt-key", default="1", help="prompts.yaml versions[KEY] to load.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "ollama/qwen3:8b"))
    parser.add_argument("--api-base", default=os.getenv("OLLAMA_BASE", "http://202.92.159.241:11434"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "0.0")))
    parser.add_argument("--mcp-url", default=os.getenv("MCP_URL", "http://localhost:8000/mcp"))
    parser.add_argument("--mcp-transport", default=os.getenv("MCP_TRANSPORT", "streamable-http"))
    parser.add_argument("--prompts-yaml", default=os.getenv("PROMPTS_YAML", "prompts.yaml"))
    args = parser.parse_args()

    # Resolve prompt
    user_prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            user_prompt = f.read().strip()
    if not user_prompt:
        user_prompt = "Can you locate, grasp, and pick the marker pen from the scene?"

    # Load system prompt
    system_prompt = load_system_prompt(args.prompts_yaml, args.prompt_key)

    server = {"url": args.mcp_url, "transport": args.mcp_transport}

    allow = {
        "capture_frame","detect_object","segment_object","compute_grasp_geometry",
        "detect_container","compute_midpoint","map_pixels_to_world",
        "plan_pick","compute_drop_height","plan_place",
        "execute_pick_and_place","execute_motion","visualize_trajectory","final_answer","echo_tool"
    }

    with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
        tools = [t for t in tc.tools if t.name in allow]

        model = LiteLLMModel(
            model_id=args.model_id,
            api_base=args.api_base,
            temperature=args.temperature,
            # qwen tool-use likes JSON-style outputs; keep if it works for you
            model_kwargs={
                "format": "json",   # keep this, we’ll harden around it
           #     "stream": False,    # IMPORTANT: avoid partial JSON in streams
           #     "options": {"mirostat": 0, "num_ctx": 8192}
            },
            #stop=["```"]            # guard against code fences
        )

        agent = ToolCallingAgent(tools=tools, model=model)
        agent.prompt_templates["system_prompt"] = system_prompt

        # Fire!
        agent.run(user_prompt)

if __name__ == "__main__":
    main()


#  uv run agent.py "Can you pick the marker pen?"
#  uv run agent.py --prompt-key 1 --model-id ollama/qwen3:32b   "Can you pick the marker pen?" 

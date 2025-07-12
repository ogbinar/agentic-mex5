
# agent.py

from smolagents import ToolCollection, ToolCallingAgent, LiteLLMModel
from smolagents.agents import PromptTemplates

server = {"url": "http://localhost:8000/mcp", "transport": "streamable-http"}

with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
    model = LiteLLMModel(
        model_id="ollama/qwen3:4b",
        api_base="http://localhost:11434",
        temperature=0.2,
    )

    system_prompt = """
    You are a robotic grasping system.

    You have access to these tools:

    1. detect_tool(object_name: str) -> dict
    Returns: {"object_name": ..., "position": {"x": float, "y": float, "z": float}}

    2. plan_tool(detection_result: dict) -> dict
    Takes as input: the full detection result dict returned by detect_tool.

    3. execute_tool(trajectory_plan: dict) -> dict

    **Rules:**

    - Always call detect_tool first.
    - Take the full output dict from detect_tool, and directly pass it into plan_tool as `detection_result`.
    - Take the full output dict from plan_tool, and directly pass it into execute_tool as `trajectory_plan`.
    - Do not modify the returned dicts.
    - Always wrap tool calls in this format:
    ```json
    {
      "name": "TOOL_NAME",
      "arguments": { ... }
    }
    ```
    - Do not output raw observations as tool calls.
    """

    #prompts = PromptTemplates(system_prompt=system_prompt)

    agent = ToolCallingAgent(tools=[*tc.tools], model=model)#, prompt_templates=prompts)
    agent.prompt_templates["system_prompt"] = system_prompt;

    print(agent.run("Pick up the remote control"))

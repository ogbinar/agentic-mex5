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
    import yaml
    yaml_path = "prompts.yaml"
    with open(yaml_path, 'r') as f:
            prompts = yaml.safe_load(f)

    
    # works: 0,1 (stable), 6    
    system_prompt = prompts['versions'][8]

    agent.prompt_templates["system_prompt"] = system_prompt

    # Launch the agent with an initial user query
    agent.run("Can you locate, grasp, and pick the marker pen from the scene?")

# test_mcp_tool.py

import asyncio
from fastmcp import Client
from mcp_server import mcp  # make sure this points to your FastMCP instance

async def main():
    # Connect to the in‑memory server
    async with Client(mcp) as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print("  -", t.name)

        # Call the 'capture_frame' tool (which returns image_path & depth_path)
        result = await client.call_tool("capture_frame_realsense", {})
        print("\nResult of capture_frame:")
        print(result.data)  # should be something like {'image_path': 'rgb_pic.png', 'depth_path': 'depth_map.npy'}

if __name__ == "__main__":
    asyncio.run(main())

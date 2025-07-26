from fastmcp import Client
from mcp_server import mcp
import asyncio

class SyncMCPClient:
    """Synchronous wrapper for the FastMCP async client"""
    
    def __init__(self):
        self._client = None
        
    def __enter__(self):
        # Create a new event loop for synchronous use
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Start the async client
        self._client = self._loop.run_until_complete(Client(mcp).__aenter__())
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up the async client
        if self._client:
            self._loop.run_until_complete(self._client.__aexit__(exc_type, exc_val, exc_tb))
        self._loop.close()
        
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Synchronous tool caller"""
        async def _call():
            return await self._client.call_tool(tool_name, arguments)
        return self._loop.run_until_complete(_call()).data

def run_pipeline():
    # Configuration
    TARGET_CLASS = "remote"
    Y_LIMITS = [-0.33, 0.33]
    X_LIMITS = [0.31, 0.57]

    try:
        with SyncMCPClient() as client:
            # 1. Capture frame
            print("Capturing frame...")
            paths = client.call_tool("capture_frame_realsense", {})
            
            # 2. Load inputs
            print("Loading image data...")
            data = client.call_tool("load_inputs", paths)
            img = data.get("img")
            depth = data.get("depth")
            img_shape = data.get("img_shape", img.shape if img else (480, 640, 3))

            # [Rest of your pipeline steps...]
            # Example for one more step:
            print("Detecting object...")
            det = client.call_tool("detect_object", {"img": img, "target_class": TARGET_CLASS})
            print(f"Detection result: {det}")

            # Continue with other steps...
            
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    # Simple test
    with SyncMCPClient() as client:
        print("Testing connection...")
        tools = client.list_tools()
        print("Available tools:", tools)
        
    # Full pipeline
    run_pipeline()
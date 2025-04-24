import os
import asyncio
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_groq import ChatGroq

async def run_memory_chat():
    """
    This function runs a memory-enabled chat with the MCP agent.
    It uses the Groq API key and the browser_mcp.json config file.
    """
    # Get Groq API key from environment
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    # Config file path - change this to your config file
    config_file = "browser_mcp.json"
    
    print("Initializing chat...")
    
    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="qwen-qwq-32b")
    
    # Create agent with memory_enabled=True
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,  # Enable built-in conversation memory
    )
    
    print("\n===== Interactive MCP Chat =====")
    
    # Interactive chat loop
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Exiting the chat...")
                break
            
            # Process the user input and get response
            try:
                response = await agent.run(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        if client and client.sessions():
            await client.close_sessions()

if __name__ == "__main__":
    # Run the async chat function
    asyncio.run(run_memory_chat())
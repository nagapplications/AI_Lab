# my_agent.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv
import json, os, asyncio, sys

# Go up one level from mcp-agent/ to AI_Lab/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from setup.config import OPENAI_API_KEY

load_dotenv()
client = OpenAI(api_key=OPENAI_API_KEY)


async def run_agent(user_input):
    # 🔌 Connect to YOUR MCP server
    server_params = StdioServerParameters(command="python", args=["my_mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 📋 Discover tools automatically from server
            tools_result = await session.list_tools()

            # Convert MCP tools to OpenAI format
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools_result.tools
            ]

            print(f"🔌 Connected! Available tools: {[t.name for t in tools_result.tools]}")

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]

            for step in range(5):
                print(f"\n--- Step {step + 1} ---")

                response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
                msg = response.choices[0].message

                if msg.tool_calls:
                    messages.append(msg)

                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        print(f"🔧 Calling MCP tool: {tool_name} → {args}")

                        # ⚡ Call tool on MCP server
                        result = await session.call_tool(tool_name, args)
                        result_text = result.content[0].text

                        print(f"📦 Result: {result_text[:200]}...")

                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
                else:
                    print(f"\n✅ Final Answer: {msg.content}")
                    return msg.content


asyncio.run(
    run_agent("search the latest on lpg stock market now and a calculate 123 * 45 and save both the info to notes"))
asyncio.run(run_agent("search the latest on ev vehicles prices in US now and save the info to notes"))
asyncio.run(run_agent("now read the notes and categorise the data and display in tabular format"))

from openai import OpenAI
from tavily import TavilyClient
import json

from setup.config import OPENAI_API_KEY, TAVILY_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# 🛠️ Tool 1 - Simulated search (replace with real API later)
def search_web(query):
    print("called search_web...")
    response = tavily_client.search(query=query, max_results=3)

    # Extract and join the content from top results
    results = [r["content"] for r in response["results"]]
    return "\n\n".join(results)


# 🛠️ Tool 2 - Summarise any text
def summarise(text):
    print("called summarise...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "summarise the content in one sentence."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


# 🧾 Tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information about a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": "Summarise a long piece of text into a concise version",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }
]


# 🔁 Agent loop
def agent(user_input):
    messages = [
        {"role": "system",
         "content": ""},
        {"role": "user", "content": user_input}
    ]

    for step in range(5):
        print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling: {tool_name} → {args}")

                if tool_name == "search_web":
                    result = search_web(**args)
                elif tool_name == "summarise":
                    result = summarise(**args)
                else:
                    result = f"Unknown tool: {tool_name}"

                print(f"📦 Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        else:
            print(f"\n✅ Final Answer: {msg.content}")
            return msg.content

    return "Stopped (too many steps)"


# ▶️ Run
agent("What is the latest news on iran war today?")

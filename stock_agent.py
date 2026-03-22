import os, json, anthropic
from tavily import TavilyClient

client = anthropic.Anthropic()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

tools = [{
    "name": "web_search",
    "description": "Search the web for current stock prices, news, and analyst ratings.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"]
    }
}]

def web_search(query):
    results = tavily.search(query=query, max_results=5)["results"]
    return json.dumps([{"title": r["title"], "content": r["content"][:400], "url": r["url"]} for r in results])

question = "Should I buy Apple stock today?"
print(f"\n🧑  User: {question}\n")
messages = [{"role": "user", "content": question}]

while True:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    for block in response.content:
        if block.type == "text" and block.text:
            print(f"🤖 Claude: {block.text}\n")

    if response.stop_reason == "end_turn":
        print("✅ Done!")
        break

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"🔧 Tool call: {block.name}  →  \"{block.input['query']}\"")
            data = web_search(block.input["query"])
            print(f"📦 Data returned: {len(data)} chars\n")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": data})

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})

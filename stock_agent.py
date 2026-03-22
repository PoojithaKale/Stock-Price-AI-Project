import os, json, anthropic
from tavily import TavilyClient

_tf = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE", "")
_auth = open(_tf).read().strip() if _tf else None
client = anthropic.Anthropic(auth_token=_auth) if _auth else anthropic.Anthropic()
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None

tools = [{"name": "web_search", "description": "Search web for stock prices, news, and analyst ratings.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]

MOCK = json.dumps([
    {"title": "Apple AAPL $213.49 +0.8%", "content": "AAPL at $213.49. P/E 33.1. Analyst avg target $237. Consensus: BUY (28 buy, 10 hold, 4 sell). 52-wk range $164–$260.", "url": "https://finance.yahoo.com/quote/AAPL"},
    {"title": "Apple Q1 FY2025 Earnings Beat", "content": "Revenue $124.3B vs $121.8B est. iPhone +1.5% YoY. Services record $26.3B. EPS $2.40 vs $2.35 expected.", "url": "https://reuters.com"},
    {"title": "Apple Intelligence Driving Upgrade Cycle", "content": "Apple AI features spurring strongest iPhone upgrade cycle in 3 years. Analysts project 8% unit growth FY2025.", "url": "https://techcrunch.com"},
])

def web_search(query):
    if tavily:
        try:
            r = tavily.search(query=query, max_results=5)["results"]
            return json.dumps([{"title": x["title"], "content": x["content"][:400], "url": x["url"]} for x in r])
        except Exception as e:
            print(f"   ⚠️  Tavily error ({e.__class__.__name__}) — using mock data")
    return MOCK

question = "Should I buy Apple stock today?"
print(f"\n🧑  User: {question}\n")
messages = [{"role": "user", "content": question}]

while True:
    response = client.messages.create(
        model="claude-opus-4-6", max_tokens=4096, tools=tools, messages=messages
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

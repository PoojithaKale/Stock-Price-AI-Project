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
    {"title": "Tesla TSLA $248.71 +2.3%", "content": "TSLA at $248.71. P/E 72. Analyst avg target $275. Consensus: BUY (18 buy, 9 hold, 6 sell). 52-wk range $138–$300.", "url": "https://finance.yahoo.com/quote/TSLA"},
    {"title": "Tesla Q4 2024 Beat Estimates", "content": "Revenue $25.7B vs $25.1B est. Deliveries 495,570 +2% YoY. Energy division record $3.0B. EPS $0.73 vs $0.71 expected.", "url": "https://reuters.com"},
    {"title": "Tesla FSD & Robotaxi 2025", "content": "FSD v13 strong results. Robotaxi launch Austin June 2025. Optimus robot at 1,000/week production. $TSLA bullish catalysts.", "url": "https://techcrunch.com"},
])

def web_search(query):
    if tavily:
        try:
            r = tavily.search(query=query, max_results=5)["results"]
            return json.dumps([{"title": x["title"], "content": x["content"][:400], "url": x["url"]} for x in r])
        except Exception as e:
            print(f"   ⚠️  Tavily error ({e.__class__.__name__}) — using mock data")
    return MOCK

question = "Should I buy Tesla stock today?"
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

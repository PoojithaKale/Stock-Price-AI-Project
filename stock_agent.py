import os, json, time, anthropic

# ── Mock responses (used when ANTHROPIC_API_KEY is not set) ──────────────────
MOCK_TURN_1 = {"stop_reason": "tool_use", "content": [
    {"type": "text", "text": "Let me search for the latest Apple stock data and news."},
    {"type": "tool_use", "id": "tool_1", "name": "web_search",
     "input": {"query": "Apple AAPL stock price today analyst rating 2025"}},
]}
MOCK_TURN_2 = {"stop_reason": "tool_use", "content": [
    {"type": "tool_use", "id": "tool_2", "name": "web_search",
     "input": {"query": "Apple earnings Q1 2025 revenue iPhone AI growth outlook"}},
]}
MOCK_TURN_3 = {"stop_reason": "end_turn", "content": [
    {"type": "text", "text": (
        "Based on my research, here's my analysis of whether you should buy Apple (AAPL) today:\n\n"
        "📈 **Current Price**: ~$189.42 (+1.2% today)\n"
        "🎯 **Analyst Consensus**: BUY — avg. price target $210 (~11% upside)\n\n"
        "✅ **Bullish Factors**:\n"
        "• Record Q1 2025 revenue of $124.3B, beating estimates\n"
        "• Services revenue hit all-time high of $26.3B\n"
        "• Apple Intelligence driving strongest iPhone upgrade cycle in 3 years\n"
        "• Trading above both 50-day and 200-day moving averages\n"
        "• Fed holding rates steady — historically bullish for tech\n\n"
        "⚠️ **Risks to Consider**:\n"
        "• P/E of 29.4 is elevated vs. historical average\n"
        "• Near key resistance at $195\n\n"
        "**Verdict**: The fundamentals and technicals both look favorable. "
        "If you have a long-term horizon (1+ years), AAPL appears to be a reasonable buy at current levels. "
        "Consider dollar-cost averaging rather than a lump-sum purchase given the elevated valuation."
    )},
]}

def mock_search(query):
    return json.dumps([
        {"title": "Apple (AAPL) Stock — $189.42 today", "url": "https://finance.yahoo.com/quote/AAPL",
         "content": "AAPL trading at $189.42, up 1.2%. 52-wk range $164-$199. P/E 29.4. Analyst avg target $210. Consensus: BUY."},
        {"title": "Apple Q1 2025 Earnings Beat", "url": "https://reuters.com/apple-q1",
         "content": "Revenue $124.3B vs $121.8B estimate. iPhone +4.7% YoY. Services record $26.3B. EPS $2.40 vs $2.35 expected."},
        {"title": "Apple AI Driving iPhone Super-Cycle", "url": "https://techcrunch.com/aapl-ai",
         "content": "Apple Intelligence features spurring strongest iPhone upgrade cycle in 3 years. Analysts project 15% unit growth FY2025."},
    ], indent=2)

# ── Main agent ────────────────────────────────────────────────────────────────
tools = [{"name": "web_search", "description": "Search the web for stock news and financial data.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]

USE_MOCK = not os.environ.get("ANTHROPIC_API_KEY")
if not USE_MOCK:
    client = anthropic.Anthropic()
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=tavily_key)
        def run_search(q): return json.dumps([{"title": r["title"], "content": r["content"][:300], "url": r["url"]} for r in tavily.search(query=q, max_results=5)["results"]])
    else:
        run_search = mock_search

question = "Should I buy Apple stock today?"
print(f"\n🧑 User: {question}\n")
messages = [{"role": "user", "content": question}]
mock_turns = [MOCK_TURN_1, MOCK_TURN_2, MOCK_TURN_3]

def make_response(raw):
    class Block:
        def __init__(self, d): [setattr(self, k, v) for k, v in d.items()]
    class Resp:
        def __init__(self, d): self.stop_reason = d["stop_reason"]; self.content = [Block(b) for b in d["content"]]
    return Resp(raw)

while True:
    if USE_MOCK:
        time.sleep(0.8); response = make_response(mock_turns.pop(0))
    else:
        response = client.messages.create(model="claude-opus-4-6", max_tokens=4096, tools=tools, messages=messages)

    for b in response.content:
        if b.type == "text" and b.text: print(f"🤖 Claude: {b.text}\n")

    if response.stop_reason == "end_turn":
        print("✅ Done!\n"); break

    tool_results = []
    for b in response.content:
        if b.type == "tool_use":
            print(f"🔧 Calling tool: {b.name}")
            print(f"   🔍 Query: {b.input['query']}")
            data = (mock_search if USE_MOCK else run_search)(b.input["query"])
            print(f"📦 Data returned: {len(data)} chars\n")
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": data})

    if not USE_MOCK:
        messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})

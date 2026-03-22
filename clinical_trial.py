#!/usr/bin/env python3
"""
Clinical Trial Finder — AI Agent Demo
======================================
Finds recruiting clinical trials within 50 miles of a US ZIP code.

Stack
-----
  • Anthropic Claude (claude-opus-4-6)  — AI agent with tool use
  • ClinicalTrials.gov API v2           — live trial data
  • Flask + SSE                         — browser UI (dropdown + ZIP input)

Quick start
-----------
  pip install anthropic requests flask pgeocode
  export ANTHROPIC_API_KEY="sk-ant-..."
  python3 clinical_trial.py
  # Open http://localhost:5000 in your browser
"""

import json
import math
import os
import threading
import time

import requests
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

import anthropic

# ── optional pgeocode for offline ZIP geocoding ──────────────────────────────
try:
    import pgeocode as _pg
    _nomi = _pg.Nominatim("us")
    _HAS_PGEOCODE = True
except Exception:
    _HAS_PGEOCODE = False

# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov helpers
# ─────────────────────────────────────────────────────────────────────────────

CT_BASE = "https://clinicaltrials.gov/api/v2"
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "ClinicalTrialFinderDemo/1.0"})

# Built-in fallback list (used when the API fetch fails)
_FALLBACK_CONDITIONS = sorted([
    "Alzheimer's Disease", "Amyotrophic Lateral Sclerosis",
    "Anxiety Disorder", "Asthma", "Atrial Fibrillation",
    "Autism Spectrum Disorder", "Bipolar Disorder", "Breast Cancer",
    "Cervical Cancer", "Chronic Kidney Disease",
    "Chronic Obstructive Pulmonary Disease", "Colorectal Cancer",
    "COVID-19", "Crohn's Disease", "Depression",
    "Diabetes Mellitus, Type 1", "Diabetes Mellitus, Type 2",
    "Epilepsy", "Heart Failure", "HIV Infections", "Hypertension",
    "Hypothyroidism", "Inflammatory Bowel Disease", "Leukemia",
    "Lung Cancer", "Lupus", "Melanoma", "Multiple Myeloma",
    "Multiple Sclerosis", "Obesity", "Osteoarthritis", "Osteoporosis",
    "Ovarian Cancer", "Parkinson Disease", "Prostate Cancer",
    "Psoriasis", "Rheumatoid Arthritis", "Schizophrenia",
    "Sickle Cell Disease", "Sleep Apnea", "Stroke",
    "Ulcerative Colitis",
])

# Cache so we only fetch once per server run
_conditions_cache: list[str] | None = None
_conditions_lock = threading.Lock()


def get_conditions() -> list[str]:
    """Return a sorted list of condition names, fetched once from ClinicalTrials.gov."""
    global _conditions_cache
    with _conditions_lock:
        if _conditions_cache is not None:
            return _conditions_cache
        try:
            r = _HTTP.get(
                f"{CT_BASE}/stats/fieldValues/ConditionSearch",
                params={"pageSize": 1000},
                timeout=20,
            )
            r.raise_for_status()
            vals = [
                item["value"]
                for item in r.json().get("fieldValues", [])
                if item.get("value")
            ]
            if vals:
                _conditions_cache = sorted(set(vals))
                print(f"[conditions] loaded {len(_conditions_cache)} from API")
                return _conditions_cache
        except Exception as exc:
            print(f"[conditions] API failed ({exc}), using built-in list")
        _conditions_cache = _FALLBACK_CONDITIONS
        return _conditions_cache


def zipcode_to_latlon(zipcode: str) -> tuple[float, float]:
    """Return (lat, lon) for a US ZIP code."""
    # 1. pgeocode (offline, fast)
    if _HAS_PGEOCODE:
        try:
            row = _nomi.query_postal_code(zipcode)
            lat, lon = float(row.latitude), float(row.longitude)
            if not (math.isnan(lat) or math.isnan(lon)):
                return lat, lon
        except Exception:
            pass
    # 2. zippopotam.us (online, free fallback)
    try:
        r = _HTTP.get(f"https://api.zippopotam.us/us/{zipcode}", timeout=10)
        r.raise_for_status()
        place = r.json()["places"][0]
        return float(place["latitude"]), float(place["longitude"])
    except Exception as exc:
        raise ValueError(
            f"Could not geocode ZIP code '{zipcode}'. "
            "Make sure it is a valid US ZIP code."
        ) from exc


def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def search_trials(
    condition: str,
    lat: float,
    lon: float,
    radius_mi: int = 50,
    max_results: int = 30,
) -> list[dict]:
    """Query ClinicalTrials.gov v2 and return deduplicated, distance-sorted results."""
    params = {
        "query.cond": condition,
        "filter.geo": f"distance({lat},{lon},{radius_mi}mi)",
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "format": "json",
    }
    r = _HTTP.get(f"{CT_BASE}/studies", params=params, timeout=25)
    r.raise_for_status()

    rows: list[dict] = []
    seen: set[tuple] = set()

    for study in r.json().get("studies", []):
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        loc_mod = proto.get("contactsLocationsModule", {})

        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "Untitled Study")

        for loc in loc_mod.get("locations", []):
            geo = loc.get("geoPoint") or {}
            slat, slon = geo.get("lat"), geo.get("lon")
            if slat is None or slon is None:
                continue
            dist = _haversine_mi(lat, lon, float(slat), float(slon))
            if dist > radius_mi:
                continue
            key = (nct_id, loc.get("facility", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "nct_id": nct_id,
                "title": title,
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "zip": loc.get("zip", ""),
                "distance_miles": round(dist, 1),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })

    rows.sort(key=lambda x: x["distance_miles"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic agent
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "search_clinical_trials",
        "description": (
            "Search ClinicalTrials.gov for currently RECRUITING clinical trials "
            "matching a medical condition within a given radius of a US ZIP code. "
            "Returns trial name, facility, city/state, distance in miles, and URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Medical condition to search (e.g. 'Type 2 Diabetes')",
                },
                "zipcode": {
                    "type": "string",
                    "description": "US 5-digit ZIP code for the search centre",
                },
                "radius_miles": {
                    "type": "integer",
                    "description": "Search radius in miles (default 50)",
                    "default": 50,
                },
            },
            "required": ["condition", "zipcode"],
        },
    }
]


def _run_tool(name: str, inputs: dict) -> str:
    if name != "search_clinical_trials":
        return json.dumps({"error": f"Unknown tool '{name}'"})

    condition = inputs.get("condition", "")
    zipcode = inputs.get("zipcode", "")
    radius = int(inputs.get("radius_miles", 50))

    try:
        lat, lon = zipcode_to_latlon(zipcode)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        trials = search_trials(condition, lat, lon, radius_mi=radius)
    except Exception as exc:
        return json.dumps({"error": f"ClinicalTrials.gov error: {exc}"})

    payload: dict = {
        "condition": condition,
        "zipcode": zipcode,
        "user_location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "radius_miles": radius,
        "count": len(trials),
        "trials": trials,
    }
    if not trials:
        payload["message"] = (
            f"No currently recruiting trials found for '{condition}' "
            f"within {radius} miles of ZIP {zipcode}."
        )
    return json.dumps(payload)


def agent_stream(condition: str, zipcode: str):
    """
    Generator that yields SSE-formatted chunks produced by the Claude agent.
    Each yielded string is: 'data: <json>\\n\\n'
    """

    def emit(event: str, data: str):
        return f"data: {json.dumps({'event': event, 'text': data})}\n\n"

    client = anthropic.Anthropic()

    prompt = (
        f"Find currently recruiting clinical trials for **{condition}** "
        f"within 50 miles of ZIP code {zipcode}.\n\n"
        f"For each trial, present:\n"
        f"  • Trial name\n"
        f"  • Facility name and city, state\n"
        f"  • Distance from ZIP {zipcode} (miles)\n"
        f"  • Link to ClinicalTrials.gov\n\n"
        f"If no trials are found, say so clearly and suggest the user "
        f"try a broader search or different condition."
    )

    messages = [{"role": "user", "content": prompt}]

    yield emit("status", f"🤖 Claude is searching for '{condition}' near {zipcode}…\n")

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=_TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text:
                yield emit("text", block.text)

        if response.stop_reason == "end_turn":
            yield emit("done", "")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    yield emit(
                        "status",
                        f"\n🛠 Calling {block.name}(condition={block.input.get('condition')!r}, "
                        f"zipcode={block.input.get('zipcode')!r})…\n",
                    )
                    result = _run_tool(block.name, block.input)
                    yield emit("status", "✅ Data received. Preparing summary…\n\n")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
        else:
            yield emit("done", "")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Flask web application
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Clinical Trial Finder</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #eef2f7;
    min-height: 100vh;
  }

  header {
    background: #1565c0;
    color: white;
    padding: 18px 30px;
  }
  header h1 { font-size: 1.6rem; }
  header p  { font-size: 0.85rem; color: #90caf9; margin-top: 4px; }

  .controls {
    background: white;
    padding: 18px 30px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
    border-bottom: 1px solid #dde3ec;
  }

  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 0.82rem; font-weight: 600; color: #444; }

  select, input[type=text] {
    height: 38px;
    border: 1px solid #bbb;
    border-radius: 5px;
    padding: 0 10px;
    font-size: 0.95rem;
    outline: none;
    transition: border-color .2s;
  }
  select { min-width: 260px; }
  input[type=text] { width: 130px; }
  select:focus, input:focus { border-color: #1565c0; }

  button {
    height: 38px;
    background: #1565c0;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0 20px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s;
  }
  button:hover   { background: #0d47a1; }
  button:disabled { background: #90a4ae; cursor: not-allowed; }

  #status-bar {
    padding: 6px 30px;
    font-size: 0.82rem;
    font-style: italic;
    color: #555;
    background: #f5f7fa;
    border-bottom: 1px solid #dde3ec;
    min-height: 28px;
  }

  #results {
    margin: 24px 30px;
    background: white;
    border: 1px solid #dde3ec;
    border-radius: 8px;
    padding: 20px 24px;
    min-height: 200px;
    font-family: 'Courier New', monospace;
    font-size: 0.88rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
    color: #1a1a1a;
    max-height: 65vh;
    overflow-y: auto;
  }

  .placeholder {
    color: #9e9e9e;
    font-style: italic;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
</style>
</head>
<body>

<header>
  <h1>🏥 Clinical Trial Finder</h1>
  <p>Powered by Anthropic Claude &nbsp;·&nbsp; Data from ClinicalTrials.gov</p>
</header>

<div class="controls">
  <div class="field">
    <label for="cond">Medical Condition</label>
    <select id="cond">
      <option value="">Loading conditions…</option>
    </select>
  </div>
  <div class="field">
    <label for="zip">ZIP Code</label>
    <input type="text" id="zip" placeholder="e.g. 90210" maxlength="5" />
  </div>
  <button id="btn" onclick="findTrials()">🔍 Find Trials</button>
</div>

<div id="status-bar">Select a condition and enter your ZIP code to search.</div>

<div id="results">
  <span class="placeholder">Results will appear here after you search.</span>
</div>

<script>
/* ── load conditions on page ready ───────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  const sel = document.getElementById('cond');
  try {
    const r = await fetch('/api/conditions');
    const data = await r.json();
    sel.innerHTML = data.conditions
      .map(c => `<option value="${escHtml(c)}">${escHtml(c)}</option>`)
      .join('');
    setStatus(`${data.conditions.length} conditions loaded. Enter your ZIP code and search.`);
  } catch(e) {
    sel.innerHTML = '<option value="">Failed to load conditions</option>';
    setStatus('Could not load conditions from ClinicalTrials.gov.');
  }
});

/* ── search ──────────────────────────────────────────────────────────────── */
async function findTrials() {
  const condition = document.getElementById('cond').value.trim();
  const zip       = document.getElementById('zip').value.trim();

  if (!condition) { alert('Please select a condition.'); return; }
  if (!/^\\d{5}$/.test(zip)) { alert('Please enter a valid 5-digit US ZIP code.'); return; }

  const btn = document.getElementById('btn');
  const out = document.getElementById('results');
  btn.disabled = true;
  btn.textContent = 'Searching…';
  out.innerHTML = '';
  setStatus(`Searching for "${condition}" near ZIP ${zip}…`);

  try {
    const url = `/api/search?condition=${encodeURIComponent(condition)}&zipcode=${encodeURIComponent(zip)}`;
    const es  = new EventSource(url);
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.event === 'done') {
        es.close();
        btn.disabled = false;
        btn.textContent = '🔍 Find Trials';
        setStatus('Search complete.');
        return;
      }
      if (msg.event === 'error') {
        out.textContent += '\\n❌ ' + msg.text + '\\n';
        es.close();
        btn.disabled = false;
        btn.textContent = '🔍 Find Trials';
        setStatus('Search failed.');
        return;
      }
      out.textContent += msg.text;
      out.scrollTop = out.scrollHeight;
    };
    es.onerror = () => {
      es.close();
      btn.disabled = false;
      btn.textContent = '🔍 Find Trials';
      setStatus('Connection error.');
    };
  } catch(e) {
    out.textContent = 'Error: ' + e.message;
    btn.disabled = false;
    btn.textContent = '🔍 Find Trials';
    setStatus('Error.');
  }
}

function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTML_PAGE


@app.get("/api/conditions")
def api_conditions():
    conditions = get_conditions()
    return jsonify({"conditions": conditions, "count": len(conditions)})


@app.get("/api/search")
def api_search():
    condition = request.args.get("condition", "").strip()
    zipcode = request.args.get("zipcode", "").strip()

    if not condition:
        return jsonify({"error": "condition is required"}), 400
    if not zipcode or not zipcode.isdigit() or len(zipcode) != 5:
        return jsonify({"error": "invalid US ZIP code"}), 400
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY environment variable is not set"}), 500

    def generate():
        try:
            yield from agent_stream(condition, zipcode)
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'text': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY is not set — searches will fail.")
        print("   Set it with:  export ANTHROPIC_API_KEY='sk-ant-...'")
    else:
        print("✅  ANTHROPIC_API_KEY found.")

    # Pre-warm the conditions list in the background
    threading.Thread(target=get_conditions, daemon=True).start()

    print("\n🏥  Clinical Trial Finder")
    print("   Open your browser at:  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

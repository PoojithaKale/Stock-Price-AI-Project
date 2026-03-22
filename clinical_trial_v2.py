#!/usr/bin/env python3
"""
Clinical Trial Finder v2 — AI Agent Demo
=========================================
Finds recruiting clinical trials anywhere in the USA within a user-chosen
radius (50 / 100 / 150 / 200 miles) of a ZIP code.

Output per trial
----------------
  • Trial name
  • Facility, city, state
  • Distance from the user's ZIP code
  • ClinicalTrials.gov link
  • 5-line plain-English summary (written by Claude)

Stack
-----
  • Anthropic Claude (claude-opus-4-6) — AI agent with tool use + summaries
  • ClinicalTrials.gov API v2          — live trial data including descriptions
  • Flask + Server-Sent Events        — streaming browser UI
  • pgeocode / zippopotam.us          — ZIP-to-lat/lon geocoding

Quick start
-----------
  pip install anthropic requests flask pgeocode
  export ANTHROPIC_API_KEY="sk-ant-..."
  python3 clinical_trial_v2.py
  # Open http://localhost:5000 in your browser
"""

import json
import math
import os
import threading

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
_HTTP.headers.update({"User-Agent": "ClinicalTrialFinderV2/1.0"})

# ── fallback condition list (used when the live API is unreachable) ───────────
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

# ── shared conditions cache ───────────────────────────────────────────────────
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
            print(f"[conditions] API unavailable ({exc}), using built-in list")
        _conditions_cache = _FALLBACK_CONDITIONS
        return _conditions_cache


def zipcode_to_latlon(zipcode: str) -> tuple[float, float]:
    """Return (lat, lon) for a US ZIP code."""
    # 1. pgeocode — offline, fast
    if _HAS_PGEOCODE:
        try:
            row = _nomi.query_postal_code(zipcode)
            lat, lon = float(row.latitude), float(row.longitude)
            if not (math.isnan(lat) or math.isnan(lon)):
                return lat, lon
        except Exception:
            pass
    # 2. zippopotam.us — free online fallback
    try:
        r = _HTTP.get(f"https://api.zippopotam.us/us/{zipcode}", timeout=10)
        r.raise_for_status()
        place = r.json()["places"][0]
        return float(place["latitude"]), float(place["longitude"])
    except Exception as exc:
        raise ValueError(
            f"Could not geocode ZIP code '{zipcode}'. "
            "Please enter a valid US ZIP code."
        ) from exc


def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
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
    max_results: int = 20,
) -> list[dict]:
    """
    Query ClinicalTrials.gov v2 for RECRUITING studies matching `condition`
    within `radius_mi` miles of (lat, lon).  Returns enriched trial dicts
    that include brief_summary, phases, and eligibility snippets so Claude
    can write informed 5-line summaries.
    """
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

        # ── identity ──────────────────────────────────────────────────────────
        id_mod = proto.get("identificationModule", {})
        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "Untitled Study")

        # ── description / summary ─────────────────────────────────────────────
        desc_mod = proto.get("descriptionModule", {})
        brief_summary = desc_mod.get("briefSummary", "").strip()
        # Truncate to keep tool result size manageable
        if len(brief_summary) > 800:
            brief_summary = brief_summary[:797] + "..."

        # ── conditions & phase ────────────────────────────────────────────────
        cond_mod = proto.get("conditionsModule", {})
        study_conditions = cond_mod.get("conditions", [])

        design_mod = proto.get("designModule", {})
        phases = design_mod.get("phases", [])

        # ── eligibility ───────────────────────────────────────────────────────
        elig_mod = proto.get("eligibilityModule", {})
        elig_criteria = elig_mod.get("eligibilityCriteria", "").strip()
        if len(elig_criteria) > 500:
            elig_criteria = elig_criteria[:497] + "..."
        min_age = elig_mod.get("minimumAge", "")
        max_age = elig_mod.get("maximumAge", "")
        sex = elig_mod.get("sex", "")

        # ── locations ─────────────────────────────────────────────────────────
        loc_mod = proto.get("contactsLocationsModule", {})
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
                # enrichment for Claude's summaries
                "brief_summary": brief_summary,
                "conditions": study_conditions,
                "phases": phases,
                "eligibility_criteria": elig_criteria,
                "min_age": min_age,
                "max_age": max_age,
                "sex": sex,
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
            "that match a medical condition and are within a given radius of a "
            "US ZIP code. Returns trial name, facility, city/state, distance in "
            "miles, URL, brief summary, phase, and eligibility details."
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
                    "description": "US 5-digit ZIP code for the centre of the search",
                },
                "radius_miles": {
                    "type": "integer",
                    "description": "Search radius in miles — 50, 100, 150, or 200",
                    "enum": [50, 100, 150, 200],
                    "default": 50,
                },
            },
            "required": ["condition", "zipcode", "radius_miles"],
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


_SYSTEM_PROMPT = """\
You are a clinical-trial research assistant.
When you receive trial data from the search tool, present EACH trial in this
exact format — no markdown headers, just clean numbered output:

─────────────────────────────────────────────
Trial #N
─────────────────────────────────────────────
Title       : <full title>
Facility    : <facility name>, <city>, <state>
Distance    : <X.X> miles from ZIP <zipcode>
Link        : https://clinicaltrials.gov/study/<NCT_ID>

Summary (5 lines):
1. <What the study is investigating>
2. <Who it is for (age, sex, condition)>
3. <Key intervention or treatment being tested>
4. <Phase and what that means for participants>
5. <Why someone might consider enrolling>

Write the 5-line summary in plain English — no jargon — so any patient can
understand it. Base it on the brief_summary, conditions, phases, eligibility,
and age/sex data returned by the tool.

After listing all trials, add a short closing line with the total count.
If no trials are found, say so clearly and suggest broadening the search radius.
"""


def agent_stream(condition: str, zipcode: str, radius_miles: int):
    """
    Generator yielding SSE chunks: 'data: <json>\\n\\n'
    Each JSON object has {event: 'text'|'status'|'done'|'error', text: str}
    """

    def emit(event: str, text: str) -> str:
        return f"data: {json.dumps({'event': event, 'text': text})}\n\n"

    client = anthropic.Anthropic()

    prompt = (
        f"Please find recruiting clinical trials for **{condition}** "
        f"within {radius_miles} miles of ZIP code {zipcode}. "
        f"Use the search tool, then present every result using the exact "
        f"format described in your instructions, including the 5-line summary."
    )

    messages = [{"role": "user", "content": prompt}]

    yield emit(
        "status",
        f"🤖  Searching for '{condition}' trials within {radius_miles} mi of {zipcode}…\n",
    )

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            system=_SYSTEM_PROMPT,
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
                        f"\n🛠  {block.name}("
                        f"condition={block.input.get('condition')!r}, "
                        f"zipcode={block.input.get('zipcode')!r}, "
                        f"radius={block.input.get('radius_miles')} mi)…\n",
                    )
                    result = _run_tool(block.name, block.input)
                    count = json.loads(result).get("count", "?")
                    yield emit("status", f"✅  {count} trial location(s) found. Writing summaries…\n\n")
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

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Clinical Trial Finder v2</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#eef2f7;min-height:100vh}

/* ── header ── */
header{background:#0d47a1;color:#fff;padding:20px 32px}
header h1{font-size:1.7rem;letter-spacing:.3px}
header p{font-size:.85rem;color:#90caf9;margin-top:5px}

/* ── control bar ── */
.bar{background:#fff;padding:16px 32px;display:flex;flex-wrap:wrap;
     gap:18px;align-items:flex-end;border-bottom:1px solid #dde3ec;
     box-shadow:0 1px 4px rgba(0,0,0,.07)}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:.78rem;font-weight:700;color:#555;text-transform:uppercase;
             letter-spacing:.4px}
select,input[type=text]{
  height:40px;border:1.5px solid #c5cdd8;border-radius:6px;
  padding:0 12px;font-size:.95rem;background:#fafcff;
  outline:none;transition:border-color .2s,box-shadow .2s}
select:focus,input:focus{border-color:#1565c0;
  box-shadow:0 0 0 3px rgba(21,101,192,.15)}
select#cond{min-width:280px}
select#miles{min-width:160px}
input#zip{width:120px}

button{height:40px;background:#1565c0;color:#fff;border:none;border-radius:6px;
       padding:0 22px;font-size:.95rem;font-weight:700;cursor:pointer;
       transition:background .18s;white-space:nowrap}
button:hover{background:#0d47a1}
button:disabled{background:#90a4ae;cursor:not-allowed}

/* ── status bar ── */
#sb{padding:7px 32px;font-size:.82rem;font-style:italic;color:#555;
    background:#f5f7fa;border-bottom:1px solid #dde3ec;min-height:30px}

/* ── results ── */
.out-wrap{margin:24px 32px}
.out-wrap h2{font-size:1rem;font-weight:700;color:#333;margin-bottom:8px}
#results{
  background:#fff;border:1px solid #dde3ec;border-radius:8px;
  padding:22px 26px;min-height:220px;
  font-family:'Courier New',Courier,monospace;font-size:.87rem;
  line-height:1.65;white-space:pre-wrap;word-wrap:break-word;
  color:#1a1a1a;max-height:68vh;overflow-y:auto}
.placeholder{color:#9e9e9e;font-style:italic;
             font-family:'Segoe UI',Arial,sans-serif;font-size:.93rem}
</style>
</head>
<body>

<header>
  <h1>🏥 Clinical Trial Finder <sup style="font-size:.6em;opacity:.8">v2</sup></h1>
  <p>Powered by Anthropic Claude &nbsp;·&nbsp; Data from ClinicalTrials.gov
     &nbsp;·&nbsp; Includes 5-line plain-English summaries</p>
</header>

<div class="bar">
  <div class="field">
    <label for="cond">Medical Condition</label>
    <select id="cond"><option value="">Loading…</option></select>
  </div>
  <div class="field">
    <label for="zip">ZIP Code</label>
    <input type="text" id="zip" placeholder="e.g. 90210" maxlength="5"/>
  </div>
  <div class="field">
    <label for="miles">Search Radius</label>
    <select id="miles">
      <option value="50">50 miles</option>
      <option value="100">100 miles</option>
      <option value="150">150 miles</option>
      <option value="200">200 miles</option>
    </select>
  </div>
  <button id="btn" onclick="findTrials()">🔍&nbsp; Find Trials</button>
</div>

<div id="sb">Select a condition, enter your ZIP code, choose a radius, then click Find Trials.</div>

<div class="out-wrap">
  <h2>Results</h2>
  <div id="results"><span class="placeholder">Your results will appear here.</span></div>
</div>

<script>
/* ── load conditions ──────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  const sel = document.getElementById('cond');
  try {
    const r    = await fetch('/api/conditions');
    const data = await r.json();
    sel.innerHTML = data.conditions
      .map(c => `<option value="${esc(c)}">${esc(c)}</option>`)
      .join('');
    sb(`${data.conditions.length} conditions loaded. Enter ZIP, pick a radius, and search.`);
  } catch(e) {
    sel.innerHTML = '<option value="">Failed to load conditions</option>';
    sb('Could not load conditions.');
  }
});

/* ── search ──────────────────────────────────────────────────────────────── */
async function findTrials() {
  const condition = document.getElementById('cond').value.trim();
  const zip       = document.getElementById('zip').value.trim();
  const miles     = document.getElementById('miles').value;

  if (!condition)          { alert('Please select a condition.');               return; }
  if (!/^\\d{5}$/.test(zip)) { alert('Please enter a valid 5-digit ZIP code.'); return; }

  const btn = document.getElementById('btn');
  const out = document.getElementById('results');
  btn.disabled = true;
  btn.textContent = 'Searching…';
  out.textContent = '';
  sb(`Searching for "${condition}" within ${miles} miles of ZIP ${zip}…`);

  const url = `/api/search?condition=${encodeURIComponent(condition)}&zipcode=${encodeURIComponent(zip)}&miles=${miles}`;
  const es  = new EventSource(url);

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.event === 'done') {
      es.close(); reset(btn); sb('Search complete.'); return;
    }
    if (msg.event === 'error') {
      out.textContent += '\\n❌ ' + msg.text + '\\n';
      es.close(); reset(btn); sb('Search failed.'); return;
    }
    out.textContent += msg.text;
    out.scrollTop = out.scrollHeight;
  };

  es.onerror = () => { es.close(); reset(btn); sb('Connection error.'); };
}

function reset(btn) { btn.disabled = false; btn.textContent = '🔍\\u00A0 Find Trials'; }
function sb(msg)    { document.getElementById('sb').textContent = msg; }
function esc(s)     { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return _HTML


@app.get("/api/conditions")
def api_conditions():
    conditions = get_conditions()
    return jsonify({"conditions": conditions, "count": len(conditions)})


@app.get("/api/search")
def api_search():
    condition = request.args.get("condition", "").strip()
    zipcode   = request.args.get("zipcode", "").strip()
    miles_raw = request.args.get("miles", "50").strip()

    if not condition:
        return jsonify({"error": "condition is required"}), 400
    if not zipcode or not zipcode.isdigit() or len(zipcode) != 5:
        return jsonify({"error": "invalid US ZIP code"}), 400
    try:
        radius = int(miles_raw)
        if radius not in (50, 100, 150, 200):
            radius = 50
    except ValueError:
        radius = 50
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY environment variable is not set"}), 500

    def generate():
        try:
            yield from agent_stream(condition, zipcode, radius)
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'text': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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

    threading.Thread(target=get_conditions, daemon=True).start()

    print("\n🏥  Clinical Trial Finder v2")
    print("   Open your browser at:  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

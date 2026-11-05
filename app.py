# app.py
# Neurosynth functional-dissociation backend (Flask)
# Based on https://github.com/spiketren/ns-nano
#
# Endpoints
#   GET /                                         health check + usage
#   GET /img                                      sample image
#   GET /test_db                                  DB connectivity test
#   GET /terms/<term>/studies                     studies mentioning <term>
#   GET /locations/<x_y_z>/studies                studies reporting [x, y, z]
#   GET /dissociate/terms/<term_a>/<term_b>       studies with term_a but NOT term_b
#   GET /dissociate/locations/<x1_y1_z1>/<x2_y2_z2>
#                                                 studies with [x1,y1,z1] but NOT [x2,y2,z2]
#
# Optional query parameters
#   ?both=1      also return the reverse direction (B \ A) in the same response
#   ?radius=R    (locations only) match coordinates within R mm (default 0 = exact match)

import os
from flask import Flask, jsonify, request, send_file
from sqlalchemy import create_engine, text

SRID = 4326
_engine = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL (or DATABASE_URL) environment variable.")
    # Normalize old 'postgres://' scheme to 'postgresql://'
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    # Use psycopg2 explicitly (newer SQLAlchemy defaults to psycopg v3,
    # which is not in requirements.txt)
    if db_url.startswith("postgresql://"):
        db_url = "postgresql+psycopg2://" + db_url[len("postgresql://"):]
    _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


# -----------------------------
# Helpers
# -----------------------------
class BadRequest(ValueError):
    pass


def normalize_term(term: str) -> str:
    """'posterior_cingulate' / 'Posterior Cingulate' -> 'posterior cingulate'"""
    t = " ".join(term.replace("_", " ").split()).lower()
    if not t:
        raise BadRequest("Empty term.")
    return t


def parse_coords(coords: str):
    """'0_-52_26' -> (0.0, -52.0, 26.0)"""
    parts = coords.replace(",", "_").split("_")
    if len(parts) != 3:
        raise BadRequest(f"Coordinates must be x_y_z, got '{coords}'.")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise BadRequest(f"Coordinates must be numbers, got '{coords}'.")


def get_radius() -> float:
    try:
        r = float(request.args.get("radius", 0))
    except ValueError:
        raise BadRequest("radius must be a number.")
    if r < 0:
        raise BadRequest("radius must be >= 0.")
    return r


def want_both() -> bool:
    return request.args.get("both", "").lower() in ("1", "true", "yes")


# Study metadata (one row per study_id)
META_CTE = """
meta AS (
    SELECT DISTINCT ON (study_id) study_id, title, authors, journal, year
    FROM ns.metadata
    ORDER BY study_id
)
"""

# Study IDs that mention a term.
# In the DB, terms may be stored with the Neurosynth prefix
# (e.g. 'terms_abstract_tfidf__amygdala'), so match both forms.
TERM_PREFIX = "terms_abstract_tfidf__"
TERM_SET = ("SELECT DISTINCT study_id FROM ns.annotations_terms "
            "WHERE term IN (:{p}, :{p}_full)")


def term_params(name: str, term: str) -> dict:
    return {name: term, f"{name}_full": TERM_PREFIX + term}

# Study IDs that report a coordinate (within radius mm)
LOC_SET = """
SELECT DISTINCT study_id FROM ns.coordinates
WHERE ST_3DDWithin(
    geom,
    ST_SetSRID(ST_MakePoint(:{p}x, :{p}y, :{p}z), {srid}),
    :radius
)
"""


def run_query(set_a_sql: str, set_b_sql: str, params: dict):
    """Return studies in set A but not in set B (A \\ B), with metadata.
    The set difference is done inside PostgreSQL with EXCEPT."""
    sql = f"""
    WITH diff AS (
        ({set_a_sql})
        EXCEPT
        ({set_b_sql})
    ),
    {META_CTE}
    SELECT d.study_id, m.title, m.authors, m.journal, m.year
    FROM diff d
    LEFT JOIN meta m USING (study_id)
    ORDER BY m.year DESC NULLS LAST, d.study_id
    """
    with get_engine().begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def run_single(set_sql: str, params: dict):
    sql = f"""
    WITH a AS ({set_sql}),
         {META_CTE}
    SELECT a.study_id, m.title, m.authors, m.journal, m.year
    FROM a
    LEFT JOIN meta m USING (study_id)
    ORDER BY m.year DESC NULLS LAST, a.study_id
    """
    with get_engine().begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def pack(a, b, studies):
    return {
        "a": a,
        "b": b,
        "description": f"Studies that mention {a} but NOT {b}",
        "count": len(studies),
        "studies": studies,
    }


# -----------------------------
# Interactive page (/app): type terms or coordinates, see the results
# -----------------------------
UI_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neurosynth Dissociation</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --text:#1f2328; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --accent2:#7c3aed; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif; }
  main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .tabs { display:flex; gap:8px; margin-bottom: 12px; }
  .tab { border:1px solid var(--line); background:#fff; padding:8px 14px; border-radius:999px; cursor:pointer; font-size:.95rem; }
  .tab.on { background: var(--text); color:#fff; border-color: var(--text); }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  label { display:block; font-size:.85rem; color: var(--muted); margin-bottom: 4px; }
  input[type=text], input[type=number] { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px; font-size:1rem; }
  .opts { display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-top:12px; font-size:.95rem; }
  .opts input[type=number] { width: 90px; }
  button.go { margin-top: 14px; background: var(--accent); color:#fff; border:0; padding:10px 18px; border-radius:8px; font-size:1rem; cursor:pointer; }
  button.go:disabled { opacity:.6; cursor:wait; }
  .examples { margin-top: 10px; font-size:.85rem; color: var(--muted); }
  .examples a { color: var(--accent); cursor:pointer; margin-right: 10px; }
  .summary { display:flex; flex-wrap:wrap; gap:12px; margin-bottom: 12px; }
  .stat { flex:1 1 200px; border-left: 4px solid var(--accent); padding: 6px 12px; background: #f3f6ff; border-radius: 6px; }
  .stat.b { border-color: var(--accent2); background: #f6f2ff; }
  .stat .n { font-size: 1.6rem; font-weight: 700; }
  .stat .d { font-size: .85rem; color: var(--muted); }
  table { width:100%; border-collapse: collapse; font-size: .9rem; }
  th, td { text-align:left; padding: 8px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  th:first-child, td.y { width: 64px; white-space: nowrap; color: var(--muted); }
  .err { color: #b91c1c; }
  .muted { color: var(--muted); font-size: .85rem; }
  h3 { margin: 18px 0 8px; font-size: 1.05rem; }
  @media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Neurosynth Functional Dissociation</h1>
  <p class="sub">輸入兩個腦區詞彙或 MNI 座標，找出「提到 A 但沒提到 B」的研究。</p>

  <div class="card">
    <div class="tabs">
      <button class="tab on" data-mode="terms">用詞彙 Terms</button>
      <button class="tab" data-mode="locations">用座標 Locations</button>
    </div>

    <div class="row">
      <div><label id="la">詞彙 A</label><input type="text" id="a"></div>
      <div><label id="lb">詞彙 B</label><input type="text" id="b"></div>
    </div>

    <div class="opts">
      <label style="margin:0"><input type="checkbox" id="both" checked> 同時顯示反方向（B 但不是 A）</label>
      <span id="radiusBox" style="display:none">半徑 <input type="number" id="radius" value="0" min="0" step="1"> mm</span>
    </div>

    <button class="go" id="go">查詢</button>
    <div class="examples" id="examples"></div>
  </div>

  <div id="out"></div>
  <p class="muted">API：<code id="apiUrl"></code></p>
</main>

<script>
const EX = {
  terms: [["posterior cingulate", "ventromedial prefrontal"], ["amygdala", "hippocampus"]],
  locations: [["0_-52_26", "-2_50_-6"], ["-22_0_-20", "22_0_-20"]]
};
let mode = "terms";
const $ = id => document.getElementById(id);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function setMode(m) {
  mode = m;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.mode === m));
  $("la").textContent = m === "terms" ? "詞彙 A" : "座標 A（x_y_z）";
  $("lb").textContent = m === "terms" ? "詞彙 B" : "座標 B（x_y_z）";
  $("radiusBox").style.display = m === "locations" ? "" : "none";
  $("a").value = EX[m][0][0]; $("b").value = EX[m][0][1];
  $("examples").innerHTML = "範例：" + EX[m].map((p, i) =>
    `<a data-i="${i}">${esc(p[0])} vs ${esc(p[1])}</a>`).join("");
  $("examples").querySelectorAll("a").forEach(el => el.onclick = () => {
    const p = EX[mode][el.dataset.i]; $("a").value = p[0]; $("b").value = p[1]; run();
  });
}

function table(studies) {
  if (!studies.length) return '<p class="muted">沒有符合的研究。</p>';
  const rows = studies.slice(0, 200).map(s => `
    <tr>
      <td class="y">${esc(s.year ?? "")}</td>
      <td><a href="https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(s.study_id)}/" target="_blank" rel="noopener">${esc(s.title || s.study_id)}</a>
          <div class="muted">${esc(s.authors || "")}${s.journal ? " · " + esc(s.journal) : ""}</div></td>
    </tr>`).join("");
  const more = studies.length > 200 ? `<p class="muted">只顯示前 200 筆（共 ${studies.length} 筆）。</p>` : "";
  return `<table><thead><tr><th>年份</th><th>研究</th></tr></thead><tbody>${rows}</tbody></table>${more}`;
}

async function run() {
  const a = $("a").value.trim(), b = $("b").value.trim();
  if (!a || !b) { $("out").innerHTML = '<div class="card err">請輸入 A 和 B。</div>'; return; }
  const enc = s => encodeURIComponent(mode === "terms" ? s.replace(/\s+/g, "_") : s.replace(/[,\s]+/g, "_"));
  const q = new URLSearchParams();
  if ($("both").checked) q.set("both", "1");
  if (mode === "locations" && Number($("radius").value) > 0) q.set("radius", $("radius").value);
  const url = `/dissociate/${mode}/${enc(a)}/${enc(b)}` + (q.toString() ? "?" + q : "");
  $("apiUrl").textContent = url;
  $("go").disabled = true;
  $("out").innerHTML = '<div class="card muted">查詢中…（伺服器休眠時第一次可能要等快一分鐘）</div>';
  try {
    const r = await fetch(url);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.status);
    const ab = d.a_minus_b || d, ba = d.b_minus_a;
    const name = x => Array.isArray(x) ? "[" + x.join(", ") + "]" : x;
    let html = '<div class="card"><div class="summary">' +
      `<div class="stat"><div class="n">${ab.count}</div><div class="d">提到 <b>${esc(name(ab.a))}</b> 但沒提到 <b>${esc(name(ab.b))}</b></div></div>` +
      (ba ? `<div class="stat b"><div class="n">${ba.count}</div><div class="d">提到 <b>${esc(name(ba.a))}</b> 但沒提到 <b>${esc(name(ba.b))}</b></div></div>` : "") +
      "</div>" +
      `<h3>${esc(name(ab.a))} \\ ${esc(name(ab.b))}</h3>` + table(ab.studies) +
      (ba ? `<h3>${esc(name(ba.a))} \\ ${esc(name(ba.b))}</h3>` + table(ba.studies) : "") +
      "</div>";
    $("out").innerHTML = html;
  } catch (e) {
    $("out").innerHTML = `<div class="card err">查詢失敗：${esc(e.message)}</div>`;
  } finally {
    $("go").disabled = false;
  }
}

document.querySelectorAll(".tab").forEach(t => t.onclick = () => setMode(t.dataset.mode));
$("go").onclick = run;
["a", "b"].forEach(id => $(id).addEventListener("keydown", e => { if (e.key === "Enter") run(); }));
setMode("terms");
</script>
</body>
</html>
"""


# -----------------------------
# App
# -----------------------------
def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(Exception)
    def handle_error(e):
        code = getattr(e, "code", 500)
        if not isinstance(code, int):
            code = 500
        return jsonify({"error": str(e)}), code

    @app.get("/", endpoint="health")
    def health():
        return """
        <h2>Neurosynth Functional Dissociation API</h2>
        <p>Server working!</p>
        <p><a href="/app"><b>&rarr; Interactive search page (/app)</b></a></p>
        <ul>
          <li><a href="/dissociate/terms/posterior_cingulate/ventromedial_prefrontal">/dissociate/terms/posterior_cingulate/ventromedial_prefrontal</a></li>
          <li><a href="/dissociate/terms/ventromedial_prefrontal/posterior_cingulate">/dissociate/terms/ventromedial_prefrontal/posterior_cingulate</a></li>
          <li><a href="/dissociate/locations/0_-52_26/-2_50_-6">/dissociate/locations/0_-52_26/-2_50_-6</a></li>
          <li><a href="/dissociate/locations/-2_50_-6/0_-52_26">/dissociate/locations/-2_50_-6/0_-52_26</a></li>
          <li><a href="/dissociate/terms/posterior_cingulate/ventromedial_prefrontal?both=1">... ?both=1</a> (A\\B and B\\A in one response)</li>
          <li><a href="/dissociate/locations/0_-52_26/-2_50_-6?radius=6">... ?radius=6</a> (coordinates within 6 mm)</li>
          <li>Sanity check (amygdala, [-22,0,-20]):
              <a href="/terms/amygdala/studies">/terms/amygdala/studies</a> ·
              <a href="/locations/-22_0_-20/studies?radius=6">/locations/-22_0_-20/studies?radius=6</a></li>
          <li><a href="/test_db">/test_db</a> · <a href="/img">/img</a></li>
        </ul>
        """

    @app.get("/app", endpoint="ui")
    def ui():
        return UI_HTML

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file("amygdala.gif", mimetype="image/gif")

    # ---- single-set queries ----
    @app.get("/terms/<path:term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        t = normalize_term(term)
        studies = run_single(TERM_SET.format(p="t"), term_params("t", t))
        return jsonify({"term": t, "count": len(studies), "studies": studies})

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        x, y, z = parse_coords(coords)
        r = get_radius()
        studies = run_single(LOC_SET.format(p="a", srid=SRID),
                             {"ax": x, "ay": y, "az": z, "radius": r})
        return jsonify({"location": [x, y, z], "radius_mm": r,
                        "count": len(studies), "studies": studies})

    # ---- dissociation: terms ----
    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        a, b = normalize_term(term_a), normalize_term(term_b)
        params = {**term_params("ta", a), **term_params("tb", b)}
        set_a, set_b = TERM_SET.format(p="ta"), TERM_SET.format(p="tb")

        a_minus_b = pack(a, b, run_query(set_a, set_b, params))
        if not want_both():
            return jsonify(a_minus_b)
        b_minus_a = pack(b, a, run_query(set_b, set_a, params))
        return jsonify({"a_minus_b": a_minus_b, "b_minus_a": b_minus_a})

    # ---- dissociation: MNI coordinates ----
    @app.get("/dissociate/locations/<coords_a>/<coords_b>", endpoint="dissociate_locations")
    def dissociate_locations(coords_a, coords_b):
        xa, ya, za = parse_coords(coords_a)
        xb, yb, zb = parse_coords(coords_b)
        r = get_radius()
        params = {"ax": xa, "ay": ya, "az": za,
                  "bx": xb, "by": yb, "bz": zb, "radius": r}
        set_a = LOC_SET.format(p="a", srid=SRID)
        set_b = LOC_SET.format(p="b", srid=SRID)
        la, lb = [xa, ya, za], [xb, yb, zb]

        a_minus_b = pack(la, lb, run_query(set_a, set_b, params))
        a_minus_b["radius_mm"] = r
        if not want_both():
            return jsonify(a_minus_b)
        b_minus_a = pack(lb, la, run_query(set_b, set_a, params))
        b_minus_a["radius_mm"] = r
        return jsonify({"a_minus_b": a_minus_b, "b_minus_a": b_minus_a})

    # ---- smoke test ----
    @app.get("/test_db", endpoint="test_db")
    def test_db():
        eng = get_engine()
        payload = {"ok": False, "dialect": eng.dialect.name}
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()
                rows = conn.execute(text(
                    "SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z FROM ns.coordinates LIMIT 3"
                )).mappings().all()
                payload["coordinates_sample"] = [dict(r) for r in rows]
                rows = conn.execute(text(
                    "SELECT study_id, contrast_id, term, weight FROM ns.annotations_terms LIMIT 3"
                )).mappings().all()
                payload["annotations_terms_sample"] = [dict(r) for r in rows]
            payload["ok"] = True
            return jsonify(payload), 200
        except Exception as e:
            payload["error"] = str(e)
            return jsonify(payload), 500

    return app


# WSGI entry point: gunicorn app:app --bind 0.0.0.0:$PORT
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)

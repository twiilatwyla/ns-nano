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

# Study IDs that mention a term
TERM_SET = "SELECT DISTINCT study_id FROM ns.annotations_terms WHERE term = :{p}"

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

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file("amygdala.gif", mimetype="image/gif")

    # ---- single-set queries ----
    @app.get("/terms/<path:term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        t = normalize_term(term)
        studies = run_single(TERM_SET.format(p="t"), {"t": t})
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
        params = {"ta": a, "tb": b}
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

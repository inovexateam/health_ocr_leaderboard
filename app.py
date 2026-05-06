"""
Health OCR Leaderboard - Flask App
Upload health screenshots → OCR → auto leaderboard
No paid APIs. Uses Tesseract + OpenCV.
"""

import os, json, uuid, sqlite3, hashlib
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ocr_engine import extract_health_data

BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
DB_PATH = os.path.join(BASE, "health_data.db")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


# ── DB ────────────────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                member_name TEXT NOT NULL,
                date TEXT NOT NULL,
                steps INTEGER,
                calories INTEGER,
                distance_km REAL,
                active_minutes INTEGER,
                sleep_hours REAL,
                heart_rate INTEGER,
                floors INTEGER,
                source_app TEXT,
                confidence TEXT,
                raw_text TEXT,
                image_path TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS members (
                phone TEXT PRIMARY KEY,
                display_name TEXT NOT NULL
            )
        """)
        con.commit()


def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_image():
    """Receive an image + member name, run OCR, store result."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    member = request.form.get("member_name", "Unknown").strip()
    entry_date = request.form.get("date", str(date.today()))

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save upload
    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = str(uuid.uuid4())
    fname = f"{uid}.{ext}"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)

    # Run OCR
    try:
        data = extract_health_data(fpath)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    # Store in DB
    row_id = uid
    with get_db() as con:
        con.execute("""
            INSERT INTO entries (id, member_name, date, steps, calories,
              distance_km, active_minutes, sleep_hours, heart_rate, floors,
              source_app, confidence, raw_text, image_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (row_id, member, entry_date, data.steps, data.calories,
              data.distance_km, data.active_minutes, data.sleep_hours,
              data.heart_rate, data.floors, data.source_app,
              data.confidence, data.raw_text[:1000], fname))
        con.commit()

    return jsonify({
        "id": row_id,
        "member": member,
        "date": entry_date,
        "extracted": data.to_dict(),
        "fields_found": data.fields_found(),
    })


@app.route("/api/leaderboard")
def leaderboard():
    """Return ranked leaderboard for a given date range."""
    period = request.args.get("period", "daily")
    ref_date = request.args.get("date", str(date.today()))
    metric = request.args.get("metric", "steps")

    if metric not in ("steps", "calories", "distance_km", "active_minutes"):
        metric = "steps"

    if period == "weekly":
        date_filter = "date >= date(?, '-6 days') AND date <= ?"
        params = (ref_date, ref_date)
    elif period == "monthly":
        date_filter = "strftime('%Y-%m', date) = strftime('%Y-%m', ?)"
        params = (ref_date,)
    else:  # daily
        date_filter = "date = ?"
        params = (ref_date,)

    sql = f"""
        SELECT member_name,
               SUM(steps)          AS steps,
               SUM(calories)       AS calories,
               SUM(distance_km)    AS distance_km,
               SUM(active_minutes) AS active_minutes,
               COUNT(*)            AS posts,
               MAX(source_app)     AS source_app
        FROM entries
        WHERE {date_filter}
        GROUP BY member_name
        ORDER BY {metric} DESC NULLS LAST
    """
    with get_db() as con:
        rows = con.execute(sql, params).fetchall()

    results = []
    for i, row in enumerate(rows):
        d = dict(row)
        d["rank"] = i + 1
        if d["distance_km"]:
            d["distance_km"] = round(d["distance_km"], 2)
        results.append(d)

    return jsonify({"period": period, "date": ref_date, "metric": metric, "rows": results})


@app.route("/api/entries")
def entries():
    """All raw entries for a date (for the detail table)."""
    target = request.args.get("date", str(date.today()))
    with get_db() as con:
        rows = con.execute("""
            SELECT id, member_name, date, steps, calories, distance_km,
                   active_minutes, source_app, confidence, image_path, created_at
            FROM entries WHERE date = ?
            ORDER BY created_at DESC
        """, (target,)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/ocr_preview", methods=["POST"])
def ocr_preview():
    """Quick OCR without saving to DB — for the test/preview panel."""
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid type"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp = os.path.join(UPLOAD_FOLDER, f"preview_{uuid.uuid4()}.{ext}")
    file.save(tmp)
    try:
        data = extract_health_data(tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

    return jsonify(data.to_dict())


@app.route("/api/sample_images")
def sample_images():
    samples_dir = os.path.join(BASE, "static", "samples")
    files = [f for f in os.listdir(samples_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return jsonify([{"filename": f, "url": f"/static/samples/{f}"} for f in sorted(files)])


@app.route("/api/test_sample", methods=["POST"])
def test_sample():
    """Run OCR on a bundled sample image and return results."""
    data = request.get_json()
    filename = data.get("filename", "")
    member = data.get("member_name", "Test User")
    entry_date = data.get("date", str(date.today()))

    safe = secure_filename(filename)
    fpath = os.path.join(BASE, "static", "samples", safe)
    if not os.path.exists(fpath):
        return jsonify({"error": "Sample not found"}), 404

    try:
        result = extract_health_data(fpath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Also save to DB so leaderboard populates
    uid = str(uuid.uuid4())
    with get_db() as con:
        con.execute("""
            INSERT INTO entries (id, member_name, date, steps, calories,
              distance_km, active_minutes, sleep_hours, heart_rate, floors,
              source_app, confidence, raw_text, image_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (uid, member, entry_date, result.steps, result.calories,
              result.distance_km, result.active_minutes, result.sleep_hours,
              result.heart_rate, result.floors, result.source_app,
              result.confidence, result.raw_text[:1000],
              f"samples/{safe}"))
        con.commit()

    return jsonify({
        "id": uid,
        "member": member,
        "date": entry_date,
        "extracted": result.to_dict(),
        "fields_found": result.fields_found(),
    })


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_db()
    print("\n🏃 Health OCR Leaderboard running on http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)

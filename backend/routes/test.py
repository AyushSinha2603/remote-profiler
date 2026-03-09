import time
from flask import Blueprint, jsonify
from utils.sheets import append_to_sheet, fetch_all_logs

test_bp = Blueprint("test", __name__)


@test_bp.route("/test/sheets", methods=["POST"])
def test_sheets():
    """
    Writes a dummy detection row to Google Sheets, then reads back
    all rows to confirm round-trip connectivity.

    Returns:
      - written_row : the payload that was appended
      - row_count   : total rows in the sheet (excluding header)
      - last_rows   : up to the 3 most recent rows read back
      - error       : present only on failure
    """
    dummy = {
        "timestamp": int(time.time()),
        "lat": 12.9716,
        "lng": 77.5946,
        "area_m2": 0.042,
        "depth_m": 0.075,
        "volume_m3": 0.0022,
        "volume_liters": 2.205,
        "confidence": 0.91,
    }

    try:
        append_to_sheet(dummy)
    except Exception as e:
        return jsonify({"status": "error", "stage": "write", "error": str(e)}), 500

    try:
        all_rows = fetch_all_logs()
    except Exception as e:
        return jsonify({"status": "error", "stage": "read", "error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "written_row": dummy,
        "row_count": len(all_rows),
        "last_rows": all_rows[-3:] if len(all_rows) >= 3 else all_rows,
    }), 200

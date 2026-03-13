from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_cors import CORS
from routes.detect import detect_bp
from routes.logs import logs_bp
from routes.test import test_bp
from utils.material_estimator import enable_ml_mode

app = Flask(__name__)
CORS(app)

# Enable ML-based material prediction (uses rule-based as fallback if models unavailable)
ml_status = enable_ml_mode('backend/models')
if not ml_status:
    print("⚠ ML models not available. Using rule-based estimation. Run: python backend/utils/ml_trainer.py --train backend/data/repair_data.csv")

@app.get("/")
def index():
    return {"message": "Hello, World!"}

app.register_blueprint(detect_bp, url_prefix="/api")
app.register_blueprint(logs_bp, url_prefix="/api")
app.register_blueprint(test_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

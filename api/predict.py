import pickle
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from http.server import BaseHTTPRequestHandler
import json
import io

# Load model
with open("/content/drive/MyDrive/Data/model.pkl", "rb") as f:
    model = pickle.load(f)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        df = pd.read_csv(io.BytesIO(body))

        preds = model.predict(df)
        probs = model.predict_proba(df)

        failure_confidence = np.mean(probs[:,1]) * 100

        beta = 2.0
        eta = 1000

        F = failure_confidence / 100
        rul = eta * (-np.log(1 - F))**(1/beta)

        response = {
            "confidence": round(failure_confidence,2),
            "rul": round(rul,2)
        }

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

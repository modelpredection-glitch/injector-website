from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

beta = 3.15
eta = 80000


def pipeline(file, odometer):

    df = pd.read_csv(file)

    X = df.values

    pred = model.predict(X)

    prob = model.predict_proba(X)

    confidence = prob.max(axis=1)

    avg_confidence = confidence.mean()

    t = eta * (-np.log(avg_confidence)) ** (1 / beta)

    rul = t - odometer

    return pred, avg_confidence, rul


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        file = request.files["file"]

        odometer = float(request.form["odometer"])

        pred, conf, rul = pipeline(file, odometer)

        final_pred = int(pd.Series(pred).mode()[0])

        return render_template(
            "result.html",
            prediction=final_pred,
            confidence=round(conf*100,2),
            rul=int(rul)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
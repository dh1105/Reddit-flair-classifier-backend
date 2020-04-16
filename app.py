from flask import Flask, jsonify, request, json, session, redirect, url_for, Response
from flask_cors import CORS
import TextClassifier as textclassifier

app = Flask(__name__)
app.secret_key = "1234"
CORS(app, supports_credentials=True)
predictions = textclassifier.TextClassifier()


@app.route("/predict", methods=['POST'])
def predict():
    details = request.get_json()
    if 'url' in details:
        url = details['url']
        try:
            flair = predictions.logreg_predict_class(url)
            return jsonify(flair)
        except Exception as e:
            print(e)
            return str(e), 400
    else:
        return "Oops! Incorrect format", 400

@app.route("/automated_testing", methods=['POST'])
def automated_testing():
    if 'file' in request.files:
        f = request.files['file']
        model = "log"
        if 'model' in request.form:
            model = request.form['model']
        flair_predictions = {}
        for line in f.read().decode().split():
            try:
                flair_predictions[line] = predictions.logreg_predict_class(line)
            except Exception as e:
                flair_predictions[line] = str(e)
        return jsonify(flair_predictions)
    else:
        return "Oops! Incorrect format", 400

if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)

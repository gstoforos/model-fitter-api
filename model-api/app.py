# app.py
from flask import Flask, request, jsonify
from model_fitter import fit_models  # Replace with your logic function

app = Flask(__name__)

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    result = fit_models(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run()

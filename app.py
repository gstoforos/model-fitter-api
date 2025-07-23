from flask import Flask, request, jsonify
from model_bingham import fit_bingham_model

app = Flask(__name__)

@app.route('/fit', methods=['POST'])
def fit_bingham():
    try:
        data = request.get_json()
        shear_rates = data.get("shear_rates", [])
        shear_stresses = data.get("shear_stresses", [])
        flow_rate = float(data.get("flow_rate", 1))
        diameter = float(data.get("diameter", 1))
        density = float(data.get("density", 1))

        if not (len(shear_rates) == len(shear_stresses) and len(shear_rates) > 1):
            return jsonify({"error": "Shear rate and stress arrays must be of equal length and contain at least two values."}), 400

        result = fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






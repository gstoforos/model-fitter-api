
from flask import Flask, request, jsonify
from scipy.optimize import curve_fit
import numpy as np

app = Flask(__name__)

def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# Rheological model functions
def model_newtonian(g, mu): return mu * g
def model_power(g, k, n): return k * g ** n
def model_hb(g, tau0, k, n): return tau0 + k * g ** n
def model_bingham(g, tau0, mu): return tau0 + mu * g
def model_casson(g, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * g)) ** 2

@app.route("/fit", methods=["POST"])
def fit_models():
    try:
        data = request.get_json()
        gamma = np.array(data["shear_rates"], dtype=float)
        tau = np.array(data["shear_stresses"], dtype=float)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    results = {}

    # Newtonian
    try:
        popt, _ = curve_fit(model_newtonian, gamma, tau)
        pred = model_newtonian(gamma, *popt)
        results["Newtonian"] = {
            "model": "Newtonian",
            "tau0": 0,
            "k": float(popt[0]),
            "n": 1.0,
            "r2": r_squared(tau, pred)
        }
    except:
        results["Newtonian"] = {
            "model": "Newtonian",
            "tau0": 0, "k": 0, "n": 1.0, "r2": 0
        }

    # Power Law
    try:
        popt, _ = curve_fit(model_power, gamma, tau, bounds=(0, np.inf))
        pred = model_power(gamma, *popt)
        results["Power-Law"] = {
            "model": "Power-Law",
            "tau0": 0,
            "k": float(popt[0]),
            "n": float(popt[1]),
            "r2": r_squared(tau, pred)
        }
    except:
        results["Power-Law"] = {
            "model": "Power-Law",
            "tau0": 0, "k": 0, "n": 1.0, "r2": 0
        }

    # Herschel–Bulkley
    try:
        popt, _ = curve_fit(model_hb, gamma, tau, bounds=(0, np.inf))
        pred = model_hb(gamma, *popt)
        results["Herschel-Bulkley"] = {
            "model": "Herschel-Bulkley",
            "tau0": float(popt[0]),
            "k": float(popt[1]),
            "n": float(popt[2]),
            "r2": r_squared(tau, pred)
        }
    except:
        results["Herschel-Bulkley"] = {
            "model": "Herschel-Bulkley",
            "tau0": 0, "k": 0, "n": 1.0, "r2": 0
        }

    # Bingham
    try:
        popt, _ = curve_fit(model_bingham, gamma, tau, bounds=(0, np.inf))
        pred = model_bingham(gamma, *popt)
        results["Bingham"] = {
            "model": "Bingham",
            "tau0": float(popt[0]),
            "k": float(popt[1]),
            "n": 1.0,
            "r2": r_squared(tau, pred)
        }
    except:
        results["Bingham"] = {
            "model": "Bingham",
            "tau0": 0, "k": 0, "n": 1.0, "r2": 0
        }

    # Casson
    try:
        popt, _ = curve_fit(model_casson, gamma, tau, bounds=(0, np.inf))
        pred = model_casson(gamma, *popt)
        results["Casson"] = {
            "model": "Casson",
            "tau0": float(popt[0]),
            "k": float(popt[1]),
            "n": 0.5,
            "r2": r_squared(tau, pred)
        }
    except:
        results["Casson"] = {
            "model": "Casson",
            "tau0": 0, "k": 0, "n": 0.5, "r2": 0
        }

    # Best model logic
    r2s = {k: v["r2"] for k, v in results.items()}
    best_model_name = max(r2s, key=r2s.get)
    best_model = results[best_model_name]

    # Override rules
    try:
        all_r2 = list(r2s.values())
        if all(r >= 0.99 for r in all_r2) and results["Newtonian"]["r2"] >= 0.99:
            best_model = results["Newtonian"]
        elif results["Bingham"]["r2"] >= 0.99 and results["Herschel-Bulkley"]["r2"] >= 0.99:
            best_model = results["Bingham"]
        elif results["Power-Law"]["r2"] >= 0.99 and results["Herschel-Bulkley"]["r2"] >= 0.99:
            best_model = results["Power-Law"]
    except:
        pass

    return jsonify({
        "best_model": best_model,
        "all_models": results
    })

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        gamma_dot = np.array(data['shear_rates'], dtype=float)
        sigma = np.array(data['shear_stresses'], dtype=float)
        result = fit_all_models(gamma_dot, sigma)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

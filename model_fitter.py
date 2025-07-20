from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
import math

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

@app.route('/fit', methods=['POST'])
def fit_models():
    data = request.get_json()

    shear_rates = np.array(data['shear_rates'])
    shear_stresses = np.array(data['shear_stresses'])
    flow_rate = float(data.get('flow_rate', 1))
    diameter = float(data.get('diameter', 1))
    density = float(data.get('density', 1))

    if len(shear_rates) < 2 or len(shear_stresses) < 2:
        return jsonify({"error": "Need at least two data points."}), 400

    models = []
    v = 4 * flow_rate / (math.pi * diameter ** 2)  # Average velocity

    # Newtonian
    def model_newtonian(gamma_dot, mu):
        return mu * gamma_dot

    try:
        popt, _ = curve_fit(model_newtonian, shear_rates, shear_stresses)
        mu = popt[0]
        pred = model_newtonian(shear_rates, *popt)
        r2 = r_squared(shear_stresses, pred)
        re = (density * v * diameter) / mu
        models.append({
            "name": "Newtonian",
            "equation": "\u03c4 = \u03bc \u22c5 \u0263\u0307",
            "k": mu,
            "n": 1,
            "tau0": 0,
            "r2": r2,
            "mu_app": mu,
            "re": re
        })
    except:
        pass

    # Power Law
    def model_power(gamma_dot, k, n):
        return k * gamma_dot ** n

    try:
        popt, _ = curve_fit(model_power, shear_rates, shear_stresses, bounds=(0, np.inf))
        k, n = popt
        pred = model_power(shear_rates, *popt)
        r2 = r_squared(shear_stresses, pred)
        re = (8 * density * v ** (2 - n) * diameter ** n) / (k * 3 ** n)
        mu_app = k * shear_rates.mean() ** (n - 1)
        models.append({
            "name": "Power Law",
            "equation": "\u03c4 = K \u22c5 \u0263\u0307^n",
            "k": k,
            "n": n,
            "tau0": 0,
            "r2": r2,
            "mu_app": mu_app,
            "re": re
        })
    except:
        pass

    # Herschel-Bulkley
    def model_hb(gamma_dot, tau0, k, n):
        return tau0 + k * gamma_dot ** n

    try:
        popt, _ = curve_fit(model_hb, shear_rates, shear_stresses, bounds=(0, np.inf))
        tau0, k, n = popt
        pred = model_hb(shear_rates, *popt)
        r2 = r_squared(shear_stresses, pred)
        mu_app = (tau0 / shear_rates.mean()) + k * shear_rates.mean() ** (n - 1)
        re = (density * v * diameter) / mu_app
        models.append({
            "name": "Herschel-Bulkley",
            "equation": "\u03c4 = \u03c4\u2080 + K \u22c5 \u0263\u0307^n",
            "k": k,
            "n": n,
            "tau0": tau0,
            "r2": r2,
            "mu_app": mu_app,
            "re": re
        })
    except:
        pass

    # Bingham Plastic
    def model_bingham(gamma_dot, tau0, mu):
        return tau0 + mu * gamma_dot

    try:
        popt, _ = curve_fit(model_bingham, shear_rates, shear_stresses, bounds=(0, np.inf))
        tau0, mu = popt
        pred = model_bingham(shear_rates, *popt)
        r2 = r_squared(shear_stresses, pred)
        mu_app = tau0 / shear_rates.mean() + mu
        re = (density * v * diameter) / mu * (1 - (4 / 3) * (tau0 / (mu * shear_rates.mean())))
        models.append({
            "name": "Bingham Plastic",
            "equation": "\u03c4 = \u03c4\u2080 + \u03bc \u22c5 \u0263\u0307",
            "k": mu,
            "n": 1,
            "tau0": tau0,
            "r2": r2,
            "mu_app": mu_app,
            "re": re
        })
    except:
        pass

    # Casson
    def model_casson(gamma_dot, tau0, mu):
        return (np.sqrt(tau0) + np.sqrt(mu * gamma_dot)) ** 2

    try:
        popt, _ = curve_fit(model_casson, shear_rates, shear_stresses, bounds=(0, np.inf))
        tau0, mu = popt
        pred = model_casson(shear_rates, *popt)
        r2 = r_squared(shear_stresses, pred)
        mu_app = ((np.sqrt(tau0) + np.sqrt(mu * shear_rates.mean())) ** 2) / shear_rates.mean()
        re = (density * v * diameter) / mu_app
        models.append({
            "name": "Casson",
            "equation": "\u03c4 = (\u221a\u03c4\u2080 + \u221a(\u03bc \u22c5 \u0263\u0307))^2",
            "k": mu,
            "n": 1,
            "tau0": tau0,
            "r2": r2,
            "mu_app": mu_app,
            "re": re
        })
    except:
        pass

    best = max(models, key=lambda m: m['r2']) if models else None
    return jsonify({"best_model": best, "all_models": models})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

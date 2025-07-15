from flask import Flask, request, jsonify
from scipy.optimize import curve_fit
import numpy as np

app = Flask(__name__)

def newtonian(gamma, mu):
    return mu * gamma

def power_law(gamma, k, n):
    return k * gamma**n

def herschel_bulkley(gamma, tau0, k, n):
    return tau0 + k * gamma**n

def bingham(gamma, tau0, mu):
    return tau0 + mu * gamma

def casson(gamma, a, b):
    return (np.sqrt(a) + np.sqrt(b * gamma))**2

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma = np.array(data.get("shear_rate", []), dtype=float)
    sigma = np.array(data.get("shear_stress", []), dtype=float)

    if len(gamma) < 2 or len(sigma) < 2:
        return jsonify({"error": "Insufficient data"}), 400

    models = {}

    try:
        popt, _ = curve_fit(newtonian, gamma, sigma)
        pred = newtonian(gamma, *popt)
        models["Newtonian"] = {"k": popt[0], "n": 1, "sigma0": 0, "r2": r2(sigma, pred)}
    except: pass

    try:
        popt, _ = curve_fit(power_law, gamma, sigma, bounds=(0, np.inf))
        pred = power_law(gamma, *popt)
        models["Power-Law"] = {"k": popt[0], "n": popt[1], "sigma0": 0, "r2": r2(sigma, pred)}
    except: pass

    try:
        popt, _ = curve_fit(herschel_bulkley, gamma, sigma, bounds=(0, np.inf))
        pred = herschel_bulkley(gamma, *popt)
        models["Herschel–Bulkley"] = {"sigma0": popt[0], "k": popt[1], "n": popt[2], "r2": r2(sigma, pred)}
    except: pass

    try:
        popt, _ = curve_fit(bingham, gamma, sigma, bounds=(0, np.inf))
        pred = bingham(gamma, *popt)
        models["Bingham Plastic"] = {"sigma0": popt[0], "k": popt[1], "n": 1, "r2": r2(sigma, pred)}
    except: pass

    try:
        popt, _ = curve_fit(casson, gamma, sigma, bounds=(0, np.inf))
        pred = casson(gamma, *popt)
        models["Casson"] = {"sigma0": popt[0], "k": popt[1], "n": 1, "r2": r2(sigma, pred)}
    except: pass

    # ✅ Safe defaults
    for m in models.values():
        for key in ["k", "n", "sigma0", "r2"]:
            if key not in m or m[key] is None or np.isnan(m[key]):
                m[key] = 0 if key != "n" else 1

    # ✅ Final best-fit logic
    best = max(models.items(), key=lambda x: x[1]["r2"], default=(None, {}))
    best_model, best_params = best if best else ("Unknown", {"k": 0, "n": 1, "sigma0": 0, "r2": 0})

    all_r2 = [m["r2"] for m in models.values()]
    if all(r >= 0.99 for r in all_r2) and "Newtonian" in models:
        best_model = "Newtonian"
        best_params = models["Newtonian"]
    elif "Bingham Plastic" in models and "Herschel–Bulkley" in models:
        if models["Bingham Plastic"]["r2"] >= 0.99 and models["Herschel–Bulkley"]["r2"] >= 0.99:
            best_model = "Bingham Plastic"
            best_params = models["Bingham Plastic"]
    elif "Power-Law" in models and "Herschel–Bulkley" in models:
        if models["Power-Law"]["r2"] >= 0.99 and models["Herschel–Bulkley"]["r2"] >= 0.99:
            best_model = "Power-Law"
            best_params = models["Power-Law"]

    best_params["model"] = best_model
    return jsonify(best_params)

if __name__ == '__main__':
    app.run()

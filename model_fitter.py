import numpy as np
from flask import Flask, request, jsonify
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os

app = Flask(__name__)

# Rheological model equations
def newtonian(gamma, mu):
    return mu * gamma

def power_law(gamma, k, n):
    return k * gamma**n

def herschel_bulkley(gamma, tau0, k, n):
    return tau0 + k * gamma**n

def bingham(gamma, tau0, mu):
    return tau0 + mu * gamma

def casson(gamma, tau0, k):
    return (np.sqrt(tau0) + np.sqrt(k * gamma))**2

# Fitting helper
def fit_model(model_func, gamma, tau, p0=None, bounds=(-np.inf, np.inf)):
    try:
        popt, _ = curve_fit(model_func, gamma, tau, p0=p0, bounds=bounds, maxfev=10000)
        predictions = model_func(gamma, *popt)
        r2 = r2_score(tau, predictions)
        return list(popt), r2
    except Exception:
        return None, -1

@app.route("/fit", methods=["POST"])
def fit_all_models():
    data = request.get_json(force=True)
    gamma = np.array(data.get("shear_rates", []), dtype=np.float64)
    tau = np.array(data.get("shear_stresses", []), dtype=np.float64)

    if len(gamma) < 2 or len(tau) < 2:
        return jsonify({"error": "Insufficient data points"}), 400

    results = {}

    # Newtonian
    (mu,), r2_n = fit_model(newtonian, gamma, tau, p0=[1])
    results["Newtonian"] = {"k": mu, "n": 1, "sigma0": 0, "r2": r2_n}

    # Power Law
    (k_pl, n_pl), r2_pl = fit_model(power_law, gamma, tau, p0=[1, 1])
    results["Power-Law"] = {"k": k_pl, "n": n_pl, "sigma0": 0, "r2": r2_pl}

    # Herschel–Bulkley
    (tau0_hb, k_hb, n_hb), r2_hb = fit_model(herschel_bulkley, gamma, tau, p0=[0.1, 1, 1])
    results["Herschel-Bulkley"] = {"k": k_hb, "n": n_hb, "sigma0": tau0_hb, "r2": r2_hb}

    # Bingham
    (tau0_b, mu_b), r2_b = fit_model(bingham, gamma, tau, p0=[0.1, 1])
    results["Bingham-Plastic"] = {"k": mu_b, "n": 1, "sigma0": tau0_b, "r2": r2_b}

    # Casson
    (tau0_c, k_c), r2_c = fit_model(casson, gamma, tau, p0=[1, 1])
    results["Casson"] = {"k": k_c, "n": 1, "sigma0": tau0_c, "r2": r2_c}

    # Selection logic
    best_model = "Unknown"
    best_r2 = max(v["r2"] for v in results.values())

    # Smart override rules
    if all(v["r2"] >= 0.99 for v in results.values()) and results["Newtonian"]["r2"] >= 0.99:
        best_model = "Newtonian"
    elif results["Herschel-Bulkley"]["r2"] >= 0.99 and results["Bingham-Plastic"]["r2"] >= 0.99:
        best_model = "Bingham-Plastic"
    elif results["Herschel-Bulkley"]["r2"] >= 0.99 and results["Power-Law"]["r2"] >= 0.99:
        best_model = "Power-Law"
    else:
        best_model = max(results.items(), key=lambda x: x[1]["r2"])[0]

    final = results[best_model]

    # Ensure null-safe numeric output
    for key in ["k", "n", "sigma0"]:
        if final[key] is None or np.isnan(final[key]):
            final[key] = 0 if key != "n" else 1

    return jsonify({
        "model": best_model,
        "k": final["k"],
        "n": final["n"],
        "sigma0": final["sigma0"],
        "r2": final["r2"]
    })

# ✅ Required for Render.com or Railway.app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

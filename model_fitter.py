from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

def safe_float(val, default=1.0):
    try:
        return float(val)
    except:
        return default

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json(force=True)
        logging.info("Received JSON: %s", data)

        shear_rates = np.array(data.get('shear_rates', []), dtype=float)
        shear_stresses = np.array(data.get('shear_stresses', []), dtype=float)

        if len(shear_rates) < 2 or len(shear_stresses) < 2:
            return jsonify({"error": "Need at least two data points."}), 400

        flow_rate = safe_float(data.get('flow_rate'), 1)
        diameter = safe_float(data.get('diameter'), 1)
        density = safe_float(data.get('density'), 1)

        use_re = all(x > 0 for x in [flow_rate, diameter, density])
        v = 4 * flow_rate / (np.pi * diameter ** 2) if use_re else 1

        models = []

        def add_model(name, equation, k, n, tau0, mu_app, r2, re, desc=""):
            models.append({
                "model": name,
                "equation": equation,
                "k": k,
                "n": n,
                "tau0": tau0,
                "mu_app": mu_app,
                "r2": r2,
                "re": re,
                "description": desc
            })

        # Newtonian
        def model_newtonian(gamma_dot, mu): return mu * gamma_dot
        try:
            popt, _ = curve_fit(model_newtonian, shear_rates, shear_stresses)
            mu = popt[0]
            r2 = r_squared(shear_stresses, model_newtonian(shear_rates, *popt))
            mu_app = mu
            re = (density * v * diameter) / mu if use_re else 0
            add_model("Newtonian", "τ = μ · γ̇", mu, 1, 0, mu_app, r2, re, "Linear relation between stress and rate.")
        except Exception as e:
            logging.warning(f"Newtonian fit failed: {e}")

        # Power Law
        def model_power(gamma_dot, k, n): return k * gamma_dot ** n
        try:
            popt, _ = curve_fit(model_power, shear_rates, shear_stresses, bounds=(0, np.inf))
            k, n = popt
            r2 = r_squared(shear_stresses, model_power(shear_rates, *popt))
            mu_app = k * shear_rates.mean() ** (n - 1)
            re = (8 * density * v ** (2 - n) * diameter ** n) / (k * 3 ** n) if use_re else 0
            add_model("Power Law", "τ = K · γ̇ⁿ", k, n, 0, mu_app, r2, re, "No yield stress; shear thinning/thickening.")
        except Exception as e:
            logging.warning(f"Power Law fit failed: {e}")

        # Herschel–Bulkley
        def model_hb(gamma_dot, tau0, k, n): return tau0 + k * gamma_dot ** n
        try:
            popt, _ = curve_fit(model_hb, shear_rates, shear_stresses, bounds=(0, np.inf))
            tau0, k, n = popt
            r2 = r_squared(shear_stresses, model_hb(shear_rates, *popt))
            mu_app = (tau0 / shear_rates.mean()) + k * shear_rates.mean() ** (n - 1)
            re = (density * v * diameter) / mu_app if use_re else 0
            add_model("Herschel–Bulkley", "τ = τ₀ + K · γ̇ⁿ", k, n, tau0, mu_app, r2, re,
                      "Yield stress fluid with shear thinning/thickening.")
        except Exception as e:
            logging.warning(f"Herschel–Bulkley fit failed: {e}")

        # Bingham Plastic
        def model_bingham(gamma_dot, tau0, mu): return tau0 + mu * gamma_dot
        try:
            popt, _ = curve_fit(model_bingham, shear_rates, shear_stresses, bounds=(0, np.inf))
            tau0, mu = popt
            r2 = r_squared(shear_stresses, model_bingham(shear_rates, *popt))
            mu_app = tau0 / shear_rates.mean() + mu
            correction = 1 - (4 / 3) * (tau0 / (mu * shear_rates.mean()))
            re = ((density * v * diameter) / mu) * correction if use_re else 0
            add_model("Bingham Plastic", "τ = τ₀ + μ · γ̇", mu, 1, tau0, mu_app, r2, re,
                      "Has yield stress. Linear after yield.")
        except Exception as e:
            logging.warning(f"Bingham fit failed: {e}")

        # Casson
        def model_casson(gamma_dot, tau0, mu):
            safe_gamma = np.clip(gamma_dot, 1e-6, None)
            return (np.sqrt(tau0) + np.sqrt(mu * safe_gamma)) ** 2
        try:
            popt, _ = curve_fit(model_casson, shear_rates, shear_stresses, bounds=(0, np.inf))
            tau0, mu = popt
            r2 = r_squared(shear_stresses, model_casson(shear_rates, *popt))
            mu_app = ((np.sqrt(tau0) + np.sqrt(mu * shear_rates.mean())) ** 2) / shear_rates.mean()
            re = (density * v * diameter) / mu_app if use_re else 0
            add_model("Casson", "τ = (√τ₀ + √(μ · γ̇))²", mu, 1, tau0, mu_app, r2, re,
                      "Empirical. Used for chocolate, blood.")
        except Exception as e:
            logging.warning(f"Casson fit failed: {e}")

        if not models:
            return jsonify({"error": "Model fitting failed for all models."}), 400

        best_model = max(models, key=lambda m: m['r2'])

        return jsonify({
            "best_model": best_model,
            "all_models": models
        })

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma)
        return {'model': 'Newtonian', 'mu': popt[0], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Newtonian fit failed: {e}")
        return {'model': 'Newtonian', 'mu': 1.0, 'r2': 0}

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Power Law fit failed: {e}")
        return {'model': 'Power Law', 'k': 1.0, 'n': 1.0, 'r2': 0}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Herschel–Bulkley fit failed: {e}")
        return {'model': 'Herschel–Bulkley', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'r2': 0}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'n': 1.0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Casson fit failed: {e}")
        return {'model': 'Casson', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'r2': 0}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'mu': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Bingham Plastic fit failed: {e}")
        return {'model': 'Bingham Plastic', 'sigma0': 0.0, 'mu': 1.0, 'r2': 0}

def fit_all_models(gamma_dot, sigma):
    models = {
        "Newtonian": fit_newtonian(gamma_dot, sigma),
        "Power Law": fit_power_law(gamma_dot, sigma),
        "Herschel–Bulkley": fit_herschel_bulkley(gamma_dot, sigma),
        "Casson": fit_casson(gamma_dot, sigma),
        "Bingham Plastic": fit_bingham(gamma_dot, sigma)
    }

    # Normalize parameters to always include sigma0, k, n
    for model in models.values():
        model["k"] = model.get("k", model.get("mu", 1.0))      # Use mu if k missing
        model["n"] = model.get("n", 1.0)
        model["sigma0"] = model.get("sigma0", 0.0)

    return {"models": list(models.values())}

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rate'], dtype=float)
    sigma = np.array(data['shear_stress'], dtype=float)
    result = fit_all_models(gamma_dot, sigma)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

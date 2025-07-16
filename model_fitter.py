from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    return {
        'model': 'Newtonian',
        'tau0': 0,
        'k': float(popt[0]),
        'n': 1.0,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Power-Law',
        'tau0': 0,
        'k': float(popt[0]),
        'n': float(popt[1]),
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, tau0, k, n): return tau0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Herschel-Bulkley',
        'tau0': float(popt[0]),
        'k': float(popt[1]),
        'n': float(popt[2]),
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, tau0, mu): return tau0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Bingham',
        'tau0': float(popt[0]),
        'k': float(popt[1]),
        'n': 1.0,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Casson',
        'tau0': float(popt[0]),
        'k': float(popt[1]),
        'n': 0.5,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_all_models(gamma_dot, sigma):
    models = []

    try: models.append(fit_newtonian(gamma_dot, sigma))
    except: pass
    try: models.append(fit_power_law(gamma_dot, sigma))
    except: pass
    try: models.append(fit_herschel_bulkley(gamma_dot, sigma))
    except: pass
    try: models.append(fit_bingham(gamma_dot, sigma))
    except: pass
    try: models.append(fit_casson(gamma_dot, sigma))
    except: pass

    if not models:
        return {'error': 'No model could be fitted'}

    best = max(models, key=lambda m: m['r2'])
    return {
        'best_model': best,
        'all_models': models
    }

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

from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import logging
import math

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def safe(val, default=1e-6):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except:
        return default

def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma)
        mu = safe(popt[0])
        return {
            'model': 'Newtonian',
            'sigma0': 0.0,
            'k': mu,
            'n': 1.0,
            'mu': mu,
            'r2': safe(r2_score(sigma, model(gamma_dot, mu)))
        }
    except Exception as e:
        logging.error(f"Newtonian fit failed: {e}")
        return {
            'model': 'Newtonian',
            'sigma0': 1e-6,
            'k': 1e-6,
            'n': 1.0,
            'mu': 1e-6,
            'r2': 0.0
        }

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {
            'model': 'Power Law',
            'sigma0': 0.0,
            'k': safe(popt[0]),
            'n': safe(popt[1]),
            'mu': 1.0,
            'r2': safe(r2_score(sigma, model(gamma_dot, *popt)))
        }
    except Exception as e:
        logging.error(f"Power Law fit failed: {e}")
        return {
            'model': 'Power Law',
            'sigma0': 1e-6,
            'k': 1e-6,
            'n': 1.0,
            'mu': 1.0,
            'r2': 0.0
        }

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {
            'model': 'Herschel–Bulkley',
            'sigma0': safe(popt[0]),
            'k': safe(popt[1]),
            'n': safe(popt[2]),
            'mu': 1.0,
            'r2': safe(r2_score(sigma, model(gamma_dot, *popt)))
        }
    except Exception as e:
        logging.error(f"Herschel–Bulkley fit failed: {e}")
        return {
            'model': 'Herschel–Bulkley',
            'sigma0': 1e-6,
            'k': 1e-6,
            'n': 1.0,
            'mu': 1.0,
            'r2': 0.0
        }

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k):
        return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot)) ** 2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {
            'model': 'Casson',
            'sigma0': safe(popt[0]),
            'k': safe(popt[1]),
            'n': 1.0,
            'mu': 1.0,
            'r2': safe(r2_score(sigma, model(gamma_dot, *popt)))
        }
    except Exception as e:
        logging.error(f"Casson fit failed: {e}")
        return {
            'model': 'Casson',
            'sigma0': 1e-6,
            'k': 1e-6,
            'n': 1.0,
            'mu': 1.0,
            'r2': 0.0
        }

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {
            'model': 'Bingham Plastic',
            'sigma0': safe(popt[0]),
            'k': safe(popt[1]),
            'n': 1.0,
            'mu': safe(popt[1]),
            'r2': safe(r2_score(sigma, model(gamma_dot, *popt)))
        }
    except Exception as e:
        logging.error(f"Bingham fit failed: {e}")
        return {
            'model': 'Bingham Plastic',
            'sigma0': 1e-6,
            'k': 1e-6,
            'n': 1.0,
            'mu': 1e-6,
            'r2': 0.0
        }

def fit_all_models(gamma_dot, sigma):
    models = [
        fit_newtonian(gamma_dot, sigma),
        fit_power_law(gamma_dot, sigma),
        fit_herschel_bulkley(gamma_dot, sigma),
        fit_casson(gamma_dot, sigma),
        fit_bingham(gamma_dot, sigma)
    ]

    valid = [m for m in models if m['r2'] >= 0]

    # Selection logic
    if all(m['r2'] > 0.99 for m in valid):
        best = next((m for m in valid if m['model'] == 'Newtonian'), valid[0])
    else:
        power = next((m for m in valid if m['model'] == 'Power Law'), None)
        hb = next((m for m in valid if m['model'] == 'Herschel–Bulkley'), None)
        bingham = next((m for m in valid if m['model'] == 'Bingham Plastic'), None)

        if power and hb and abs(power['r2'] - hb['r2']) < 0.01:
            best = power
        elif bingham and hb and abs(bingham['r2'] - hb['r2']) < 0.01:
            best = bingham
        else:
            best = max(valid, key=lambda m: m['r2'])

    return best, models

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        gamma_dot = np.array(data['shear_rate'], dtype=float)
        sigma = np.array(data['shear_stress'], dtype=float)
        best, all_models = fit_all_models(gamma_dot, sigma)
        return jsonify({
            "best_model": best['model'],
            "tau0": best['sigma0'],
            "k": best['k'],
            "n": best['n'],
            "mu": best['mu'],
            "r2": best['r2'],
            "all_models": all_models
        })
    except Exception as e:
        logging.error(f"Failed to process request: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

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
        mu = popt[0]
        return {
            'model': 'Newtonian',
            'sigma0': 0.0,
            'k': mu,
            'n': 1.0,
            'mu': mu,
            'r2': r2_score(sigma, model(gamma_dot, mu))
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
            'k': popt[0],
            'n': popt[1],
            'mu': 1.0,
            'r2': r2_score(sigma, model(gamma_dot, *popt))
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
            'sigma0': popt[0],
            'k': popt[1],
            'n': popt[2],
            'mu': 1.0,
            'r2': r2_score(sigma, model(gamma_dot, *popt))
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
            'sigma0': popt[0],
            'k': popt[1],
            'n': 1.0,
            'mu': 1.0,
            'r2': r2_score(sigma, model(gamma_dot, *popt))
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
            'sigma0': popt[0],
            'k': popt[1],
            'n': 1.0,
            'mu': popt[1],
            'r2': r2_score(sigma, model(gamma_dot, *popt))
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

def fit():
    try:
        data = request.get_json()
        gamma_dot = np.array(data['shear_rate'], dtype=float)
        sigma = np.array(data['shear_stress'], dtype=float)

        models = [
            fit_newtonian(gamma_dot, sigma),
            fit_power_law(gamma_dot, sigma),
            fit_herschel_bulkley(gamma_dot, sigma),
            fit_casson(gamma_dot, sigma),
            fit_bingham(gamma_dot, sigma)
        ]

def select_best_model(models):
    valid = [m for m in models if m['r2'] is not None and not np.isnan(m['r2'])]

    if not valid:
        return {'model': 'None', 'r2': 0.0}

    # Rule 1: If all models R² > 0.99 → Newtonian
    if all(m['r2'] > 0.99 for m in valid):
        return next((m for m in valid if m['model'] == 'Newtonian'), max(valid, key=lambda m: m['r2']))

    if abs(r2_power - r2_hb) < 1e-4 and r2_power > max(v for k, v in r2s.items() if k not in ['Power Law', 'Herschel–Bulkley']):
        return next(m for m in models if m['model'] == 'Power Law')

    if abs(r2_bingham - r2_hb) < 1e-4 and r2_bingham > max(v for k, v in r2s.items() if k not in ['Bingham Plastic', 'Herschel–Bulkley']):
        return next(m for m in models if m['model'] == 'Bingham Plastic')


    # Default: best R²
    return max(valid, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rate'], dtype=float)
    sigma = np.array(data['shear_stress'], dtype=float)
    result = fit_all_models(gamma_dot, sigma)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


 





   

    

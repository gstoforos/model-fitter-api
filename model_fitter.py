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
        return {'model': 'Newtonian', 'sigma0': 0.0, 'k': popt[0], 'n': 1.0, 'mu': popt[0], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Newtonian fit failed: {e}")
        return {'model': 'Newtonian', 'sigma0': 1e-6, 'k': 1e-6, 'n': 1.0, 'mu': None, 'r2': 0.0}

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Power Law', 'sigma0': 0.0, 'k': popt[0], 'n': popt[1], 'mu': 1.0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Power Law fit failed: {e}")
        return {'model': 'Power Law', 'sigma0': 1e-6, 'k': 1e-6, 'n': 1.0, 'mu': None, 'r2': 0.0}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'mu': 1.0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Herschel–Bulkley fit failed: {e}")
        return {'model': 'Herschel–Bulkley', 'sigma0': 1e-6, 'k': 1e-6, 'n': 1.0, 'mu': None, 'r2': 0.0}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k):
        valid = (sigma0 >= 0) and np.all(k * gamma_dot >= 0)
        if not valid: raise ValueError("Invalid sqrt in Casson")
        return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot)) ** 2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'n': 1.0, 'mu': 1.0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Casson fit failed: {e}")
        return {'model': 'Casson', 'sigma0': 1e-6, 'k': 1e-6, 'n': 1.0, 'mu': None, 'r2': 0.0}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'k': popt[1], 'n': 1.0, 'mu': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}
    except Exception as e:
        logging.error(f"Bingham fit failed: {e}")
        return {'model': 'Bingham Plastic', 'sigma0': 1e-6, 'k': 1e-6, 'n': 1.0, 'mu': None, 'r2': 0.0}

def select_best_model(models):
    valid = [m for m in models if m['r2'] is not None and not np.isnan(m['r2'])]

    if not valid:
        return {'model': 'None', 'r2': 0.0}

    if all(m['r2'] > 0.99 for m in valid):
        return next((m for m in valid if m['model'] == 'Newtonian'), valid[0])

    pl = next((m for m in valid if m['model'] == 'Power Law'), None)
    hb = next((m for m in valid if m['model'] == 'Herschel–Bulkley'), None)
    bp = next((m for m in valid if m['model'] == 'Bingham Plastic'), None)

    if pl and hb and abs(pl['r2'] - hb['r2']) < 0.01 and pl['r2'] > 0.97:
        return pl
    if bp and hb and abs(bp['r2'] - hb['r2']) < 0.01 and bp['r2'] > 0.97:
        return bp

    return max(valid, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
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

        return jsonify({
            'best_model': select_best_model(models),
            'all_models': models
        })

    except Exception as e:
        logging.error(f"Request failed: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

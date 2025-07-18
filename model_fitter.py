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
        r2 = r2_score(sigma, model(gamma_dot, mu))
        return {
            'model': 'Newtonian',
            'sigma0': 0.0,
            'k': mu,
            'n': 1.0,
            'mu': mu,
            'r2': r2
        }
    except Exception as e:
        logging.error(f"Newtonian fit failed: {e}")
        return {
            'model': 'Newtonian',
            'sigma0': 0.000001,
            'k': 0.000001,
            'n': 0.000001,
            'mu': None,
            'r2': 0.0
        }

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Power Law',
        'sigma0': 0.0,
        'k': popt[0],
        'n': popt[1],
        'mu': 1.0,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Herschel–Bulkley',
        'sigma0': popt[0],
        'k': popt[1],
        'n': popt[2],
        'mu': 1.0,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Casson',
        'sigma0': popt[0],
        'k': popt[1],
        'n': 1.0,
        'mu': 1.0,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Bingham Plastic',
        'sigma0': popt[0],
        'k': popt[1],
        'n': 1.0,
        'mu': popt[1],
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_all_models(gamma_dot, sigma):
    newtonian = fit_newtonian(gamma_dot, sigma)
    power_law = fit_power_law(gamma_dot, sigma)
    hb = fit_herschel_bulkley(gamma_dot, sigma)
    casson = fit_casson(gamma_dot, sigma)
    bingham = fit_bingham(gamma_dot, sigma)

    models = [newtonian, power_law, hb, casson, bingham]
    valid_models = [m for m in models if m['r2'] is not None and not np.isnan(m['r2'])]

    if not valid_models:
        logging.error("All model fits failed.")
        return {'model': 'None', 'r2': 0.0}

    # Rule 1: All R² > 0.99 → Newtonian
    if all(m['r2'] > 0.99 for m in models):
        return newtonian

    # Rule 2: Power Law vs HB
    if (power_law['r2'] > 0.97 and
        abs(power_law['r2'] - hb['r2']) < 0.01 and
        power_law['r2'] >= max(m['r2'] for m in models if m['model'] != 'Herschel–Bulkley')):
        return power_law

    # Rule 3: Bingham vs HB
    if (bingham['r2'] > 0.97 and
        abs(bingham['r2'] - hb['r2']) < 0.01 and
        bingham['r2'] >= max(m['r2'] for m in models if m['model'] != 'Herschel–Bulkley')):
        return bingham

    # Rule 4: Fallback → best R²
    return max(valid_models, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rate'], dtype=float)
    sigma = np.array(data['shear_stress'], dtype=float)
    result = fit_all_models(gamma_dot, sigma)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

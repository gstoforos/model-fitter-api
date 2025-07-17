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
        return {'model': 'Newtonian', 'sigma0': 0.0, 'k': mu, 'n': 1.0, 'mu': mu, 'r2': r2}
    except Exception as e:
        logging.error(f"Newtonian fit failed: {e}")
        return {'model': 'Newtonian', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'mu': 1.0, 'r2': 0.0}

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        k, n = popt
        r2 = r2_score(sigma, model(gamma_dot, k, n))
        return {'model': 'Power Law', 'sigma0': 0.0, 'k': k, 'n': n, 'mu': 1.0, 'r2': r2}
    except Exception as e:
        logging.error(f"Power Law fit failed: {e}")
        return {'model': 'Power Law', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'mu': 1.0, 'r2': 0.0}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        sigma0, k, n = popt
        r2 = r2_score(sigma, model(gamma_dot, sigma0, k, n))
        return {'model': 'Herschel–Bulkley', 'sigma0': sigma0, 'k': k, 'n': n, 'mu': 1.0, 'r2': r2}
    except Exception as e:
        logging.error(f"Herschel–Bulkley fit failed: {e}")
        return {'model': 'Herschel–Bulkley', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'mu': 1.0, 'r2': 0.0}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        sigma0, k = popt
        r2 = r2_score(sigma, model(gamma_dot, sigma0, k))
        return {'model': 'Casson', 'sigma0': sigma0, 'k': k, 'n': 1.0, 'mu': 1.0, 'r2': r2}
    except Exception as e:
        logging.error(f"Casson fit failed: {e}")
        return {'model': 'Casson', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'mu': 1.0, 'r2': 0.0}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        sigma0, mu = popt
        r2 = r2_score(sigma, model(gamma_dot, sigma0, mu))
        return {'model': 'Bingham Plastic', 'sigma0': sigma0, 'k': mu, 'n': 1.0, 'mu': mu, 'r2': r2}
    except Exception as e:
        logging.error(f"Bingham Plastic fit failed: {e}")
        return {'model': 'Bingham Plastic', 'sigma0': 0.0, 'k': 1.0, 'n': 1.0, 'mu': 1.0, 'r2': 0.0}

@app.route('/fit', methods=['POST'])
def fit():
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

    return jsonify({"models": models})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

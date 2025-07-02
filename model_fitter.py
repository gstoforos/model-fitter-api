from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

# Model fit functions
def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    r2 = r2_score(sigma, model(gamma_dot, *popt))
    return {'model': 'Newtonian', 'mu': popt[0], 'r2': r2}

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    r2 = r2_score(sigma, model(gamma_dot, *popt))
    return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'r2': r2}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    r2 = r2_score(sigma, model(gamma_dot, *popt))
    return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'r2': r2}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    r2 = r2_score(sigma, model(gamma_dot, *popt))
    return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'r2': r2}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    r2 = r2_score(sigma, model(gamma_dot, *popt))
    return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'mu': popt[1], 'r2': r2}

def fit_all_models(gamma_dot, sigma):
    fits = [
        fit_newtonian(gamma_dot, sigma),
        fit_power_law(gamma_dot, sigma),
        fit_herschel_bulkley(gamma_dot, sigma),
        fit_casson(gamma_dot, sigma),
        fit_bingham(gamma_dot, sigma)
    ]
    return max(fits, key=lambda x: x['r2'])

# API endpoint
@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rate'], dtype=float)
    sigma = np.array(data['shear_stress'], dtype=float)
    result = fit_all_models(gamma_dot, sigma)
    return jsonify(result)

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

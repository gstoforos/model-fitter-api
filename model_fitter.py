from flask import Flask, request, jsonify 
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    return {'model': 'Newtonian', 'mu': popt[0], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'mu': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_all_models(gamma_dot, sigma):
    def safe_call(fit_func):
        try:
            result = fit_func(gamma_dot, sigma)
            # Ensure all expected keys exist, even if not used by the model
            result.setdefault('mu', 0.0)
            result.setdefault('k', 0.0)
            result.setdefault('n', 1.0)
            result.setdefault('sigma0', 0.0)
            result['r2'] = result.get('r2') if result.get('r2') is not None else 0.0
            return result
        except Exception as e:
            logging.error(f"{fit_func.__name__} failed: {e}")
            return {
                'model': fit_func.__name__.replace('fit_', '').replace('_', ' ').title(),
                'mu': 0.0,
                'k': 0.0,
                'n': 1.0,
                'sigma0': 0.0,
                'r2': 0.0
            }

    models = [
        safe_call(fit_newtonian),
        safe_call(fit_power_law),
        safe_call(fit_herschel_bulkley),
        safe_call(fit_casson),
        safe_call(fit_bingham)
    ]
    return max(models, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rate'], dtype=float)
    sigma = np.array(data['shear_stress'], dtype=float)
    result = fit_all_models(gamma_dot, sigma)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

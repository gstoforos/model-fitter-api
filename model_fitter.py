from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def fit_newtonian(gamma_dot, sigma):
    def model(g, mu): return mu * g
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma)
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Newtonian', 'k': popt[0], 'n': 1.0, 'tau0': 0, 'r2': r2}
    except:
        return {'model': 'Newtonian', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

def fit_power_law(gamma_dot, sigma):
    def model(g, k, n): return k * g ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'tau0': 0, 'r2': r2}
    except:
        return {'model': 'Power Law', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(g, tau0, k, n): return tau0 + k * g ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Herschel-Bulkley', 'k': popt[1], 'n': popt[2], 'tau0': popt[0], 'r2': r2}
    except:
        return {'model': 'Herschel-Bulkley', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

def fit_casson(gamma_dot, sigma):
    def model(g, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * g)) ** 2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Casson', 'k': popt[1], 'n': 1.0, 'tau0': popt[0], 'r2': r2}
    except:
        return {'model': 'Casson', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

def fit_bingham(gamma_dot, sigma):
    def model(g, tau0, mu): return tau0 + mu * g
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Bingham Plastic', 'k': popt[1], 'n': 1.0, 'tau0': popt[0], 'r2': r2}
    except:
        return {'model': 'Bingham Plastic', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        gamma_dot = np.array(data['shear_rates'], dtype=float)
        sigma = np.array(data['shear_stresses'], dtype=float)
        models = [
            fit_newtonian(gamma_dot, sigma),
            fit_power_law(gamma_dot, sigma),
            fit_herschel_bulkley(gamma_dot, sigma),
            fit_casson(gamma_dot, sigma),
            fit_bingham(gamma_dot, sigma)
        ]
        return jsonify({'all_models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

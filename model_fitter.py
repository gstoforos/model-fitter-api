from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

# Newtonian model: τ = μ * γ̇
def fit_newtonian(gamma_dot, sigma):
    def model(g, mu): return mu * g
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma)
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Newtonian', 'k': popt[0], 'n': 1.0, 'tau0': 0, 'r2': r2}
    except Exception:
        return {'model': 'Newtonian', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

# Power-law model: τ = K * γ̇ⁿ
def fit_power_law(gamma_dot, sigma):
    def model(g, k, n): return k * g ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'tau0': 0, 'r2': r2}
    except Exception:
        return {'model': 'Power Law', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

# Herschel-Bulkley: τ = τ₀ + K * γ̇ⁿ
def fit_herschel_bulkley(gamma_dot, sigma):
    def model(g, tau0, k, n): return tau0 + k * g ** n
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Herschel-Bulkley', 'k': popt[1], 'n': popt[2], 'tau0': popt[0], 'r2': r2}
    except Exception:
        return {'model': 'Herschel-Bulkley', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

# Casson: √τ = √τ₀ + √(K * γ̇)
def fit_casson(gamma_dot, sigma):
    def model(g, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * g)) ** 2
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Casson', 'k': popt[1], 'n': 1.0, 'tau0': popt[0], 'r2': r2}
    except Exception:
        return {'model': 'Casson', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

# Bingham: τ = τ₀ + μ * γ̇
def fit_bingham(gamma_dot, sigma):
    def model(g, tau0, mu): return tau0 + mu * g
    try:
        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        r2 = r2_score(sigma, model(gamma_dot, *popt))
        return {'model': 'Bingham Plastic', 'k': popt[1], 'n': 1.0, 'tau0': popt[0], 'r2': r2}
    except Exception:
        return {'model': 'Bingham Plastic', 'k': 0, 'n': 1.0, 'tau0': 0, 'r2': -1.0}

# Try all models and return the one with highest R²
def fit_all_models(gamma_dot, sigma):
    models = [
        fit_newtonian(gamma_dot, sigma),
        fit_power_law(gamma_dot, sigma),
        fit_herschel_bulkley(gamma_dot, sigma),
        fit_casson(gamma_dot, sigma),
        fit_bingham(gamma_dot, sigma)
    ]

    # Extract models by name
    model_dict = {m['model']: m for m in models}
    r2_all = [m['r2'] for m in models]

    # Rule 1: If all R² > 0.99 and Newtonian is included, choose Newtonian
    if all(r2 > 0.99 for r2 in r2_all) and model_dict['Newtonian']['r2'] > 0.99:
        best = model_dict['Newtonian']
    # Rule 2: If both Bingham and Herschel–Bulkley have R² > 0.99, choose Bingham
    elif model_dict['Bingham Plastic']['r2'] > 0.99 and model_dict['Herschel-Bulkley']['r2'] > 0.99:
        best = model_dict['Bingham Plastic']
    # Rule 3: If both Power Law and Herschel–Bulkley have R² > 0.99, choose Power Law
    elif model_dict['Power Law']['r2'] > 0.99 and model_dict['Herschel-Bulkley']['r2'] > 0.99:
        best = model_dict['Power Law']
    else:
        best = max(models, key=lambda m: m['r2'])

    return {'best_model': best, 'all_models': models}

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

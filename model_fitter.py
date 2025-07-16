ffrom flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def safe_fit(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None
    return wrapper

@safe_fit
def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    return {'model': 'Newtonian', 'k': popt[0], 'n': 1, 'sigma0': 0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}

@safe_fit
def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'sigma0': 0, 'r2': r2_score(sigma, model(gamma_dot, *popt))}

@safe_fit
def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

@safe_fit
def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'n': 1, 'r2': r2_score(sigma, model(gamma_dot, *popt))}

@safe_fit
def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'k': popt[1], 'n': 1, 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_all_models(gamma_dot, sigma):
    candidates = {}
    for fit_fn in [fit_newtonian, fit_power_law, fit_herschel_bulkley, fit_casson, fit_bingham]:
        result = fit_fn(gamma_dot, sigma)
        if result:
            candidates[result['model']] = result

    if not candidates:
        return {'model': 'Unknown', 'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

    r2s = {name: r['r2'] for name, r in candidates.items()}
    best = max(r2s, key=r2s.get)

    # Override rules
    all_r2s_above_99 = all(r >= 0.99 for r in r2s.values())
    if all_r2s_above_99 and "Newtonian" in r2s:
        best = "Newtonian"
    elif r2s.get("Bingham Plastic", 0) >= 0.99 and r2s.get("Herschel–Bulkley", 0) >= 0.99:
        best = "Bingham Plastic"
    elif r2s.get("Power Law", 0) >= 0.99 and r2s.get("Herschel–Bulkley", 0) >= 0.99:
        best = "Power Law"

    return candidates[best]

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json(force=True)
        gamma_dot = np.array(data['shear_rate'], dtype=float)
        sigma = np.array(data['shear_stress'], dtype=float)

        result = fit_all_models(gamma_dot, sigma)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


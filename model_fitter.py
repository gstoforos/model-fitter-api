from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def fit_newtonian(gamma_dot, sigma):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    return {
        'model': 'Newtonian',
        'sigma0': 0,
        'mu': popt[0],
        'k': None,
        'n': 1,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_power_law(gamma_dot, sigma):
    def model(gamma_dot, k, n): return k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Power Law',
        'sigma0': 0,
        'mu': None,
        'k': popt[0],
        'n': popt[1],
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Bingham Plastic',
        'sigma0': popt[0],
        'mu': popt[1],
        'k': None,
        'n': 1,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Herschel–Bulkley',
        'sigma0': popt[0],
        'mu': None,
        'k': popt[1],
        'n': popt[2],
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {
        'model': 'Casson',
        'sigma0': popt[0],
        'mu': None,
        'k': popt[1],
        'n': 0.5,
        'r2': r2_score(sigma, model(gamma_dot, *popt))
    }

def select_best_model(models):
    r2s = {m['model']: m['r2'] for m in models}
    ranked = sorted(models, key=lambda m: m['r2'], reverse=True)

    r2_newtonian = r2s.get('Newtonian', 0)
    r2_power = r2s.get('Power Law', 0)
    r2_bingham = r2s.get('Bingham Plastic', 0)
    r2_hb = r2s.get('Herschel–Bulkley', 0)
    r2_casson = r2s.get('Casson', 0)

    all_high = all(r2 > 0.99 for r2 in r2s.values())
    if all_high:
        return next(m for m in models if m['model'] == 'Newtonian')

    if abs(r2_power - r2_hb) < 1e-4 and r2_power > max(v for k, v in r2s.items() if k not in ['Power Law', 'Herschel–Bulkley']):
        return next(m for m in models if m['model'] == 'Power Law')

    if abs(r2_bingham - r2_hb) < 1e-4 and r2_bingham > max(v for k, v in r2s.items() if k not in ['Bingham Plastic', 'Herschel–Bulkley']):
        return next(m for m in models if m['model'] == 'Bingham Plastic')

    return ranked[0]

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        gamma_dot = np.array(data['shear_rates'], dtype=float)
        sigma = np.array(data['shear_stresses'], dtype=float)

        models = [
            fit_newtonian(gamma_dot, sigma),
            fit_power_law(gamma_dot, sigma),
            fit_bingham(gamma_dot, sigma),
            fit_herschel_bulkley(gamma_dot, sigma),
            fit_casson(gamma_dot, sigma),
        ]
        best_model = select_best_model(models)

        return jsonify({
            'best_model': best_model,
            'all_models': models
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

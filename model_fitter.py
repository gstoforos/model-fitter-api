from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def fit_newtonian(g, s):
    def model(g, mu): return mu * g
    popt, _ = curve_fit(model, g, s)
    return {'model': 'Newtonian', 'mu': popt[0], 'sigma0': 0, 'k': None, 'n': 1, 'r2': r2_score(s, model(g, *popt))}

def fit_power_law(g, s):
    def model(g, k, n): return k * g**n
    popt, _ = curve_fit(model, g, s, bounds=(0, np.inf))
    return {'model': 'Power Law', 'k': popt[0], 'n': popt[1], 'sigma0': 0, 'mu': None, 'r2': r2_score(s, model(g, *popt))}

def fit_bingham(g, s):
    def model(g, sigma0, mu): return sigma0 + mu * g
    popt, _ = curve_fit(model, g, s, bounds=(0, np.inf))
    return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'mu': popt[1], 'k': None, 'n': 1, 'r2': r2_score(s, model(g, *popt))}

def fit_herschel_bulkley(g, s):
    def model(g, sigma0, k, n): return sigma0 + k * g**n
    popt, _ = curve_fit(model, g, s, bounds=(0, np.inf))
    return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'mu': None, 'r2': r2_score(s, model(g, *popt))}

def fit_casson(g, s):
    def model(g, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * g))**2
    popt, _ = curve_fit(model, g, s, bounds=(0, np.inf))
    return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'n': 0.5, 'mu': None, 'r2': r2_score(s, model(g, *popt))}

def select_best_model(models):
    r2s = {m['model']: m['r2'] for m in models}
    r_newtonian = r2s['Newtonian']
    r_hb = r2s['Herschel–Bulkley']
    r_power = r2s['Power Law']
    r_bingham = r2s['Bingham Plastic']
    r_casson = r2s['Casson']

    if all(r > 0.99 for r in r2s.values()):
        return next(m for m in models if m['model'] == 'Newtonian')
    if abs(r_power - r_hb) < 1e-4 and r_power > max(r_newtonian, r_bingham, r_casson):
        return next(m for m in models if m['model'] == 'Power Law')
    if abs(r_bingham - r_hb) < 1e-4 and r_bingham > max(r_newtonian, r_power, r_casson):
        return next(m for m in models if m['model'] == 'Bingham Plastic')
    return max(models, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        g = np.array(data['shear_rate'], dtype=float)
        s = np.array(data['shear_stress'], dtype=float)
        models = [
            fit_newtonian(g, s),
            fit_power_law(g, s),
            fit_bingham(g, s),
            fit_herschel_bulkley(g, s),
            fit_casson(g, s),
        ]
        return jsonify({
            'all_models': models,
            'best_model': select_best_model(models)
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

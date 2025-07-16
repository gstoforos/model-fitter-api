from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
import os

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json(force=True)
        gamma = np.array(data['shear_rates'], dtype=np.float64)
        tau = np.array(data['shear_stresses'], dtype=np.float64)

        models = {}

        # 1. Newtonian
        def model_newtonian(g, mu): return mu * g
        try:
            popt, _ = curve_fit(model_newtonian, gamma, tau)
            pred = model_newtonian(gamma, *popt)
            models['Newtonian'] = {
                'model': 'Newtonian',
                'sigma0': 0.0,
                'k': float(popt[0]),
                'n': 1.0,
                'r2': r_squared(tau, pred)
            }
        except:
            models['Newtonian'] = {'model': 'Newtonian', 'sigma0': 0.0, 'k': 0.0, 'n': 1.0, 'r2': 0.0}

        # 2. Power Law
        def model_power(g, k, n): return k * g ** n
        try:
            popt, _ = curve_fit(model_power, gamma, tau, bounds=(0, np.inf))
            pred = model_power(gamma, *popt)
            models['Power-Law'] = {
                'model': 'Power-Law',
                'sigma0': 0.0,
                'k': float(popt[0]),
                'n': float(popt[1]),
                'r2': r_squared(tau, pred)
            }
        except:
            models['Power-Law'] = {'model': 'Power-Law', 'sigma0': 0.0, 'k': 0.0, 'n': 1.0, 'r2': 0.0}

        # 3. Herschel–Bulkley
        def model_hb(g, tau0, k, n): return tau0 + k * g ** n
        try:
            popt, _ = curve_fit(model_hb, gamma, tau, bounds=(0, np.inf))
            pred = model_hb(gamma, *popt)
            models['Herschel-Bulkley'] = {
                'model': 'Herschel-Bulkley',
                'sigma0': float(popt[0]),
                'k': float(popt[1]),
                'n': float(popt[2]),
                'r2': r_squared(tau, pred)
            }
        except:
            models['Herschel-Bulkley'] = {'model': 'Herschel-Bulkley', 'sigma0': 0.0, 'k': 0.0, 'n': 1.0, 'r2': 0.0}

        # 4. Bingham
        def model_bingham(g, tau0, mu): return tau0 + mu * g
        try:
            popt, _ = curve_fit(model_bingham, gamma, tau)
            pred = model_bingham(gamma, *popt)
            models['Bingham'] = {
                'model': 'Bingham',
                'sigma0': float(popt[0]),
                'k': float(popt[1]),
                'n': 1.0,
                'r2': r_squared(tau, pred)
            }
        except:
            models['Bingham'] = {'model': 'Bingham', 'sigma0': 0.0, 'k': 0.0, 'n': 1.0, 'r2': 0.0}

        # 5. Casson
        def model_casson(g, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * g)) ** 2
        try:
            popt, _ = curve_fit(model_casson, gamma, tau, bounds=(0, np.inf))
            pred = model_casson(gamma, *popt)
            models['Casson'] = {
                'model': 'Casson',
                'sigma0': float(popt[0]),
                'k': float(popt[1]),
                'n': 0.5,
                'r2': r_squared(tau, pred)
            }
        except:
            models['Casson'] = {'model': 'Casson', 'sigma0': 0.0, 'k': 0.0, 'n': 0.5, 'r2': 0.0}

        # Best-fit logic
        best = max(models.values(), key=lambda m: m['r2'])

        all_r2 = [m['r2'] for m in models.values()]
        all_r2_high = all(r2 >= 0.99 for r2 in all_r2)

        if all_r2_high and 'Newtonian' in models:
            best = models['Newtonian']
        elif models['Bingham']['r2'] >= 0.99 and models['Herschel-Bulkley']['r2'] >= 0.99:
            best = models['Bingham']
        elif models['Power-Law']['r2'] >= 0.99 and models['Herschel-Bulkley']['r2'] >= 0.99:
            best = models['Power-Law']

        return jsonify(best)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

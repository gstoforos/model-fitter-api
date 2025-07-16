from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json()
        gamma = np.array(data['shear_rates'])
        tau = np.array(data['shear_stresses'])

        models = {}

        # 1. Newtonian
        def model_newtonian(g, mu): return mu * g
        try:
            popt_new, _ = curve_fit(model_newtonian, gamma, tau)
            pred = model_newtonian(gamma, *popt_new)
            models['Newtonian'] = {
                'model': 'Newtonian',
                'tau0': 0,
                'k': float(popt_new[0]),
                'n': 1.0,
                'r2': r_squared(tau, pred)
            }
        except:
            pass

        # 2. Power Law
        def model_power(g, k, n): return k * g ** n
        try:
            popt_pow, _ = curve_fit(model_power, gamma, tau, bounds=(0, np.inf))
            pred = model_power(gamma, *popt_pow)
            models['Power-Law'] = {
                'model': 'Power-Law',
                'tau0': 0,
                'k': float(popt_pow[0]),
                'n': float(popt_pow[1]),
                'r2': r_squared(tau, pred)
            }
        except:
            pass

        # 3. Herschel-Bulkley
        def model_hb(g, tau0, k, n): return tau0 + k * g ** n
        try:
            popt_hb, _ = curve_fit(model_hb, gamma, tau, bounds=(0, np.inf))
            pred = model_hb(gamma, *popt_hb)
            models['Herschel-Bulkley'] = {
                'model': 'Herschel-Bulkley',
                'tau0': float(popt_hb[0]),
                'k': float(popt_hb[1]),
                'n': float(popt_hb[2]),
                'r2': r_squared(tau, pred)
            }
        except:
            pass

        # 4. Bingham
        def model_bingham(g, tau0, mu): return tau0 + mu * g
        try:
            popt_b, _ = curve_fit(model_bingham, gamma, tau, bounds=(0, np.inf))
            pred = model_bingham(gamma, *popt_b)
            models['Bingham'] = {
                'model': 'Bingham',
                'tau0': float(popt_b[0]),
                'k': float(popt_b[1]),
                'n': 1.0,
                'r2': r_squared(tau, pred)
            }
        except:
            pass

        # 5. Casson
        def model_casson(g, tau0, k): return (tau0**0.5 + (k * g) ** 0.5) ** 2
        try:
            popt_cas, _ = curve_fit(model_casson, gamma, tau, bounds=(0, np.inf))
            pred = model_casson(gamma, *popt_cas)
            models['Casson'] = {
                'model': 'Casson',
                'tau0': float(popt_cas[0]),
                'k': float(popt_cas[1]),
                'n': 0.5,
                'r2': r_squared(tau, pred)
            }
        except:
            pass

        # Best-fit model selection
        all_r2 = [m['r2'] for m in models.values()]
        all_r2_high = all(r2 >= 0.99 for r2 in all_r2)

        if all_r2_high and 'Newtonian' in models:
            best = models['Newtonian']
        elif 'Bingham' in models and 'Herschel-Bulkley' in models:
            if models['Bingham']['r2'] >= 0.99 and models['Herschel-Bulkley']['r2'] >= 0.99:
                best = models['Bingham']
            else:
                best = max(models.values(), key=lambda m: m['r2'])
        elif 'Power-Law' in models and 'Herschel-Bulkley' in models:
            if models['Power-Law']['r2'] >= 0.99 and models['Herschel-Bulkley']['r2'] >= 0.99:
                best = models['Power-Law']
            else:
                best = max(models.values(), key=lambda m: m['r2'])
        else:
            best = max(models.values(), key=lambda m: m['r2'])

        return jsonify({
            'best_model': best,
            'all_models': models
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

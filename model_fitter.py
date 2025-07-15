from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit

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

        # Newtonian: τ = μ * γ̇
        def newtonian(g, mu): return mu * g
        try:
            popt, _ = curve_fit(newtonian, gamma, tau)
            y_pred = newtonian(gamma, *popt)
            models['Newtonian'] = {
                'k': popt[0], 'n': 1, 'sigma0': 0,
                'r2': r_squared(tau, y_pred)
            }
        except Exception:
            models['Newtonian'] = {'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Power-Law: τ = K * γ̇^n
        def power_law(g, K, n): return K * g**n
        try:
            popt, _ = curve_fit(power_law, gamma, tau, bounds=(0, np.inf))
            y_pred = power_law(gamma, *popt)
            models['Power-Law'] = {
                'k': popt[0], 'n': popt[1], 'sigma0': 0,
                'r2': r_squared(tau, y_pred)
            }
        except Exception:
            models['Power-Law'] = {'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Herschel–Bulkley: τ = τ₀ + K * γ̇^n
        def herschel_bulkley(g, tau0, K, n): return tau0 + K * g**n
        try:
            popt, _ = curve_fit(herschel_bulkley, gamma, tau, bounds=(0, np.inf))
            y_pred = herschel_bulkley(gamma, *popt)
            models['Herschel–Bulkley'] = {
                'sigma0': popt[0], 'k': popt[1], 'n': popt[2],
                'r2': r_squared(tau, y_pred)
            }
        except Exception:
            models['Herschel–Bulkley'] = {'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Bingham: τ = τ₀ + μ * γ̇
        def bingham(g, tau0, mu): return tau0 + mu * g
        try:
            popt, _ = curve_fit(bingham, gamma, tau)
            y_pred = bingham(gamma, *popt)
            models['Bingham Plastic'] = {
                'sigma0': popt[0], 'k': popt[1], 'n': 1,
                'r2': r_squared(tau, y_pred)
            }
        except Exception:
            models['Bingham Plastic'] = {'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Casson: sqrt(τ) = sqrt(τ₀) + sqrt(K*γ̇)
        def casson(g, tau0, K): return (np.sqrt(tau0) + np.sqrt(K * g))**2
        try:
            popt, _ = curve_fit(casson, gamma, tau, bounds=(0, np.inf))
            y_pred = casson(gamma, *popt)
            models['Casson'] = {
                'sigma0': popt[0], 'k': popt[1], 'n': 1,
                'r2': r_squared(tau, y_pred)
            }
        except Exception:
            models['Casson'] = {'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Choose best model
        r2_values = {m: v['r2'] for m, v in models.items()}
        best_model = max(r2_values, key=r2_values.get)

        # Override logic
        all_r2_1 = all(round(v['r2'], 8) == 1 for v in models.values())
        if all_r2_1 and 'Newtonian' in models:
            best_model = 'Newtonian'
        elif models['Bingham Plastic']['r2'] >= 0.99 and models['Herschel–Bulkley']['r2'] >= 0.99:
            best_model = 'Bingham Plastic'
        elif models['Power-Law']['r2'] >= 0.99 and models['Herschel–Bulkley']['r2'] >= 0.99:
            best_model = 'Power-Law'

        response = models[best_model].copy()
        response['model'] = best_model
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json(force=True)
        gamma = np.array(data['shear_rate'], dtype=np.float64)
        tau = np.array(data['shear_stress'], dtype=np.float64)

        models = {}

        # Newtonian: τ = μ * γ̇
        def newtonian(g, mu): return mu * g
        try:
            popt, _ = curve_fit(newtonian, gamma, tau)
            y_pred = newtonian(gamma, *popt)
            models['Newtonian'] = {
                'model': 'Newtonian',
                'k': float(popt[0]), 'n': 1, 'sigma0': 0,
                'r2': r_squared(tau, y_pred)
            }
        except:
            models['Newtonian'] = {'model': 'Newtonian', 'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Power Law: τ = K * γ̇^n
        def power_law(g, k, n): return k * g ** n
        try:
            popt, _ = curve_fit(power_law, gamma, tau, bounds=(0, np.inf))
            y_pred = power_law(gamma, *popt)
            models['Power-Law'] = {
                'model': 'Power-Law',
                'k': float(popt[0]), 'n': float(popt[1]), 'sigma0': 0,
                'r2': r_squared(tau, y_pred)
            }
        except:
            models['Power-Law'] = {'model': 'Power-Law', 'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Herschel–Bulkley: τ = τ₀ + K * γ̇^n
        def herschel_bulkley(g, tau0, k, n): return tau0 + k * g ** n
        try:
            popt, _ = curve_fit(herschel_bulkley, gamma, tau, bounds=(0, np.inf))
            y_pred = herschel_bulkley(gamma, *popt)
            models['Herschel–Bulkley'] = {
                'model': 'Herschel–Bulkley',
                'sigma0': float(popt[0]), 'k': float(popt[1]), 'n': float(popt[2]),
                'r2': r_squared(tau, y_pred)
            }
        except:
            models['Herschel–Bulkley'] = {'model': 'Herschel–Bulkley', 'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Bingham: τ = τ₀ + μ * γ̇
        def bingham(g, tau0, mu): return tau0 + mu * g
        try:
            popt, _ = curve_fit(bingham, gamma, tau, bounds=(0, np.inf))
            y_pred = bingham(gamma, *popt)
            models['Bingham Plastic'] = {
                'model': 'Bingham Plastic',
                'sigma0': float(popt[0]), 'k': float(popt[1]), 'n': 1,
                'r2': r_squared(tau, y_pred)
            }
        except:
            models['Bingham Plastic'] = {'model': 'Bingham Plastic', 'k': 0, 'n': 1, 'sigma0': 0, 'r2': 0}

        # Casson: τ = (√τ₀ + √(K * γ̇))²
        def casson(g, tau0, k): return (np.sqrt(tau0) + np.sqrt(k * g)) ** 2
        try:
            popt, _ = curve_fit(casson, gamma, tau, bounds=(0, np.inf))
            y_pred = casson(gamma, *popt)
            models['Casson'] = {
                'model': 'Casson',
                'sigma0': float(popt[0]), 'k': float(popt[1]), 'n': 0.5,
                'r2': r_squared(tau, y_pred)
            }
        except:
            models['Casson'] = {'model': 'Casson', 'k': 0, 'n': 0.5, 'sigma0': 0, 'r2': 0}

        # Return all models (let frontend or app decide best fit)
        return jsonify(models)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

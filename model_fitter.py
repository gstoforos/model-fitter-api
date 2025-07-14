import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define models
def newtonian(g, mu):
    return mu * g

def power_law(g, k, n):
    return k * g**n

def herschel_bulkley(g, tau0, k, n):
    return tau0 + k * g**n

def bingham(g, tau0, mu):
    return tau0 + mu * g

def casson(g, tau0, k):
    return (np.sqrt(tau0) + np.sqrt(k * g))**2

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json()
        g_dot = np.array(data['shear_rate'], dtype=np.float64)
        tau = np.array(data['shear_stress'], dtype=np.float64)

        # Always patch the last point with +0.00001 to ensure API safety
        if len(tau) >= 2:
            tau[-1] += 0.00001

        results = {}

        # Try Newtonian
        try:
            popt, _ = curve_fit(newtonian, g_dot, tau, maxfev=10000)
            pred = newtonian(g_dot, *popt)
            results['Newtonian'] = {
                'model': 'Newtonian',
                'sigma0': 0,
                'k': popt[0],
                'n': 1,
                'r2': r2_score(tau, pred)
            }
        except:
            results['Newtonian'] = {'model': 'Newtonian', 'r2': 0, 'sigma0': 0, 'k': 0, 'n': 1}

        # Power Law
        try:
            popt, _ = curve_fit(power_law, g_dot, tau, maxfev=10000)
            pred = power_law(g_dot, *popt)
            results['Power-Law'] = {
                'model': 'Power-Law',
                'sigma0': 0,
                'k': popt[0],
                'n': popt[1],
                'r2': r2_score(tau, pred)
            }
        except:
            results['Power-Law'] = {'model': 'Power-Law', 'r2': 0, 'sigma0': 0, 'k': 0, 'n': 1}

        # Herschel–Bulkley
        try:
            popt, _ = curve_fit(herschel_bulkley, g_dot, tau, maxfev=10000)
            pred = herschel_bulkley(g_dot, *popt)
            results['Herschel–Bulkley'] = {
                'model': 'Herschel–Bulkley',
                'sigma0': popt[0],
                'k': popt[1],
                'n': popt[2],
                'r2': r2_score(tau, pred)
            }
        except:
            results['Herschel–Bulkley'] = {'model': 'Herschel–Bulkley', 'r2': 0, 'sigma0': 0, 'k': 0, 'n': 1}

        # Bingham
        try:
            popt, _ = curve_fit(bingham, g_dot, tau, maxfev=10000)
            pred = bingham(g_dot, *popt)
            results['Bingham Plastic'] = {
                'model': 'Bingham Plastic',
                'sigma0': popt[0],
                'k': popt[1],
                'n': 1,
                'r2': r2_score(tau, pred)
            }
        except:
            results['Bingham Plastic'] = {'model': 'Bingham Plastic', 'r2': 0, 'sigma0': 0, 'k': 0, 'n': 1}

        # Casson
        try:
            popt, _ = curve_fit(casson, g_dot, tau, maxfev=10000)
            pred = casson(g_dot, *popt)
            results['Casson'] = {
                'model': 'Casson',
                'sigma0': popt[0],
                'k': popt[1],
                'n': 1,
                'r2': r2_score(tau, pred)
            }
        except:
            results['Casson'] = {'model': 'Casson', 'r2': 0, 'sigma0': 0, 'k': 0, 'n': 1}

        # Model decision based on your rules
        newton = results['Newtonian']
        bingham = results['Bingham Plastic']
        power = results['Power-Law']
        hb = results['Herschel–Bulkley']

        best = None
        if all(m['r2'] >= 0.99 for m in [newton, power, bingham, hb]):
            best = newton
        elif bingham['r2'] >= 0.99 and hb['r2'] >= 0.99:
            best = bingham
        elif power['r2'] >= 0.99 and hb['r2'] >= 0.99:
            best = power
        else:
            best = max(results.values(), key=lambda x: x['r2'])

        return jsonify({
            'model': best['model'],
            'sigma0': best['sigma0'],
            'k': best['k'],
            'n': best['n'],
            'r2': best['r2']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

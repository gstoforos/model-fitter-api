from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def newtonian(gamma, k):
    return k * gamma

def power_law(gamma, k, n):
    return k * gamma**n

def herschel_bulkley(gamma, sigma0, k, n):
    return sigma0 + k * gamma**n

def casson(gamma, sigma0, k):
    return (np.sqrt(sigma0) + np.sqrt(k * gamma))**2

def bingham(gamma, sigma0, k):
    return sigma0 + k * gamma

def safe_fit(model_func, xdata, ydata, p0):
    try:
        popt, _ = curve_fit(model_func, xdata, ydata, p0=p0, maxfev=10000)
        predictions = model_func(xdata, *popt)
        r2 = r2_score(ydata, predictions)
        return popt, r2
    except:
        return None, 0

def calc_reynolds(flow_rate, diameter, density, k, n):
    # flow_rate: m³/s, diameter: m, density: kg/m³, k: Pa·sⁿ, n: unitless
    area = np.pi * (diameter/2)**2
    velocity = flow_rate / area
    re = (density * velocity**(2-n) * diameter**n) / k
    return re

@app.route('/fit', methods=['POST'])
def fit_models():
    data = request.get_json()
    gamma = np.array(data['shear_rate'])
    sigma = np.array(data['shear_stress'])
    flow_rate = float(data.get('flow_rate', 1))
    diameter = float(data.get('diameter', 1))
    density = float(data.get('density', 1))

    results = []

    # Newtonian
    popt, r2 = safe_fit(newtonian, gamma, sigma, [1.0])
    if popt is not None:
        k = popt[0]
        results.append({"model": "Newtonian", "sigma0": 0, "k": k, "n": 1.0, "r2": r2, "re": calc_reynolds(flow_rate, diameter, density, k, 1.0)})

    # Power Law
    popt, r2 = safe_fit(power_law, gamma, sigma, [1.0, 1.0])
    if popt is not None:
        k, n = popt
        results.append({"model": "Power Law", "sigma0": 0, "k": k, "n": n, "r2": r2, "re": calc_reynolds(flow_rate, diameter, density, k, n)})

    # Herschel–Bulkley
    popt, r2 = safe_fit(herschel_bulkley, gamma, sigma, [0.1, 1.0, 1.0])
    if popt is not None:
        sigma0, k, n = popt
        results.append({"model": "Herschel–Bulkley", "sigma0": sigma0, "k": k, "n": n, "r2": r2, "re": calc_reynolds(flow_rate, diameter, density, k, n)})

    # Casson
    popt, r2 = safe_fit(casson, gamma, sigma, [0.1, 1.0])
    if popt is not None:
        sigma0, k = popt
        results.append({"model": "Casson", "sigma0": sigma0, "k": k, "n": 1.0, "r2": r2, "re": calc_reynolds(flow_rate, diameter, density, k, 1.0)})

    # Bingham
    popt, r2 = safe_fit(bingham, gamma, sigma, [0.1, 1.0])
    if popt is not None:
        sigma0, k = popt
        results.append({"model": "Bingham Plastic", "sigma0": sigma0, "k": k, "n": 1.0, "r2": r2, "re": calc_reynolds(flow_rate, diameter, density, k, 1.0)})

    # Select best model
    best = max(results, key=lambda x: x['r2']) if results else {"model": "None", "sigma0": 0, "k": 0, "n": 1, "r2": 0, "re": 0}

    return jsonify({
        "best_model": best,
        "all_models": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

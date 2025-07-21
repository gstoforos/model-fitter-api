from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from math import pi

app = Flask(__name__)

def fit_newtonian(gamma_dot, sigma, Q=1, D=1, rho=1):
    def model(gamma_dot, mu): return mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma)
    mu = popt[0]
    r2 = r2_score(sigma, model(gamma_dot, *popt))

    idx = len(gamma_dot) // 2
    mu_app = sigma[idx] / gamma_dot[idx] if gamma_dot[idx] != 0 else mu

    # Reynolds calculation
    area = pi * (D / 2) ** 2
    v = Q / area if area != 0 else 0
    Re = (rho * v * D) / mu_app if mu_app != 0 else 0

    return {
        'model': 'Newtonian',
        'mu': mu,
        'mu_app': mu_app,
        'Re': Re,
        'r2': r2,
        'equation': f"σ = {mu:.4g}·γ̇"
    }

    # (previous code unchanged)

# (previous code unchanged)

# Power Law
try:
    def power(g, k, n): return k * g**n
    popt, _ = curve_fit(power, gamma, sigma, bounds=(0, [np.inf, np.inf]))
    k, n = popt
    sigma_fit = power(gamma, k, n)
    r2 = r2_score(sigma, sigma_fit)
    mu_app = k * np.mean(gamma) ** (n - 1)
    Re = None  # ← removed Reynolds calculation for Power Law
    models["Power Law"] = {
        "mu": None,
        "k": k,
        "n": n,
        "tau0": 0,
        "r2": r2,
        "mu_app": mu_app,
        "Re": Re,
        "equation": f"σ = {k:.3f} γ̇^{n:.3f}"
    }
except:
    models["Power Law"] = {
        "mu": None,
        "k": None,
        "n": None,
        "tau0": 0,
        "r2": 0,
        "mu_app": None,
        "Re": None,
        "equation": "N/A"
    }

# (rest of models unchanged)

    

def fit_herschel_bulkley(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k, n): return sigma0 + k * gamma_dot**n
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Herschel–Bulkley', 'sigma0': popt[0], 'k': popt[1], 'n': popt[2], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_casson(gamma_dot, sigma):
    def model(gamma_dot, sigma0, k): return (np.sqrt(sigma0) + np.sqrt(k * gamma_dot))**2
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Casson', 'sigma0': popt[0], 'k': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_bingham(gamma_dot, sigma):
    def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot
    popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
    return {'model': 'Bingham Plastic', 'sigma0': popt[0], 'mu': popt[1], 'r2': r2_score(sigma, model(gamma_dot, *popt))}

def fit_all_models(gamma_dot, sigma, Q=1, D=1, rho=1):
    models = [
        fit_newtonian(gamma_dot, sigma, Q, D, rho),
        fit_power_law(gamma_dot, sigma),
        fit_herschel_bulkley(gamma_dot, sigma),
        fit_casson(gamma_dot, sigma),
        fit_bingham(gamma_dot, sigma)
    ]
    return max(models, key=lambda m: m['r2'])

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data.get('shear_rate', []), dtype=float)
    sigma = np.array(data.get('shear_stress', []), dtype=float)
    Q = float(data.get("flow_rate", 1))
    D = float(data.get("pipe_diameter", 1))
    rho = float(data.get("density", 1))

    result = fit_all_models(gamma_dot, sigma, Q, D, rho)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

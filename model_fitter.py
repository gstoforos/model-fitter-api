from flask import Flask, request, jsonify
from scipy.optimize import curve_fit
import numpy as np

app = Flask(__name__)

@app.route('/fit', methods=['POST'])
def fit():
    data = request.get_json()
    gamma_dot = np.array(data['shear_rates'])
    sigma = np.array(data['shear_stresses'])

    flow_rate = float(data.get("flow_rate", 1))
    pipe_diameter = float(data.get("pipe_diameter", 1))
    density = float(data.get("density", 1))

    v = (4 * flow_rate) / (np.pi * pipe_diameter**2) if flow_rate and pipe_diameter else 1

    models = {}

    # Newtonian
    def newtonian(gamma_dot, mu):
        return mu * gamma_dot

    try:
        popt, _ = curve_fit(newtonian, gamma_dot, sigma, bounds=(0, np.inf))
        mu = popt[0]
        predicted = newtonian(gamma_dot, mu)
        r2 = 1 - np.sum((sigma - predicted)**2) / np.sum((sigma - np.mean(sigma))**2)
        Re_newtonian = (density * v * pipe_diameter) / mu if mu > 0 else 0
        models["Newtonian"] = {
            "mu": mu,
            "k": None,
            "n": None,
            "tau0": 0,
            "r2": r2,
            "equation": f"σ = {mu:.3f} γ̇",
            "mu_app": mu,
            "Re": Re_newtonian
        }
    except:
        models["Newtonian"] = {
            "mu": None, "k": None, "n": None, "tau0": None,
            "r2": 0, "equation": "N/A", "mu_app": None, "Re": None
        }

    # Power Law
    def power_law(gamma_dot, k, n):
        return k * gamma_dot**n

    try:
        popt, _ = curve_fit(power_law, gamma_dot, sigma, bounds=(0, [np.inf, np.inf]))
        k, n = popt
        predicted = power_law(gamma_dot, k, n)
        r2 = 1 - np.sum((sigma - predicted)**2) / np.sum((sigma - np.mean(sigma))**2)
        gamma_mean = np.mean(gamma_dot)
        mu_app = k * gamma_mean**(n - 1)
        if k > 0 and n > 0:
            Re_powerlaw = (8 * density * v**(2 - n) * pipe_diameter**n) / (k * ((3 * n + 1) / (4 * n)))
        else:
            Re_powerlaw = 0
        models["Power Law"] = {
            "mu": None,
            "k": k,
            "n": n,
            "tau0": 0,
            "r2": r2,
            "equation": f"σ = {k:.3f} γ̇^{n:.3f}",
            "mu_app": mu_app,
            "Re": Re_powerlaw
        }
    except:
        models["Power Law"] = {
            "mu": None, "k": None, "n": None, "tau0": None,
            "r2": 0, "equation": "N/A", "mu_app": None, "Re": None
        }

    # Herschel–Bulkley
    def herschel_bulkley(gamma_dot, tau0, k, n):
        return tau0 + k * gamma_dot**n

    try:
        popt, _ = curve_fit(herschel_bulkley, gamma_dot, sigma, bounds=(0, [np.inf, np.inf, np.inf]))
        tau0, k, n = popt
        predicted = herschel_bulkley(gamma_dot, tau0, k, n)
        r2 = 1 - np.sum((sigma - predicted)**2) / np.sum((sigma - np.mean(sigma))**2)
        models["Herschel-Bulkley"] = {
            "mu": None,
            "k": k,
            "n": n,
            "tau0": tau0,
            "r2": r2,
            "equation": f"σ = {tau0:.3f} + {k:.3f} γ̇^{n:.3f}",
            "mu_app": None,
            "Re": None
        }
    except:
        models["Herschel-Bulkley"] = {
            "mu": None, "k": None, "n": None, "tau0": None,
            "r2": 0, "equation": "N/A", "mu_app": None, "Re": None
        }

    # Casson
    def casson(gamma_dot, tau0, k):
        return (tau0**0.5 + (k * gamma_dot)**0.5)**2

    try:
        popt, _ = curve_fit(casson, gamma_dot, sigma, bounds=(0, [np.inf, np.inf]))
        tau0, k = popt
        predicted = casson(gamma_dot, tau0, k)
        r2 = 1 - np.sum((sigma - predicted)**2) / np.sum((sigma - np.mean(sigma))**2)
        models["Casson"] = {
            "mu": None,
            "k": k,
            "n": None,
            "tau0": tau0,
            "r2": r2,
            "equation": f"σ^0.5 = {tau0**0.5:.3f} + ({k:.3f} γ̇)^0.5",
            "mu_app": None,
            "Re": None
        }
    except:
        models["Casson"] = {
            "mu": None, "k": None, "n": None, "tau0": None,
            "r2": 0, "equation": "N/A", "mu_app": None, "Re": None
        }

    # Bingham
    def bingham(gamma_dot, tau0, mu):
        return tau0 + mu * gamma_dot

    try:
        popt, _ = curve_fit(bingham, gamma_dot, sigma, bounds=(0, [np.inf, np.inf]))
        tau0, mu = popt
        predicted = bingham(gamma_dot, tau0, mu)
        r2 = 1 - np.sum((sigma - predicted)**2) / np.sum((sigma - np.mean(sigma))**2)
        models["Bingham"] = {
            "mu": mu,
            "k": None,
            "n": None,
            "tau0": tau0,
            "r2": r2,
            "equation": f"σ = {tau0:.3f} + {mu:.3f} γ̇",
            "mu_app": None,
            "Re": None
        }
    except:
        models["Bingham"] = {
            "mu": None, "k": None, "n": None, "tau0": None,
            "r2": 0, "equation": "N/A", "mu_app": None, "Re": None
        }

    return jsonify({
        "models": models,
        "best_model": "Newtonian"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

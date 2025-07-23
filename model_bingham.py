import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bingham_model(gamma_dot, tau0, mu):
    return tau0 + mu * gamma_dot

def fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density):
    shear_rates = np.array(shear_rates)
    shear_stresses = np.array(shear_stresses)

    # Ensure positive values
    if np.any(shear_rates <= 0) or np.any(shear_stresses <= 0):
        raise ValueError("Shear rates and stresses must be positive.")

    # Initial guesses
    initial_tau0 = np.min(shear_stresses) * 0.5
    initial_mu = (shear_stresses[-1] - shear_stresses[0]) / (shear_rates[-1] - shear_rates[0])
    popt, _ = curve_fit(bingham_model, shear_rates, shear_stresses, p0=[initial_tau0, initial_mu])

    tau0, mu = popt
    predicted = bingham_model(shear_rates, tau0, mu)
    r2 = r2_score(shear_stresses, predicted)

    # Apparent viscosity at mean shear rate
    gamma_mean = np.mean(shear_rates)
    mu_app = tau0 / gamma_mean + mu

    # Reynolds number (optional)
    if flow_rate > 0 and diameter > 0 and density > 0:
        Q = flow_rate
        D = diameter
        rho = density
        Re = (4 * rho * Q) / (np.pi * D * mu)
    else:
        Re = None

    return {
        "model": "Bingham Plastic",
        "tau0": tau0,
        "mu": mu,
        "mu_app": mu_app,
        "r2": r2,
        "re": Re,
        "equation": "τ = τ₀ + μ·γ̇"
    }

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bingham_model(gamma_dot, tau0, mu):
    return tau0 + mu * gamma_dot

def fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density):
    shear_rates = np.array(shear_rates)
    shear_stresses = np.array(shear_stresses)

    # Initial guesses: tau0 = min stress, mu = slope
    initial_tau0 = np.min(shear_stresses)
    initial_mu = (np.max(shear_stresses) - initial_tau0) / (np.max(shear_rates) - np.min(shear_rates))
    try:
        popt, _ = curve_fit(bingham_model, shear_rates, shear_stresses, p0=[initial_tau0, initial_mu])
        tau0, mu = popt
        predicted = bingham_model(shear_rates, tau0, mu)
        r2 = r2_score(shear_stresses, predicted)
    except Exception:
        tau0 = 0.0
        mu = 1.0
        r2 = 0.0
        predicted = bingham_model(shear_rates, tau0, mu)

    # Apparent viscosity at mean shear rate
    mean_gamma_dot = np.mean(shear_rates)
    tau = bingham_model(mean_gamma_dot, tau0, mu)
    mu_app = tau / mean_gamma_dot if mean_gamma_dot != 0 else 0.0

    # Reynolds number
    if flow_rate > 0 and diameter > 0 and density > 0:
        Q = flow_rate
        D = diameter
        rho = density
        try:
            Re = (4 * rho * Q) / (np.pi * D * mu)
        except Exception:
            Re = 0.0
    else:
        Re = 0.0

    # Clean NaN values
    if np.isnan(mu): mu = 1.0
    if np.isnan(tau0): tau0 = 0.0
    if np.isnan(mu_app): mu_app = 0.0
    if np.isnan(r2): r2 = 0.0
    if Re is None or np.isnan(Re): Re = 0.0

    return {
        "model": "Bingham Plastic",
        "tau0": tau0,
        "mu": mu,
        "mu_app": mu_app,
        "r2": r2,
        "re": Re,
        "equation": "τ = τ₀ + μ·γ̇"
    }

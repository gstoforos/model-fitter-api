import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bingham_model(gamma_dot, tau0, mu):
    return tau0 + mu * gamma_dot

def fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density):
    shear_rates = np.array(shear_rates, dtype=float)
    shear_stresses = np.array(shear_stresses, dtype=float)

    try:
        # Initial guess for fitting
        initial_tau0 = np.min(shear_stresses)
        initial_mu = (np.max(shear_stresses) - initial_tau0) / max(np.max(shear_rates) - np.min(shear_rates), 1e-6)

        # Fit the Bingham model
        popt, _ = curve_fit(bingham_model, shear_rates, shear_stresses, p0=[initial_tau0, initial_mu], maxfev=10000)
        tau0, mu = popt

        # Predicted stresses for R²
        predicted = bingham_model(shear_rates, tau0, mu)
        r2 = r2_score(shear_stresses, predicted)

        # Apparent viscosity
        mean_gamma_dot = np.mean(shear_rates)
        tau = bingham_model(mean_gamma_dot, tau0, mu)
        mu_app = tau / mean_gamma_dot if mean_gamma_dot != 0 else 1.0

        # Reynolds number for Bingham flow
        if flow_rate > 0 and diameter > 0 and density > 0:
            re = (4 * density * flow_rate) / (np.pi * diameter * mu)
        else:
            re = 0.0

    except Exception as e:
        print("Exception in Bingham model:", str(e))
        tau0, mu, mu_app, r2, re = 0.0, 1.0, 1.0, 0.0, 0.0

    # Final safety cleanup
    tau0 = float(np.nan_to_num(tau0, nan=0.0))
    mu = float(np.nan_to_num(mu, nan=1.0))
    mu_app = float(np.nan_to_num(mu_app, nan=1.0))
    r2 = float(np.nan_to_num(r2, nan=0.0))
    re = float(np.nan_to_num(re, nan=0.0))

    return {
        "model": "Bingham Plastic",
        "tau0": round(tau0, 6),
        "mu": round(mu, 6),
        "mu_app": round(mu_app, 6),
        "r2": round(r2, 6),
        "re": round(re, 2),
        "equation": "τ = τ₀ + μ·γ̇"
    }

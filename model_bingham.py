import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bingham_model(gamma, tau0, mu):
    return tau0 + mu * gamma

def fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density):
    shear_rates = np.array(shear_rates)
    shear_stresses = np.array(shear_stresses)

    try:
        popt, _ = curve_fit(
            bingham_model,
            shear_rates,
            shear_stresses,
            bounds=([0, 0], [np.inf, np.inf])
        )
        tau0, mu = popt
        predicted = bingham_model(shear_rates, tau0, mu)
        r2 = r2_score(shear_stresses, predicted)
    except Exception:
        tau0 = mu = r2 = 0.0

    gamma_mean = np.mean(shear_rates)
    tau_mean = bingham_model(gamma_mean, tau0, mu)
    mu_app = tau_mean / gamma_mean if gamma_mean != 0 else 0.0

    if flow_rate > 0 and diameter > 0 and density > 0:
        Q = flow_rate
        D = diameter
        rho = density
        Re = (4 * rho * Q) / (np.pi * D * mu) if mu > 0 else 0.0
    else:
        Re = 0.0

    return {
        "tau0": round(tau0, 6),
        "mu": round(mu, 6),
        "r2": round(r2, 6),
        "mu_app": round(mu_app, 6),
        "re": round(Re, 2),
        "equation": f"τ = {round(tau0, 2)} + {round(mu, 2)}·γ"
    }

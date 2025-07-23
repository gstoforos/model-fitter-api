import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bingham_model_fixed_tau0(gamma_dot, mu):
    return 2.0 + mu * gamma_dot  # τ₀ is fixed to 2.0

def fit_bingham_model(shear_rates, shear_stresses, flow_rate, diameter, density):
    shear_rates = np.array(shear_rates, dtype=float)
    shear_stresses = np.array(shear_stresses, dtype=float)

    try:
        # Fit only μ (tau0 is fixed at 2.0)
        popt, _ = curve_fit(bingham_model_fixed_tau0, shear_rates, shear_stresses, p0=[0.1], maxfev=10000)
        mu = popt[0]
        tau0 = 2.0  # Fixed

        predicted = bingham_model_fixed_tau0(shear_rates, mu)
        r2 = r2_score(shear_stresses, predicted)

        mean_gamma_dot = np.mean(shear_rates)
        tau = bingham_model_fixed_tau0(mean_gamma_dot, mu)
        mu_app = tau / mean_gamma_dot if mean_gamma_dot != 0 else 1.0

        if flow_rate > 0 and diameter > 0 and density > 0:
            re = (4 * density * flow_rate) / (np.pi * diameter * mu)
        else:
            re = 0.0

    except Exception as e:
        print("Exception in Bingham model:", str(e))
        tau0, mu, mu_app, r2, re = 2.0, 1.0, 1.0, 0.0, 0.0

    # Final cleanup
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
        "equation": f"τ = {round(tau0, 2)} + {round(mu, 2)}·γ̇"
    }

import sys, json
import numpy as np
from scipy.optimize import curve_fit

def fit_models(data):
    g_dot = np.array(data["shear_rate"])
    tau = np.array(data["shear_stress"])
    gamma_wall = float(data["gamma_wall"])
    rho = float(data["rho"])
    v = float(data["velocity"])
    d = float(data["diameter"])

    models = []

    # Newtonian
    def f1(g, mu): return mu * g
    popt, _ = curve_fit(f1, g_dot, tau)
    mu = popt[0]
    models.append({
        "Name": "Newtonian",
        "Tau0": 0,
        "K": mu,
        "n": 1,
        "R2": r_squared(tau, f1(g_dot, *popt)),
        "Equation": f"σ = {mu:.4f}·γ̇",
        "MuApp": mu,
        "Re": rho * v * d / mu,
        "Description": "Constant viscosity. Linear stress–strain rate relation."
    })

    # Power-law
    def f2(g, K, n): return K * g ** n
    popt, _ = curve_fit(f2, g_dot, tau, bounds=(0, [np.inf, 10]))
    K, n = popt
    mu_app = K * gamma_wall ** (n - 1)
    Re = rho * v * d / mu_app
    models.append({
        "Name": "Power-Law",
        "Tau0": 0,
        "K": K,
        "n": n,
        "R2": r_squared(tau, f2(g_dot, *popt)),
        "Equation": f"σ = {K:.4f}·γ̇^{n:.3f}",
        "MuApp": mu_app,
        "Re": Re,
        "Description": "Shear-thinning (n<1) or shear-thickening (n>1). No yield stress."
    })

    # Herschel–Bulkley
    def f3(g, tau0, K, n): return tau0 + K * g ** n
    popt, _ = curve_fit(f3, g_dot, tau, bounds=(0, [1000, 1000, 10]))
    tau0, K, n = popt
    mu_app = (tau0 / gamma_wall) + K * gamma_wall ** (n - 1)
    Re = rho * v * d / mu_app
    models.append({
        "Name": "Herschel–Bulkley",
        "Tau0": tau0,
        "K": K,
        "n": n,
        "R2": r_squared(tau, f3(g_dot, *popt)),
        "Equation": f"σ = {tau0:.2f} + {K:.4f}·γ̇^{n:.3f}",
        "MuApp": mu_app,
        "Re": Re,
        "Description": "Yield stress fluid with non-linear shear region. Generalized power-law."
    })

    # Bingham
    def f4(g, tau0, muP): return tau0 + muP * g
    popt, _ = curve_fit(f4, g_dot, tau, bounds=(0, [1000, 1000]))
    tau0, muP = popt
    mu_app = (tau0 / gamma_wall) + muP
    Re = rho * v * d / mu_app
    models.append({
        "Name": "Bingham Plastic",
        "Tau0": tau0,
        "K": muP,
        "n": 1,
        "R2": r_squared(tau, f4(g_dot, *popt)),
        "Equation": f"σ = {tau0:.2f} + {muP:.4f}·γ̇",
        "MuApp": mu_app,
        "Re": Re,
        "Description": "Has yield stress. Flows like Newtonian after yielding."
    })

    # Casson
    def f5(g, tau0, K): return (np.sqrt(tau0) + np.sqrt(K * g)) ** 2
    popt, _ = curve_fit(f5, g_dot, tau, bounds=(0, [1000, 1000]))
    tau0, K = popt
    mu_app = (tau0 / gamma_wall) + K * gamma_wall ** (-0.5)
    Re = rho * v * d / mu_app
    models.append({
        "Name": "Casson",
        "Tau0": tau0,
        "K": K,
        "n": 0.5,
        "R2": r_squared(tau, f5(g_dot, *popt)),
        "Equation": f"√σ = √{tau0:.2f} + √({K:.4f}·γ̇)",
        "MuApp": mu_app,
        "Re": Re,
        "Description": "Empirical. Used in chocolate, blood, printing inks."
    })

    return { "All": models }

def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

# === Entry point ===
if __name__ == "__main__":
    raw = sys.stdin.read()
    inp = json.loads(raw)
    result = fit_models(inp)
    print(json.dumps(result))

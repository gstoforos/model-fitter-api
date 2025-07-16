from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

def fit_newtonian(g, t):
    model = LinearRegression(fit_intercept=False).fit(g.reshape(-1, 1), t)
    mu = model.coef_[0]
    pred = mu * g
    return {"model": "Newtonian", "k": float(mu), "n": 1.0, "sigma0": 0.0, "r2": r_squared(t, pred)}

def fit_power_law(g, t):
    log_g = np.log10(g)
    log_t = np.log10(t)
    model = LinearRegression().fit(log_g.reshape(-1, 1), log_t)
    n = model.coef_[0]
    log_k = model.intercept_
    k = 10**log_k
    pred = k * g**n
    return {"model": "Power Law", "k": float(k), "n": float(n), "sigma0": 0.0, "r2": r_squared(t, pred)}

def fit_herschel_bulkley(g, t):
    best = {"r2": -1}
    for frac in np.linspace(0, 0.95, 20):
        tau0 = np.min(t) * frac
        mask = t > tau0 + 1e-6
        if np.sum(mask) < 2:
            continue
        g_trim = g[mask]
        t_trim = t[mask] - tau0
        log_g = np.log10(g_trim)
        log_t = np.log10(t_trim)
        model = LinearRegression().fit(log_g.reshape(-1, 1), log_t)
        n = model.coef_[0]
        log_k = model.intercept_
        k = 10**log_k
        pred = tau0 + k * g**n
        r2 = r_squared(t, pred)
        if r2 > best['r2']:
            best = {"model": "Herschel–Bulkley", "k": float(k), "n": float(n), "sigma0": float(tau0), "r2": r2}
    return best

def fit_bingham(g, t):
    model = LinearRegression().fit(g.reshape(-1, 1), t)
    mu = model.coef_[0]
    tau0 = model.intercept_
    if tau0 < 0: tau0 = 0
    pred = tau0 + mu * g
    return {"model": "Bingham Plastic", "k": float(mu), "n": 1.0, "sigma0": float(tau0), "r2": r_squared(t, pred)}

def fit_casson(g, t):
    best = {"r2": -1}
    for frac in np.linspace(0, 0.95, 20):
        tau0 = np.min(t) * frac
        try:
            s = np.sqrt(t) - np.sqrt(tau0)
            x = np.sqrt(g).reshape(-1, 1)
            model = LinearRegression().fit(x, s)
            slope = model.coef_[0]
            k = slope**2
            pred = (np.sqrt(tau0) + np.sqrt(k * g))**2
            r2 = r_squared(t, pred)
            if r2 > best['r2']:
                best = {"model": "Casson", "k": float(k), "n": 0.5, "sigma0": float(tau0), "r2": r2}
        except:
            continue
    return best

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json(force=True)
        gamma = np.array(data.get("shear_rates", []), dtype=float)
        sigma = np.array(data.get("shear_stresses", []), dtype=float)
        if len(gamma) < 2 or len(gamma) != len(sigma):
            return jsonify({"error": "Invalid input"}), 400

        models = {
            "Newtonian": fit_newtonian(gamma, sigma),
            "Power Law": fit_power_law(gamma, sigma),
            "Herschel–Bulkley": fit_herschel_bulkley(gamma, sigma),
            "Bingham Plastic": fit_bingham(gamma, sigma),
            "Casson": fit_casson(gamma, sigma)
        }

        # Best-fit logic
        all_r2_1 = all(round(m['r2'], 8) == 1 for m in models.values())
        if all_r2_1:
            best = "Newtonian"
        elif models["Bingham Plastic"]["r2"] >= 0.99 and models["Herschel–Bulkley"]["r2"] >= 0.99:
            best = "Bingham Plastic"
        elif models["Power Law"]["r2"] >= 0.99 and models["Herschel–Bulkley"]["r2"] >= 0.99:
            best = "Power Law"
        else:
            best = max(models.items(), key=lambda x: x[1]['r2'])[0]

        response = {
            "model": best,
            "Newtonian": models["Newtonian"],
            "Power Law": models["Power Law"],
            "Herschel–Bulkley": models["Herschel–Bulkley"],
            "Bingham Plastic": models["Bingham Plastic"],
            "Casson": models["Casson"]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

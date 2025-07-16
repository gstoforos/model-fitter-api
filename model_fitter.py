from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

app = Flask(__name__)

def safe_r2(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred)
    except:
        return 0.0

def fit_model(name, func, gamma, tau, bounds=None, param_names=[]):
    try:
        popt, _ = curve_fit(func, gamma, tau, bounds=bounds if bounds else (-np.inf, np.inf))
        pred = func(gamma, *popt)
        result = {'model': name, 'r2': safe_r2(tau, pred)}
        for i, pname in enumerate(param_names):
            result[pname] = float(popt[i])
        return result
    except:
        result = {'model': name, 'r2': 0.0}
        for pname in param_names:
            result[pname] = 0.0
        return result

@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        gamma = np.array(data['shear_rates'], dtype=float)
        tau = np.array(data['shear_stresses'], dtype=float)

        models = []

        models.append(fit_model("Newtonian", lambda g, mu: mu * g, gamma, tau, param_names=["k"]))
        models[-1]["n"] = 1.0
        models[-1]["tau0"] = 0.0

        models.append(fit_model("Power Law", lambda g, k, n: k * g**n, gamma, tau, bounds=(0, np.inf), param_names=["k", "n"]))
        models[-1]["tau0"] = 0.0

        models.append(fit_model("Herschel-Bulkley", lambda g, tau0, k, n: tau0 + k * g**n, gamma, tau, bounds=(0, np.inf), param_names=["tau0", "k", "n"]))

        models.append(fit_model("Casson", lambda g, tau0, k: (np.sqrt(tau0) + np.sqrt(k * g))**2, gamma, tau, bounds=(0, np.inf), param_names=["tau0", "k"]))
        models[-1]["n"] = 0.5

        models.append(fit_model("Bingham Plastic", lambda g, tau0, mu: tau0 + mu * g, gamma, tau, bounds=(0, np.inf), param_names=["tau0", "k"]))
        models[-1]["n"] = 1.0

        return jsonify({"all_models": models})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

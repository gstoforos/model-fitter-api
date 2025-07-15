from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

app = Flask(__name__)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

def model_newtonian(gamma, mu):
    return mu * gamma

def model_power_law(gamma, K, n):
    return K * gamma ** n

def model_bingham(gamma, tau0, mu_p):
    return tau0 + mu_p * gamma

def model_herschel_bulkley(gamma, tau0, K, n):
    return tau0 + K * gamma ** n

def model_casson(gamma, tau0, K):
    return (np.sqrt(tau0) + np.sqrt(K * gamma)) ** 2

@app.route('/fit', methods=['POST'])
def fit_models():
    try:
        data = request.get_json()
        gamma = np.array(data['shear_rate'], dtype=np.float64)
        tau = np.array(data['shear_stress'], dtype=np.float64)

        models = {}

        # 1. Newtonian
        reg = LinearRegression()
        reg.fit(gamma.reshape(-1, 1), tau)
        pred = reg.predict(gamma.reshape(-1, 1))
        models['Newtonian'] = {
            'model': 'Newtonian',
            'sigma0': 0,
            'k': float(reg.coef_[0]),
            'n': 1,
            'r2': r_squared(tau, pred)
        }

        # 2. Power Law
        try:
            log_gamma = np.log(gamma)
            log_tau = np.log(tau)
            reg_pw = LinearRegression().fit(log_gamma.reshape(-1, 1), log_tau)
            n_pw = float(reg_pw.coef_[0])
            K_pw = float(np.exp(reg_pw.intercept_))
            pred_pw = K_pw * gamma ** n_pw
            models['Power-Law'] = {
                'model': 'Power-Law',
                'sigma0': 0,
                'k': K_pw,
                'n': n_pw,
                'r2': r_squared(tau, pred_pw)
            }
        except:
            pass

        # 3. Bingham Plastic
        try:
            popt_b, _ = curve_fit(model_bingham, gamma, tau, bounds=(0, np.inf))
            pred_b = model_bingham(gamma, *popt_b)
            models['Bingham'] = {
                'model': 'Bingham Plastic',
                'sigma0': float(popt_b[0]),
                'k': float(popt_b[1]),
                'n': 1,
                'r2': r_squared(tau, pred_b)
            }
        except:
            pass

        # 4. Herschel–Bulkley
        try:
            popt_hb, _ = curve_fit(model_herschel_bulkley, gamma, tau, bounds=(0, np.inf))
            pred_hb = model_herschel_bulkley(gamma, *popt_hb)
            models['Herschel–Bulkley'] = {
                'model': 'Herschel–Bulkley',
                'sigma0': float(popt_hb[0]),
                'k': float(popt_hb[1]),
                'n': float(popt_hb[2]),
                'r2': r_squared(tau, pred_hb)
            }
        except:
            pass

        # 5. Casson
        try:
            popt_cas, _ = curve_fit(model_casson, gamma, tau, bounds=(0, np.inf))
            pred_cas = model_casson(gamma, *popt_cas)
            models['Casson'] = {
                'model': 'Casson',
                'sigma0': float(popt_cas[0]),
                'k': float(popt_cas[1]),
                'n': 0.5,
                'r2': r_squared(tau, pred_cas)
            }
        except:
            pass

        # Best-fit logic
        best = max(models.values(), key=lambda m: m['r2'])

        all_r2 = [m['r2'] for m in models.values()]
        all_r2_high = all(r2 >= 0.99 for r2 in all_r2)

        if all_r2_high and 'Newtonian' in models:
            best = models['Newtonian']
        elif 'Bingham' in models and 'Herschel–Bulkley' in models:
            if models['Bingham']['r2'] >= 0.99 and models['Herschel–Bulkley']['r2'] >= 0.99:
                best = models['Bingham']
        elif 'Power-Law' in models and 'Herschel–Bulkley' in models:
            if models['Power-Law']['r2'] >= 0.99 and models['Herschel–Bulkley']['r2'] >= 0.99:
                best = models['Power-Law']

        return jsonify(best)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/fit', methods=['POST'])
def fit_dummy():
    # Always return dummy safe values for all models
    dummy_model = {
        "mu": 1.0,
        "k": 1.0,
        "n": 1.0,
        "tau0": 1.0,
        "r2": 1.0,
        "equation": "σ = 1.0",
        "mu_app": 1.0,
        "Re": 1.0
    }

    return jsonify({
        "models": {
            "Newtonian": dummy_model,
            "Power Law": dummy_model,
            "Herschel-Bulkley": dummy_model,   # Replaced EN DASH with hyphen
            "Casson": dummy_model,
            "Bingham Plastic": dummy_model
        },
        "best_model": "Newtonian"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

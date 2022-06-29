from flask import Flask, render_template, request
import joblib
import numpy as np

ml_model = joblib.load('test_model')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_cancer():
    clump_thickness = int(request.form.get('clump_thickness'))
    cell_size = int(request.form.get('cell_size'))
    cell_shape = int(request.form.get('cell_shape'))
    marginal_adhesion = int(request.form.get('marginal_adhesion'))
    single_epithelial_cell_size = int(request.form.get('single_epithelial_cell_size'))
    bare_nuclei = float(request.form.get('bare_nuclei'))
    normal_nucleoli = int(request.form.get('normal_nucleoli'))
    bland_chromatin = int(request.form.get('bland_chromatin'))
    mitoses = int(request.form.get('mitoses'))

    # make prediction
    result = ml_model.predict(np.array([clump_thickness, cell_size, cell_shape, marginal_adhesion, single_epithelial_cell_size, bare_nuclei,normal_nucleoli, bland_chromatin, mitoses]).reshape(1, 9))
    
    if result[0] == 2:
        result = "Benign refers to a condition, tumor, or growth that is not cancerous"

    else:
        result = "Malignant cells grow in an uncontrolled way and can invade nearby tissues and spread to other parts of the body through the blood and lymph system."

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
mul_reg = open("gradient_boost.pkl", "rb")
ml_model = joblib.load(mul_reg)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('LIMIT_BAL'))
        try:
            LIMIT_BAL = float(request.form['LIMIT_BAL'])
            SEX = float(request.form['SEX'])
            EDUCATION = float(request.form['EDUCATION'])
            MARRIAGE = float(request.form['MARRIAGE'])
            AGE = float(request.form['AGE'])
            PAY_0 = float(request.form['PAY_0'])
            PAY_2 = float(request.form['PAY_2'])
            PAY_3 = float(request.form['PAY_3'])
            PAY_4 = float(request.form['PAY_4'])
            PAY_5 = float(request.form['PAY_5'])
            PAY_6 = float(request.form['PAY_6'])
            BILL_AMT1 = float(request.form['BILL_AMT1'])
            BILL_AMT2 = float(request.form['BILL_AMT2'])
            BILL_AMT3 = float(request.form['BILL_AMT3'])
            BILL_AMT4 = float(request.form['BILL_AMT4'])
            BILL_AMT5 = float(request.form['BILL_AMT5'])
            BILL_AMT6 = float(request.form['BILL_AMT6'])
            PAY_AMT1 = float(request.form['PAY_AMT1'])
            PAY_AMT2 = float(request.form['PAY_AMT2'])
            PAY_AMT3 = float(request.form['PAY_AMT3'])
            PAY_AMT4 = float(request.form['PAY_AMT4'])
            PAY_AMT5 = float(request.form['PAY_AMT5'])
            PAY_AMT6 = float(request.form['PAY_AMT6'])

            pred_args = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1,
                         BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4,
                         PAY_AMT5, PAY_AMT6]

            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            # mul_reg = open("multiple_regression_model.pkl", "rb")
            # ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction=model_prediction)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

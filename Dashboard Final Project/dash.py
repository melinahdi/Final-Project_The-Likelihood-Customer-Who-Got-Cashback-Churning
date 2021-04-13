from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

#halaman home
@app.route('/')
def home():
    return render_template('home.html')

#halaman dataset
@app.route('/database', methods=['POST', 'GET'])
def dataset():
    return render_template('dataset.html')

# #halaman visualisasi
@app.route('/visualize', methods=['POST', 'GET'])
def visual():
    return render_template('plot.html')

# #halaman input prediksi
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    return render_template('predict.html')

# #halaman hasil prediksi
@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form

        df_predict = pd.DataFrame({
            'Tenure':[input['Tenure']],
            'CityTier' : [input['City Tier']],
            'Gender' : [input['Gender']],
            'NumberOfDeviceRegistered' : [input['Number of Device Registered']],
            'PreferedOrderCat' : [input['Prefered Order Category']],
            'SatisfactionScore' : [input['Satisfaction Score']],
            'MaritalStatus' : [input['Marital Status']],
            'NumberOfAddress' : [input['Number Of Address']],
            'Complain' : [input['Complain']],
            'CouponUsed' : [input['Coupon Used']],
            'OrderCount' : [input['Order Count']],
            'DaySinceLastOrder' : [input['Day Since Last Order']],
            'CashbackAmount' : [input['Cashback Amount']],
        })

        #     df_predict=pd.DataFrame({
        #     'Tenure' : [0],
        #     'CityTier' : [1],
        #     'Gender' : ['Male'],
        #     'NumberOfDeviceRegistered' : [3],
        #     'PreferedOrderCat' : ['Mobile Phone'],
        #     'SatisfactionScore' : [2],
        #     'MaritalStatus': ['Divorced'],
        #     'NumberOfAddress' : [3],
        #     'Complain' : [1],
        #     'CouponUsed' : [2],
        #     'OrderCount': [2],
        #     'DaySinceLastOrder': [0],
        #     'CashbackAmount' : [123]
        # })

        prediksi = model.predict_proba(df_predict)[0][1]

        if prediksi > 0.5:
            quality = "Churn"
        else:
            quality = "Not Churn"

        return render_template('result.html',
            data=input, pred=quality)

if __name__ == '__main__':
    # model = joblib.load('model_joblib')

    filename = 'C:/Users/user/Desktop/Purwadhika/Final Project/Submit/Dashboard Final Project/ModelChurnFinal.sav'
    model = pickle.load(open(filename,'rb'))

    app.run(debug=True)
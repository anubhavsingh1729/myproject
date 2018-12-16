from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

app = Flask(__name__)
gbr = joblib.load('model.pkl')

@app.route("/")
def main():
    return render_template('index.html')

    # user_input = 48,138,344,643,646,1999
def input_to_one_hot(dt):
    enc_input = np.zeros(19)
    enc_input[0] = dt['bedrooms']
    #enc_input[1] = dt['bathrooms']
    enc_input[2] = dt['sqft_living']
    #enc_input[3] = dt['sqft_lot']
    enc_input[4] = dt['floors']
    #enc_input[5] = dt['grade']
    #enc_input[6] = dt['sqft_above']
    #enc_input[7] = dt['sqft_basement']
    enc_input[8] = dt['yr_built']
    #enc_input[9] = dt['yr_renovated']
    enc_input[10] = dt['zipcode']
    enc_input[11] = dt['latitude']
    enc_input[12] = dt['longitude']
    return enc_input

@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    bedrooms = result['bedrooms']
    yr_built = result['yr_built']
    sqft_living = result['sqft_living']
    floors = result['floors']
    zipcode = result['zipcode']
    latitude = result['latitude']
    longitude = result['longitude']

    user_input = {"bedrooms": bedrooms, "sqft_living": sqft_living, "floors": floors,"yr_built": yr_built, "zipcode": zipcode,"latitude":latitude,"longitude":longitude}
    a = input_to_one_hot(user_input)
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    return json.dumps({'price': price_pred})

if __name__  == "__main__":
    app.run(port=8080, debug=True)
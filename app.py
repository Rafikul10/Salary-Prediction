# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:12:53 2022

@author: RAFIKUL
"""

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app= Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    
    int_features =[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template("index.html", prediction_text="Employee salary should be $ {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
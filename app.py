import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import lightgbm

st.set_page_config(page_title="Seoul Bike Trip Duration Prediction App")

st.title('Seoul Bike Trip Duration Prediction')

fNames = pk.load(open("Feature_Names.sav", 'rb'))
scalarX = pk.load(open("scaler_x.sav", 'rb'))
scalary = pk.load(open("scaler_y.sav", 'rb'))
assert isinstance(scalarX, MinMaxScaler)
scalarX.clip = False
assert isinstance(scalary, MinMaxScaler)
scalary.clip = False

lgbm = pk.load(open("lgbm_model_final.sav", 'rb'))


st.write("------------")

l = ["Distance", "PLong", "DLong", "Haversine", "Pmonth", "Phour", "PDweek", "Dmonth", "Dhour", "DDweek", "Temp", "Wind", "Humid", "Solar", "GroundTemp", "Dust"]

st.write("Enter input values:")
dist = st.slider(l[0], 0, 10000, 10)

plong = st.slider(l[1], 37.35, 37.75, 37.5)
dlong = st.slider(l[2], 37.35, 37.75, 37.5)

hsine = st.slider(l[3], 0, 25, 10)

Pmon = st.slider(l[4], 0, 11, 5)
Phr = st.slider(l[5], 0, 23, 9)
Pweek = st.slider(l[6], 0, 6, 5)

Dmon = st.slider(l[7], 0, 11, 5)
Dhr = st.slider(l[8], 0, 23, 9)
Dweek = st.slider(l[9], 0, 6, 5)

t = st.slider(l[10], -20, 41, 17)
w = st.slider(l[11], 0, 8, 5)
h = st.slider(l[12], 10, 100, 45)
s = st.slider(l[13], 0.0, 3.6, 0.9)
g = st.slider(l[14], -15, 62, 17)
d = st.slider(l[15], 0, 300, 50)
# submit_val = st.form_submit_button("Predict Duration")

if st.button('Predict'):	
	fArray = np.array([[dist, plong, 5, dlong, 5, hsine, Pmon, 5, Phr, 5, Pweek, Dmon, 5, Dhr, 5, Dweek, t, w, h, s, g, d]])
	notReqCol = [2, 4, 7, 9, 12, 14]
	X_test = scalarX.transform(fArray)
	X_test = np.delete(X_test,notReqCol,1)

	y_pred = lgbm.predict(X_test).reshape(-1,1)
	y = scalary.inverse_transform(y_pred)

	st.write("Predicted duration: " + str(round(y[0][0])) + " mins")
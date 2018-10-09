import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


from flask import Flask,render_template,url_for,request

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():

	# pdwork = pd.read_csv('FinalDataSheetWork.csv')
	# # Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
	# mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
	# # Train the imputor on the df dataset
	# pdwork["parking"]=mean_imputer.fit_transform(pdwork[["parking"]]).ravel()
	# pdwork["bathrooms"]=mean_imputer.fit_transform(pdwork[["bathrooms"]]).ravel()
	# #labelencode
	# label_encoder = LabelEncoder()
	# pdwork['location'] = label_encoder.fit_transform(pdwork['location'])
	# pdwork['furnishingLevel'] = label_encoder.fit_transform(pdwork['furnishingLevel'])
	# y = pdwork['price']
	# x = pdwork.iloc[:,:-1]
	# X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

	# from sklearn.ensemble import RandomForestClassifier
	# model = RandomForestClassifier(warm_start = True,max_depth = 5,random_state=0).fit(X_train,y_train)
	# tpred = model.predict(X_test)
	# print ('rforest ',accuracy_score(y_test, tpred))

	##Alternative Use of Saved Model
	#modelkl = open("model64bit.joblib","rb")
	tpred = joblib.load("model64bit.joblib")

	if request.method == 'POST':
		location = request.form[location]
		furnishingLevel = request.form[furnishingLevel]
		bedrooms = request.form[bedrooms]
		bathrooms = request.form[bathrooms]
		parking = request.form[parking]

		dat = [location,furnishingLevel,bedrooms,parking,parking]

		data = np.asarray(dat)
		#data = label_encoder.fit_transform(data)

		# data['location'] = label_encoder.fit_transform(data['location'])
		# data['furnishingLevel'] = label_encoder.fit_transform(data['furnishingLevel'])

		# modData = ''
		my_prediction = tpred.predict([data])
	return render_template('results.html',prediction = my_prediction)



if __name__== '__main__':
	app.run(debug=True)
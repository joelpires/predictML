from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

	if request.method == 'POST':
		age = int(request.form['age'])
		if (age == '' or age<0 or age >99):
			age = 36
		workclass = request.form['workclass']
		if (workclass == ''):
			workclass = 'Private'

		fnlwgt = 189778

		education = request.form['education']
		education_num = 0
		if(education == "Preschool"):
			education_num = 0
		elif(education == "1st-4th"):
			education_num = 1
		elif(education == "5th-6th"):
			education_num = 2
		elif(education == "7th-8th"):
			education_num = 3
		elif(education == "9th"):
			education_num = 4
		elif(education == "10th"):
			education_num = 5
		elif(education == "11th"):
			education_num = 6
		elif(education == "12th"):
			education_num = 7
		elif(education == "High School Graduated"):
			education_num = 8
			education = 'HS-grad'
		elif(education == "Some College"):
			education_num = 9
			education = 'Some-college'
		elif(education == "Associate Acdm"):
			education_num = 10
			education = 'Assoc-acdm'
		elif(education == "Associate Voc"):
			education_num = 11
			education = 'Assoc-voc'
		elif(education == "Bachelors"):
			education_num = 12
		elif(education == "Professor at School"):
			education_num = 13
			education = 'Prof-school'
		elif(education == "Masters"):
			education_num = 14
		elif(education == "Doctorate"):
			education_num = 15
		else:
			education = 'HS-grad'
			education_num = 8

		marital_status = request.form['marital_status']
		if(marital_status not in ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']):
			marital_status = 'Married-civ-spouse'

		occupation = request.form['occupation']
		if(occupation not in ['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty', 'Other-service', 'Adm-clerical', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners', 'Sales', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']):
			occupation = 'Prof-specialty'

		relationship = 'Husband'

		race = request.form['race']
		if(race not in ['White', 'Black', 'Asian-Pac-Islander', 'Other','Amer-Indian-Eskimo']):
			race = 'White'

		sex = request.form['sex']
		if(sex not in ['Female', 'Male']):
			sex = 'Male'

		hours_per_week = int(request.form['hours_per_week'])
		if (hours_per_week == '' or hours_per_week<0 or hours_per_week >50):
			age = 40

		native_country = request.form['native_country']
		if(native_country not in ['United-States','Mexico', 'Greece', 'Vietnam', 'China',
       'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada',
       'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran',
       'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba',
       'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
       'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
       'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland',
       'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)']):
			native_country = 'United-States'

		capital_gain = 0
		capital_loss = 0


		classifier = request.form['classifier']

		if classifier == 'Decision Tree':
			classifier = 'DT'

		#importing our model
		model = joblib.load('./models/census/' + classifier +'_model.sav')

		#spliting the columns
		original_order = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
		categories = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
		numerical = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

		new_data = [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country]
		print(new_data)

		features = pd.DataFrame(columns=original_order).append(pd.Series(new_data, index=original_order), ignore_index=True)

		print(features)
		"""
		#import preprocessing pipeline
		pipeline = joblib.load('./models/census/preprocessing_pipeline.pkl')

		#preprocess the features of our new data
		features_transformed = pipeline.transform(features)
		columns = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names(['age']).tolist() + numerical + pipeline.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot'].get_feature_names(categories).tolist()
		features_transformed = pd.DataFrame(features_transformed.toarray(), columns=columns)

		if model == 'KNN' or model == 'SVC' or model == 'Naive Bayes':
		    prediction = model.predict(features_transformed.toarray()) #there's a problem with EnsembleVoter and Blend - fix TODO
		else:
		    prediction = model.predict(features_transformed)

		if int(prediction) == 0:
		       answer = 'Equal or Less than 50.000$'
		else:
		       answer = 'Greater than 50.000$'
		"""
	return render_template('results.html',prediction = prediction,name = answer)


if __name__ == '__main__':
	app.run(debug=True)

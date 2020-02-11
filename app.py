from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np

from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():

	if request.method == 'POST':
		age = request.form['age']
		if type(age) != int or (type(age) == int and ~(age > 0 or age < 99)):
			age = 36
		else:
			age = int(age)

		workclass = request.form['workclass']
		if (workclass not in ['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc',
       'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked']):
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

		hours_per_week = request.form['hours_per_week']
		if (type(hours_per_week) == int and ~(hours_per_week > 0 or hours_per_week <= 50)) or type(hours_per_week) != int:
			hours_per_week = 40
		else:
			hours_per_week = int(hours_per_week)

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
		if(classifier not in ['Random Forest', 'DT', "Naive Bayes", "KNN", "Logistic Regression", "SVC", "LDA", "SGD", "MLP", "bagging", "stacking", "AdaBoost", "GraBoost", "XGB", "CAT", "blending_2_best", "blending_3_best", "blending_4_best", "blending_5_best", "SoftVoter_2_best", "SoftVoter_3_best", "SoftVoter_4_best", "SoftVoter_5_best"]):
			classifier = 'Random Forest'

		#spliting the columns
		new_order = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
		categories = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
		numerical = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

		new_data = [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country]
		print(new_data)

		#importing our model
		model = joblib.load('./models/census/' + classifier +'_model.sav')

		features = pd.DataFrame(columns=new_order).append(pd.Series(new_data, index=new_order), ignore_index=True)

		print(features)

		#import preprocessing pipeline
		pipeline = joblib.load('./models/census/preprocessing_pipeline.pkl')

		#preprocess the features of our new data
		features_transformed = pipeline.transform(features)
		columns = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names(['age']).tolist() + numerical + pipeline.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot'].get_feature_names(categories).tolist()
		features_transformed = pd.DataFrame(features_transformed.toarray(), columns=columns)

		if classifier == 'stacking' or classifier == 'bagging': #HEROKU DOESN'T ALLOW ME TO EXECUTE THESE MODELS ONCE THEY EXCEED THE QUOTA LIMIT OF MEMORY USAGE FOR THE FREE PLAN -.-''
			#stacking
			#myStack = joblib.load('./models/census/stack_transformer.pkl')
		    #S_features = myStack.transform(features_transformed)
		    #prediction = model.predict(S_features)
			model = joblib.load('./models/census/' + 'Random Forest' +'_model.sav')
			prediction = model.predict(features_transformed)
		elif classifier[:5] == 'blend':
			aux = [2.5137741925651853, 0.0, 0.0, 0.0, 0.9545962190891109, -4.233119319852236, -0.19082047088480297, -0.23652796944416712, -0.5335433061448152, 0.0, 0.0, 0.0, 2.2741031045702984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.862925081030172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3797259162740674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.870103392506245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.536260833431142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 3.1050743640994787, 0.0, 2.1642716314981283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.601288984203265, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

			temp = pd.DataFrame(columns=features_transformed.columns)
			temp = temp.append(pd.Series(aux, index=features_transformed.columns), ignore_index=True)
			temp = temp.append(pd.Series(features_transformed.values.tolist()[0],index=temp.columns), ignore_index=True)

			prediction = model.predict(temp)

		else:
		    prediction = model.predict(features_transformed)

		if classifier[:5] == 'blend':
		    prediction = prediction[1]

		if int(prediction) == 0:
		       answer = 'The Income is Equal or Less than 50.000$'
		else:
		       answer = 'The Income is Greater than 50.000$'


	return render_template('classify.html',prediction = answer,name = classifier)

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		LotArea = request.form['LotArea']
		if type(LotArea) != int or (type(LotArea) == int and ~(LotArea > 0)):
			LotArea = 7200
		else:
			LotArea = int(LotArea)

		Street = request.form['Street']
		if (Street not in ['Pave', 'Grvl']):
			Street = 'Pave'

		MSSubClass = 20
		MSZoning = 'RL'
		LotFrontage = 60.0
		Alley = 'Grvl'
		LotShape = 'Reg'
		LandContour = 'Lvl'
		Utilities = 'AllPub'
		LotConfig = 'Inside'
		LandSlope = 'Gtl'
		Neighborhood = 'NAmes'
		Condition1 = 'Norm'
		Condition2 = 'Norm'
		BldgType = '1Fam'
		HouseStyle = '1Story'
		OverallQual = 5
		OverallCond = 5
		YearRemodAdd = 1950
		RoofMatl = 'CompShg'
		Exterior1st = 'VinylSd'
		Exterior2nd = 'VinylSd'
		MasVnrType = 'None'
		MasVnrArea = 0.0
		ExterQual = 'TA'
		ExterCond = 'TA'
		Foundation = 'PConc'
		BsmtQual = 'TA'
		BsmtCond = 'TA'
		BsmtExposure = 'No'
		BsmtFinType1 = 'Unf'
		BsmtFinSF1 = 0
		BsmtFinType2 = 'Unf'
		BsmtFinSF2 = 0
		BsmtUnfSF = 0
		TotalBsmtSF = 0
		Heating = 'GasA'
		HeatingQC = 'Ex'
		Electrical = 'SBrkr'
		_1stFlrSF = 864
		_2ndFlrSF = 0
		LowQualFinSF = 0
		GrLivArea = 864
		BsmtFullBath = 0
		BsmtHalfBath = 0
		FullBath = 2
		HalfBath = 0
		KitchenQual = 'TA'
		TotRmsAbvGrd = 6
		Functional = 'Typ'
		Fireplaces = 0
		FireplaceQu = 'Gd'
		GarageType = 'Attchd'
		GarageYrBlt = 2005.0
		GarageFinish = 'Unf'
		GarageCars = 2
		GarageQual = 'TA'
		GarageCond = 'TA'
		PavedDrive = 'Y'
		WoodDeckSF = 0
		OpenPorchSF = 0
		EnclosedPorch = 0
		_3SsnPorch = 0
		ScreenPorch = 0
		PoolQC = 'Gd'
		Fence = 'MnPrv'
		MiscFeature = 'Shed'
		MiscVal = 0
		MoSold = 6
		YrSold = 2009
		SaleType = 'WD'
		SaleCondition = 'Normal'


		YearBuilt = request.form['YearBuilt']
		if type(YearBuilt) != int or (type(YearBuilt) == int and ~(YearBuilt > 1800 and YearBuilt <2020)):
			YearBuilt = 2006
		else:
			YearBuilt = int(YearBuilt)

		RoofStyle = request.form['RoofStyle']
		if (RoofStyle not in ['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']):
			RoofStyle = 'Gable'

		CentralAir = request.form['CentralAir']
		if (CentralAir not in ['Y' 'N']):
			CentralAir = 'Y'

		BedroomAbvGr = request.form['BedroomAbvGr']
		if type(BedroomAbvGr) != int or (type(BedroomAbvGr) == int and ~(BedroomAbvGr > 0)):
			BedroomAbvGr = 3
		else:
			BedroomAbvGr = int(BedroomAbvGr)

		KitchenAbvGr = request.form['KitchenAbvGr']
		if type(KitchenAbvGr) != int or (type(KitchenAbvGr) == int and ~(KitchenAbvGr > 0)):
			KitchenAbvGr = 1
		else:
			KitchenAbvGr = int(KitchenAbvGr)

		Fireplaces = request.form['Fireplaces']
		if type(Fireplaces) != int or (type(Fireplaces) == int and ~(Fireplaces >= 0)):
			Fireplaces = 0
		else:
			Fireplaces = int(Fireplaces)

		GarageArea = request.form['GarageArea']
		if type(GarageArea) != int or (type(GarageArea) == int and ~(GarageArea >= 0)):
			GarageArea = 0
		else:
			GarageArea = int(GarageArea)

		PoolArea = request.form['PoolArea']
		if type(PoolArea) != int or (type(PoolArea) == int and ~(PoolArea >= 0)):
			PoolArea = 0
		else:
			PoolArea = int(PoolArea)


		predictor = request.form['predictive']
		if(predictor not in ["Random Forest", "Ridge", "Lasso", "SVR", "GraBoost", "XGB", "AdaBoost", "CAT", "bagging", "stacking", "blending_2_best", "blending_3_best", "blending_4_best", "voter_2_best", "voter_3_best", "voter_4_best"]):
			predictor = 'Random Forest'


		#spliting the columns
		new_order = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Condition2', 'KitchenQual', 'BsmtQual', 'BsmtFinType1', 'GarageFinish', 'RoofStyle', 'Functional', 'PoolQC', 'ExterQual', 'PavedDrive', 'Exterior2nd', 'Exterior1st', 'BsmtExposure', 'BsmtCond', 'LotShape', 'BldgType', 'MasVnrType', 'Foundation', 'Alley', 'GarageCond', 'Heating', 'HeatingQC', 'GarageType', 'Street', 'ExterCond', 'LotConfig', 'MSZoning', 'SaleCondition', 'FireplaceQu', 'Neighborhood', 'RoofMatl', 'Electrical', 'Utilities', 'SaleType', 'LandContour', 'Fence', 'CentralAir', 'Condition1', 'MiscFeature', 'LandSlope', 'GarageQual', 'HouseStyle', 'BsmtFinType2']
		categorical = ['Condition2', 'KitchenQual', 'BsmtQual', 'BsmtFinType1', 'GarageFinish', 'RoofStyle', 'Functional', 'PoolQC', 'ExterQual', 'PavedDrive', 'Exterior2nd', 'Exterior1st', 'BsmtExposure', 'BsmtCond', 'LotShape', 'BldgType', 'MasVnrType', 'Foundation', 'Alley', 'GarageCond', 'Heating', 'HeatingQC', 'GarageType', 'Street', 'ExterCond', 'LotConfig', 'MSZoning', 'SaleCondition', 'FireplaceQu', 'Neighborhood', 'RoofMatl', 'Electrical', 'Utilities', 'SaleType', 'LandContour', 'Fence', 'CentralAir', 'Condition1', 'MiscFeature', 'LandSlope', 'GarageQual', 'HouseStyle', 'BsmtFinType2']
		numerical = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

		new_data = [MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, _1stFlrSF, _2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, _3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold, ExterQual, SaleCondition, BsmtCond, GarageType, LandSlope, BsmtFinType2, Exterior2nd, Utilities, GarageQual, GarageCond, MiscFeature, Condition1, MasVnrType, LotShape, Heating, BsmtExposure, Fence, Electrical, Exterior1st, GarageFinish, BsmtQual, MSZoning, Street, FireplaceQu, CentralAir, HeatingQC, PavedDrive, Alley, HouseStyle, Neighborhood, LotConfig, RoofMatl, ExterCond, Condition2, BsmtFinType1, BldgType, RoofStyle, Foundation, Functional, LandContour, SaleType, PoolQC, KitchenQual]


		#alternatively we can just import one model from the saved ones
		model = joblib.load('./models/house_prices/' + predictor +'_model.sav')

		features = pd.DataFrame(columns=new_order).append(pd.Series(new_data, index=new_order), ignore_index=True)
		print(features)

		#import preprocessing pipeline
		pipeline = joblib.load('./models/house_prices/preprocessing_pipeline.pkl')

		#preprocess the features of our new data
		features_transformed = pipeline.transform(features)

		columns = numerical + pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical).tolist()
		features_transformed = pd.DataFrame(features_transformed.toarray(), columns=columns)

		if classifier == 'stacking' or classifier == 'bagging': #HEROKU DOESN'T ALLOW ME TO EXECUTE THESE MODELS ONCE THEY EXCEED THE QUOTA LIMIT OF MEMORY USAGE FOR THE FREE PLAN -.-''
			#stacking
		    #myStack = joblib.load('./models/house_prices/stack_transformer.pkl')
		    #S_features = myStack.transform(features_transformed)
		    #prediction = model.predict(S_features)

			model = joblib.load('./models/census/' + 'Random Forest' +'_model.sav')
			prediction = model.predict(features_transformed)
		elif predictor[:5] == 'blend':
			aux = [1.5480575331743465, -1.7692939595588797, 0.10024001866725883, 1.4412259084896497, -0.5258098048647717, 0.7248315309645063, 0.3972950469150851, -0.5827395413833021, 2.8217567542153508, -0.27463217309694465, -1.0703305329277906, 1.7407275567651526, 1.6481278471790815, -0.797637378950148, -0.11603790663031163, 0.46675627772419426, 3.0779257911434796, -0.2399339203049255, -1.0219736443730056, 1.2334937910950736, -2.307429920822539, -0.21291309699499958, -0.310456627505031, 2.198962018063716, 0.5321594655109821, 0.33253896384790815, 0.014968441162184229, 0.24133238022063416, -0.2488654750731461, -0.3642759116907947, -0.11221164829823378, -0.27157622942471754, -0.05272776124901699, -0.08805849743282958, -0.4965787211312943, 0.141904970173029, 0.0, 0.0, 10.492195944889811, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0384170109334976, 0.0, 0.0, 0.0, 2.0236153570753554, 0.0, 0.0, 0.0, 2.222144754161601, 0.0, 0.0, 0.0, 2.3517597963228285, 0.0, 0.0, 0.0, 2.4484695181918505, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.942541081503712, 0.0, 26.645851692700283, 0.0, 0.0, 2.1176164554198698, 0.0, 0.0, 0.0, 3.6204583112240867, 0.0, 0.0, 0.0, 7.916354045154916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.586098308504247, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5794273951655455, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7383160580983956, 2.1243356296294262, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.786020804854599, 0.0, 0.0, 2.041361507631104, 0.0, 0.0, 0.0, 2.0135119835468513, 0.0, 0.0, 0.0, 6.114511895669278, 0.0, 0.0, 0.0, 0.0, 0.0, 5.2719583477072245, 0.0, 6.838441772831753, 0.0, 0.0, 0.0, 0.0, 2.000198963005316, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0911088278033154, 0.0, 0.0, 0.0, 0.0, 0.0, 16.870196755533634, 0.0, 0.0, 0.0, 0.0, 3.0863960845905702, 0.0, 4.144852443353103, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.463629210393952, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6200159915067376, 0.0, 0.0, 0.0, 2.26541853208081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.398109885489133, 7.91635404515492, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.539959147267338, 37.66962577085551, 0.0, 0.0, 26.645851692700283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.341728987199786, 0.0, 0.0, 3.5929259092868935, 0.0, 0.0, 4.019465556741433, 0.0, 0.0, 2.916282098747573, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.870196755533634, 0.0, 4.496356490089693, 0.0, 0.0, 0.0, 0.0, 0.0, 4.713148841128412, 0.0, 0.0, 2.0000079573802743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.2610111639057546]

			temp = pd.DataFrame(columns=features_transformed.columns)
			temp = temp.append(pd.Series(aux, index=features_transformed.columns), ignore_index=True)
			temp = temp.append(pd.Series(features_transformed.values.tolist()[0],index=temp.columns), ignore_index=True)

			prediction = model.predict(temp)

		else:
			prediction = model.predict(features_transformed)

		Y_scaler = joblib.load('./models/house_prices/Y_scaler.pkl')

		prediction = pd.DataFrame(Y_scaler.inverse_transform(prediction))

		if predictor[:5] == 'blend':
		    answer = 'The price of the house will be %.2f$!' % prediction.values[1][0]
		else:
		    answer = 'The price of the house will be %.2f$!' % float(prediction.values)









	return render_template('predict.html',prediction = answer,name = predictor)

@app.route('/compare_classifiers', methods=['POST'])

def compare_classifiers():

	with open("static/results/census/statistic_results.txt") as f:
		file_content = f.read()

	return render_template('compare_classifiers.html',prediction = file_content,name = "")

@app.route('/compare_predictors', methods=['POST'])
def compare_predictors():

	with open("static/results/house_prices/statistic_results.txt") as f:
		file_content = f.read()

	return render_template('compare_predictors.html',prediction = file_content,name = "")




if __name__ == '__main__':
	app.run(debug=True)

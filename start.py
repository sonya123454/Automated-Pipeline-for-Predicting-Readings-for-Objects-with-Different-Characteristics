#import data
import json
%run libs/libs.py
%run libs/data.py
#import os
#os.system('python libs.py')

###### Глобальные переменные. 

config = "C:/Users/msson/energy_artefacts/conf/gisee_test.json.example"
engine = db.db_connect(config)

saving_path = '/.pipeline/'
	
def start_pipline(target = 'ec_d'):

	timeseries = data.get_data(engine)
	# добавляем признаки
	timeseries_new = []
	for df in timeseries:
		timeseries_new.append(get_features(df, engine))
	timeseries = timeseries_new
	# удаляем аномалии
	timeseries_new = []
	for df in timeseries:
		timeseries_new.append(delete_anomalies_all(df, plt = False))
	timeseries = timeseries_new
	# кластеризуем и сохраним kmeans
	clusters = clusterization(timeseries)
	# выберем репрезентативные ряды из кластеров
	representatives = []
	for i in range(clusters.max()):
		representatives.append(get_cluster_representative(timeseries, i, clusters)[0])
	
	models = {}
	
	# выбираем признаки
	representatives_new = []
	for i in range(len(representatives)):
		representative = representatives[i]
		selected = select_features(representative, engine)
		representatives_new.append(representative[np.append(selected, target)])
		models[i] = {'features': selected.tolist()}
	representatives = representatives_new	
	
	# выбираем модель
	for i in range(clusters.max()):
		models[i]['model'] = select_model(representatives[i], plt = False)
	
	# сохраняем подобранные модель и признаки у кластеров
	with open(saving_path + 'cluster_models_info.json', 'w') as fp:
		json.dump(models, fp)
		
def get_prediction(dt_start_forecast, dt_finish_forecast):
	# получаем временной ряд для прогноза
	df = get_data_for_prediction(engine)
	# получаем доп данные
	df = get_features(df, engine)
	# определяем кластер, признаки и модель
	new_object, columns = get_features_for_clusterization(df)
	new_object = np.array(new_object).reshape(1,-1)
	new_object = pd.DataFrame(data = new_object, columns = columns)
	clst, features, model = find_cluster(new_object, plt = False)
	# оставляем только нужные признаки
	df = df[np.append(features, target)]
	# удаляем аномалии
	df = delete_anomalies_all(df)
	
	
	# получаем exog
	dates = pd.date_range(dt_start_forecast, dt_finish_forecast, freq='D')
	exog = pd.DataFrame(index = dates)
	exog = get_features(exog, engine, lags = False, target = target)
	exog = exog[features]
	X_y_train_features =  df
	y_train = df[target]
	X_train = df.drop(target, axis = 1)
	# подбираем гиперпараметры, обучаем модель и делаем предсказание
	if (model == 'xgb'): 
		prediction = XGB_forecast(X_train, exog, y_train, y_train)
	if (model == 'prophet'): 
		prediction = prophet_forecast(X_y_train_features, exog, features)
	if (model == 'sarimax'):
		predicrion = SARIMAX_forecast(X_train, exog, y_train, y_train)
		
	return prediction

	
start_pipline(target = 'ec_d')
get_prediction( num_pred = 365)
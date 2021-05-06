import data
%run libs/libs.py
%run libs/data.py

###### Глобальные переменные. Они будут переопределены при рестарте пайплайна
clusters = []
cluster_with_features = []
cluster_models = {}

config = "C:/Users/msson/energy_artefacts/conf/gisee_test.json.example"
engine = db.db_connect(config)

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
		timeseries_new.append(delete_anomalies_all(df, engine))
	timeseries = timeseries_new
	# кластеризуем
	clusters = clusterization(timeseries)
	representatives = []
	for i in range(clusters.max()):
		representatives.append(get_claster(timeseries, i, clusters)[0])
	
	# выбираем признаки
	representatives_new = []
	for representative in representatives:
		selected = select_features(representative, engine)
		representatives_new.append(representative[np.append(selected, target)])
	representatives = representatives_new	
	# выбираем модель
	models = {}
	for i in range(clusters.max()):
		models[i] = select_model(representatives[i], plt = True)
	
	
	'''
	for df in timeseries:
		df = delete_anomalies(df, anomalies, type = 'average', num = 7)
		
	clusters = clustering(timeseries)
	
	clusters_with_features = []
	for cluster in clusters:
		clusters_with_features.append(create_features(cluster))
	
	for cluster in clusters_with_features:
		get_features(cluster, engine, lags = False, target = "ec_d")
	'''	
	
	
def get_prediction(df, period):
	df = delete_anomalies(df, anomalies, type = 'average', num = 7)
	cluster, features = get_cluster(clusters)
	df = get_features(df, cluster)
	model = cluster_models[cluster]
	model.fit()
	prediction = model.predict(period)
	
	return prediction

	
	
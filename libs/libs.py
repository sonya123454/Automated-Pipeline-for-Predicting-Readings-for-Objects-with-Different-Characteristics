#graphs
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as scs
from statsmodels import tsa
from adtk.visualization import plot

#math
import pandas as pd
import numpy as np
import operator
import math

#sys
import imp
import os
import sys
import traceback
import itertools

#time
import time
import datetime 

#own libraries
import prediction as p
from my_db import init_config 
import my_db as db 

#machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression#, LogisticRegression

#time series machine learning
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet

# anomalies
from adtk.data import validate_series
from adtk.pipe import Pipeline
from adtk.detector import QuantileAD

# saving
import pickle
import json

saving_path = './.pipeline/'



###################################################################
## ФУНЦИИ ДЛЯ ОТРИСОВКИ ГРАФИКОВ

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()        
        print("Критерий Дики-Фуллера: p=%f" % tsa.stattools.adfuller(y)[1])

    return 
	
	
def tsplot_only(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)   
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        plt.tight_layout()        
    return 
	
	
def model_result_plot(train_pred, y_train, test_pred, y_test):
    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    
    plt.figure(figsize=(15, 5))
    plt.plot(train_pred, label="prediction")
    plt.plot(y_train, label = 'y_train')
    plt.axis('tight')
    plt.grid(True)
    plt.legend()

    # запоминаем ошибку на трейне
    scale = 1
    deviation = mean_absolute_error(train_pred, y_train)
    # и на тестовом
    
    lower = test_pred-scale*deviation
    upper = test_pred+scale*deviation

    Anomalies = np.array([np.NaN]*len(test_pred))
    Anomalies[y_test<lower] = y_test[y_test<lower]

    plt.figure(figsize=(15, 5))
    plt.plot(test_pred, label="prediction")
    #plt.plot(lower, "r--", label="upper bond / lower bond")
    #plt.plot(upper, "r--")
    plt.plot(y_test, label="y_test")
    #plt.plot(Anomalies, "ro", markersize=10)
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Mean absolute error {}".format(mean_absolute_error(test_pred, y_test)))#round()
    plt.grid(True)
    plt.legend()



####################################################################
## ФУНКЦИИ ДЛЯ АНОМАЛИЙ

def distributions_division(data, plt = False):
    # для разделения вероятностей строим распределение и находим значение, которое вроятнее всего разделяет распределения
    # для этого находим 2 самых популярных бина и между ними самый непопулярный
    nbins = 10
    n, bins = np.histogram(data, nbins, density=1)

    argmax1 = n[:int(nbins/2)].argmax()

    argmax2 = n[int(nbins/2):].argmax() + int(nbins/2)
    argmin1 = n[argmax1:argmax2].argmin() + argmax1

    min1 = bins[argmin1]
    if (plt):
        tsplot_only(np.array(data.loc[data > min1])) 
        tsplot_only(np.array(data.loc[data < min1])) 

    return data.loc[data > min1], data.loc[data < min1]


def delete_anomalies(df, anomalies, type = 'average', num = 7):## Аномалии только для 2 сезонов!
    
    if (type == 'average'):
        df_ans = df.copy()
        new_df = (df_ans.shift(1) + df_ans.shift(-1)) / 2
        df_ans.loc[anomalies.ec_d == True] = new_df.loc[anomalies.ec_d == True]
        return df_ans
    if (type == 'moving_average'):
        df_ans = df.copy()
        new_df = df_ans.rolling(num, min_periods=1).mean()
        #print(new_df)
        df_ans.loc[anomalies.ec_d == True] = new_df.loc[anomalies.ec_d == True]
        return df_ans
		
		
def delete_anomalies_all(df, plt = False):

	similarity = 0.85
	steps = [
		("quantile_ad", QuantileAD(high=0.95, low=0.10))
	]
	pipeline = Pipeline(steps)
	s = validate_series(df[df.index[0]: df.index[-1]])
	anomalies_ts1 = pipeline.fit_detect(s)
	if (plt):
		plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');

	winter, summer = distributions_division(df.ec_d)
	if (plt):
		tsplot_only(np.array(df.ec_d))
	
	s = validate_series(winter)
	anomalies_ts2 = pipeline.fit_detect(s)
	if (plt):
		plot(s, anomaly=anomalies_ts2, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	s = validate_series(summer)
	anomalies_ts3 = pipeline.fit_detect(s)
	if (plt):
		plot(s, anomaly=anomalies_ts3, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	anomalies_ts1.ec_d[anomalies_ts2.index[0]:anomalies_ts2.index[-1]] = anomalies_ts1.ec_d[anomalies_ts2.index[0]:anomalies_ts2.index[-1]] | anomalies_ts2
	anomalies_ts1.ec_d[anomalies_ts3.index[0]:anomalies_ts3.index[-1]] = anomalies_ts1.ec_d[anomalies_ts3.index[0]:anomalies_ts3.index[-1]] | anomalies_ts3
	
	s = validate_series(df[df.index[0]: df.index[-1]])
	if (plt):
		plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	# считаем корреляцию
	# Set window size to compute moving window synchrony.
	r_window_size = 5
	# Compute rolling window synchrony
	rolling_r = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	if (plt):
		tsplot_only(np.array(rolling_r))
	
	anomalies_ts1.loc[abs(rolling_r) > similarity] = False
	
	####################### доработать не реализованы лаги
	'''
	df['temperature'] = np.array(exog_temp.shift(1).fillna(0).temperature)
	rolling_r1 = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	tsplot_only(np.array(rolling_r))
	anomalies_ts1.loc[abs(rolling_r1) > similarity] = False
	
	
	df['temperature'] = np.array(exog_temp.shift(2).fillna(0).temperature)
	rolling_r2 = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	tsplot_only(np.array(rolling_r))
	anomalies_ts1.loc[abs(rolling_r2) > similarity] = False
	'''
	s = validate_series(df[df.index[0]: df.index[-1]])
	if (plt):
		plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red')
	
	new_df = delete_anomalies(df, anomalies_ts1, type = 'moving_average')
	if (plt):
		tsplot_only(np.array(df.ec_d))
		tsplot_only(np.array(new_df.ec_d))
	
	return new_df
	
'''	
def temperature_corr(data, engine):
	df = pd.DataFrame(data.copy())
	exog_temp = db.get_temperature(engine)

	exog_temp.index = pd.to_datetime(exog_temp.index)
	exog = pd.DataFrame(index = exog_temp.index.values, columns=['temperature'])
	exog_temp = exog_temp.groupby(exog_temp.index).first()
	exog_temp = exog_temp.reindex(pd.date_range(exog_temp.index[0], exog_temp.index[-1], freq='D'), fill_value=0)
	exog_temp = exog_temp[df.index[0]: df.index[-1]]
	
	df['temperature'] = np.array(exog_temp.temperature)
	
	return df.corr()
	
'''



####################################################################
### ФУНКЦИИ ДЛЯ ПРИЗНАКОВ

def day_length(day_of_year, latitude):
    P = math.asin(0.39795 * math.cos(0.2163108 + 2 * math.atan(0.9671396 * math.tan(.00860 * (day_of_year - 186)))))
    pi = math.pi
    day_light_hours = 24 - (24 / pi) * math.acos((math.sin(0.8333 * pi / 180) + math.sin(latitude * pi / 180) * math.sin(P)) / (math.cos(latitude * pi / 180) * math.cos(P)))
    return day_light_hours 
	# formula per Ecological Modeling, volume 80 (1995) pp. 87-95, called "A Model Comparison for Daylength as a Function of Latitude and Day of the Year."
	# see more details - http://mathforum.org/library/drmath/view/56478.html
	# Latitude in degrees, postive for northern hemisphere, negative for southern
	# Day 1 = Jan 1
	
	
def heating_switched_off(pred):
    cnt_8 = 0
    heat = []
    for i, row in pred.iteritems():
        fl = 0
        if (i.month >= 4 and i.month <= 9):
            fl = 1
        if (fl and row > 8):
            cnt_8 += 1 
        elif (cnt_8 < 14 or fl == 0): 
            cnt_8 = 0 
        if (i.month == 6 or i.month == 7 or i.month == 8 or i.month == 9 or cnt_8 > 14):
            heat.append(0)
        else:
            heat.append(1)
    return heat

import sqlalchemy as sa

def fill_missing_weather(engine, temperature, day_start, day_finish):
	temperature_2018 = sa.sql.text(''' SELECT * FROM weather_temperature;''')
	temperature_2018 = pd.read_sql_query(temperature_2018, con=engine, index_col='id')
	temperature_2018 = temperature_2018.loc[temperature_2018.year == 2018]
	dates_past = pd.date_range(day_start + datetime.timedelta(days=1), temperature.index.min(), freq='D')	
	dates_future = pd.date_range(temperature.index.max() + datetime.timedelta(days=1), day_finish, freq='D')
	#################### заполнение прошлых значений
	
	temperature_new = []
	for key in dates_past:
		a = np.asarray(temperature_2018.loc[(temperature_2018.month == key.month) & (temperature_2018.day == key.day)]['temperature'])
		if (len(a) == 0):
			temperature_new.append(-15)
		else:
			temperature_new.append(a[0])#
	temperature_new = np.append(temperature_new, temperature.values.reshape(-1).tolist()).tolist()
	#################### заполнение будущих значений
	for key in dates_future:
		a = np.asarray(temperature_2018.loc[(temperature_2018.month == key.month) & (temperature_2018.day == key.day)]['temperature'])
		if (len(a) == 0):
			temperature_new.append(-15)
		else:
			temperature_new.append(a[0])#
	return temperature_new
	
def get_features(df, engine, lags = False, target = "ec_d"): # проблема - при предсказание урезается на 7 изза лагов
	day_start, day_finish = df.index.min(), df.index.max()
	
	# holidays
	holidays = db.get_holidays(engine)
	exog = pd.DataFrame(index = holidays.index.values, columns=df.columns.values)
	exog['is_holiday'] = 1;
	dates = pd.date_range(day_start, day_finish, freq='D')
	exog = exog.reindex(dates, fill_value=0)
	exog[df.columns.values] = df
	## temperature
	exog_temp = db.get_temperature(engine)
	exog_temp.index = pd.to_datetime(exog_temp.index)
	exog_temp = exog_temp.groupby(exog_temp.index).first()
	exog_temp = exog_temp.reindex(pd.date_range(exog_temp.index[0], exog_temp.index[-1], freq='D'), fill_value=0)
	exog_temp = exog_temp[day_start: day_finish]
	
	#filling missing temperature values 	
	exog_temp = fill_missing_weather(engine, exog_temp, day_start, day_finish)
	exog['temperature'] = exog_temp
	
	# day_length
	dayofyear = [(exog.index[i] - datetime.datetime(exog.index[i].year, 1, 1)).days + 1 for i in range(len(exog))]
	days_length = [day_length(i, 55.4424) for i in dayofyear]
	exog['sun_time'] = days_length
	
	# weekday
	exog['weekday'] = exog.index.weekday
	
	# lags
	for j in ['is_holiday', 'temperature', 'sun_time']:
		for i in range(1, 7):
			lag = j + 'lag' + str(i)
			exog[lag] = np.append(np.repeat(np.nan, i), exog[j][:-i].values)
			
	# autoregression
	if (lags == True):
		for i in range(1, 365):
			lag = 'lag' + str(i)
			exog[lag] = np.append(np.repeat(np.nan, i), exog[target][:-i].values)
			
	# heat off
	exog['heat_off'] = heating_switched_off(exog.temperature)
	exog = exog.dropna()
	
	return exog
	
	
def select_features(df, engine, max_features = 10, target = "ec_d"):
    Z = []
    X_train, _, y_train, _ = train_test_split(df, 0)

    ### xgb
    Z.append(XGB_features(df))
    ### corr
    cor = df.corr()
    feature_corr = dict(sorted(np.abs(cor[target]).to_dict().items(), key=operator.itemgetter(1), reverse=True)[:max_features])
    Z.append(feature_corr)
    ### forward
    try:
        feature_fw = forward_selection(X_train, y_train, significance_level=0.05)
        Z.append(feature_fw)
    except:
        a = 1
    ### lasso
    scaler = StandardScaler()
    scaler.fit(X_train.fillna(0))

    sel_ = SelectFromModel(Lasso(), max_features = max_features)
    sel_.fit(scaler.transform(X_train), y_train)
    feature_score = pd.DataFrame(data = sel_.get_support().astype(int), columns = ['is_important'])
    feature_score.index = X_train.columns.values
    feature_lasso = feature_score.loc[feature_score['is_important'] > 0]['is_important'].to_dict()

    Z.append(feature_lasso)

    #### 
    features = []
    for z in Z:
        new_z = []
        for j in X_train.columns:
            if j in z:
                new_z.append(1)
            else:
                new_z.append(0)
        features.append(new_z)
    features = pd.DataFrame(features, columns = X_train.columns)
    features_ = features.columns[features.sum() >= len(Z) / 2].values

    d = features.shape[1]
    kbar = features.sum(1).mean();
    stability = 1 - features.var(0, ddof=1).mean() / ((kbar/d)*(1-kbar/d))
    #print('stability', stability)
    return features_
	
	
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)> 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features
	
	
def XGB_features(data, scale=1.96, max_features = 10):
    # исходные данные
    X_train, _, y_train, _ = train_test_split(data,  0)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # задаём параметры
    params = {
        'objective': 'reg:squarederror',#'objective': 'reg:linear',
        'booster':'gbtree'#'booster':'gblinear'
    }
    trees = 100

    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics = ('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())
    
    feature_score = bst.get_score()
    feature_top_10 = dict(sorted(feature_score.items(), key=operator.itemgetter(1), reverse=True)[:max_features])

    return feature_top_10
	
	
	
###################################################################
# ФУНКЦИИ ДЛЯ КЛАСТЕРИЗАЦИИ

def summer_consumption(data):
    # считаем среднее летнее потребление, чтобы отсеить тех, для кого летнее тоже надо предсказывать
    
    counter = data.loc[data.index.month == 7].size + data.loc[data.index.month == 6].size + data.loc[data.index.month == 8].size
    summ = data.loc[data.index.month == 7].sum() + data.loc[data.index.month == 6].sum() + data.loc[data.index.month == 8].sum()
   
    if (counter == 0):
        return 0
    return summ / counter


def trend(data, plt = False):
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(np.array(range(data.size)).reshape(-1, 1), data) 
    y_pred = regr.predict(np.array(range(data.size)).reshape(-1, 1))
    if (plt):
        plt.scatter(range(data.size), np.array(data).reshape(-1, 1),  color='black')
        plt.plot(range(data.size), y_pred, color='blue', linewidth=3)
        plt.show()
    return regr.coef_[0]
	
	
def get_cluster_representative(timeseries, claster, fclust, plt = False):
	clst = []
	for i in range(len(fclust)):
		if (fclust[i] != claster):
			continue
		else:
			clst.append(timeseries[i])
			if (plt):
				print(i)
				tsplot_only(np.array(timeseries[i].ec_d))
	return clst
	
	
def find_cluster(df, plt = False):
	# загрузим кластеризующую модель и найдем кластер у df
	loaded_model = pickle.load(open(saving_path + 'kmeans_model.sav', 'rb'))
	result = loaded_model.predict(df)[0]
	# загрузим информацию о признаках и модели данного кластера
	with open(saving_path + 'cluster_models_info.json', 'r') as fp:
		savings = json.load(fp)
	features = savings[str(result)]['features']
	model = savings[str(result)]['model']
	return result, features, model


def get_features_for_clusterization(df):
    winter_df, summer_df = distributions_division(df.ec_d)
    summer0 = summer_consumption(df.ec_d)
    trendwinter = trend(winter_df)
    trendsummer = trend(summer_df)
    correlation = df.corr().ec_d.values[1:]
    try:
        pacf_winter = pacf(np.array(winter_df))[1:7]
    except:
        pacf_winter = np.array([-1 for i in range(6)])
        
    try:
        pacf_summer = pacf(np.array(summer_df))[1:7]
    except:
        pacf_summer = np.array([-1 for i in range(6)])

    new_represents = np.append([summer0, trendwinter, trendsummer], correlation)
    new_represents = np.append(new_represents, pacf_winter)
    new_represents = np.append(new_represents, pacf_summer)
	
    columns = ['summer_consuption', 'trendwinter', 'trendsummer']
    columns = np.append(columns, df.corr().ec_d.index.values[1:])
    columns = np.append(columns, ['pacfwinter' + str(i) for i in range(6)])
    columns = np.append(columns, ['pacfsummer' + str(i) for i in range(6)])
    return new_represents, columns


def clusterization(timeseries):
	
	pacf0 = []
	summer0 = []
	trend0 = []
	trendwinter = []
	trendsummer = []
	pacf_summer = []
	pacf_winter = []
	correlation = []
	cnt = 0
	pacf0 = []
	
	new_objects = []
	columns = []
	for i in range(len(timeseries)):
		new_object, columns = get_features_for_clusterization(timeseries[i])
		new_objects.append(new_object)
	
	new_objects = np.array(new_objects).reshape(len(timeseries),-1)
	classification = pd.DataFrame(data = new_objects, columns = columns)
	
	kmeans = KMeans( random_state=0).fit(classification)
	## сохраним модель 
	filename = saving_path + 'kmeans_model.sav'
	pickle.dump(kmeans, open(filename, 'wb'))
	labels = kmeans.labels_
	return labels
	


############################################
# ФУНКЦИИ ДЛЯ ПРЕДСКАЗЫВАЮЩИХ МОДЕЛЕЙ 

def train_test_split(data, test_size=0.15):

    data = pd.DataFrame(data.copy())
    data = data.dropna()
    data = data.reset_index(drop=True)
    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    test_index = int(len(data)*(1-test_size))
    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["ec_d"], axis=1)
    y_train = data.loc[:test_index]["ec_d"]
    X_test = data.loc[test_index:].drop(["ec_d"], axis=1)
    y_test = data.loc[test_index:]["ec_d"]

    return X_train, X_test, y_train, y_test
	
	
def fit_sarimax(X_train, X_test, y_train, y_test):
	print("FITTING SARIMAX")

	p = d = q = range(0, 2)
	pdq = [(x[0], 1,  2) for x in list(itertools.product(p))]#,7
	seasonal_pdq = [(x[0], 1, 1, 7) for x in list(itertools.product(p))]
	res = []

	for param in pdq:
		for param_seasonal in seasonal_pdq:
			try:
				print(param)
				mod = sm.tsa.statespace.SARIMAX(y_train,
												order=param,
												exog=X_train,
												seasonal_order=param_seasonal,
												enforce_stationarity=False,
												enforce_invertibility=False)

				results = mod.fit(disp=False)


				pred = results.get_prediction(end=results.nobs + len(X_test) - 1, exog = X_test)

				y_pred = pred.predicted_mean
				y_obs  = y_test#[-100:]


				mse  = ((y_pred - y_obs) ** 2).mean()
				rmse = math.sqrt(mse)
				cv   = 100 * rmse / y_obs.mean()
				mape = 100 *((y_pred - y_obs)/y_obs).abs().mean()

				res.append([param, param_seasonal, results.aic, cv, mape, mse, rmse, results])

			except Exception as e:
				print('SARIMAX:\n', traceback.format_exc())
				continue
	return(res)
	
def SARIMAX_forecast(X_train, X_test, y_train, y_test, plt = False):   # надо выбирать с мин акаике
	model = fit_sarimax(X_train, X_test, y_train, y_test)[0][-1]
	train_pred = model.get_prediction(end=model.nobs - 1).predicted_mean
	test_pred = model.get_prediction(end=model.nobs + len(X_test) - 1, exog = X_test).predicted_mean[-len(X_test):]
	if (plt):
		model_result_plot(train_pred, y_train, test_pred, y_test)
	return test_pred

def XGB_forecast(X_train, X_test, y_train, y_test , plt = False):

    # исходные данные
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    # задаём параметры
    params = {
        'objective': 'reg:squarederror',#'objective': 'reg:linear',
        'booster':'gbtree'#'booster':'gblinear'
    }
    trees = 100
    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics = ('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)
    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())
    # запоминаем ошибку на кросс-валидации
    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]
    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    prediction_train = bst.predict(dtrain)
    prediction_test = []
    pred = y_train.values[-1]
    
    if ('lag1' in X_test.columns):
        for ind, row in X_test.iterrows():
            ww = []
            for i in row.values:
                ww.append(i)
            cols = []
            for i in X_test.columns.values:
                cols.append(i)
            pdtest = pd.DataFrame([ww], columns = cols)
            pdtest['lag1'] = pred
            dtest = xgb.DMatrix(pdtest)
            pred = bst.predict(dtest)
            prediction_test.append(pred[0])
    else:
        prediction_test = bst.predict(dtest)
    if (plt):
        model_result_plot(prediction_train, y_train.values, prediction_test, y_test.values)
    return prediction_test
    
def prophet_forecast(X_y_train_features, X_y_test_features, y_train, y_test, selected_features, target = 'ec_d', plt = False):
	X_y_train_features_prophet = X_y_train_features.rename(columns={target: "y" })
	X_y_train_features_prophet = X_y_train_features_prophet.reset_index()
	X_y_train_features_prophet = X_y_train_features_prophet.rename(columns={'index': "ds" })

	X_y_test_features_prophet = X_y_test_features.reset_index()
	X_y_test_features_prophet = X_y_test_features_prophet.rename(columns={'index': "ds" })
	
	m = Prophet()
	for feature in selected_features:
		m.add_regressor(feature, prior_scale=0.5, mode='multiplicative')
	m.yearly_seasonality = True
	m.weekly_seasonality = True
	m.fit(X_y_train_features_prophet)
	
	test_pred = m.predict(X_y_test_features_prophet).yhat.values
	train_pred = m.predict(X_y_train_features_prophet).yhat.values
	
	if (plt):
		model_result_plot(train_pred, y_train.values, test_pred, y_test.values)
		
	return test_pred
	
def select_model(X_y,  metric = mean_absolute_error, selected_features = None, target = 'ec_d', test_size = 200,  plt = False): # добавить GAM
	if (selected_features == None):
		selected_features = X_y.drop(target, axis = 1).columns.values
		
	X_y_train_features, X_y_test_features =  X_y[:-test_size], X_y[-test_size:] 

	y_train, y_test = X_y_train_features.ec_d, X_y_test_features.ec_d
	X_train = X_y_train_features.drop('ec_d', axis = 1)
	X_test = X_y_test_features.drop('ec_d', axis = 1)
	############# SARIMAX
	test_pred = SARIMAX_forecast(X_train, X_test, y_train, y_test, plt = plt)
	error_sar = metric(test_pred, y_test)
	
	############# PROPHET
	test_pred = prophet_forecast(X_y_train_features, X_y_test_features, y_train, y_test, selected_features, target = target, plt = plt) 
	error_pr = metric(test_pred, y_test)
	
	################ XGB
	test_pred = XGB_forecast(X_train, X_test, y_train, y_test) 
	error_xgb = metric(test_pred, y_test)
	
	################ GAM
	
	if (error_sar <= error_pr and error_sar <= error_xgb):
		return 'sarimax'
	if (error_pr <= error_sar and error_pr <= error_xgb):
		return 'prophet'
	return 'xgb'
	
	
	
###################################################################
# ФУНКЦИИ ДЛЯ ЗАПОЛНЕНИЯ ПРОПУЩЕННЫХ ДАННЫХ
#Усреднение по двум соседним точкам.          averge
#Скользящая средняя.                          moving_average
#Результаты предыдущего периода.              last_period 
#Данные за аналогичный период в прошлые годы. last_year_period
#Экспертное заключение.

def fill_in_missing_values(s):
    return(s.interpolate(method='linear'))


def resample(df):
    dft = df['ec_h'].resample('D').sum().to_frame().rename(columns={'ec_h': 'ec_d'}).copy()
    return(dft)

	
def preparing_data(df):
	print("PREPARING DATA")
	if (df.shape[0] < 100):
		print("too little data to analyze")
		return (0, "too little data to analyze")
	df2 = pd.DataFrame(index=df['dt_reading'])
	df2['ec_h'] = df['consumption'].values
	df = df2
	
	df.ec_h = fill_in_missing_values(df.ec_h)
	df = resample(df)

	print("first date for which data are available:          ", min(df.index.values))
	print("total number of days for which there is data:     ", df.shape[0])

	df.loc[df['ec_d'] < 0, 'ec_d'] = 0
	return (df, "")
	
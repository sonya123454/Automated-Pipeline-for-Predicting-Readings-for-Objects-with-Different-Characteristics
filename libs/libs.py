#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(0, "C:/Users/msson/gisee_md_forecasts/system_forecast/" )

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
#sys
import imp
import os
import sys
#time
import time
#own libraries
import prediction as p
from my_db import init_config #libs.
import my_db as db #from libs 
#import prediction as p
#machine learning
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import pacf
#time series machine learning
from statsmodels.tsa.seasonal import seasonal_decompose
# anomalies
from adtk.data import validate_series
from adtk.pipe import Pipeline
from adtk.detector import QuantileAD






###################################################################
## Графики
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
	
####################################################################
## Деление на 2 сезона

def distributions_division(data):
    # для разделения вероятностей строим распределение и находим значение, которое вроятнее всего разделяет распределения
    # для этого находим 2 самых популярных бина и между ними самый непопулярный
    nbins = 10
    n, bins = np.histogram(data, nbins, density=1)

    argmax1 = n[:int(nbins/2)].argmax()

    argmax2 = n[int(nbins/2):].argmax() + int(nbins/2)
    argmin1 = n[argmax1:argmax2].argmin() + argmax1

    min1 = bins[argmin1]
    
    tsplot_only(np.array(data.loc[data > min1])) 
    tsplot_only(np.array(data.loc[data < min1])) 

    return data.loc[data > min1], data.loc[data < min1]


####################################################################
## Аномалии только для 2 сезонов!
def delete_anomalies(df, anomalies, type = 'average', num = 7):
    
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
		
		
def delete_anomalies_all(df, engine):

	similarity = 0.85
	
	#day_correl = 5
	
	steps = [
		("quantile_ad", QuantileAD(high=0.95, low=0.10))
	]
	pipeline = Pipeline(steps)
	s = validate_series(df[df.index[0]: df.index[-1]])
	anomalies_ts1 = pipeline.fit_detect(s)
	#plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');

	winter, summer = distributions_division(df.ec_d)
	#tsplot_only(np.array(df.ec_d))
	
	s = validate_series(winter)
	anomalies_ts2 = pipeline.fit_detect(s)
	#plot(s, anomaly=anomalies_ts2, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	s = validate_series(summer)
	anomalies_ts3 = pipeline.fit_detect(s)
	#plot(s, anomaly=anomalies_ts3, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	anomalies_ts1.ec_d[anomalies_ts2.index[0]:anomalies_ts2.index[-1]] = anomalies_ts1.ec_d[anomalies_ts2.index[0]:anomalies_ts2.index[-1]] | anomalies_ts2
	anomalies_ts1.ec_d[anomalies_ts3.index[0]:anomalies_ts3.index[-1]] = anomalies_ts1.ec_d[anomalies_ts3.index[0]:anomalies_ts3.index[-1]] | anomalies_ts3
	
	s = validate_series(df[df.index[0]: df.index[-1]])
	#anomalies_ts1 = pipeline.fit_detect(s)
	#plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');
	
	exog_temp = db.get_temperature(engine)

	exog_temp.index = pd.to_datetime(exog_temp.index)
	exog = pd.DataFrame(index = exog_temp.index.values, columns=['temperature'])
	exog_temp = exog_temp.groupby(exog_temp.index).first()
	exog_temp = exog_temp.reindex(pd.date_range(exog_temp.index[0], exog_temp.index[-1], freq='D'), fill_value=0)
	exog_temp = exog_temp[df.index[0]: df.index[-1]]
	
	df['temperature'] = np.array(exog_temp.temperature)
	# считаем корреляцию
	# Set window size to compute moving window synchrony.
	r_window_size = 5
	# Compute rolling window synchrony
	rolling_r = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	#tsplot_only(np.array(rolling_r))
	
	anomalies_ts1.loc[abs(rolling_r) > similarity] = False
	
	df['temperature'] = np.array(exog_temp.shift(1).fillna(0).temperature)
	rolling_r1 = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	#tsplot_only(np.array(rolling_r))
	anomalies_ts1.loc[abs(rolling_r1) > similarity] = False
	
	
	df['temperature'] = np.array(exog_temp.shift(2).fillna(0).temperature)
	rolling_r2 = df['ec_d'].rolling(window=r_window_size, center=True).corr(df['temperature']).fillna(0)
	#tsplot_only(np.array(rolling_r))
	anomalies_ts1.loc[abs(rolling_r2) > similarity] = False
	
	s = validate_series(df[df.index[0]: df.index[-1]])
	#anomalies_ts1 = pipeline.fit_detect(s)
	plot(s, anomaly=anomalies_ts1, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red')
	
	new_df = delete_anomalies(df, anomalies_ts1, type = 'moving_average')
	
	tsplot_only(np.array(df.ec_d))
	tsplot_only(np.array(new_df.ec_d))
	
	return new_df
	
	

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
	
	
####################################################################
#функции для машинного обучения

def train_test_split(data, test_size=0.15):#, lag_start=5, lag_end=20

    data = pd.DataFrame(data.copy())
    #data.columns = ["ec_d"]
    data = data.dropna()
    data = data.reset_index(drop=True)
    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    test_index = int(len(data)*(1-test_size))
    print(len(data))
    print(test_index)
    # добавляем лаги исходного ряда в качестве признаков

    

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["ec_d"], axis=1)
    y_train = data.loc[:test_index]["ec_d"]
    X_test = data.loc[test_index:].drop(["ec_d"], axis=1)
    y_test = data.loc[test_index:]["ec_d"]

    return X_train, X_test, y_train, y_test

################################################################
electricity_db = {
'18431': '44044',
}
heat_db = {
'6140': '48457',
'8706': '44883',
'9044': '43824',
'9292': '43831',
'11470': '43827',
'11493': '43832',
'13287': '43821',
'13406':'43828',
'13447':'43826',
'13732':'43823',
'18217':'48378',
'24764':'45138',
'26065':'48363',
'28288':'48354',
'37510':'48470',
#'37598':'48470',
#'37919':'61543',
'38486':'48472',
'38497':'48473'}
#'54108':'48932',
#'54116':'48932',
    
hot_water_db = {
    '8801': '44883',
'8909':'44881',
'12689':'48646',
'12741':'48643',
'12750':'48631',
'15499':'48933',
#'18274':'48315',
'18883':'48364',
'22265':'43633',
'22323':'43633',
'22454':'43633',
'22467':'43633',
'22913'	:'49046',
'23121':	'43661',
'26032':	'48374',
'26108':	'48373',
'28064':	'48361',
'28122':	'48353',
'28426':	'48356',
'31134':	'48887',
#'31139':	'48889',
'34564':	'43595',
'34606':	'43594',
'34616':	'43597',
#'34624':	'43585',
'34632':	'43590',
'34639':	'43584',
'34648':	'43592',
'37012':	'49041',
'37016':	'49045',
'52177':	'44349',
'54687':	'45199',
'54712':	'45202',
'54717':	'45203',
'54786':	'45250',
'54791':	'45217',
'54793':	'45200',
}

###################################################################
## Получение данных
def get_data(engine, plot = False):
	timeseries_water = []
	for i in hot_water_db:
		df = db.get_informaion_from_first_forecast_qisee(engine, 20, i, hot_water_db[i])
		try:
			df, res = p.preparing_data(df)
			if (plot):
				fig = plt.figure(figsize=[15, 10])
				tsplot_only(np.array(df.ec_d))
				plt.show()
			timeseries_water.append(df)
		except:
			print(i)
			continue
	timeseries_el = []
	for i in electricity_db:
		df = db.get_informaion_from_first_forecast_qisee(engine, 30, i, electricity_db[i])
		try:
			df, res = p.preparing_data(df)
			if (plot):
				fig = plt.figure(figsize=[15, 10])
				tsplot_only(np.array(df.ec_d))
				plt.show()
			timeseries_el.append(df)
		except:
			print(i)
			continue
	timeseries_heat = []
	for i in heat_db:
		df = db.get_informaion_from_first_forecast_qisee(engine, 10, i, heat_db[i])
		try:
			df, res = p.preparing_data(df)
			winter_df, summer_df = distributions_division(df.ec_d)
			if (plot):
				tsplot(np.array(df.ec_d))
				plt.show()
			timeseries_heat.append(df)
		except:
			print(i)
			continue
	return timeseries_water, timeseries_el, timeseries_heat

####################################################################
### features
import math

# formula per Ecological Modeling, volume 80 (1995) pp. 87-95, called "A Model Comparison for Daylength as a Function of Latitude and Day of the Year."
# see more details - http://mathforum.org/library/drmath/view/56478.html
# Latitude in degrees, postive for northern hemisphere, negative for southern
# Day 1 = Jan 1
def day_length(day_of_year, latitude):
    P = math.asin(0.39795 * math.cos(0.2163108 + 2 * math.atan(0.9671396 * math.tan(.00860 * (day_of_year - 186)))))
    pi = math.pi
    day_light_hours = 24 - (24 / pi) * math.acos((math.sin(0.8333 * pi / 180) + math.sin(latitude * pi / 180) * math.sin(P)) / (math.cos(latitude * pi / 180) * math.cos(P)))
    return day_light_hours 
	
##
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
            #pred.at[i] = 0
        else:
            heat.append(1)
    return heat
	
import datetime 
def get_features(df, engine, lags = False):

	cols = df.columns.values
	cols = np.append(cols, ['is_holiday', 'temperature', 'sun_time'])
	holidays = db.get_holidays(engine)
	exog = pd.DataFrame(index = holidays.index.values, columns=cols)#, 'sun_time', 'temperature'
	# holidays
	exog['is_holiday'] = 1;
	dates = pd.date_range(df.index.min(), df.index.max(), freq='D')
	exog = exog.reindex(dates, fill_value=0)
	exog[df.columns.values] = df
	## temperature
	exog_temp = db.get_temperature(engine)
	exog_temp.index = pd.to_datetime(exog_temp.index)
	exog_temp = exog_temp.groupby(exog_temp.index).first()
	exog_temp = exog_temp.reindex(pd.date_range(exog_temp.index[0], exog_temp.index[-1], freq='D'), fill_value=0)
	exog_temp = exog_temp[df.index[0]: df.index[-1]]
	exog['temperature'] = np.array(exog_temp.temperature)
	# day_length
	dayofyear = [(exog.index[i] - datetime.datetime(exog.index[i].year, 1, 1)).days + 1 for i in range(len(exog))]
	days_length = [day_length(i, 55.4424) for i in dayofyear]
	exog['sun_time'] = days_length
	# weekday
	exog['weekday'] = exog.index.weekday
	# lags
	for j in ['is_holiday', 'temperature', 'sun_time']:
		for i in range(1, 7):
			lag = 'lag' + str(i)
			exog[j + lag] = np.append(np.repeat(np.nan, i), exog[j][:-i].values)
	# autoregression
	
	if (lags == True):
		for i in range(1, 365):
			lag = 'lag' + str(i)
			exog[lag] = np.append(np.repeat(np.nan, i), exog.ec_d[:-i].values)

	exog['heat_off'] = heating_switched_off(exog.temperature)
	exog = exog.dropna()
	return exog

	
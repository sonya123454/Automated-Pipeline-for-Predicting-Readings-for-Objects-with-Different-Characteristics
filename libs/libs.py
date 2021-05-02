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
from my_db import init_config 
import my_db as db 
#machine learning
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import pacf
import xgboost as xgb

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
def get_features(df, engine, lags = False, target = "ec_d"):

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
			exog[lag] = np.append(np.repeat(np.nan, i), exog[target][:-i].values)

	exog['heat_off'] = heating_switched_off(exog.temperature)
	exog = exog.dropna()
	return exog
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso#, LogisticRegression

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
    print(Z)
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
	
import statsmodels.api as sm
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
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
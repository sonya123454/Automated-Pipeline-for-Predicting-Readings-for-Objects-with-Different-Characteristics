import io
import json
import math
import numpy as np
import pandas as pd
import traceback
import scipy
from scipy import stats
import statsmodels.api as sm
import itertools
import statsmodels
#from libs import my_db as db
import uuid
import warnings
import datetime
from fbprophet import Prophet

#g_start_ts = '2015-01-01' # changes
#g_end_ts   = '2018-08-01' # changes
day_test = 200


def fill_in_missing_values(s):
    return(s.interpolate(method='linear'))

def resample(df):
    dft = df['ec_h'].resample('D').sum().to_frame().rename(columns={'ec_h': 'ec_d'}).copy()
    return(dft)

#def add_features(df):
#    df['month']      = df.index.month
#    df['dayofweek']  = df.index.dayofweek
#    df['weekofyear'] = df.index.weekofyear
#    return(df)
	
def preparing_data(df):

	print("PREPARING DATA")
	if (df.shape[0] < 100):
		#print("ERROR WITH DATASTART") 
		#raise IOError("too little data to analyze")
		print("too little data to analyze")
		return (0, "too little data to analyze")
	df2 = pd.DataFrame(index=df['dt_reading'])
	df2['ec_h'] = df['consumption'].values
	df = df2
	
	df.ec_h = fill_in_missing_values(df.ec_h)
	df = resample(df)
	global g_start_ts
	g_start_ts = min(df.index.values)
	
	print("first date for which data are available:          ", g_start_ts)
	print("total number of days for which there is data:     ", df.shape[0])
	
	global g_end_ts
	g_end_ts = df.index.values[ df.shape[0] - day_test]
	print("The last date on which the training is conducted: ", g_end_ts)
		
	df.loc[df['ec_d'] < 0, 'ec_d'] = 0
	return (df, "")
	



def fit_sarimax(df, exog):
	print("FITTING SARIMAX")

	p = d = q = range(0, 2)
	pdq = [(x[0], 1, 1, 7) for x in list(itertools.product(p))]
	seasonal_pdq = [(x[0], 1, 1, 7) for x in list(itertools.product(p))]
	res = []
	
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		for param in pdq:
			for param_seasonal in seasonal_pdq:
				try:
					param_an = [1, 1, 1]
					param_seasonal_an = [0, 1, 1, 7]
					mod = sm.tsa.statespace.SARIMAX(df,
													order=param,
													exog=exog,
													seasonal_order=param_seasonal,
													enforce_stationarity=False,
													enforce_invertibility=False)

					results = mod.fit(disp=False)
					
					
					pred = results.get_prediction(start=pd.to_datetime(g_end_ts), 
												  
												  end=df.index.max(), 
												  exog=exog[-day_test:],
												  dynamic=False, 
												  full_results=True)
					
					y_pred = pred.predicted_mean
					y_obs  = df.loc[g_end_ts:]

					
					mse  = ((y_pred - y_obs) ** 2).mean()
					rmse = math.sqrt(mse)
					cv   = 100 * rmse / y_obs.mean()
					mape = 100 *((y_pred - y_obs)/y_obs).abs().mean()

					res.append([param, param_seasonal, results.aic, cv, mape, mse, rmse, results])

				except Exception as e:
					print('SARIMAX:\n', traceback.format_exc())
					continue
	return(res)

def prep_temp(exog_temp):
	
	exog_temp1 = exog_temp.copy()
	cnt_8 = 0
	for i, row in exog_temp1.iteritems():
		fl = 0
		if (i.month >= 4 and i.month <= 9):
			fl = 1
		if (fl and row > 8):
			cnt_8 += 1 
		elif ((cnt_8 < 14) or (fl == 0)): 
			cnt_8 = 0 
		if (i.month == 6 or i.month == 7 or i.month == 8 or cnt_8 > 14):
			exog_temp1.at[i] = 30
	return exog_temp1

def fit_prophet(train_df):
	
	m = Prophet()
	m.add_regressor('temp', prior_scale=0.5, mode='multiplicative')
	m.yearly_seasonality = False
	m.weekly_seasonality = False
	m.add_seasonality(name="quaterly", period = 365.25/4, fourier_order=1, prior_scale = 15)
	m.add_seasonality(name="yearly", period = 365.25, fourier_order=2, prior_scale = 30)
	m.fit(train_df)
	return m
	
def fit_prophet_el(train_df, HOLIDAYS):

	m = Prophet(holidays=HOLIDAYS, holidays_prior_scale=1)
	m.add_regressor('temp', prior_scale=0.5, mode='multiplicative')
	m.add_regressor('sun', prior_scale=0.5, mode='multiplicative')
	m.yearly_seasonality = True
	m.weekly_seasonality = True
	m.fit(train_df)
	return m	
	

def build_model(df, exog_holidays, exog_temp, exog_sun, dt_finish_forecast, id_voc_com_resource):
	
	print("BUILDING MODEL")
	print('g_start_ts:', g_start_ts)
	print('g_end_ts:', g_end_ts)
	print('dt_finish_forecast:', dt_finish_forecast)
	
	dates = pd.date_range(g_start_ts, max(dt_finish_forecast, df.index.max()), freq='D')
	dates_train = pd.date_range(df.index.min(), df.index.max(), freq='D')
	
	HOLIDAYS_train = exog_holidays
	HOLIDAYS_train['holiday'] = "holiday";
	HOLIDAYS_train = HOLIDAYS_train.reset_index()
	HOLIDAYS_train.columns = ['ds', 'holiday']
	
	exog_holidays.index = pd.to_datetime(exog_holidays.index)
	exog = pd.DataFrame(index = exog_holidays.index.values, columns=['is_holiday', 'sun_time', 'temperature'])
	exog['is_holiday'] = 1;
	exog = exog.reindex(dates, fill_value=0)
	
	#filling missing temperature values 
	exog_temp.index = pd.to_datetime(exog_temp.index).to_pydatetime()
	exog_temp = exog_temp.reindex(pd.date_range(exog_temp.index.min(), max(dt_finish_forecast, df.index.max()), freq='D'))
	for i in exog_temp.index:
		if (math.isnan(exog_temp['temperature'][i])):
			exog_temp['temperature'][i] = exog_temp.iloc[[exog_temp.index.get_loc(datetime.date(i.year - 1,i.month, min(28, i.day)))]]['temperature'][0]
	exog['temperature'] = exog_temp.reindex(dates)
	
	exog_sun.index = pd.to_datetime(exog_sun.index)
	exog_sun = exog_sun.reindex(dates, fill_value=datetime.datetime.now().time())
	exog['sun_time'] = [(datetime.datetime.combine(datetime.date.today(), exog_sun['sunset'][i]) - 
			datetime.datetime.combine(datetime.date.today(), exog_sun['sunrise'][i])).total_seconds() / 60
			for i in range(exog_sun['sunset'].size)]
			
	model = None
	if (id_voc_com_resource != db.electricity):
		exog['temperature'] = prep_temp(exog['temperature'])
	SUN_train = exog['sun_time'].loc[df.index.min():df.index.max()]
	exog_temp_train = exog['temperature'].loc[df.index.min():df.index.max()]
	
	df = df.reset_index()
	df.columns = ['ds', 'y']
	df['temp'] = exog_temp_train.values
	df['sun'] = SUN_train.values
	
	if (id_voc_com_resource == db.electricity or id_voc_com_resource == db.hws):
		model = fit_prophet_el(df, HOLIDAYS_train)
	elif (id_voc_com_resource != db.electricity):
		model = fit_prophet(df)
	return model, exog

def heating_switched_off(pred):
	
	cnt_8 = 0
	for i, row in pred.iteritems():
		fl = 0
		if (i.month >= 4 and i.month <= 9):
			fl = 1
		if (fl and row > 8):
			cnt_8 += 1 
		elif (cnt_8 < 14 or fl == 0): 
			cnt_8 = 0 
		if (i.month == 6 or i.month == 7 or i.month == 8 or i.month == 9 or cnt_8 > 14):
			pred.at[i] = 0
	return pred	
	
	
	
def get_pred(model, df, dt_start_forecast, dt_finish_forecast, exog, id_voc_com_resource, heat_for_hot_water):
	print("GETTING PREDICTION")
	dates = pd.date_range(dt_start_forecast, dt_finish_forecast, freq='D')
	future = pd.DataFrame(index = dates, columns=['temp', 'sun'])
	future = future.reset_index()
	future.columns = ['ds', 'temp', 'sun']
	future['temp'] = exog['temperature'].loc[dt_start_forecast:dt_finish_forecast].values
	future['sun'] = exog['sun_time'].loc[dt_start_forecast:dt_finish_forecast].values
	pred = model.predict(future)	
	pred.index = pred['ds']
	pred = pred['yhat']
	
	if (heat_for_hot_water):
		heating_switched_off(pred)
	return pred




def prep_to_save_data(pred, previous_id, id_org_building_forecast, id_voc_com_resource, fake, df):
	# pred.sum()
	pred_sum = pred.sum()
	print("SAVING DATA")
	pred_to_save = pd.DataFrame() 
	pred_to_save['id'] =                       [i + previous_id for i in range(pred.shape[0])]
	pred_to_save['dt'] =                       pred.index.values
	pred_to_save['id_org_building_forecast'] = id_org_building_forecast
	pred_to_save['id_voc_com_resource'] =      id_voc_com_resource 
	pred_to_save['consumption'] =              pred.values
	pred_to_save['fake'] =                     fake
	pred_to_save['not_enough'] =               0
	pred_to_save['uuid'] =                     [uuid.uuid4() for i in range(pred.shape[0])]
	return pred_to_save

def prep_to_save_no_data(pred, previous_id, id_org_building_forecast, id_voc_com_resource, fake, df):
	print("SAVING DATA")
	pred_to_save = pd.DataFrame() 
	pred_to_save['id'] =                       [previous_id ]
	pred_to_save['dt'] =                       datetime.datetime.today()#pred.index.values
	pred_to_save['id_org_building_forecast'] = id_org_building_forecast
	pred_to_save['id_voc_com_resource'] =      id_voc_com_resource 
	pred_to_save['consumption'] =              pred
	pred_to_save['fake'] =                     fake
	pred_to_save['not_enough'] =               [1]
	pred_to_save['uuid'] =                     [uuid.uuid4()]
	return pred_to_save
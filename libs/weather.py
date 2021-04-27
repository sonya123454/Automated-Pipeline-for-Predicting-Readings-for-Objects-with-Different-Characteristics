import requests
import json
import pandas as pd
import  my_db as db
import datetime
import sqlalchemy as sa
import sys, getopt
import traceback
import numpy as np

day_start = '2019-08-29' # changes
day_finish   = '2030-04-08' # changes

def put_weather(config):

	engine = db.db_connect(config)
	
	temperature_2018 = sa.sql.text(''' SELECT * FROM weather_temperature;''')
	temperature_2018 = pd.read_sql_query(temperature_2018, con=engine, index_col='id')

	#temperature_2018 = engine.execute(temperature_2018).fetchall()
	#print(temperature_2018)
	#print(temperature_2018['dt'])
	cnt_temp_sql = sa.sql.text(''' SELECT max(id) FROM weather_temperature;''')
	cnt_temp = engine.execute(cnt_temp_sql).fetchall()[0][0] 
	cnt_temp = 0 if cnt_temp is None else cnt_temp + 1

	dates = pd.date_range(day_start, day_finish, freq='D')	
	temperature_2018 = temperature_2018.loc[temperature_2018.year == 2018]

	temp = pd.DataFrame(columns=['id','temperature','forecast','id_voc_weather_location','dt','year','month',
                                 'day','dt_update','fake'])
	
	temp['dt'] = np.asarray(dates)
	temp['forecast'] = 1
	temp['day'] = dates.day
	temp['year'] = dates.year
	temp['month'] = dates.month
	temp['forecast'] = 1
	temperature = []
	for ind, key in temp.iterrows():
		#temperature.append(
		#print(key.month, key.day)
		#print(key)
		a = np.asarray(temperature_2018.loc[(temperature_2018.month == key.month) & (temperature_2018.day == key.day)]['temperature'])
		if (len(a) == 0):
			temperature.append(-15)
		else:
			temperature.append(a[0])#
		
		
	temp['temperature'] = temperature
	temp.id = range(temp.shape[0])
	temp.id = temp.id + cnt_temp
	temp.fake = 1
	temp.dt_update = datetime.date.today()
	print(temp)
	save_weather(temp, engine)
	#print(np.asarray(dates))
	
	#temp.loc[cnt_temp] = [cnt_temp, float(data['fact']['temp']),0, id_voc_location, 
    #                  datetime.date.today(), datetime.date.today().year,  datetime.date.today().month,
    #                  datetime.date.today().day, datetime.date.today(), 0]
				
def save_weather(weather, engine):
	name = "weather_temperature"
	weather.to_sql(name, engine, schema=None, if_exists='append', 
	index=False, index_label=None, chunksize=None, dtype=None)
#get_all('C:\\Users\\msson\\energy_artefacts\\conf\\test.json.example')
put_weather('C:\\Users\\msson\\energy_artefacts\\conf\\test.json.example')
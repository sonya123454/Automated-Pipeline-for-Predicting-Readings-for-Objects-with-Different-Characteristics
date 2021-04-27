import sqlalchemy as sa
import pandas as pd
import datetime
import os
import json
import sys

now_dt = datetime.datetime.now()
heating     = 10
hws         = 30
cws         = 20
ventilation = 40
electricity = 60
gas         = 50

def init_config(config_file = ''):
    if not os.path.isfile(config_file):
        print("Can't open file '" + config_file + "'")
        sys.exit(1)

    with open(config_file, encoding='utf-8') as f:
        config = json.load(f)
    print("The configuration file has been loaded.")
    return(config)
	
def db_connect(config_path):
	config = init_config(config_path)
	item = config['db']["energy"]
	
	try:
		engine = sa.create_engine(item)
	except sa.exc.NoSuchModuleError:
		logger.error('Could not load the module for creating the SQLAlchemy engine for %s' % key)
	return engine
	
def get_informaion_from_first_forecast_qisee(engine, id_voc_com_resource, id_md_building, id_org_building):
	s1 = ''''''
	s2 = ''';'''

	if (id_voc_com_resource == heating):
		s1 = ''' and md_readings.id_voc_md_volume = 17;'''

	get_information_first = sa.sql.text('''
	select md_readings.* from md_readings
		inner join md_buildings on md_buildings.id_metering_device = md_readings.id_metering_device 
        where md_buildings.id = ''' + str(id_md_building) + s1
	)
	
	df = pd.read_sql_query(get_information_first, con=engine, index_col='id')
	return(df)		   
	
def get_informaion_from_first_forecast_sas(engine, id_voc_com_resource, id_md_building, id_org_building):
	s1 = ''''''
	s2 = ''';'''
	if (id_voc_com_resource == heating):
		s1 = " and mr.id_md_source_volume = 13" # Сасдуэ

	#if (id_voc_com_resource == cws or id_voc_com_resource == hws):
	#	s1 = " and vmv.type = 'V'"
		
	if (id_md_building):
		s2 = ''' and m.id = ''' + str(id_md_building) + ''';''' #2 task prediction
	
	get_information_first = sa.sql.text('''
	select mr.* 
	
	from md_buildings m 
	inner join md_readings mr on m.id_metering_device = mr.id_metering_device
	inner join md_com_resources mc on mr.id_metering_device = mc.id_metering_device
	where mc.id_voc_com_resource = ''' + str(id_voc_com_resource) + ''' and m.id_org_building = ''' + str(id_org_building) + s1 + s2
	)
	print(get_information_first)
	df = pd.read_sql_query(get_information_first, con=engine, index_col='id')
	return(df)	


def add_dt_calc_forecast(engine, id_forecast, success):
	print(str(now_dt))
	add_dt_calc = sa.sql.text('''
	UPDATE org_building_forecasts 
	SET dt_calc = ''' + "TIMESTAMP '" + str(datetime.datetime.now()) + "'," + '''
	dt_start =  ''' + "TIMESTAMP '" + str(now_dt) + "'" + 
	''' WHERE id = ''' +  str(id_forecast))
	engine.execute(add_dt_calc)

def get_count_forecasts(engine):
	get_count_forecasts = sa.sql.text('''
		SELECT COUNT(*) FROM org_building_forecasts
		WHERE dt_calc IS NULL;
	''')
	result = engine.execute(get_count_forecasts)
	return result.fetchall()[0][0]
	
def get_max_forecasts_res(engine):
	get_max_forecasts_res = sa.sql.text('''
		SELECT max(id) FROM org_building_forecast_results
	''')
	result = engine.execute(get_max_forecasts_res)
	a = 0
	b = result.fetchall()[0][0]
	if (b):
		a = b + 1
	return a

def get_forecast_inf(engine, id_forecast):
	get_forecast_inf = sa.sql.text('''
		SELECT fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast
		FROM org_building_forecasts 
		WHERE dt_calc IS NULL
		and id = ''' + str(id_forecast)
	)
	result = engine.execute(get_forecast_inf)
	result = result.fetchall()[0]
	return result[0], result[1], result[2], result[3], result[4], result[5]


def get_inf_first_forecast_test(engine):
	get_inf_first_forecast_test = sa.sql.text('''
		SELECT id, fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast
		FROM org_building_forecasts 
		WHERE id = 161;
	''')
	result = engine.execute(get_inf_first_forecast_test)
	result = result.fetchall()[0]
	return result[0], result[1], result[2], result[3], result[4], result[5], result[6]

def get_heat_for_hot_water(engine, id_org_building):
	get_inf_first_forecast_test = sa.sql.text('''
		SELECT 
		CASE 
		WHEN id_voc_main_heating_system = 30 and (id_voc_infrastructure_hot_water = 20 or id_voc_infrastructure_hot_water = 30)
		THEN 0
		ELSE 1
		END flag 
		FROM org_buildings where id = ''' + str(id_org_building) + ''';'''
	)
	result = engine.execute(get_inf_first_forecast_test)
	result = result.fetchall()[0]
	return result[0]

	
	
def save_pred(pred, engine):
	name1 = "org_building_forecast_results"
	print('save_pred123')
	pred.to_sql(name1, engine, schema=None, if_exists='append', 
	index=False, index_label=None, chunksize=None, dtype=None)
	
def get_holidays(engine):
	get_holidays = sa.sql.text('''
		SELECT dt
		FROM holidays
	''')
	df = pd.read_sql_query(get_holidays, con=engine, index_col='dt')
	return df
	
def get_weather_sun(engine):
	get_weather_sun = sa.sql.text('''
		SELECT dt, sunset, sunrise
		FROM weather_sun
	''')
	df = pd.read_sql_query(get_weather_sun, con=engine, index_col='dt')
	return df
	
def get_temperature(engine):
	get_temperature = sa.sql.text('''
		SELECT DISTINCT dt, temperature
		FROM weather_temperature
		where id_voc_weather_location = 0 and 
		((forecast = 0) or ((forecast = 1) and (dt_update = '''   + "TIMESTAMP '" + str(datetime.datetime.now()) + "')))"
	)
	df = pd.read_sql_query(get_temperature, con=engine, index_col='dt')
	return df	


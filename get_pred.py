import prediction as p
from libs import my_db as db
import pandas as pd
import sys, getopt
import traceback

# key options example
# python get_pred.py -c C:/Users/msson/energy_artefacts/conf/gisee_test.json.example -i 249

def usage():
    print('Usage:')
    print('    python3 get_pred.py -c config.json')
    sys.exit()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'hc:i:v', ['help', 'config=', 'id_forecast=','verbose'])
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit()
	
config = None
verbose = False
id_forecast = None

for k, v in opts:
    if k == '-v':
        verbose = True
    elif k in ('-h', '--help'):
        usage()
        sys.exit()
    elif k in ('-c', '--config'):
        config = v
    elif k in ('-i', '--id_forecast'):
        id_forecast = v
    else:
        assert False, 'unhandled option'

if not config or not id_forecast:
    usage()
    sys.exit()
# use all_forecasts()

def forecast(df, count_res_md, fake, dt_start_forecast, dt_finish_forecast, engine, id_voc_com_resource, id_org_building):
	try:
		df, res = p.preparing_data(df)
		if (res == "too little data to analyze"):
			
			pred = [-1]
			pred = p.prep_to_save_no_data(pred, count_res_md, id_forecast, id_voc_com_resource, fake, df)
			
			#save to data base
			db.save_pred(pred, engine)
			db.add_dt_calc_forecast(engine, id_forecast, 0)
		
		else:
			exog_holidays = db.get_holidays(engine)
			exog_temp = db.get_temperature(engine)
			exog_sun = db.get_weather_sun(engine)
			
			heat_for_hot_water = 0;
			if (id_voc_com_resource == db.heating):
				heat_for_hot_water = db.get_heat_for_hot_water(engine, id_org_building)
			
			model, exog = p.build_model(df, exog_holidays, exog_temp, exog_sun, dt_finish_forecast, id_voc_com_resource)
			pred = p.get_pred(model, df, dt_start_forecast, dt_finish_forecast, exog, id_voc_com_resource, heat_for_hot_water)
			pred = p.prep_to_save_data(pred, count_res_md, id_forecast, id_voc_com_resource, fake, df)
			
			#save to data base
			db.save_pred(pred, engine)
			db.add_dt_calc_forecast(engine, id_forecast, 1)
	
	except Exception as e:
		print("ERROR IN PREDICTION")
		print('ERROR:\n', traceback.format_exc())
		db.add_dt_calc_forecast(engine, id_forecast, 0)
	
def first_forecasts(engine, fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast):
	df = db.get_informaion_from_first_forecast_qisee(engine, id_voc_com_resource, id_md_building, id_org_building)
	# get some additional information
	count_res_md =  db.get_max_forecasts_res(engine) 
	# prediction
	forecast(df, count_res_md, fake, dt_start_forecast, dt_finish_forecast, engine, id_voc_com_resource, id_org_building)


def all_forecasts(config):
	engine = db.db_connect(config)
	fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast = db.get_forecast_inf(engine, id_forecast)
	first_forecasts(engine,  fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast)





all_forecasts(config)


#all_forecasts("C:\\Users\\msson\\energy_artefacts\\conf\\gisee_test.json.example")

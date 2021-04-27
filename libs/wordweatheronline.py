import requests
import json
import pandas as pd
import  my_db as db
import datetime
import sqlalchemy as sa
import sys, getopt
import traceback


def get_weather_data():
    #r = requests.get("http://api.worldweatheronline.com/premium/v1/weather.ashx?key=615ccbde210d4c81864133508210103&q=moscow&num_of_days=15&tp=24&format=json")
    r = requests.get("http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=615ccbde210d4c81864133508210103&q=moscow&date=2019-07-27&enddate=2019-08-01&tp=24&format=json")
    data = json.loads(r.text)
    print(data)
    print(data['data'])
    #return
    data_new = []
    for i in data['data']['weather']:
        data_new.append([i['date'], i['avgtempC']])
    return data_new
	
	
def save_weather_data(data, engine):
    cnt_temp_sql = sa.sql.text(''' SELECT max(id) FROM weather_temperature;''')
    cnt_temp = engine.execute(cnt_temp_sql).fetchall()[0][0] 
    cnt_temp = 0 if cnt_temp is None else cnt_temp + 1
        
    temp = pd.DataFrame(columns=['id','temperature','forecast','id_voc_weather_location','dt','year','month',
                                 'day','dt_update','fake'])
								 
    for i in data:
        temp_avg = i[1]
        date = datetime.datetime.strptime(i[0], "%Y-%m-%d").date()
        fl = 0 if date == datetime.date.today() else 1
        temp.loc[cnt_temp] = [cnt_temp, temp_avg, 0, 0, date, date.year, 
                         date.month,
                         date.day, datetime.date.today(), 0]
        
        cnt_temp += 1
    print(temp)  
    name = "weather_temperature"
    temp.to_sql(name, engine, schema=None, if_exists='append', index=False, index_label=None,chunksize=None, dtype=None)
    
    
def get_all(config):
    engine = db.db_connect(config)
    #locations = ''' SELECT * from voc_weather_locations'''
    #df = pd.read_sql(locations, engine)
    data = get_weather_data()
    save_weather_data(data,  engine)
    print(datetime.date.today() , "- DOWNLOADED")	

def usage():
    print('Usage:')
    print('    python3 get_pred.py -c config.json')
    sys.exit()


'''
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hc:v', ['help', 'config=', 'verbose'])
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit()
config = None
verbose = False
for k, v in opts:
    if k == '-v':
        verbose = True
    elif k in ('-h', '--help'):
        usage()
        sys.exit()
    elif k in ('-c', '--config'):
        config = v
    else:
        assert False, 'unhandled option'

if not config:
    usage()
    sys.exit()
get_all(config)
'''

get_all('C:\\Users\\msson\\energy_artefacts\\conf\\gisee_test.json.example')
#get_weather_data()
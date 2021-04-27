import requests
import json
import pandas as pd
import  my_db as db
import datetime
import sqlalchemy as sa
import sys, getopt
import traceback


def get_weather_data(latitude, longitude):
    params = {'lat': latitude, 'lon': longitude, 
              'lang': "ru_RU", 'limit': 7, 'hours':'false', "extra":"false"}
    headers = {"X-Yandex-API-Key":"3fdf63ec-2cd8-4905-8f48-7886f54256a5"}
    r = requests.get("https://api.weather.yandex.ru/v1/forecast",
                     params=params,
                     headers=headers)
    data = json.loads(r.text)
    return data

def save_weather_data(data, id_voc_location, engine):
    cnt_temp_sql = sa.sql.text(''' SELECT max(id) FROM weather_temperature;''')
    cnt_temp = engine.execute(cnt_temp_sql).fetchall()[0][0] 
    cnt_temp = 0 if cnt_temp is None else cnt_temp + 1
        
    temp = pd.DataFrame(columns=['id','temperature','forecast','id_voc_weather_location','dt','year','month',
                                 'day','dt_update','fake'])
    temp.loc[cnt_temp] = [cnt_temp, float(data['fact']['temp']),0, id_voc_location, 
                      datetime.date.today(), datetime.date.today().year,  datetime.date.today().month,
                      datetime.date.today().day, datetime.date.today(), 0]
    
    
    cnt_temp += 1
    for i in data['forecasts']:
        temp_avg = (float(i['parts']['day_short']['temp']) + float(i['parts']['night_short']['temp'])) / 2
        sunrise = i['sunrise']
        sunset = i['sunset']
        temp.loc[cnt_temp] = [cnt_temp, temp_avg,1, id_voc_location, 
                      i['date'], datetime.datetime.strptime(i['date'], "%Y-%m-%d").date().year, 
                         datetime.datetime.strptime(i['date'], "%Y-%m-%d").date().month,
                         datetime.datetime.strptime(i['date'], "%Y-%m-%d").date().day, datetime.date.today(), 0]
        
        cnt_temp += 1
		
    name1 = "weather_temperature"
    temp.to_sql(name1, engine, schema=None, if_exists='append', index=False, index_label=None,
                chunksize=None, dtype=None)
    
    
def get_all(config):
    engine = db.db_connect(config)
    locations = ''' SELECT * from voc_weather_locations'''
    df = pd.read_sql(locations, engine)
    for index, row in df.iterrows():
        data = get_weather_data(row['latitude'], row['longitude'])
        save_weather_data(data, index, engine)
    print(datetime.date.today() , "- DOWNLOADED")	

def usage():
    print('Usage:')
    print('    python3 get_pred.py -c config.json')
    sys.exit()



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


#get_all('C:\\Users\\msson\\energy_artefacts\\conf\\db.json.example')
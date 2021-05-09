import libs
import my_db as db 

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


test_db = {
	'20': {
		'6140': '48457'
	}
}

###################################################################
## Получение данных
def get_data(engine, plot = False):
	timeseries = []
	for i in hot_water_db:
		
		#try:
		df = db.get_informaion_from_first_forecast_qisee(engine, 20, i, hot_water_db[i])
		df, res = libs.preparing_data(df)#, fill_method = 'linear'
		if (plot):
			fig = plt.figure(figsize=[15, 10])
			tsplot_only(np.array(df.ec_d))
			plt.show()
		timeseries.append(df)
		#except:
		#	print(i, 'cannot get')
	for i in electricity_db:
		
		#try:
			df = db.get_informaion_from_first_forecast_qisee(engine, 30, i, electricity_db[i])
			df, res = libs.preparing_data(df)#, fill_method = 'linear'
			if (plot):
				fig = plt.figure(figsize=[15, 10])
				tsplot_only(np.array(df.ec_d))
				plt.show()
			timeseries.append(df)
		#except:
		#	print(i, 'cannot get')
	for i in heat_db:
		
		#try:
		df = db.get_informaion_from_first_forecast_qisee(engine, 10, i, heat_db[i])
		df, res = libs.preparing_data(df)#, fill_method = 'linear'
		if (plot):
			tsplot(np.array(df.ec_d))
			plt.show()
		timeseries.append(df)
		#except:
		#	print(i, 'cannot get')
			
	return timeseries
	
	
def get_data_for_prediction(engine):
	timeseries = []
	for j in test_db:
		for i in test_db[j]:
			df, res = libs.preparing_data(db.get_informaion_from_first_forecast_qisee(engine, j, i, test_db[j][i]))
			return df
			timeseries.append(df)
	return timeseries
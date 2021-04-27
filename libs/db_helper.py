
import sqlalchemy as sa
import pandas as pd
import datetime
import os
import json
import sys

# вспомогательные запросы из таблицы


# выбрать ПУ у которых показаний не менее 100
'''
select count(*) AS CNT,
md_readings.id_metering_device, md_buildings.id_org_building, org_buildings.label, orgs.label from md_readings
inner join md_buildings on md_buildings.id_metering_device = md_readings.id_metering_device
inner join org_buildings on org_buildings.id = md_buildings.id_org_building
inner join orgs on orgs.id = org_buildings.id_org 
group by md_readings.id_metering_device, md_buildings.id_org_building, org_buildings.label, orgs.label
HAVING COUNT(*) > 100;
'''

# вставить новые значения 
'''
INSERT INTO org_building_forecasts VALUES
(212, '374285b6-731c-47cc-bd9d-25863af34b2f', NULL, 0,	45200, '2020-11-24 09:06:30', 53856, NULL,	2, 10, '2020-12-31 00:00:00', '2020-12-01 00:00:00', 'month')
'''


select count(*) AS CNT,
md_readings.id_metering_device, md_buildings.id_org_building, org_buildings.label, orgs.label, md.serial_no  from md_readings
inner join metering_devices md on md.id = md_readings.id_metering_device
inner join md_buildings on md_buildings.id_metering_device = md_readings.id_metering_device
inner join org_buildings on org_buildings.id = md_buildings.id_org_building
inner join orgs on orgs.id = org_buildings.id_org 
group by md_readings.id_metering_device, md_buildings.id_org_building, org_buildings.label, orgs.label, md.serial_no 
HAVING COUNT(*) > 100;

select org_buildings.label, orgs.label from org_buildings
inner join orgs on orgs.id = org_buildings.id_org;
SELECT id, fake, id_org_building, id_md_building, id_voc_com_resource, dt_start_forecast, dt_finish_forecast 
		FROM org_building_forecasts 
		WHERE dt_calc IS NULL
		fetch first 1 row only;
		
select * from org_building_forecast_results obfr where id_org_building_forecast = 212;

INSERT INTO org_building_forecasts VALUES
(219, '374285b6-731c-47cc-bd9d-25863af34b2f', NULL, 0,	45200, '2020-11-24 09:06:30', 53856, NULL,	2, 10, '2020-12-31 00:00:00', '2020-12-01 00:00:00', 'month');
select * from org_building_forecasts;

SELECT * from metering_devices
where id = 53854;

select * from md_buildings where id_metering_device = 53854;


select * from md_readings;
select * from voc_md_volumes vmv ;
select * from voc_com_resources vcr;

select * from md_readings;
and m.id_org_building = 45200 ;mr.id_voc_md_volume = 17 ;
 select mr.*
        from md_buildings m
        inner join md_readings mr on m.id_metering_device = mr.id_metering_device
        inner join md_com_resources mc on mr.id_metering_device = mc.id_metering_device
        where mr.id = 53856;
        
       select * from md_readings
       where id_metering_device = 53856;
       
 select * from org_building_forecast_results obfr where id_org_building_forecast =218;
 
select * from org_building_forecasts;
select * from md_buildings;

truncate table org_building_forecast_results ;
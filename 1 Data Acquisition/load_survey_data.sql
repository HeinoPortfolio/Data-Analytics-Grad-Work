-- Script tht load the data fro the surveys table.
copy public.surveys_csv (patient_id, item_1, item_2, item_3, item_4, item_5, item_6, item_7, item_8) 
FROM 'C:/LabFiles/msurvey.csv' 
	DELIMITER ',' CSV HEADER;
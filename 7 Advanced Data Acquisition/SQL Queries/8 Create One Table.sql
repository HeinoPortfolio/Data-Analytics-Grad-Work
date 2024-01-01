-- Create the joined table.
DROP TABLE IF EXISTS all_medical_data;

CREATE TABLE all_medical_data AS(
	SELECT latitude,longitude, age, 
	readmis, wmd.gender, wmd.state,
	wmd.county, zip, city,wmd."Patient Income", arthritis, overweight, asthma, diabetes,
	high_blood_pressure, stroke,
	usc.state AS "Census State",
	usc.county AS "Census County", usc.total_pop as "Total Population",
	usc.income AS "Income", usc.men AS "Men Population", 
	usc.women AS "Women Population", usc.unemployment AS "Census Unemployment" 
	FROM wgu_medical_data AS wmd
	Full JOIN US_census_data as usc ON wmd.state = usc.state 
	AND wmd.county = usc.county
);
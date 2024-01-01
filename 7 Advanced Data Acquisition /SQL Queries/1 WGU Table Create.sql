DROP TABLE IF EXISTS wgu_medical_data;
CREATE TABLE wgu_medical_data AS(
	
	SELECT pt.lat AS latitude, pt.lng AS longitude,
		pt.age, pt.readmis, pt.gender, pt.income AS "Patient Income",
		loc.state, loc.county, loc.zip, loc.city,
		serv.arthritis, serv.overweight, serv.asthma, serv.diabetes,
		pt.hignblood AS high_blood_pressure, pt.stroke 
	FROM patient AS pt
	INNER JOIN location AS loc ON pt.location_id = loc.location_id
	INNER JOIN servicesaddon AS serv ON pt.patient_id = serv.patient_id 
	
);
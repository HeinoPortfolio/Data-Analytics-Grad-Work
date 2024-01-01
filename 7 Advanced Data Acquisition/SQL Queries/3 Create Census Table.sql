DROP TABLE IF EXISTS us_census_data;

CREATE TABLE us_census_data(
	countyId integer NOT NULL,
	state text,
	county text,
	total_pop integer,
	men Integer,
	women Integer,
	hispanic float,
	white float,
	black float,
	native float,
	asian float,
	pacific float,
	voting_age_citizen integer,
	income integer,
	income_error integer,
	income_per_cap integer,
	income_per_cap_err integer,
	poverty float,
	child_poverty float,
	professional float,
	service float,
	office float,
	construction float,
	production float,
	drive float,
	carpool float,
	transit float,
	walk float,
	other_trans float,
	workAtHome float,
	mean_Commute float,
	employed float,
	private_Work float,
	public_work float,
	self_employed float,
	family_work float,
	unemployment float,
	
	CONSTRAINT countyId_pkey PRIMARY KEY (countyId)
	
)
	
	
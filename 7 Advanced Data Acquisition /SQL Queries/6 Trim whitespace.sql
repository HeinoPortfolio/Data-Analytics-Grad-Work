-- TRIM whitespace
UPDATE us_census_data
SET county = TRIM (BOTH FROM county);

UPDATE wgu_medical_data
SET county = TRIM (BOTH FROM county);
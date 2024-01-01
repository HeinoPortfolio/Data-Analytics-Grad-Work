-- Change county to a single word
UPDATE us_census_data
SET county = TRIM (TRAILING 'County' FROM county);
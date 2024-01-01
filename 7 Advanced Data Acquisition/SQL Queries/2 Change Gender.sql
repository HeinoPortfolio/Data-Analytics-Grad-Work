-- Change the gender.
UPDATE wgu_medical_data
SET gender = REPLACE (gender,
					 'Prefer not to answer',
					 'Nonbinary');
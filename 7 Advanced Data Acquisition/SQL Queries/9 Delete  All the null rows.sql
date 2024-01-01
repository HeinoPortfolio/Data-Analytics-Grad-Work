-- Drop empty rows from the table.
-- The empty rows will not be needed 
-- in the final part of the assessment.
-- DROP the reows that do not have latitude.

DELETE FROM all_medical_data 
WHERE latitude IS NULL;
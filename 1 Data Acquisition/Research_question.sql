/*  
	Research Query to find the  the importance of 
	factors that matter to patients.
*/
SELECT 
	loc.state,
	ROUND(AVG(surv.item_1), 2) AS ave_item_1,
	ROUND(AVG(surv.item_2), 2) AS ave_item_2,
	ROUND(AVG(surv.item_3),2) AS ave_item_3,
	ROUND(AVG(surv.item_4),2) AS ave_item_4,
	ROUND(AVG(surv.item_5),2) AS ave_item_5,
	ROUND(AVG(surv.item_6),2) AS ave_item_6,
	ROUND(AVG(surv.item_7),2) AS ave_item_7,
	ROUND(AVG(surv.item_8),2) AS ave_item_8
FROM 
	patient AS pat
INNER JOIN location AS loc
	ON loc.location_id = pat.location_id
INNER JOIN surveys_csv as surv
	ON surv.patient_id = pat.patient_id
GROUP BY loc.state
ORDER BY loc.state DESC;


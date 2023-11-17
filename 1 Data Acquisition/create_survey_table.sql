-- Table: public.surveys_csv

DROP TABLE IF EXISTS public.surveys_csv;

CREATE TABLE IF NOT EXISTS public.surveys_csv
(
    patient_id text COLLATE pg_catalog."default" NOT NULL,
    item_1 integer,
    item_2 integer,
    item_3 integer,
    item_4 integer,
    item_5 integer,
    item_6 integer,
    item_7 integer,
    item_8 integer,
    CONSTRAINT surveys_csv_pkey PRIMARY KEY (patient_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.surveys_csv
    OWNER to postgres;
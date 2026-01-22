# NOTE: must keep sample_id for sample-level join (image/text keyed by sample_id)
TAB_KEEP_COLS = {
    "demographics": ["sample_id", "subject_id", "gender", "anchor_age", "anchor_year_group"],

    "admissions": [
        "sample_id",
        "subject_id", "hadm_id", "admittime",
        "admission_type", "admission_location",
        "insurance", "language", "marital_status",
        # "ethnicity", "edregtime", "edouttime"
    ],

    "icustays": ["sample_id", "subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],

    "transfers": ["sample_id", "subject_id", "hadm_id", "eventtype", "careunit", "intime", "outtime"],

    "prescriptions": ["sample_id", "subject_id", "hadm_id", "starttime", "drug_type", "route", "drug"],

    "procedures_icd": ["sample_id", "subject_id", "hadm_id", "chartdate", "icd_code", "icd_version"],

    # leak risk + huge -> do NOT load for now
    # "diagnoses_icd": ["sample_id", "subject_id", "hadm_id", "icd_code", "icd_version"],
}


# NOTE:
# - sample_id kept but NOT cast in dtype (string/object safer)
# - categories are fine, but if you later write to CSV, they become strings anyway
TAB_CFG = {
    "demographics": {
        "usecols": TAB_KEEP_COLS["demographics"],
        "dtype": {
            "subject_id": "int32",
            "gender": "category",
            "anchor_age": "float32",
            "anchor_year_group": "category",
        },
        "parse_dates": None,
    },

    "admissions": {
        "usecols": TAB_KEEP_COLS["admissions"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "admission_type": "category",
            "admission_location": "category",
            "insurance": "category",
            "language": "category",
            "marital_status": "category",
        },
        "parse_dates": ["admittime"],
    },

    "icustays": {
        "usecols": TAB_KEEP_COLS["icustays"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "stay_id": "int32",
            "first_careunit": "category",
        },
        "parse_dates": ["intime", "outtime"],
    },

    "transfers": {
        "usecols": TAB_KEEP_COLS["transfers"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "eventtype": "category",
            "careunit": "category",
        },
        "parse_dates": ["intime", "outtime"],
    },

    "prescriptions": {
        "usecols": TAB_KEEP_COLS["prescriptions"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "drug_type": "category",
            "drug": "category",
            "route": "category",
        },
        "parse_dates": ["starttime"],
    },

    "procedures_icd": {
        "usecols": TAB_KEEP_COLS["procedures_icd"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "icd_code": "category",
            "icd_version": "int8",
        },
        "parse_dates": ["chartdate"],
    },
}


# NOTE: must keep sample_id for sample-level join (image/text keyed by sample_id)
TS_KEEP_COLS = {
    # vitals / bedside measurements
    "chartevents": [
        "sample_id",
        "subject_id", "hadm_id", "stay_id",
        "charttime",
        "itemid",
        "valuenum",      # numeric value (preferred)
        "valueuom",      # optional
    ],

    # labs
    "labevents": [
        "sample_id",
        "subject_id", "hadm_id",
        "charttime",
        "itemid",
        "valuenum",
        "valueuom",
        "ref_range_lower", "ref_range_upper",
        "flag",
    ],

    # inputs: fluids/meds etc.
    "inputevents": [
        "sample_id",
        "subject_id", "hadm_id", "stay_id",
        "starttime", "endtime",
        "itemid",
        "amount", "amountuom",
        "rate", "rateuom",
        "patientweight",
    ],

    # outputs: urine/output volumes etc.
    "outputevents": [
        "sample_id",
        "subject_id", "hadm_id", "stay_id",
        "charttime",
        "itemid",
        "value", "valueuom",
    ],

    # procedures / interventions
    "procedureevents": [
        "sample_id",
        "subject_id", "hadm_id", "stay_id",
        "starttime", "endtime",
        "itemid",
        "value", "valueuom",
        "locationcategory",
    ],
}


# NOTE:
# - sample_id kept but NOT cast
# - outputevents.value / procedureevents.value may be mixed -> do numeric conversion later
TS_CFG = {
    "chartevents": {
        "usecols": TS_KEEP_COLS["chartevents"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "stay_id": "int32",
            "itemid": "int32",
            "valuenum": "float32",
            "valueuom": "category",
        },
        "parse_dates": ["charttime"],
    },

    "labevents": {
        "usecols": TS_KEEP_COLS["labevents"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "itemid": "int32",
            "valuenum": "float32",
            "valueuom": "category",
            "ref_range_lower": "float32",
            "ref_range_upper": "float32",
            "flag": "category",
        },
        "parse_dates": ["charttime"],
    },

    "inputevents": {
        "usecols": TS_KEEP_COLS["inputevents"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "stay_id": "int32",
            "itemid": "int32",
            "amount": "float32",
            "amountuom": "category",
            "rate": "float32",
            "rateuom": "category",
            "patientweight": "float32",
        },
        "parse_dates": ["starttime", "endtime"],
    },

    "outputevents": {
        "usecols": TS_KEEP_COLS["outputevents"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "stay_id": "int32",
            "itemid": "int32",
            "valueuom": "category",
            # value: mixed types -> convert later with pd.to_numeric(errors="coerce")
        },
        "parse_dates": ["charttime"],
    },

    "procedureevents": {
        "usecols": TS_KEEP_COLS["procedureevents"],
        "dtype": {
            "subject_id": "int32",
            "hadm_id": "int32",
            "stay_id": "int32",
            "itemid": "int32",
            "valueuom": "category",
            "locationcategory": "category",
            # value: mixed types -> convert later
        },
        "parse_dates": ["starttime", "endtime"],
    },
}




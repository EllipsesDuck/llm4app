from data.src.prepare import *

from IPython.display import Image # IPython display
pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('float_format', '{:f}'.format)
pd.options.mode.chained_assignment = None  # default='warn'
get_ipython().run_line_magic('matplotlib', 'inline')

core_mimiciv_path = 'data/physionet/files/mimiciv/1.0/'
core_mimiciv_imgcxr_path = 'data/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/'

df_admissions, df_patients, f_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitem, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_d_items, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio, df_noteevents, df_dsnotes, df_ecgnotes, df_echonotes, df_radnotes = load_mimiciv(core_mimiciv_path)

df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])

df = df_d_items
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'chart', 'category': category, 'label': item}, ignore_index=True)
		
df = df_d_labitems
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'lab', 'category': category, 'label': item}, ignore_index=True)
		
df = df_d_hcpcs
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.long_description.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'hcpcs', 'category': category, 'label': item}, ignore_index=True)
		
df_ids = pd.concat([pd.DataFrame(), df_procedureevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_outputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_inputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_icustays[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_datetimeevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_chartevents[['subject_id','hadm_id','stay_id']]], sort=True).drop_duplicates()

df_cxr_ids = pd.concat([pd.DataFrame(), df_mimic_cxr_chexpert[['subject_id']]], sort=True).drop_duplicates()

df_ids = df_ids[df_ids['subject_id'].isin(df_cxr_ids['subject_id'].unique())] 
    
df_ids.to_csv(core_mimiciv_path + 'mimiciv_key_ids.csv', index=False)


print('Unique Subjects: ' + str(len(df_patients['subject_id'].unique())))
print('Unique Subjects/HospAdmissions/Stays Combinations: ' + str(len(df_ids)))
print('Unique Subjects with Chest Xrays Available: ' + str(len(df_cxr_ids)))

   
df_ids = pd.read_csv(core_mimiciv_path + 'mimiciv_key_ids.csv')
print('Unique Records Available: ' + str(len(df_ids)))

nfiles = generate_all_mimiciv_patient_object(df_ids, core_mimiciv_path)
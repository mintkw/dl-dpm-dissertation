# Based on the "Set up and Data Organisation" notebook written by Alexandra Young

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# import statsmodels.formula.api as smf

from config import ADNIMERGE_DIR

# Read in ADNI merge and use it as a basis for joining the spreadsheets.
# This is expected to raise "DtypeWarning: Columns (19,20,21,50,51,104,105,106) have mixed types"
data_adnimerge = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "ADNIMERGE.csv"))

# Convert the problem columns to numeric - note that coercing errors leads to some nans that could be salvaged
data_adnimerge['ABETA'] = pandas.to_numeric(data_adnimerge['ABETA'], errors='coerce')
data_adnimerge['TAU'] = pandas.to_numeric(data_adnimerge['TAU'], errors='coerce')
data_adnimerge['PTAU'] = pandas.to_numeric(data_adnimerge['PTAU'], errors='coerce')
data_adnimerge['ABETA_bl'] = pandas.to_numeric(data_adnimerge['ABETA_bl'], errors='coerce')
data_adnimerge['TAU_bl'] = pandas.to_numeric(data_adnimerge['TAU_bl'], errors='coerce')
data_adnimerge['PTAU_bl'] = pandas.to_numeric(data_adnimerge['PTAU_bl'], errors='coerce')

# Get rid of columns we won't use (not really necessary?)
data_adnimerge = data_adnimerge.drop(columns=['PTETHCAT', 'PTRACCAT', 'PTMARRY', 'CDRSB',
                                              'ADAS11', 'ADAS13', 'ADASQ4', 'RAVLT_immediate',
                                              'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
                                              'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA',
                                              'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                                              'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem',
                                              'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan',
                                              'EcogSPDivatt', 'EcogSPTotal', 'mPACCdigit', 'mPACCtrailsB',
                                              'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl',
                                              'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl',
                                              'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL', 'DIGITSCOR_bl',
                                              'TRABSCOR_bl', 'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl',
                                              'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl',
                                              'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl',
                                              'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl',
                                              'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl'])

# Replace any screening visit codes with baseline
data_adnimerge['VISCODE'] = data_adnimerge['VISCODE'].replace('m0', 'bl')

# Read in ADNI 1, 2 and 3 volumetric data
data_fs43_adni1 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX_11_02_15.csv"))
data_fs51_adni2 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX51_11_08_19.csv"))
data_fs60_adni3 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX6.csv"))

# Read in the labels of the regions from the data dictionary
data_dict = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "DATADIC.csv"))
data_dict_fs43_adni1 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX']
data_dict_fs51_adni2 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX51']
data_dict_fs60_adni3 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX6']

# Remove columns that are not cortical or subcortical volumes, and rename VISCODE columns if necessary
select_columns = data_fs43_adni1.columns[~data_fs43_adni1.columns.str.contains('TS|TA|SA|HS')]
data_fs43_adni1 = data_fs43_adni1[select_columns]

select_columns = data_fs51_adni2.columns[~data_fs51_adni2.columns.str.contains('TS|TA|SA|HS')]
data_fs51_adni2 = data_fs51_adni2[select_columns]
data_fs51_adni2['VISCODE'] = data_fs51_adni2['VISCODE2']
data_fs51_adni2 = data_fs51_adni2.drop(columns=['VISCODE2'])

select_columns = data_fs60_adni3.columns[~data_fs60_adni3.columns.str.contains('TS|TA|SA|HS')]
data_fs60_adni3 = data_fs60_adni3[select_columns]
data_fs60_adni3['VISCODE'] = data_fs60_adni3['VISCODE2']
data_fs60_adni3 = data_fs60_adni3.drop(columns=['VISCODE2'])

# Filter data dictionaries to leave only cortical and subcortical volumes with a shortened "Label" descriptive field.
data_dict_fs43_adni1 = data_dict_fs43_adni1[data_dict_fs43_adni1['FLDNAME'].str.contains('CV|SV')]
data_dict_fs43_adni1['Label'] = data_dict_fs43_adni1['TEXT'].map(lambda x: x.lstrip('Volume (Cortical Parcellation) of|Volume (WM Parcellation) of'))

data_dict_fs51_adni2 = data_dict_fs51_adni2[data_dict_fs51_adni2['FLDNAME'].str.contains('CV|SV')]
data_dict_fs51_adni2['Label'] = data_dict_fs51_adni2['TEXT'].map(lambda x: x.lstrip('Cortical Volume (aparc.stats) of|Subcortical Volume (aseg.stats) of'))

data_dict_fs60_adni3 = data_dict_fs60_adni3[data_dict_fs60_adni3['FLDNAME'].str.contains('CV|SV')]
data_dict_fs60_adni3['Label'] = data_dict_fs60_adni3['TEXT'].map(lambda x: x.lstrip('Cortical Volume (aparc.stats) of|Subcortical Volume (aseg.stats) of'))

# Replace screening visit codes with baseline
data_fs43_adni1['VISCODE'] = data_fs43_adni1['VISCODE'].replace('sc', 'bl')
data_fs43_adni1 = data_fs43_adni1[data_fs43_adni1['VISCODE'] != 'f']
data_fs43_adni1 = data_fs43_adni1[~data_fs43_adni1['VISCODE'].isnull()]

data_fs51_adni2['VISCODE'] = data_fs51_adni2['VISCODE'].replace('scmri', 'bl')
data_fs51_adni2 = data_fs51_adni2[data_fs51_adni2['VISCODE'] != 'nv']
data_fs51_adni2 = data_fs51_adni2[~data_fs51_adni2['VISCODE'].isnull()]

data_fs60_adni3['VISCODE'] = data_fs60_adni3['VISCODE'].replace('sc', 'bl')
data_fs60_adni3 = data_fs60_adni3[data_fs60_adni3['VISCODE'] != 'y1']
data_fs60_adni3 = data_fs60_adni3[data_fs60_adni3['VISCODE'] != 'y2']
data_fs60_adni3 = data_fs60_adni3[data_fs60_adni3['VISCODE'] != 'nv']
data_fs60_adni3 = data_fs60_adni3[~data_fs60_adni3['VISCODE'].isnull()]

# Select cortical regions to sum over left and right
regions_cortical = ['Paracentral', 'Parahippocampal', 'ParsOpercularis', 'ParsOrbitalis',
                    'ParsTriangularis', 'Pericalcarine', 'Postcentral', 'PosteriorCingulate',
                    'Precentral', 'Precuneus', 'RostralAnteriorCingulate', 'RostralMiddleFrontal',
                    'SuperiorFrontal', 'SuperiorParietal', 'SuperiorTemporal', 'Supramarginal',
                    'TemporalPole', 'TransverseTemporal', 'Insula',
                    'Bankssts', 'CaudalAnteriorCingulate', 'CaudalMiddleFrontal',
                    'Cuneus', 'Entorhinal', 'FrontalPole', 'Fusiform',
                    'InferiorParietal', 'InferiorTemporal', 'IsthmusCingulate', 'LateralOccipital',
                    'LateralOrbitofrontal', 'Lingual', 'MedialOrbitofrontal', 'MiddleTemporal']

# Select subcortical regions to sum over left and right
regions_subcortical = ['Accumbens', 'Amygdala', 'Caudate', 'Hippocampus',
                       'Pallidum', 'Putamen', 'Thalamus']

regions_frontal = ['SuperiorFrontal', 'RostralMiddleFrontal', 'CaudalMiddleFrontal',
                   'ParsOpercularis', 'ParsTriangularis', 'ParsOrbitalis',
                   'LateralOrbitofrontal', 'MedialOrbitofrontal',
                   'Precentral', 'Paracentral', 'FrontalPole']
regions_parietal = ['SuperiorParietal', 'InferiorParietal',
                    'Supramarginal', 'Postcentral', 'Precuneus']
regions_temporal = ['SuperiorTemporal', 'MiddleTemporal', 'InferiorTemporal',
                    'Bankssts', 'Fusiform', 'TransverseTemporal',
                    'Entorhinal', 'TemporalPole', 'Parahippocampal']
regions_occipital = ['LateralOccipital', 'Lingual',
                     'Cuneus', 'Pericalcarine']
regions_cingulate = ['PosteriorCingulate', 'RostralAnteriorCingulate', 'CaudalAnteriorCingulate', 'IsthmusCingulate']
regions_insula = ['Insula']

# Sum left and right cortical and subcortical and generate lobar data
data_dicts_fs = [data_dict_fs43_adni1, data_dict_fs51_adni2, data_dict_fs60_adni3]
datasets_fs = [data_fs43_adni1, data_fs51_adni2, data_fs60_adni3]

for i in range(3):
    dataset_fs = datasets_fs[i]
    data_dict_fs = data_dicts_fs[i]

    for region in regions_cortical:
        select_region = data_dict_fs['Label'].str.contains(region)
        temp_values = dataset_fs[data_dict_fs[select_region]['FLDNAME']].to_numpy()
        temp_sum = np.sum(temp_values, axis=1)
        dataset_fs[region] = temp_sum
    for region in regions_subcortical:
        select_region = data_dict_fs['Label'].str.contains(region)
        temp_values = dataset_fs[data_dict_fs[select_region]['FLDNAME']].to_numpy()
        temp_sum = np.sum(temp_values, axis=1)
        dataset_fs[region] = temp_sum

    dataset_fs['Frontal'] = np.sum(dataset_fs[regions_frontal].to_numpy(), axis=1)
    dataset_fs['Parietal'] = np.sum(dataset_fs[regions_parietal].to_numpy(), axis=1)
    dataset_fs['Temporal'] = np.sum(dataset_fs[regions_temporal].to_numpy(), axis=1)
    dataset_fs['Occipital'] = np.sum(dataset_fs[regions_occipital].to_numpy(), axis=1)
    dataset_fs['Cingulate'] = np.sum(dataset_fs[regions_cingulate].to_numpy(), axis=1)

# Merge each dataset with the ADNI merge spreadsheet
data_subset_adni1 = pandas.merge(data_adnimerge, data_fs43_adni1, how="inner",
                                 left_on=['RID', 'VISCODE', 'ICV'], right_on=['RID', 'VISCODE', 'ST10CV'])

data_subset_adni2 = pandas.merge(data_adnimerge, data_fs51_adni2, how="inner",
                                 left_on=['RID', 'VISCODE', 'ICV'], right_on=['RID', 'VISCODE', 'ST10CV'])

data_subset_adni3 = pandas.merge(data_adnimerge, data_fs60_adni3, how="inner",
                                 left_on=['RID', 'VISCODE', 'ICV'], right_on=['RID', 'VISCODE', 'ST10CV'])

# Concatenate ADNI 1, ADNI 2 and ADNI 3 datasets
data = pandas.concat([data_subset_adni1, data_subset_adni2, data_subset_adni3], ignore_index=True)

# Select only those that pass overall quality control
data = data[data['OVERALLQC'] == 'Pass']

# Replace duplicated columns with a single column
data['EXAMDATE'] = data['EXAMDATE_x'].copy()
data['Hippocampus'] = data['Hippocampus_x'].copy()
data['Entorhinal'] = data['Entorhinal_x'].copy()
data['Fusiform'] = data['Fusiform_x'].copy()

# Select data for study
data = data[['RID', 'VISCODE', 'EXAMDATE', 'Years_bl',
             'AGE', 'PTGENDER', 'PTEDUCAT', 'DX', 'MMSE', 'APOE4',
             'FDG', 'PIB', 'AV45', 'FBB', 'ABETA', 'TAU', 'PTAU',
             'OVERALLQC', 'ICV', 'FSVERSION',
             'Frontal', 'Parietal', 'Temporal',
             'Occipital', 'Cingulate', 'Insula',
             'Accumbens', 'Amygdala', 'Caudate', 'Hippocampus',
             'Pallidum', 'Putamen', 'Thalamus']]  # removed 'ABpos'

regions = ['Frontal', 'Parietal', 'Temporal',
           'Occipital', 'Cingulate', 'Insula',
           'Accumbens', 'Amygdala', 'Caudate', 'Hippocampus',
           'Pallidum', 'Putamen', 'Thalamus']

# Keep only rows that have data for all regions
data = data[np.all(~np.isnan(data.loc[:, regions]), axis=1)]

# Keep only rows with ICV within 5 standard deviations of the mean
data = data.loc[np.abs(data['ICV'] - np.mean(data['ICV']) <= 5 * np.std(data['ICV']))]

# Create a new column 'DX_num' that holds 0, 1, or 2 for CN, MCI, Dementia respectively (as stored in column 'DX').
data.loc[data['DX'] == 'CN', 'DX_num'] = 0
data.loc[data['DX'] == 'MCI', 'DX_num'] = 1
data.loc[data['DX'] == 'Dementia', 'DX_num'] = 2

# Drop rows that are missing data for any biomarker (see the 'biomarkers' list below)
biomarkers = ['Frontal', 'Parietal', 'Temporal', 'Amygdala', 'Hippocampus', 'AGE', 'ICV', 'DX_num']
data = data.loc[np.sum(np.isnan(data[biomarkers]), axis=1) == 0]

# Add columns 'FS4', 'FS5', 'FS6' that are boolean flags indicating the FreeSurfer version.
data['FS4'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)'
data['FS5'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (5.1)'
data['FS6'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (6.0)'

# # PROVISIONAL - DATA TRANSFORMATIONS --------------------------------------------------------
# is_control = (data['DX'] == 'CN') & (data['ABpos'] == 0) & (data['VISCODE'] == 'bl')
#
# # make a copy of our dataframe (we don't want to overwrite our original data)
# zdata = pandas.DataFrame(data, copy=True)
#
# # for each region
# for region in regions:
#     mod = smf.ols('%s ~ AGE + ICV + FS4 + FS5 + FS6' % region,
#                   # fit a model finding the effect of age and headsize on biomarker
#                   data=data[is_control]  # fit this model *only* to individuals in the control group
#                   ).fit()  # fit model
#
#     # get the "predicted" values for all subjects based on the control model parameters
#     predicted = mod.predict(data)
#
#     # calculate our zscore: observed - predicted / SD of the control group residuals
#     w_score = (data.loc[:, region] - predicted) / mod.resid.std()
#
#     # save zscore back into our new (copied) dataframe
#     # multiplied by -1 for use with SuStaIn
#     zdata.loc[:, region] = -w_score
# ------------------------------------------------------------------------------------

# Create a spreadsheet of the longitudinal data
data.to_csv(os.path.join(ADNIMERGE_DIR, "adni_longitudinal_data_raw.csv"), index=False)

# zdata.to_csv('Data/longitudinal_data.csv')

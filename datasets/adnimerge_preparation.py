# Based on the "Set up and Data Organisation" notebook written by Alexandra Young

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import statsmodels.formula.api as smf

from config import ADNIMERGE_DIR


if __name__ == "__main__":
    # READ IN ADNIMERGE, WHICH WILL BE USED AS A BASIS FOR JOINING THE SPREADSHEETS. --------------------------------

    # This is expected to raise "DtypeWarning: Columns (19,20,21,50,51,104,105,106) have mixed types"
    data_adnimerge = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "ADNIMERGE.csv"))

    # Convert the problem columns to numeric - note that coercing errors leads to some nans that could be salvaged
    data_adnimerge['ABETA'] = pandas.to_numeric(data_adnimerge['ABETA'], errors='coerce')
    data_adnimerge['TAU'] = pandas.to_numeric(data_adnimerge['TAU'], errors='coerce')
    data_adnimerge['PTAU'] = pandas.to_numeric(data_adnimerge['PTAU'], errors='coerce')
    data_adnimerge['ABETA_bl'] = pandas.to_numeric(data_adnimerge['ABETA_bl'], errors='coerce')
    data_adnimerge['TAU_bl'] = pandas.to_numeric(data_adnimerge['TAU_bl'], errors='coerce')
    data_adnimerge['PTAU_bl'] = pandas.to_numeric(data_adnimerge['PTAU_bl'], errors='coerce')

    # Get rid of columns we won't use, to reduce dataframe size
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

    # Compute amyloid-beta positive or negative labels for each datapoint by fitting a mixture model.
    # Look at CSF Abeta
    data_abeta = data_adnimerge['ABETA'].to_numpy()
    has_abeta = ~np.isnan(data_abeta)
    data_abeta = data_abeta[has_abeta]
    gm = GaussianMixture(n_components=2).fit_predict(data_abeta.reshape(-1, 1))

    if np.mean(data_abeta[gm == 0]) < np.mean(data_abeta[gm == 1]):
        gm = 1 - gm

    is_positive = np.nan * np.ones(data_adnimerge.shape[0])
    is_positive[has_abeta] = gm

    overall_positive = np.nan * np.ones(data_adnimerge.shape[0])
    overall_positive[is_positive == 1] = 1
    overall_positive[is_positive == 0] = 0

    # Look at Amyloid PET
    for abeta_col in ["PIB", "AV45", "FBB"]:
        data_abeta = data_adnimerge['PIB'].to_numpy()
        has_abeta = ~np.isnan(data_abeta)
        data_abeta = data_abeta[has_abeta]
        gm = GaussianMixture(n_components=2).fit_predict(data_abeta.reshape(-1, 1))

        if np.mean(data_abeta[gm == 0]) > np.mean(data_abeta[gm == 1]):
            gm = 1 - gm

        is_positive = np.nan * np.ones(data_adnimerge.shape[0])
        is_positive[has_abeta] = gm

        overall_positive[is_positive == 1] = 1
        overall_positive[is_positive == 0] = 0

    # Add amyloid-positive column to the dataframe
    data_adnimerge['ABpos'] = overall_positive

    # READ IN ADNI 1, 2, AND 3 VOLUMETRIC DATA -----------------------------------------------
    data_fs43_adni1 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX_11_02_15.csv"))
    data_fs51_adni2 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX51_11_08_19.csv"))
    data_fs60_adni3 = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "UCSFFSX6.csv"))

    # READ IN THE LABELS OF THE REGIONS FROM THE DATA DICTIONARY ------------------------------
    data_dict = pandas.read_csv(os.path.join(ADNIMERGE_DIR, "DATADIC.csv"))
    data_dict_fs43_adni1 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX']
    data_dict_fs51_adni2 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX51']
    data_dict_fs60_adni3 = data_dict.loc[data_dict['TBLNAME'] == 'UCSFFSX6']

    # PREPARE ADNI 1, 2, 3 DATA ---------------------------------------------------------------
    # Remove columns that are not cortical or subcortical volumes
    for df in [data_fs43_adni1, data_fs51_adni2, data_fs60_adni3]:
        select_columns = df.columns[~df.columns.str.contains('TS|TA|SA|HS')]
        df = df[select_columns]

    # Also rename VISCODE columns for ADNI 2 and 3
    data_fs51_adni2['VISCODE'] = data_fs51_adni2['VISCODE2']
    data_fs51_adni2 = data_fs51_adni2.drop(columns=['VISCODE2'])

    data_fs60_adni3['VISCODE'] = data_fs60_adni3['VISCODE2']
    data_fs60_adni3 = data_fs60_adni3.drop(columns=['VISCODE2'])

    # Filter data dicts to leave only cortical and subcortical volumes, with a descriptive "Label" column.
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

    # Only for ADNI3, in which ST10CV does not perfectly match ICV stored in ADNIMERGE: Round ST10CV to the nearest 10.
    data_fs60_adni3['ST10CV'] = np.round(data_fs60_adni3['ST10CV'], -1)

    # SUM DATA OVER REGIONS. -------------------------------------------
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

    # Select lobar regions to sum over
    regions_frontal = ['SuperiorFrontal', 'RostralMiddleFrontal', 'CaudalMiddleFrontal',
                       'ParsOpercularis', 'ParsTriangularis', 'ParsOrbitalis',
                       'LateralOrbitofrontal', 'MedialOrbitofrontal',
                       'Precentral', 'Paracentral', 'FrontalPole']

    regions_parietal = ['SuperiorParietal', 'InferiorParietal', 'Supramarginal', 'Postcentral', 'Precuneus']

    regions_temporal = ['SuperiorTemporal', 'MiddleTemporal', 'InferiorTemporal',
                        'Bankssts', 'Fusiform', 'TransverseTemporal',
                        'Entorhinal', 'TemporalPole', 'Parahippocampal']

    regions_occipital = ['LateralOccipital', 'Lingual', 'Cuneus', 'Pericalcarine']

    regions_cingulate = ['PosteriorCingulate', 'RostralAnteriorCingulate',
                         'CaudalAnteriorCingulate', 'IsthmusCingulate']

    regions_insula = ['Insula']  # unused??

    # Sum over left and right cortical and subcortical regions and generate lobar data.
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
    data_subsets = []
    for df in [data_fs43_adni1, data_fs51_adni2, data_fs60_adni3]:
        data_subsets.append(pandas.merge(data_adnimerge, df, how="inner",
                                         left_on=['RID', 'VISCODE', 'ICV'], right_on=['RID', 'VISCODE', 'ST10CV']))

    # Concatenate ADNI 1, ADNI 2 and ADNI 3 datasets
    data = pandas.concat(data_subsets, ignore_index=True)

    # POST-MERGE PROCESSING ---------------------------------------------------------
    # Select only rows that pass overall quality control
    data = data[data['OVERALLQC'] == 'Pass']

    # Replace duplicated columns with a single column
    data['EXAMDATE'] = data['EXAMDATE_x'].copy()
    data['Hippocampus'] = data['Hippocampus_x'].copy()
    data['Entorhinal'] = data['Entorhinal_x'].copy()
    data['Fusiform'] = data['Fusiform_x'].copy()

    # Select data for study
    data = data[['RID', 'VISCODE', 'EXAMDATE', 'Years_bl',
                 'AGE', 'PTGENDER', 'PTEDUCAT', 'DX', 'MMSE', 'APOE4', 'ABpos',
                 'FDG', 'PIB', 'AV45', 'FBB', 'ABETA', 'TAU', 'PTAU',
                 'OVERALLQC', 'ICV', 'FSVERSION',
                 'Frontal', 'Parietal', 'Temporal',
                 'Occipital', 'Cingulate', 'Insula',
                 'Accumbens', 'Amygdala', 'Caudate', 'Hippocampus',
                 'Pallidum', 'Putamen', 'Thalamus']]

    regions = ['Frontal', 'Parietal', 'Temporal',
               'Occipital', 'Cingulate', 'Insula',
               'Accumbens', 'Amygdala', 'Caudate', 'Hippocampus',
               'Pallidum', 'Putamen', 'Thalamus']

    # Keep only rows that have data for all regions
    data = data[np.all(~np.isnan(data.loc[:, regions]), axis=1)]

    # Keep only rows with ICV within 5 standard deviations of the mean
    data = data.loc[np.abs(data['ICV'] - np.mean(data['ICV']) <= 5 * np.std(data['ICV']))]

    # Add additional columns 'FS4', 'FS5', and 'FS6' that are boolean types indicating the FreeSurfer version.
    data['FS4'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)'
    data['FS5'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (5.1)'
    data['FS6'] = data['FSVERSION'] == 'Cross-Sectional FreeSurfer (6.0)'

    # Create 3 new columns to one-hot encode CN, MCI, Dementia (as stored in column 'DX').
    data['CN'] = (data['DX'] == 'CN').astype(int)
    data['MCI'] = (data['DX'] == 'MCI').astype(int)
    data['AD'] = (data['DX'] == 'Dementia').astype(int)

    # Drop rows that are missing data for any of the following columns
    biomarkers = ['Frontal', 'Parietal', 'Temporal', 'Amygdala', 'Hippocampus', 'ICV']
    data = data.loc[np.sum(np.isnan(data[biomarkers]), axis=1) == 0]

    # Drop rows with no DX information
    data = data[~data['DX'].isnull()]

    # # Replacement: drop rows that are missing data for any field.
    # data = data.loc[np.all(~np.isnan(data), axis=1)]

    # # DATA TRANSFORMATIONS --------------------------------------------------------
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
    #
    # data = zdata  # We only need the transformed data.
    # # ------------------------------------------------------------------------------------

    # EXPERIMENT: Z SCORE TRANSFORMATIONS ---------------------------------------------------
    zdata = pandas.DataFrame(data, copy=True)
    # print(np.mean(zdata.loc[:, regions], axis=1).shape)
    # print(np.std(zdata.loc[:, regions], axis=1).shape)
    print(data.loc[:, regions].shape)
    for region in regions:
        zdata.loc[:, region] = -(zdata.loc[:, region] - np.mean(zdata.loc[:, region])) / np.std(zdata.loc[:, region])
    data = zdata
    # ---------------------------------------------------------------------------------------

    # todo: below is for an experiment - keeping only what i believe is volumetric data, and diagnosis for stratified split.
    data = data.loc[:, regions + ['CN', 'MCI', 'AD']]

    # Split the dataset into training and testing sets.
    CN_data = data[data['CN'] == 1]
    MCI_data = data[data['MCI'] == 1]
    AD_data = data[data['AD'] == 1]

    shuffled_idx_CN = np.random.permutation(len(CN_data))
    shuffled_idx_MCI = np.random.permutation(len(MCI_data))
    shuffled_idx_AD = np.random.permutation(len(AD_data))

    train_ratio = 0.8

    train_data = pandas.concat([CN_data.iloc[shuffled_idx_CN[:int(train_ratio * len(CN_data))]],
                                MCI_data.iloc[shuffled_idx_MCI[:int(train_ratio * len(MCI_data))]],
                                AD_data.iloc[shuffled_idx_AD[:int(train_ratio * len(AD_data))]]
                                ],
                               ignore_index=True)

    test_data = pandas.concat([CN_data.iloc[shuffled_idx_CN[int(train_ratio * len(CN_data)):]],
                               MCI_data.iloc[shuffled_idx_MCI[int(train_ratio * len(MCI_data)):]],
                               AD_data.iloc[shuffled_idx_AD[int(train_ratio * len(AD_data)):]]
                               ],
                              ignore_index=True)

    # print(train_data.shape, test_data.shape, data.shape)

    # Save the longitudinal data to a spreadsheet.
    # print(data.shape)
    print(data.columns.tolist())
    train_data.to_csv(os.path.join(ADNIMERGE_DIR, "adni_longitudinal_data_train.csv"), index=False)
    test_data.to_csv(os.path.join(ADNIMERGE_DIR, "adni_longitudinal_data_test.csv"), index=False)

    # print(np.mean(zdata.loc[is_control, regions].values, axis=0))
    # print(np.min(zdata.loc[is_control, regions].values, axis=0), np.max(zdata.loc[is_control, regions].values, axis=0))
    print(np.mean(zdata.loc[zdata['DX'] == 'CN', regions].values, axis=0))
    print(np.min(zdata.loc[zdata['DX'] == 'CN', regions].values, axis=0), np.max(zdata.loc[zdata['DX'] == 'CN', regions].values, axis=0))
    print(np.mean(zdata.loc[zdata['DX'] == 'Dementia', regions].values, axis=0))
    print(np.min(zdata.loc[zdata['DX'] == 'Dementia', regions].values, axis=0), np.max(zdata.loc[zdata['DX'] == 'Dementia', regions].values, axis=0))

    # data.to_csv(os.path.join(ADNIMERGE_DIR, "adni_longitudinal_data_raw.csv"), index=False)
    # zdata.to_csv(os.path.join(ADNIMERGE_DIR, "adni_longitudinal_data.csv"), index=False)

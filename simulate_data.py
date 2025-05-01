from dataset_simulation import simulateDPMdata, simulateEBMdata
import numpy as np
import os
import json


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    print("Seed set to", seed)

    # define a common configuration to use for both models.
    seq = [[3], [2], [0], [4], [1]]
    n_biomarkers = 5
    n_mci = 200
    n_controls = 200
    n_patients = 200
    means_normal = np.zeros(n_biomarkers)
    means_abnormal = np.ones(n_biomarkers)
    sds_normal = 0.05 * np.ones(n_biomarkers)
    sds_abnormal = 0.05 * np.ones(n_biomarkers)
    biomarker_labels = [chr(ord('A') + i) for i in range(n_biomarkers)]

    data_dir = os.path.join("data", "simulated")
    os.makedirs(data_dir, exist_ok=True)

    dpm_df, dpm_ks_mci, plot_vars = simulateDPMdata.simulateDPMdata(seq=seq,
                                                                    n_biomarkers=n_biomarkers,
                                                                    n_mci=n_mci,
                                                                    n_controls=n_controls,
                                                                    n_patients=n_patients,
                                                                    means_normal=means_normal,
                                                                    means_abnormal=means_abnormal,
                                                                    sds_normal=sds_normal,
                                                                    sds_abnormal=sds_abnormal,
                                                                    biomarker_labels=biomarker_labels)

    file_path = os.path.join(data_dir, f"dpm_synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}")  # without extension
    dpm_df.to_csv(file_path + ".csv", index=False)
    with open(file_path + ".json", 'w') as f:
        json.dump(list(dpm_ks_mci), f)

    # print("Stages of the MCI datapoints generated with the general DPM model:", dpm_ks_mci)
    # print(plot_vars)

    ebm_df, ebm_ks_mci = simulateEBMdata.simulateEBMdata(seq=seq,
                                                         n_mci=n_mci,
                                                         means_normal=means_normal,
                                                         means_abnormal=means_abnormal,
                                                         sds_normal=sds_normal,
                                                         sds_abnormal=sds_abnormal,
                                                         biomarker_labels=biomarker_labels,
                                                         force_uniform_stages=True,
                                                         plot=False,
                                                         n_controls=n_controls,
                                                         n_patients=n_patients)

    file_path = os.path.join(data_dir, f"ebm_synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}")

    ebm_df.to_csv(file_path + ".csv", index=False)
    # print("Stages of the MCI datapoints generated with the EBM model:", ebm_ks_mci)

    with open(file_path + ".json", 'w') as f:
        json.dump([float(k) for k in list(ebm_ks_mci)], f)

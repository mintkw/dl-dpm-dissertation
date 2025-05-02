from dataset_simulation import simulateDPMdata, simulateEBMdata
import numpy as np
import os
import json
from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    print("Seed set to", seed)

    # define a common configuration to use for both models.
    # seq = [[3], [2], [0], [4], [1]]
    n_biomarkers = 100
    seq = np.arange(n_biomarkers)[:, None]
    seq = np.random.permutation(seq).tolist()  # uniformly sample sequence
    n_mci = 1000
    n_controls = 1000
    n_patients = 1000
    means_normal = np.zeros(n_biomarkers) + 0.5
    means_abnormal = np.ones(n_biomarkers) + 0.5
    sds_normal = 0.05 * np.ones(n_biomarkers)
    sds_abnormal = 0.05 * np.ones(n_biomarkers)
    # biomarker_labels = [chr(ord('A') + i) for i in range(n_biomarkers)]
    biomarker_labels = [i for i in range(n_biomarkers)]

    os.makedirs(SIMULATED_OBSERVATIONS_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABELS_DIR, exist_ok=True)

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

    file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_dpm"  # without extension
    dpm_df.to_csv(os.path.join(SIMULATED_OBSERVATIONS_DIR, file_name + ".csv"), index=False)
    with open(os.path.join(SIMULATED_LABELS_DIR, file_name + "_stages.json"), 'w') as f:
        json.dump(list(dpm_ks_mci), f)

    with open(os.path.join(SIMULATED_LABELS_DIR, file_name + "_seq.json"), 'w') as f:
        json.dump(seq, f)

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

    file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_ebm"  # without extension
    ebm_df.to_csv(os.path.join(SIMULATED_OBSERVATIONS_DIR, file_name + ".csv"), index=False)
    with open(os.path.join(SIMULATED_LABELS_DIR, file_name + "_stages.json"), 'w') as f:
        json.dump([float(k) for k in list(ebm_ks_mci)], f)

    with open(os.path.join(SIMULATED_LABELS_DIR, file_name + "_seq.json"), 'w') as f:
        json.dump(seq, f)

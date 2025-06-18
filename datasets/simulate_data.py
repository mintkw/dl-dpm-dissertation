from datasets import simulateDPMdata, simulateEBMdata
import numpy as np
import os
import json
from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR


if __name__ == "__main__":
    # seed = 0
    # np.random.seed(seed)
    # print("Seed set to", seed)

    # define a common configuration to use for both models.
    n_biomarkers = 50
    n_mci = 1000
    n_controls = 100
    n_patients = 100
    means_normal = np.zeros(n_biomarkers)
    means_abnormal = np.ones(n_biomarkers)
    # sds_normal = 0.05 * np.ones(n_biomarkers)
    # sds_abnormal = 0.05 * np.ones(n_biomarkers)
    sds_normal = np.zeros(n_biomarkers) + 0.1
    sds_abnormal = np.zeros(n_biomarkers) + 0.1
    # biomarker_labels = [chr(ord('A') + i) for i in range(n_biomarkers)]
    biomarker_labels = [i for i in range(n_biomarkers)]

    cwd = os.getcwd()
    os.chdir("..")
    os.makedirs(SIMULATED_OBSERVATIONS_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABELS_DIR, exist_ok=True)
    os.chdir(cwd)

    # generate num_sets of datasets with their own sequences.
    num_sets = 1

    for i in range(num_sets):
        seq = np.arange(n_biomarkers)[:, None]
        seq = np.random.permutation(seq).tolist()  # uniformly sample sequence

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

        file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_dpm_{i}"  # without extension
        dpm_df.to_csv(os.path.join("..", SIMULATED_OBSERVATIONS_DIR, file_name + ".csv"), index=False)
        with open(os.path.join("..", SIMULATED_LABELS_DIR, file_name + "_stages.json"), 'w') as f:
            # have to add on stage information for CN and AD - 0.0 and 1.0, i assume
            json.dump([0.0 for _ in range(n_controls)]
                      + [float(k) for k in list(dpm_ks_mci)]
                      + [float(n_biomarkers) for _ in range(n_patients)], f)

        with open(os.path.join("..", SIMULATED_LABELS_DIR, file_name + "_seq.json"), 'w') as f:
            json.dump(seq, f)

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

        file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_ebm_{i}"  # without extension
        ebm_df.to_csv(os.path.join("..", SIMULATED_OBSERVATIONS_DIR, file_name + ".csv"), index=False)
        with open(os.path.join("..", SIMULATED_LABELS_DIR, file_name + "_stages.json"), 'w') as f:
            # have to add on stage information for CN and AD - 0.0 and num_biomarkers, i assume
            json.dump([0.0 for _ in range(n_controls)] +
                      [float(k) for k in list(ebm_ks_mci)] +
                      [float(n_biomarkers) for _ in range(n_patients)], f)

        with open(os.path.join("..", SIMULATED_LABELS_DIR, file_name + "_seq.json"), 'w') as f:
            json.dump(seq, f)

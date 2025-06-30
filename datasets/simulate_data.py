from datasets import simulateDPMdata, simulateEBMdata
import numpy as np
import os
import json
from config import SIMULATED_OBS_VAL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR


if __name__ == "__main__":
    # seed = 0
    # np.random.seed(seed)
    # print("Seed set to", seed)

    # define a common configuration to use for both models.
    n_biomarkers = 10
    n_mci = 100
    n_controls = 10
    n_patients = 10
    # means_normal = np.array([0, 0, 1, 1, 2])
    # means_abnormal = np.array([1, 1, 2, 3, 0])
    means_normal = np.zeros(n_biomarkers)
    means_abnormal = np.concatenate([np.ones(5), np.ones(5) + 1])
    sds_normal = 0.05 * np.ones(n_biomarkers)
    sds_abnormal = 0.05 * np.ones(n_biomarkers)
    # sds_normal = np.concatenate([0.05 * np.ones(15), 0.1 * np.ones(15)])
    # sds_abnormal = np.concatenate([0.05 * np.ones(15), 0.1 * np.ones(15)])
    # biomarker_labels = [chr(ord('A') + i) for i in range(n_biomarkers)]
    biomarker_labels = [str(i) for i in range(n_biomarkers)]

    cwd = os.getcwd()
    os.chdir("..")
    os.makedirs(SIMULATED_OBS_TRAIN_DIR, exist_ok=True)
    os.makedirs(SIMULATED_OBS_VAL_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABEL_TRAIN_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABEL_VAL_DIR, exist_ok=True)
    os.chdir(cwd)

    # generate num_sets of datasets with their own sequences.
    num_sets = 1

    for i in range(num_sets):
        # uniformly sample a sequence from all possible permutations
        seq = np.arange(n_biomarkers)[:, None]
        seq = np.random.permutation(seq).tolist()

        obs_directories = [SIMULATED_OBS_VAL_DIR, SIMULATED_OBS_TRAIN_DIR]
        label_directories = [SIMULATED_LABEL_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR]

        # for both EBM and DPM settings, generate a train dataset, then a val dataset
        for set in range(2):
            dpm_df, dpm_ks_mci, plot_vars = simulateDPMdata.simulateDPMdata(seq=seq,
                                                                            n_biomarkers=n_biomarkers,
                                                                            n_mci=n_mci,
                                                                            n_controls=n_controls,
                                                                            n_patients=n_patients,
                                                                            means_normal=means_normal,
                                                                            means_abnormal=means_abnormal,
                                                                            sds_normal=sds_normal,
                                                                            sds_abnormal=sds_abnormal,
                                                                            plot=True,
                                                                            biomarker_labels=biomarker_labels)

            file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_dpm_{i}"  # without extension
            dpm_df.to_csv(os.path.join("..", obs_directories[set], file_name + ".csv"), index=False)
            with open(os.path.join("..", label_directories[set], file_name + "_stages.json"), 'w') as f:
                # add on stage information for CN and AD - 0.0 and num_biomarkers
                json.dump([0.0 for _ in range(n_controls)]
                          + [float(k) for k in list(dpm_ks_mci)]
                          + [float(n_biomarkers) for _ in range(n_patients)], f)

            with open(os.path.join("..", label_directories[set], file_name + "_seq.json"), 'w') as f:
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
            ebm_df.to_csv(os.path.join("..", obs_directories[set], file_name + ".csv"), index=False)
            with open(os.path.join("..", label_directories[set], file_name + "_stages.json"), 'w') as f:
                # add on stage information for CN and AD - 0.0 and num_biomarkers
                json.dump([0.0 for _ in range(n_controls)] +
                          [float(k) for k in list(ebm_ks_mci)] +
                          [float(n_biomarkers) for _ in range(n_patients)], f)

            with open(os.path.join("..", label_directories[set], file_name + "_seq.json"), 'w') as f:
                json.dump(seq, f)

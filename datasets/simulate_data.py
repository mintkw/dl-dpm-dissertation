from datasets import simulateDPMdata, simulateEBMdata
import numpy as np
import os
import json
from config import SIMULATED_OBS_TEST_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR, ROOT_DIR


def generate_normalised_data(n_biomarkers, n_mci, n_controls, n_patients, num_sets=1, means_normal=None, means_abnormal=None,
                             sds_normal=None, sds_abnormal=None, biomarker_labels=None, onsets_stages=None):
    # Generate and save data for `n_biomarkers` biomarkers with all normal levels of 0 and abnormal levels of 1, by default.

    # Default arguments:
    if means_normal is None:
        means_normal = np.zeros(n_biomarkers)
    if means_abnormal is None:
        means_abnormal = np.ones(n_biomarkers)
    if sds_normal is None:
        sds_normal = 0.05 * np.ones(n_biomarkers)
    if sds_normal is None:
        sds_abnormal = 0.05 * np.ones(n_biomarkers)
    if biomarker_labels is None:
        # biomarker_labels = [chr(ord('A') + i) for i in range(n_biomarkers)]
        biomarker_labels = [str(i) for i in range(n_biomarkers)]

    os.chdir(ROOT_DIR)
    os.makedirs(SIMULATED_OBS_TRAIN_DIR, exist_ok=True)
    os.makedirs(SIMULATED_OBS_TEST_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABEL_TRAIN_DIR, exist_ok=True)
    os.makedirs(SIMULATED_LABEL_TEST_DIR, exist_ok=True)

    # generate num_sets of datasets with their own sequences.
    for i in range(num_sets):
        # uniformly sample a sequence from all possible permutations
        seq = np.arange(n_biomarkers)[:, None]
        seq = np.random.permutation(seq).tolist()

        obs_directories = [SIMULATED_OBS_TEST_DIR, SIMULATED_OBS_TRAIN_DIR]
        label_directories = [SIMULATED_LABEL_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR]

        # for the DPM setting, generate a train dataset, then a val dataset
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
                                                                            plot=False,
                                                                            biomarker_labels=biomarker_labels,
                                                                            onsets_stages=onsets_stages)

            file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_{i}"  # without extension
            dpm_df.to_csv(os.path.join(obs_directories[set], file_name + ".csv"), index=False)
            with open(os.path.join(label_directories[set], file_name + "_stages.json"), 'w') as f:
                # add on stage information for CN and AD - 0.0 and num_biomarkers
                json.dump([0.0 for _ in range(n_controls)]
                          + [float(k) for k in list(dpm_ks_mci)]
                          + [float(n_biomarkers) for _ in range(n_patients)], f)

            with open(os.path.join(label_directories[set], file_name + "_seq.json"), 'w') as f:
                json.dump(seq, f)

            # ebm_df, ebm_ks_mci = simulateEBMdata.simulateEBMdata(seq=seq,
            #                                                      n_mci=n_mci,
            #                                                      means_normal=means_normal,
            #                                                      means_abnormal=means_abnormal,
            #                                                      sds_normal=sds_normal,
            #                                                      sds_abnormal=sds_abnormal,
            #                                                      biomarker_labels=biomarker_labels,
            #                                                      force_uniform_stages=True,
            #                                                      plot=False,
            #                                                      n_controls=n_controls,
            #                                                      n_patients=n_patients)
            #
            # file_name = f"synthetic_{n_mci + n_patients + n_controls}_{n_biomarkers}_ebm_{i}"  # without extension
            # ebm_df.to_csv(os.path.join(obs_directories[set], file_name + ".csv"), index=False)
            # with open(os.path.join(label_directories[set], file_name + "_stages.json"), 'w') as f:
            #     # add on stage information for CN and AD - 0.0 and num_biomarkers
            #     json.dump([0.0 for _ in range(n_controls)] +
            #               [float(k) for k in list(ebm_ks_mci)] +
            #               [float(n_biomarkers) for _ in range(n_patients)], f)
            #
            # with open(os.path.join(label_directories[set], file_name + "_seq.json"), 'w') as f:
            #     json.dump(seq, f)


if __name__ == "__main__":
    # seed = 0
    # np.random.seed(seed)
    # print("Seed set to", seed)

    # Define a common configuration to use for both saved_models.
    n_biomarkers = 10
    n_mci = 100
    n_controls = 10
    n_patients = 10
    num_sets = 10

    onsets_stages = np.sort(np.random.rand(n_biomarkers))  # randomly determine

    generate_normalised_data(n_biomarkers=n_biomarkers, n_mci=n_mci, n_controls=n_controls, n_patients=n_patients, num_sets=num_sets,
                             onsets_stages=onsets_stages)


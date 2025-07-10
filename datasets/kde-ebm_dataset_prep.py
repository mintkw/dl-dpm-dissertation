import csv
import os
import pandas as pd
import numpy as np
from config import SIMULATED_OBS_DIR, SIMULATED_LABEL_DIR


def prepare_csv_for_kde_ebm(csv_path, suffix):
    # suffix: to add to the end of prepared dataset csv files

    output_path = os.path.join(csv_path.split('.csv')[0]) + f"_{suffix}.csv"

    df = pd.read_csv(csv_path)
    N = df.shape[0]  # number of datapoints
    num_biomarkers = df.shape[1] - 3   # this depends on the last three columns being 'CN', 'MCI', 'AD'

    with open(output_path, 'w', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow([N, num_biomarkers, 'CN', 'MCI', 'AD'])

        for i in range(N):
            row = list(df.iloc[i])
            measurements = row[:-3]
            label = np.argmax(row[-3:])

            # for kde-ebm, it should be 0 for CN, 1 for AD, 2 for MCI so labels need to be corrected
            if label == 1:
                label = 2
            elif label == 2:
                label = 1

            measurements.append(label)
            writer.writerow(measurements)


if __name__ == "__main__":
    # Converts all dataset csv files in the synthetic data directory that do not have the prepared data suffix
    suffix = "kde-ebm"
    simulated_obs_dir = os.path.join("..", SIMULATED_OBS_DIR)

    for filename in os.listdir(simulated_obs_dir):
        split_filename = os.path.splitext(filename)
        if split_filename[-1] == ".csv" and split_filename[0].split('_')[-1] != suffix:
            prepare_csv_for_kde_ebm(os.path.join(simulated_obs_dir, filename), suffix=suffix)

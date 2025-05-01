import csv
import os
import pandas as pd
import numpy as np
from config import DATA_DIR


def prepare_csv_for_kde_ebm(csv_path):
    output_path = os.path.join(csv_path.split('.csv')[0]) + '_kde-ebm.csv'

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
            measurements.append(label)
            writer.writerow(measurements)


simulated_data_dir = os.path.join("..", DATA_DIR, "simulated")
prepare_csv_for_kde_ebm(os.path.join("..", DATA_DIR, ) "../ebm_synthetic_600_5.csv")
prepare_csv_for_kde_ebm("../dpm_synthetic_600_5.csv")

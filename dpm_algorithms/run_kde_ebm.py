# Original code by Nicholas C. Firth <ncfirth87@gmail.com>, modified by Kwan Wynn Tan
# License: GPL-3.0

from matplotlib import pyplot as plt
import os
import time
import json
import numpy as np
import pandas as pd

from datasets.kde_ebm_dataset_prep import prepare_csv_for_kde_ebm
from kde_ebm import mixture_model, mcmc, plotting, datasets
from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, ADNI_DIR
from dpm_algorithms.evaluation import evaluate_sequence


def run_kde_ebm(file_dir, file_name, greedy_n_init=50, plot=False):
    # Get our biomarker data X, numeric disease labels y, names for each
    # numeric disease label cname, and column names of our biomarker data
    # bmname.

    # get dataset
    biomarker_names = pd.read_csv(str(os.path.join(file_dir, file_name + ".csv"))).columns[:-3].tolist()
    obs_path = os.path.join(file_dir, file_name + "_kde-ebm.csv")

    X, y, _, cname = datasets.load_synthetic(obs_path)

    time0 = time.time()

    # Fit GMM/KDE for each biomarker and plot the results
    mixture_models = mixture_model.fit_all_kde_models(X, y)
    # mixture_models = mixture_model.fit_all_gmm_models(X, y)

    if plot:
        fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                              score_names=biomarker_names,
                                              class_names=cname)
        fig.show()

    # Now we fit our disease sequence, using greedy ascent followed by
    # MCMC optimisation
    print("Fitting disease sequence with greedy ascent and MCMC")
    res = mcmc.mcmc(X, mixture_models, n_iter=10000,
                    greedy_n_iter=10000, greedy_n_init=greedy_n_init, plot=plot)

    # Then plot these using a positional variance diagram to visualise
    # any uncertainty in the sequence
    if plot:
        fig, ax = plotting.mcmc_uncert_mat(res, score_names=biomarker_names)
        fig.show()

    res.sort(reverse=True)
    ml_order = res[0]

    # Finally we can stage all our participants using the fitted EBM
    print("Staging participants with the fitted EBM")
    prob_mat = mixture_model.get_prob_mat(X, mixture_models)
    stages, stages_like = ml_order.stage_data(prob_mat)

    if plot:
        fig, ax = plotting.stage_histogram(stages, y)

    time_taken = time.time() - time0
    print(f"Time taken to run script: {time_taken} seconds.")

    if plot:
        plt.show()

    return ml_order.ordering, time_taken, stages


if __name__ == '__main__':
    # Run KDE-EBM on a specified dataset
    dataset_name = "synthetic_3000_250_0"
    file_dir = SIMULATED_OBS_TRAIN_DIR

    # Create a copy of the dataset formatted for use with the KDE-EBM
    prepare_csv_for_kde_ebm(os.path.join(file_dir, dataset_name + ".csv"), suffix="kde-ebm")

    ml_order, time_taken, stages = run_kde_ebm(file_dir=SIMULATED_OBS_TRAIN_DIR, file_name=dataset_name, plot=True)
    print(ml_order)

    # Read in ground truth sequence
    seq_label_file = os.path.join(SIMULATED_LABEL_TRAIN_DIR, dataset_name + "_seq.json")
    with open(seq_label_file, 'r') as f:
        seq_gt = json.load(f)

    seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists
    print(evaluate_sequence(seq_gt, ml_order))

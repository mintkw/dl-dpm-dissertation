# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from matplotlib import pyplot as plt
import os
import time
import json

from kde_ebm import mixture_model
from kde_ebm import mcmc
from kde_ebm import plotting
from kde_ebm import datasets
from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR


def main():
    # Get our biomarker data X, numeric disease labels y, names for each
    # numeric disease label cname, and column names of our biomarker data
    # bmname.

    # filename = "synthetic_1500_5_ebm"
    filename = "synthetic_3000_100_dpm"
    obs_path = os.path.join(SIMULATED_OBSERVATIONS_DIR, filename + "_kde-ebm.csv")

    X, y, bmname, cname = datasets.load_synthetic(obs_path)

    time0 = time.time()

    # Fit GMM for each biomarker and plot the results
    # print("Fitting KDE for each biomarker")
    mixture_models = mixture_model.fit_all_kde_models(X, y)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                          score_names=bmname,
                                          class_names=cname)
    fig.show()

    # Now we fit our disease sequence, using greedy ascent followed by
    # MCMC optimisation
    print("Fitting disease sequence with greedy ascent and MCMC")
    res = mcmc.mcmc(X, mixture_models, n_iter=10000,
                    greedy_n_iter=5000, greedy_n_init=10)

    # Then plot these using a positional variance diagram to visualise
    # any uncertainty in the sequence
    fig, ax = plotting.mcmc_uncert_mat(res, score_names=bmname)
    fig.show()

    res.sort(reverse=True)
    ml_order = res[0]
    print(f"ML ordering estimate: {ml_order.ordering}")

    label_path = os.path.join(SIMULATED_LABELS_DIR, filename + "_seq.json")
    with open(label_path, 'r') as f:
        print(f"Ground truth ordering: {json.load(f)}")

    # Finally we can stage all our participants using the fitted EBM
    print("Staging participants with the fitted EBM")
    prob_mat = mixture_model.get_prob_mat(X, mixture_models)
    stages, stages_like = ml_order.stage_data(prob_mat)

    fig, ax = plotting.stage_histogram(stages, y, )

    time_taken = time.time() - time0
    print(f"Time taken to run script: {time_taken} seconds.")
    plt.show()


if __name__ == '__main__':
    import numpy
    numpy.random.seed(42)
    main()

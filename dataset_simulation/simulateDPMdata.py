# Credit to Christopher Parker

def simulateDPMdata(seq=None, n_biomarkers=2, n_mci=0, sample_res=None, stage_res=None, n_controls=0,
                    n_patients=0, means_normal=None, means_abnormal=None, sds_normal=None,
                    sds_abnormal=None, biomarker_labels=None, label_points=True, permutation=[],
                    gradients=None, onsets=None, plot=False, plot_hist=False, plot_points=True,
                    plot_legend=True, plot_lwd=1.5, plot_nlines=10, set_seed=False, set_cn_seed=False,
                    set_mci_seed=False, set_pt_seed=False, fix_n_mci=False, force_mci_time=True,
                    jitter_times=None, model_gt_string='-GT', legend_title=None, xlabel='Disease progression',
                    ylabel='Biomarker', plot_lines=True, ylims=None, plot_plot=True, colors_mci=None, verbose=False):
    """
    This is an updated version of simulateDPMdata_orig, with constant 0 - 1 disease time axis & added functionality (samples_res & stage_res).

    Params:
        seq (list of lists): if specified, overrides n_biomarkers and onset.
        n_biomarkers (int): number of biomarkers.
        n_mci (int):
        sample_res (float?): the sampling resolution (time between MCI subject samples). If specified, overrides n_mci.
        stage_res (float?): the stage temporal 'resolution' (time between the sets of events in each respective stage). If specified, overrides onsets.
        n_controls (int): number of control subjects?
        n_patients (int): number of patient subjects?
        means_normal (np.array): means of each biomarker when normal. 0 by default.
        means_abnormal (np.array): means of each biomarker when abnormal. 1 by default.
        sds_normal (np.array): standard deviation of each biomarker when normal. 0.05 by default.
        sds_abnormal (np.array): standard deviation of each biomarker when abnormal. 0.05 by default.
        biomarker_labels
        label_points
        permutation: permutation of biomarkers. (when would we use this?)
        gradients
        onsets: onset of each biomarker.
            the function header comment said "onsets_in_list - elements must all be between 0 and 1". maybe that is this?.
            by default seems to be spaced evenly through the time frame in biomarker order.

        plot_lwd (float?):
        fix_n_mci:
        force_mci_time (bool): If sample_res not provided, forces MCI to be equally spaced in time. Else the MCI times are randomly sampled from (0,1)
        jitter_times:

        skipping over arguments that seem to be flags for plotting, plot settings, etc.

    Returns:
        (pd.DataFrame) 

    """
    # all inputs should be numpy arrays
    # mean of normal is 0
    # mean of abnormal is 1
    # timesrange is 2-dimensional np array | (kw) this doesn't get mentioned anywhere else?
    # tofix: mixture of lists and numpy arrays in inputs (e.g., means and sds)

    # Modules
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import itertools
    import pandas as pd
    import string
    import itertools
    import copy
    import pdb

    # Functions
    def sigmoid(t, a, b):
        # b - onset time
        # a - gradient
        # (kw) returns the level of a biomarker at time t given how steep the curve is (a) and the onset time (b)?
        return 1 / (1 + np.exp(-a * (t - b)))


    # Random-ness
    if set_seed:
        np.random.seed(0)

    # Disease progression time
    max_time = 1

    # MCI times
    if sample_res!=None: 
        # sample_res requested
        if sample_res > 1:
            # exit if sample res error
            print('sample_res cannot be greater than 1 (the maximum disease time) - exiting')
            exit()
        #
        # n_tp: number of timepoints to achieve effective sample_res
        # (the requested sample_res may not fit in neatly into the time frame - effective sample res always does)
        n_tp = int(np.floor(max_time / sample_res) + 1)
        sample_res_effective = 1/(n_tp-1)
        if verbose:
            print('verbose=' + str(verbose))
            print('requested sample_res: ' + str(sample_res))
            print('effective sample_res: ' + str(sample_res_effective))
        # fix the number of MCI subjects?
        if fix_n_mci:
            # exit if n_mci requested is too low for the effective sample_res
            if (n_mci < n_tp):
                print('n_mci < n_tp for the requested sample_res - exiting')
                exit()
            # generate mci along with fixed sample (time) resolution
            rep_factor = np.ceil(n_mci/n_tp)
            # breakpoint()
            times = np.repeat(np.linspace(start=0, stop=max_time, num=n_tp),rep_factor)[:n_mci]
        else:
            # mci is set equal to the number of timepoints (one mci per sample in time)
            n_mci = copy.deepcopy(n_tp)
            times = np.linspace(start=0, stop=max_time, num=n_mci)
    else:
        if force_mci_time:
            # no sample_res provided - sampling MCI equally in time
            times = np.linspace(start=0, stop=max_time, num=n_mci)
        else:
            # no sample_res provided - sampling MCI randomly in time
            times = np.random.uniform(low=0,high=1,size=n_mci)
            
    # Disease progression times of MCI
    # times = np.linspace(start=0, stop=max_time, num=n_mci)
    

    # # MCI times
    # if sample_res!=None: 
    #     if sample_res > 1:
    #         print('sample_res cannot be less than 1 (the maximum disease time) - exiting')
    #         exit()
    #     n_mci = int(np.floor(max_time / sample_res) + 1)
    #     sample_res_effective = 1/(n_mci-1)
    #     print('requested sample_res: ' + str(sample_res))
    #     print('effective sample_res: ' + str(sample_res_effective))

    # # Stage temporal resolution
    # if stage_res!=None:
    #     n_stages = len(seq)+1 if seq != None else n_biomarkers+1
    #     # time from first transition to last transition
    #     total_time_nonends = (n_stages - 2) * stage_res
    #     min_time_stage = 0 + (max_time - total_time_nonends)/2
    #     max_time_stage = max_time - (max_time - total_time_nonends)/2
    #     onsets_stages = np.linspace(min_time_stage, max_time_stage, n_stages-1) # onset time of each (non-zero) stage


    # Sequence (if specified) & onsets
    if seq == None:
        # Default Onsets for each biomarker
        if onsets==None: onsets = np.array([i*(max_time/(n_biomarkers+1)) for i in range(1,n_biomarkers+1)]) 
        # Stage temporal resolution (if specified, overrides the supplied onsets, assuming a linear order)
        if stage_res!=None:
            n_stages = n_biomarkers+1
            total_time_nonends = (n_stages - 2) * stage_res # time from first to last transition
            min_time_stage = 0 + (max_time - total_time_nonends)/2
            max_time_stage = max_time - (max_time - total_time_nonends)/2
            onsets = np.linspace(min_time_stage, max_time_stage, n_stages-1) # onset time of each (non-zero) stage (equivalently each biomarker)
        #
        # Other default parameters
        n_biomarkers = len(onsets)
        onsets_stages = np.unique(onsets) # np.unique already sorts output
    else:
        # biomarkers & onsets determined by sequence
        bioms = list(itertools.chain(*seq))
        n_biomarkers = len(bioms)
        n_stages = len(seq) + 1 # including stage 0
        onsets = np.array([np.nan for i in range(n_biomarkers)]) # onset of each biomarker
        # Stage onsets (if stage temporal resolution is specified, it determines the stage onsets, else they're determined by equidistant spacing)
        if stage_res==None:
            onsets_stages = np.array([i*(max_time/n_stages) for i in range(n_stages)])[1:] # onset time of each (non-zero) stage
        else:
            total_time_nonends = (n_stages - 2) * stage_res # time from first to last transition
            min_time_stage = 0 + (max_time - total_time_nonends)/2
            max_time_stage = max_time - (max_time - total_time_nonends)/2
            onsets_stages = np.linspace(min_time_stage, max_time_stage, n_stages-1) # onset time of each (non-zero) stage (equivalently each biomarker)
        # 
        # writing onsets based on (non-zero) stage onsets
        for i in range(n_stages-1):
            bioms_stagei = seq[i]
            for j in bioms_stagei: onsets[j] = onsets_stages[i]


    # Sigmoid gradients
    if gradients is None:
        grad_default = 30 / max_time  # 50 default approximately mimicks best case of Tandon et al
        gradients = np.ones(shape=n_biomarkers) * grad_default

    # standard deviations
    if sds_normal is None: sds_normal = np.array([0.05 for i in range(n_biomarkers)])
    if sds_abnormal is None: sds_abnormal = np.array([0.05 for i in range(n_biomarkers)])

    # Biomarker labels
    if biomarker_labels is None: biomarker_labels = list((string.ascii_uppercase * 100)[:n_biomarkers])

    # Permuting biomarker attributes
    if permutation:
        onsets = np.array([onsets[ind] for ind in permutation])
        gradients = np.array([gradients[ind] for ind in permutation])
        biomarker_labels = np.array([biomarker_labels[ind] for ind in permutation])
        sds_normal = np.array([sds_normal[ind] for ind in permutation])
        sds_abnormal = np.array([sds_abnormal[ind] for ind in permutation])


    # Mean normal & abnormal
    if means_normal is None:
        means_normal = [0] * n_biomarkers

    if means_abnormal is None:
        means_abnormal = [1] * n_biomarkers

    # Jitters on the sigmoid GT x-location (optional, mainly useful for plotting)
    if jitter_times is None: jitter_times = [0 for i in range(n_biomarkers)]

    # Generating MCI data
    X_mci_gt = np.empty(shape=(n_mci, n_biomarkers)) * np.nan
    X_mci = np.empty(shape=(n_mci, n_biomarkers)) * np.nan
    for i, grad, onset, sd_abn in zip(range(n_biomarkers), gradients, onsets, sds_abnormal):
        X_mci_gt[:, i] = sigmoid(t=times + jitter_times[i], a=grad, b=onset) * means_abnormal[i]
        if set_mci_seed:
            np.random.seed(1)
        X_mci[:, i] = X_mci_gt[:, i] + np.random.normal(loc=0, scale=sd_abn, size=n_mci)


    ## Control data
    if n_controls == 0:
        X_controls = np.array([]).reshape(-1,n_biomarkers)
    else:
        X_controls = np.empty((n_controls, n_biomarkers)) * np.nan
        for i in range(n_biomarkers):
            if set_cn_seed:
                np.random.seed(2)
            X_controls[:,i] = means_normal[i] + np.random.normal(loc=0, scale=sds_normal[i], size=n_controls)

    ## Patient data
    if n_patients == 0:
        X_patients = np.array([]).reshape(-1,n_biomarkers)
    else:
        X_patients = np.empty((n_patients, n_biomarkers)) * np.nan
        for i in range(n_biomarkers):
            if set_pt_seed:
                np.random.seed(3)
            X_patients[:, i] = means_abnormal[i] + np.random.normal(loc=0, scale=sds_abnormal[i], size=n_patients)

    # Plot
    if plot and (n_biomarkers<20):
        # Colour pallette
        colpal = ['C'+str(c_ind) for c_ind in range(0,n_biomarkers)]

        # Fake data for plotting ground truth trajectories
        n_gt_times = 100
        times_gtplot = np.linspace(0, max_time, n_gt_times)
        mci_gtplot = np.empty(shape=(n_gt_times, n_biomarkers)) * np.nan
        for i, grad, onset, sd_abn in zip(range(n_biomarkers), gradients, onsets, sds_abnormal):
            mci_gtplot[:, i] = sigmoid(t=times_gtplot + jitter_times[i], a=grad, b=onset) * means_abnormal[i]
            
        # MCI time course (GT & observed)
        colors_mci = colpal[:n_biomarkers] if colors_mci is None else colors_mci # ['b', 'r'] #['darkorange', 'orangered']
        # breakpoint()
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(n_biomarkers):
            if (i+1)>plot_nlines:
                print('Reached max number of lines to plot - existing plot')
                break
            # ax.plot(times, X_mci_gt[:, i], color=colors_mci[i], label=biomarker_labels[i] + '-GT',linewidth=plot_lwd)
            if plot_lines:
                ax.plot(times_gtplot, mci_gtplot[:, i], color=colors_mci[i], label=biomarker_labels[i] + model_gt_string,linewidth=plot_lwd)
            #
            if plot_points:
                if label_points:
                    ax.plot(times, X_mci[:, i], '.', color=colors_mci[i], label=biomarker_labels[i])
                else:
                    ax.plot(times, X_mci[:, i], '.', color=colors_mci[i])

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(ylims)
        if plot_legend: plt.legend(loc="upper left", title=legend_title)
        if plot_plot:
            plot_vars = None
            plt.show()
        else:
            plot_vars = (fig, ax)

        if plot & plot_hist:
            # MCI histogram
            plt.hist(X_mci, color=colors_mci[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
            plt.legend(loc="upper left")
            plt.show()

            # Controls histogram
            if n_controls !=0:
                colors_controls = ['C'+str(c_ind) for c_ind in range(0,n_biomarkers)] # ['green','magenta'] #['steelblue', 'deepskyblue']
                plt.hist(X_controls, color=colors_controls[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
                plt.legend(loc="upper left")
                plt.show()

            # Patients histogram
            if n_patients != 0:
                colors_patients = ['C'+str(c_ind) for c_ind in range(0,n_biomarkers)]
                plt.hist(X_patients, color=colors_patients[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
                plt.legend(loc="upper left")
                plt.show()
    else:
        plot_vars = None

    # Returning data
    # biomarker columns
    X = np.concatenate((X_controls,X_mci,X_patients),axis=0)
    # diagnosis columns
    cn_col = np.ones(n_controls + n_mci + n_patients) * 0
    cn_col[:n_controls] = 1
    mci_col = np.ones(n_controls + n_mci + n_patients) * 0
    mci_col[n_controls:(n_controls+n_mci)] = 1
    pat_col = np.ones(n_controls + n_mci + n_patients) * 0
    pat_col[(n_controls+n_mci):] = 1
    X = np.append(X, np.column_stack((cn_col, mci_col, pat_col)), axis=1)

    # Stages of mci subjects
    ks_mci = np.ones(n_mci) * np.nan
    onsets_stages_inclend = np.concatenate((onsets_stages, [1]))
    for j in range(n_mci):
        j_time = times[j]
        j_dists_onsets = j_time - onsets_stages_inclend
        j_stage = np.where(j_dists_onsets<=0)[0][0] # stage = first negative index
        # print(onsets_stages_inclend)
        # print(j_dists_onsets)
        # print('')
        ks_mci[j] = j_stage


    # Convert to pandas array
    X = pd.DataFrame(data=X, columns=biomarker_labels[:n_biomarkers] + ['CN'] + ['MCI'] + ['AD'])
    return X, ks_mci, plot_vars

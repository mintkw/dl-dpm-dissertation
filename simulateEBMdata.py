def simulateEBMdata(seq, n_mci, means_normal, means_abnormal, sds_normal, sds_abnormal, biomarker_labels, force_uniform_stages=True, plot=False, n_controls=0, n_patients=0):
    # SEQ: in partial order format i.e. [[0],[1,2]]
    # Modules
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import string
    # prev_dir = os.getcwd()
    # script_dir = '/Users/christopherparker/Documents/GitHubProjects/dDPM/python'
    # os.chdir(script_dir)
    from seq2stages import seq2stages
    # os.chdir(prev_dir)
    #
    stages = seq2stages(seq)
    n_stages = len(stages)
    n_subjects = n_controls + n_mci + n_patients
    seq_vec = [item for sublist in seq for item in sublist]
    n_biomarkers = len(seq_vec)
    #
    #- Controls
    X_controls = np.ones((n_controls,n_biomarkers))
    for i in range(n_biomarkers):
        X_controls[:,i] = np.random.normal(means_normal[i],sds_normal[i],n_controls)
    #- Patients
    X_patients = np.ones((n_patients, n_biomarkers))
    for i in range(n_biomarkers):
        X_patients[:, i] = np.random.normal(loc=means_abnormal[i], scale=sds_abnormal[i], size=n_patients)
    # - MCI
    X_mci = np.ones((n_mci,n_biomarkers)) * np.nan
    if force_uniform_stages==True:
        ks_mci_pre = np.linspace(start=0,stop=n_stages,num=n_mci,endpoint=False)
        ks_mci = np.floor(ks_mci_pre).astype(int)
    else:
        ks_mci_pre = np.random.uniform(low=0, high=n_stages, size=n_mci)
        ks_mci = np.floor(ks_mci_pre).astype(int)
    for i in range(n_mci):
        stage_id = ks_mci[i]
        stage = stages[stage_id]
        for j in range(n_biomarkers):
            if stage[j]==0:
                val = np.random.normal(loc=means_normal[j], scale=sds_normal[j], size=1)
            else:
                val = np.random.normal(loc=means_abnormal[j], scale=sds_abnormal[j], size=1)
            X_mci[i,j] = val
    # - Plotting
    if plot and (n_biomarkers < 20):
        # Colour pallette
        colpal = ['C' + str(c_ind) for c_ind in range(0, n_biomarkers)]
        # Building MCI ground truth
        n_ks_plot = 1000
        X_mci_gt = np.ones((n_ks_plot, n_biomarkers)) * np.nan
        ks_mci_plot_pre = np.linspace(start=0, stop=n_stages, num=n_ks_plot, endpoint=False)
        ks_mci_plot = np.floor(ks_mci_plot_pre).astype(int)
        for i in range(n_ks_plot):
            stage_id = ks_mci_plot[i]
            stage = stages[stage_id]
            for j in range(n_biomarkers):
                if stage[j] == 0:
                    val_gt = means_normal[j]
                else:
                    val_gt = means_abnormal[j]
                X_mci_gt[i, j] = val_gt
        # MCI time course
        colors_mci = colpal[:n_biomarkers]  # ['b', 'r'] #['darkorange', 'orangered']
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(n_biomarkers):
            ax.plot(ks_mci_plot_pre, X_mci_gt[:, i], color=colors_mci[i], label=biomarker_labels[i] + '-GT')
            ax.plot(ks_mci_pre, X_mci[:, i], '.', color=colors_mci[i], label=biomarker_labels[i])
        ax.set_xlabel("Disease Stage", fontsize=20)
        ax.set_ylabel("Biomarker", fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        # MCI histogram
        plt.hist(X_mci, color=colors_mci[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
        plt.legend(loc="upper left")
        plt.show()
        # MCI stage
        plt.hist(ks_mci,bins=n_stages)
        plt.xlabel('MCI stage')
        plt.show()
        # Controls histogram
        if n_controls != 0:
            colors_controls = ['C' + str(c_ind) for c_ind in
                               range(0, n_biomarkers)]  # ['green','magenta'] #['steelblue', 'deepskyblue']
            plt.hist(X_controls, color=colors_controls[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
            plt.legend(loc="upper left")
            plt.show()
        # Patients histogram
        if n_patients != 0:
            colors_patients = ['C' + str(c_ind) for c_ind in range(0, n_biomarkers)]
            plt.hist(X_patients, color=colors_patients[:n_biomarkers], label=biomarker_labels[:n_biomarkers])
            plt.legend(loc="upper left")
            plt.show()
    # - Full dataset
    # biomarker columns
    X = np.concatenate((X_controls, X_mci, X_patients), axis=0)
    # diagnosis columns
    cn_col = np.ones(n_subjects) * 0
    cn_col[:n_controls] = 1
    mci_col = np.ones(n_subjects) * 0
    mci_col[n_controls:(n_controls + n_mci)] = 1
    pat_col = np.ones(n_subjects) * 0
    pat_col[(n_controls + n_mci):] = 1
    X = np.append(X, np.column_stack((cn_col, mci_col, pat_col)), axis=1)
    # Convert to pandas array
    X = pd.DataFrame(data=X, columns=biomarker_labels[:n_biomarkers] + ['CN'] + ['MCI'] + ['AD'])
    return X, ks_mci

## To add:
# plotting function



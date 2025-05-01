def seq2stages(seq):
    # SEQ is in PO format i.e. list ofl ists
    seq_vec = [item for sublist in seq for item in sublist]
    n_biomarkers = len(seq_vec)
    n_stages = len(seq)+1
    stages = [[0] * n_biomarkers] + [[0]*n_biomarkers for _ in range(n_stages-1)]
    for i in range(1,n_stages):
        event_inds_stage_i = seq[i-1]
        for event_ind in event_inds_stage_i:
            for k in range(i,n_stages):
                stages[k][event_ind] = 1
    return stages



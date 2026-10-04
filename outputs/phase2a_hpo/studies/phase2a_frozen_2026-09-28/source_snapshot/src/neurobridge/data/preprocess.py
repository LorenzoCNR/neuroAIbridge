import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def create_rats_trial(behav_data: np.ndarray):
    """
    Create trial identifiers from a discrete behavioral stream (CEBRA rats style).
    Increments trial id every two changes in the variable.

    Returns:
        trial_ids: int array [N]
        c_t: list of start indices for each trial
    """
    behav_data = np.asarray(behav_data).ravel()
    trial_ids = np.zeros(len(behav_data), dtype=int)
    c_t = [0]
    current_trial = 1
    trial_ids[0] = current_trial
    change_count = 0
    for i in range(1, len(behav_data)):
        if behav_data[i] != behav_data[i - 1]:
            change_count += 1
            if change_count == 2:
                current_trial += 1
                c_t.append(i)
                change_count = 0
        trial_ids[i] = current_trial
    return trial_ids, c_t


def create_trials_id(trial_id, y):
# functuion created for the simil-hasson paper
    original_label_order =  np.sort(np.unique(y))  # [0,1,2,3,4,5,6,7,8,]
    len_y=len(y)

    change_idx = np.where(np.diff(trial_id) != 0)[0] + 1
    change_idx
    print('ciao')
    # the c_t vector tells the starting and endiing points of every trial
    c_t=np.concatenate([[0], change_idx,[len_y]], dtype=int)

    ## list of list of starting and ending points for trials
    c_t_list=[]
    c_t_list = [(c_t[i], c_t[i+1] - 1) for i in range(len(c_t) - 1)]
    n_trials=len(c_t)-1

    ### check trial length (useful for graphics)
    trial_len=np.diff(c_t)
    trial_length=trial_len[0]
    """
    Returns:
        c_t (list of start and end points of trials)
        trial_length (scalar)
        
    """
    
    return c_t_list, trial_length, n_trials


def f_resample(datasets, trials, step, overlap, methods, 
    mode="overlapping", normalization=False):
    """
    Resample datasets into windows defined by trials.
    - Supports 'center', 'mean', 'sum'.
    - Mode: 'disjoint' (non overlapping) or 'overlapping'.
    - Normalization avoids bias from overlapping sums.
    
    Args:
        datasets (list): list of arrays to resample.
        trials (list): list of (start, end) indices for each trial.
        step (int): window length.
        overlap (int): number of overlapping points (if mode='overlapping').
        methods (dict): mapping dataset index -> resampling method.
        mode (str): 'disjoint' or 'overlapping'.
        normalization (bool): normalize sums if True.
        
    Returns:
        resampled_datasets (list)
        new_trial_lengths (list)
        new_trials_indices (list)
    """
    
    # checks
    if mode not in ("disjoint", "overlapping"):
        raise ValueError("mode must be 'disjoint' or 'overlapping'")
    if step <= 0:
        raise ValueError("step must be > 0")
    stride = step if mode == "disjoint" else step - overlap
    if mode == "overlapping" and not (0 <= overlap < step):
        raise ValueError("overlap must satisfy 0 <= overlap < step")

    resampled_datasets = []
    new_trials_indices = []
    new_trial_lengths = []

    for i, dataset in enumerate(datasets):
        resampled_trials = []
        trial_lengths = []
        trial_indices = []
        method = methods.get(i, "center")
        current_start = 0

        for start_trial, end_trial in trials:
            trial_data = dataset[start_trial:end_trial]
            n_rows = len(trial_data)
            resampled = []

            if n_rows >= step:
                for start in range(0, n_rows - step + 1, stride):
                    end = start + step
                    if method == "center":
                        idx = min(start + step // 2, n_rows - 1)
                        resampled.append(trial_data[idx])
                    elif method == "mean":
                        resampled.append(np.mean(trial_data[start:end], axis=0))
                    elif method == "sum":
                        sum_val = np.sum(trial_data[start:end], axis=0)
                        if normalization:
                           # corrected_sum = sum_val / step
                            # decide we
                            corrected_sum=(sum_val/step)*stride
                            resampled.append(corrected_sum)
                        else:
                            resampled.append(sum_val)

            if resampled:
                resampled_array = np.array(resampled)
                resampled_trials.append(resampled_array)
                trial_len = resampled_array.shape[0]
                trial_lengths.append(trial_len)
                trial_indices.append((current_start, current_start + trial_len - 1))
                current_start += trial_len

        if resampled_trials:
            non_empty = [arr for arr in resampled_trials if arr.size > 0]
            if non_empty:
                resampled_concat = np.concatenate(non_empty, axis=0)
            else:
                resampled_concat = np.empty((0, dataset.shape[1])) if dataset.ndim > 1 else np.empty((0,))
            resampled_datasets.append(resampled_concat)
        else:
            resampled_datasets.append(np.array([]))

        new_trial_lengths.append(trial_lengths)
        new_trials_indices.append(trial_indices)

    return resampled_datasets, new_trial_lengths, new_trials_indices

def split_data_trials(
    data: Dict[str, Any],
    case: int = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    update_original: bool = True,
):
    """
    Split data into train/val/test based on trial ids (no leakage).

    Cases:
      1 -> train/val/test
      2 -> train/val
      3 -> train (subtrain+val) / test

    Assumes data contains: 'X', 'y', 'trials'.
    """
    X, y, trials = data["X"], data["y"], data["trials"]
    uniq = np.unique(trials)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq)

    if case == 1:
        test_ratio = 1 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError("Train+Val leave no Test set.")
        n_tr = int(round(train_ratio * len(uniq)))
        n_va = int(round(val_ratio * len(uniq)))
        tr_trials = uniq[:n_tr]
        va_trials = uniq[n_tr:n_tr + n_va]
        te_trials = uniq[n_tr + n_va:]
        sub_trials = None
    elif case == 2:
        n_tr = int(round(train_ratio * len(uniq)))
        tr_trials = uniq[:n_tr]
        va_trials = uniq[n_tr:]
        te_trials = None
        sub_trials = None
    elif case == 3:
        n_tr = int(round(train_ratio * len(uniq)))
        tr_trials = uniq[:n_tr]
        te_trials = uniq[n_tr:]
        n_sub = int(round((1 - val_ratio) * len(tr_trials)))
        sub_trials = tr_trials[:n_sub]
        va_trials = tr_trials[n_sub:]
    else:
        raise ValueError("case must be 1|2|3")

    def idx_of(tr_arr, chosen):
        return None if chosen is None else np.isin(tr_arr, chosen).nonzero()[0]

    idx_tr = idx_of(trials, tr_trials)
    idx_va = idx_of(trials, va_trials)
    idx_te = idx_of(trials, te_trials)
    idx_sub = idx_of(trials, sub_trials) if case == 3 else idx_tr

    split = dict(
        X_train=X[idx_tr], y_train=y[idx_tr],
        X_val=X[idx_va],   y_val=y[idx_va],
        train_trials=tr_trials, val_trials=va_trials,
    )
    if idx_te is not None:
        split.update(X_test=X[idx_te], y_test=y[idx_te], test_trials=te_trials)
    if case == 3:
        split.update(X_subtrain=X[idx_sub], y_subtrain=y[idx_sub], subtrain_trials=sub_trials)

    if update_original:
        data.update(split)
        return data
    return split


def circular_initial_split(trials: np.ndarray, train_ratio: float, seed: Optional[int] = None):
    """
    Initial circular split: split unique trials into train and val as one contiguous circular slice.
    """
    np.random.seed(seed)
    unique_trials = np.unique(trials)
    total = len(unique_trials)
    n_train = int(round(train_ratio * total))
    start = np.random.randint(0, total)
    tr_idx = (np.arange(start, start + n_train) % total)
    va_idx = (np.arange(start + n_train, start + total) % total)
    return unique_trials[tr_idx], unique_trials[va_idx]

def circular_cv_split(trials: np.ndarray, train_trials: np.ndarray, val_ratio: float, split_idx: int):
    """
    Circular CV split within the training trials: rotates a validation slice across train_trials.
    """
    uniq_train = np.unique(train_trials)
    total = len(uniq_train)
    n_val = int(round(val_ratio * total))
    n_sub = total - n_val
    start = (split_idx * n_val) % total
    sub_idx = (np.arange(start, start + n_sub) % total)
    val_idx = (np.arange(start + n_sub, start + n_sub + n_val) % total)
    return uniq_train[sub_idx], uniq_train[val_idx]

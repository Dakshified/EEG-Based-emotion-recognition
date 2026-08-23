import os
import time
import numpy as np

def main():
    dataset_path = "seed_iv_processed.npz"
    data = np.load(dataset_path)
    features = data["features"]
    subject_ids = data["subject_ids"]
    session_nums = data["session_nums"]
    trial_ids = data["trial_ids"]
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys, trial_indices_map = np.unique(trial_keys, return_index=True)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_trial_idx, _ = next(kf.split(np.arange(len(unique_trial_keys))))
    train_trials_set = set(unique_trial_keys[train_trial_idx])
    train_indices = [idx for idx in range(len(features)) if trial_keys[idx] in train_trials_set]
    
    trials = {}
    for idx in train_indices:
        key = (subject_ids[idx], session_nums[idx], trial_ids[idx])
        if key not in trials:
            trials[key] = []
        trials[key].append((idx, features[idx].mean(axis=-1)))
        
    trial_series = []
    for key, val in trials.items():
        val_sorted = sorted(val, key=lambda x: x[0])
        ts = np.array([x[1] for x in val_sorted])
        if ts.shape[0] > 5:
            trial_series.append(ts)
            
    if len(trial_series) > 30:
        np.random.seed(42)
        indices = np.random.choice(len(trial_series), 30, replace=False)
        trial_series = [trial_series[i] for i in indices]
        
    # We will compute the full 62x62 F-matrix using Method 4 (Precompute restricted + lstsq for unrestricted)
    # and time it.
    t0 = time.time()
    num_channels = 62
    
    # 1. Precompute restricted RSS for all channels i
    print("Precomputing restricted model RSS...")
    rss_rest_array = np.zeros(num_channels)
    for i in range(num_channels):
        rss_sum = 0.0
        for ts in trial_series:
            y_i = ts[1:, i]
            y_i_lag1 = ts[:-1, i]
            L = len(y_i)
            if L < 5:
                continue
            X_rest = np.column_stack((np.ones(L), y_i_lag1))
            _, residuals, _, _ = np.linalg.lstsq(X_rest, y_i, rcond=None)
            if len(residuals) > 0:
                rss_sum += residuals[0]
            else:
                pred_rest = X_rest @ np.linalg.lstsq(X_rest, y_i, rcond=None)[0]
                rss_sum += np.sum((y_i - pred_rest)**2)
        rss_rest_array[i] = rss_sum
        
    print(f"Precomputed restricted RSS in {time.time() - t0:.2f}s")
    
    # 2. Compute unrestricted RSS and F-statistic
    t_unrest = time.time()
    F_matrix = np.zeros((num_channels, num_channels))
    n_obs_total = sum(len(ts) - 1 for ts in trial_series if len(ts) > 5)
    
    print("Computing unrestricted models...")
    for i in range(num_channels):
        rss_rest_total = rss_rest_array[i]
        
        # Prepare target and lag matrices for channel i across all trials
        # to see if we can optimize the LSTSQ calls
        for j in range(num_channels):
            if i == j:
                continue
                
            rss_unrest_total = 0.0
            for ts in trial_series:
                y_i = ts[1:, i]
                y_i_lag1 = ts[:-1, i]
                y_j_lag1 = ts[:-1, j]
                L = len(y_i)
                if L < 5:
                    continue
                    
                X_unrest = np.column_stack((np.ones(L), y_i_lag1, y_j_lag1))
                _, residuals, _, _ = np.linalg.lstsq(X_unrest, y_i, rcond=None)
                if len(residuals) > 0:
                    rss_unrest_total += residuals[0]
                else:
                    pred_unrest = X_unrest @ np.linalg.lstsq(X_unrest, y_i, rcond=None)[0]
                    rss_unrest_total += np.sum((y_i - pred_unrest)**2)
                    
            den = rss_unrest_total / (n_obs_total - 3)
            if den > 1e-10:
                F_val = (rss_rest_total - rss_unrest_total) / den
                F_matrix[i, j] = max(0.0, F_val)
                
        if (i + 1) % 10 == 0 or i == 0 or i == 61:
            print(f"  Completed channel {i+1}/62 (Elapsed: {time.time() - t_unrest:.2f}s)")
            
    print(f"\nUnrestricted computed in {time.time() - t_unrest:.2f}s")
    print(f"Total time: {time.time() - t0:.2f}s")
    
    # Check pair (0, 1) against Method 1
    # Run Method 1 check for pair (0, 1)
    rss_rest_m1 = 0.0
    rss_unrest_m1 = 0.0
    for ts in trial_series:
        y_i = ts[1:, 0]
        y_i_lag1 = ts[:-1, 0]
        y_j_lag1 = ts[:-1, 1]
        L = len(y_i)
        if L < 5:
            continue
        X_rest = np.column_stack((np.ones(L), y_i_lag1))
        beta_rest, _, _, _ = np.linalg.lstsq(X_rest, y_i, rcond=None)
        pred_rest = X_rest @ beta_rest
        rss_rest_m1 += np.sum((y_i - pred_rest)**2)
        
        X_unrest = np.column_stack((np.ones(L), y_i_lag1, y_j_lag1))
        beta_unrest, _, _, _ = np.linalg.lstsq(X_unrest, y_i, rcond=None)
        pred_unrest = X_unrest @ beta_unrest
        rss_unrest_m1 += np.sum((y_i - pred_unrest)**2)
        
    den_m1 = rss_unrest_m1 / (n_obs_total - 3)
    F_m1 = (rss_rest_m1 - rss_unrest_m1) / den_m1
    
    print(f"\nMethod 1 F-val (0, 1): {F_m1:.10f}")
    print(f"Method 4 F-val (0, 1): {F_matrix[0, 1]:.10f}")
    print(f"Difference: {abs(F_m1 - F_matrix[0, 1]):.15f}")

if __name__ == "__main__":
    main()

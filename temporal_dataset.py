"""
Temporal Sequence Dataset Generator for SEED-IV EEG
===================================================
Groups consecutive 1-second Differential Entropy (DE) frames into temporal sequence
windows of length T = 8 with sliding stride = 2.

CRITICAL ZERO-LEAKAGE CONSTRAINT:
Sequences NEVER cross trial boundaries (Subject x Session x Trial). Slicing is performed
strictly within each trial independently, verified with explicit assertions.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from spatial_mapping import transform_to_2d_grid_numpy, transform_to_2d_grid_torch

class EEGSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding (T, 5, 9, 9) spatial-temporal EEG tensors.
    """
    def __init__(self, features_grid, labels, subject_ids, session_nums, trial_ids, domain_labels=None):
        """
        Parameters:
            features_grid: np.ndarray or torch.Tensor of shape (N, T, 5, 9, 9)
            labels: np.ndarray or torch.Tensor of shape (N,)
            subject_ids: np.ndarray or torch.Tensor of shape (N,)
            session_nums: np.ndarray or torch.Tensor of shape (N,)
            trial_ids: np.ndarray or torch.Tensor of shape (N,)
            domain_labels: np.ndarray or torch.Tensor of shape (N,) [Optional, for CDAN: 1=Source, 0=Target]
        """
        if isinstance(features_grid, np.ndarray):
            self.features = torch.tensor(features_grid, dtype=torch.float32)
        else:
            self.features = features_grid.float()
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()
            
        if isinstance(subject_ids, np.ndarray):
            self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)
        else:
            self.subject_ids = subject_ids.long()
            
        if isinstance(session_nums, np.ndarray):
            self.session_nums = torch.tensor(session_nums, dtype=torch.long)
        else:
            self.session_nums = session_nums.long()
            
        if isinstance(trial_ids, np.ndarray):
            self.trial_ids = torch.tensor(trial_ids, dtype=torch.long)
        else:
            self.trial_ids = trial_ids.long()
            
        if domain_labels is not None:
            if isinstance(domain_labels, np.ndarray):
                self.domain_labels = torch.tensor(domain_labels, dtype=torch.float32)
            else:
                self.domain_labels = domain_labels.float()
        else:
            self.domain_labels = torch.zeros(len(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],             # (T, 5, 9, 9)
            'label': self.labels[idx],                   # int (0..3)
            'subject_id': self.subject_ids[idx],         # int (1..15)
            'session_num': self.session_nums[idx],       # int (1..3)
            'trial_id': self.trial_ids[idx],             # int (1..24)
            'domain_label': self.domain_labels[idx]      # float (0.0 or 1.0)
        }

def build_trial_quarantined_sequences(dataset_path="seed_iv_processed.npz", T=4, stride=1):
    """
    Extracts sliding temporal sequence windows of length T from the SEED-IV dataset,
    strictly enforcing that no window spans across multiple trials.
    
    Default: T = 4, stride = 1 (Dense, high-yield temporal slicing without boundary crossing).
    
    Returns:
        seq_dict: dict containing:
            'features_raw': np.ndarray of shape (N_seq, T, 62, 5)
            'labels': np.ndarray of shape (N_seq,)
            'subject_ids': np.ndarray of shape (N_seq,)
            'session_nums': np.ndarray of shape (N_seq,)
            'trial_ids': np.ndarray of shape (N_seq,)
            'trial_keys': np.ndarray of shape (N_seq,)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'")
        
    data = np.load(dataset_path)
    features = data['features']      # (37575, 62, 5)
    labels = data['labels']          # (37575,)
    subject_ids = data['subject_ids']# (37575,)
    session_nums = data['session_nums'] # (37575,)
    trial_ids = data['trial_ids']    # (37575,)
    
    # Construct unique composite trial identifier
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trials = np.unique(trial_keys)
    assert len(unique_trials) == 1080, f"Expected 1080 unique trials (15 subjects x 3 sessions x 24 trials), found {len(unique_trials)}"
    
    seq_features = []
    seq_labels = []
    seq_subs = []
    seq_sess = []
    seq_trials = []
    seq_keys = []
    
    for u_trial in unique_trials:
        mask = (trial_keys == u_trial)
        trial_feats = features[mask]  # (L, 62, 5)
        trial_label_arr = labels[mask]
        trial_sub_arr = subject_ids[mask]
        trial_ses_arr = session_nums[mask]
        trial_t_arr = trial_ids[mask]
        
        # Explicit Trial Boundary Assertions
        assert np.all(trial_label_arr == trial_label_arr[0]), f"Inconsistent label within trial {u_trial}!"
        assert np.all(trial_sub_arr == trial_sub_arr[0]), f"Inconsistent subject within trial {u_trial}!"
        assert np.all(trial_ses_arr == trial_ses_arr[0]), f"Inconsistent session within trial {u_trial}!"
        assert np.all(trial_t_arr == trial_t_arr[0]), f"Inconsistent trial ID within trial {u_trial}!"
        
        L = len(trial_feats)
        assert L >= T, f"Trial {u_trial} has length {L} < T ({T}). Minimum expected is 10."
        
        t_label = trial_label_arr[0]
        t_sub = trial_sub_arr[0]
        t_ses = trial_ses_arr[0]
        t_trial = trial_t_arr[0]
        
        # Sliding sequence slicing strictly within trial
        num_windows_added = 0
        for start in range(0, L - T + 1, stride):
            end = start + T
            window = trial_feats[start:end]
            assert window.shape == (T, 62, 5), f"Window shape mismatch: {window.shape}"
            
            seq_features.append(window)
            seq_labels.append(t_label)
            seq_subs.append(t_sub)
            seq_sess.append(t_ses)
            seq_trials.append(t_trial)
            seq_keys.append(u_trial)
            num_windows_added += 1
            
        # Append terminal window if residual frames exist
        if (L - T) % stride != 0 and L >= T:
            terminal_window = trial_feats[-T:]
            assert terminal_window.shape == (T, 62, 5)
            seq_features.append(terminal_window)
            seq_labels.append(t_label)
            seq_subs.append(t_sub)
            seq_sess.append(t_ses)
            seq_trials.append(t_trial)
            seq_keys.append(u_trial)
            num_windows_added += 1
            
        assert num_windows_added > 0, f"Zero windows generated for trial {u_trial}!"
        
    seq_dict = {
        'features_raw': np.array(seq_features, dtype=np.float32), # (N_seq, T, 62, 5)
        'labels': np.array(seq_labels, dtype=np.int64),
        'subject_ids': np.array(seq_subs, dtype=np.int64),
        'session_nums': np.array(seq_sess, dtype=np.int64),
        'trial_ids': np.array(seq_trials, dtype=np.int64),
        'trial_keys': np.array(seq_keys, dtype=np.int64)
    }
    
    return seq_dict

def prepare_scaled_spatial_datasets(seq_dict, train_mask, test_mask, val_mask=None, target_unlabeled_mask=None):
    """
    Applies strict zero-leakage feature scaling (StandardScaler fit ONLY on train_mask)
    and transforms the scaled sequences into (N, T, 5, 9, 9) spatial grid representations.
    
    Parameters:
        seq_dict: dict from build_trial_quarantined_sequences
        train_mask: np.ndarray bool mask of training sequences
        test_mask: np.ndarray bool mask of test sequences
        val_mask: np.ndarray bool mask of validation sequences (optional)
        target_unlabeled_mask: np.ndarray bool mask of unlabeled target sequences for CDAN (optional)
        
    Returns:
        train_dataset: EEGSequenceDataset
        test_dataset: EEGSequenceDataset
        val_dataset: EEGSequenceDataset (or None)
        target_unlabeled_dataset: EEGSequenceDataset (or None)
    """
    raw_feats = seq_dict['features_raw'] # (N_seq, T, 62, 5)
    labels = seq_dict['labels']
    subs = seq_dict['subject_ids']
    sess = seq_dict['session_nums']
    trials = seq_dict['trial_ids']
    
    N_seq, T, C, B = raw_feats.shape
    
    # 1. Flatten frames for standard scaling
    train_raw = raw_feats[train_mask] # (N_train, T, 62, 5)
    train_frames = train_raw.reshape(-1, C * B) # (N_train * T, 310)
    
    # 2. Fit scaler strictly on training frames
    scaler = StandardScaler()
    scaler.fit(train_frames)
    
    # 3. Transform and map training set
    train_scaled = scaler.transform(train_frames).reshape(-1, T, C, B)
    train_grid = transform_to_2d_grid_numpy(train_scaled) # (N_train, T, 5, 9, 9)
    train_dataset = EEGSequenceDataset(
        train_grid, labels[train_mask], subs[train_mask], sess[train_mask], trials[train_mask],
        domain_labels=np.ones(np.sum(train_mask)) # 1.0 = Source domain
    )
    
    # 4. Transform and map test set
    test_raw = raw_feats[test_mask]
    test_frames = test_raw.reshape(-1, C * B)
    test_scaled = scaler.transform(test_frames).reshape(-1, T, C, B)
    test_grid = transform_to_2d_grid_numpy(test_scaled)
    test_dataset = EEGSequenceDataset(
        test_grid, labels[test_mask], subs[test_mask], sess[test_mask], trials[test_mask],
        domain_labels=np.zeros(np.sum(test_mask)) # 0.0 = Target domain
    )
    
    # 5. Transform and map validation set if provided
    val_dataset = None
    if val_mask is not None and np.sum(val_mask) > 0:
        val_raw = raw_feats[val_mask]
        val_frames = val_raw.reshape(-1, C * B)
        val_scaled = scaler.transform(val_frames).reshape(-1, T, C, B)
        val_grid = transform_to_2d_grid_numpy(val_scaled)
        val_dataset = EEGSequenceDataset(
            val_grid, labels[val_mask], subs[val_mask], sess[val_mask], trials[val_mask],
            domain_labels=np.ones(np.sum(val_mask))
        )
        
    # 6. Transform and map target unlabeled set if provided (for CDAN domain discriminator)
    target_unlabeled_dataset = None
    if target_unlabeled_mask is not None and np.sum(target_unlabeled_mask) > 0:
        target_raw = raw_feats[target_unlabeled_mask]
        target_frames = target_raw.reshape(-1, C * B)
        target_scaled = scaler.transform(target_frames).reshape(-1, T, C, B)
        target_grid = transform_to_2d_grid_numpy(target_scaled)
        target_unlabeled_dataset = EEGSequenceDataset(
            target_grid, labels[target_unlabeled_mask], subs[target_unlabeled_mask], sess[target_unlabeled_mask], trials[target_unlabeled_mask],
            domain_labels=np.zeros(np.sum(target_unlabeled_mask)) # 0.0 = Target domain
        )
        
    return train_dataset, test_dataset, val_dataset, target_unlabeled_dataset

def get_stratified_session_trial_splits(session_seq_dict, n_splits=4, seed=42):
    """
    Computes Stratified 4-Fold cross-validation splits at the TRIAL level for a single session.
    Guarantees:
    1. Zero trial leakage (train trials and test trials are mutually exclusive).
    2. Balanced emotion class distribution in every training fold (18 trials) and test fold (6 trials).
    
    Parameters:
        session_seq_dict: dict or mask containing sequence metadata for a single subject & session.
        n_splits: int (default 4)
        seed: int (default 42)
        
    Returns:
        folds: list of dicts [{'train_trials': np.ndarray, 'test_trials': np.ndarray, 'train_mask': np.ndarray, 'test_mask': np.ndarray}, ...]
    """
    from sklearn.model_selection import StratifiedKFold
    
    trials = session_seq_dict['trial_ids']
    labels = session_seq_dict['labels']
    
    # Extract unique trial IDs and their corresponding single emotion label
    unique_trials = np.unique(trials)
    assert len(unique_trials) == 24, f"Expected 24 trials in session, found {len(unique_trials)}"
    
    trial_labels = np.array([labels[trials == t][0] for t in unique_trials])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(unique_trials, trial_labels)):
        train_trials = unique_trials[train_idx]
        test_trials = unique_trials[test_idx]
        
        train_mask = np.isin(trials, train_trials)
        test_mask = np.isin(trials, test_trials)
        
        # Verify strict quarantine and coverage
        assert len(np.intersect1d(train_trials, test_trials)) == 0, "Trial overlap between train and test!"
        assert np.sum(train_mask) + np.sum(test_mask) == len(trials), "Incomplete sequence coverage in fold!"
        
        folds.append({
            'fold': fold_idx + 1,
            'train_trials': train_trials,
            'test_trials': test_trials,
            'train_mask': train_mask,
            'test_mask': test_mask
        })
        
    return folds

if __name__ == '__main__':
    print("Testing temporal sequence dataset generation (T=4, stride=1)...")
    seq_dict = build_trial_quarantined_sequences("seed_iv_processed.npz", T=4, stride=1)
    print(f"Generated {len(seq_dict['labels'])} sequences of shape {seq_dict['features_raw'].shape}")
    
    # Test Stratified 4-Fold Trial CV for Subject 1 Session 1
    s1_ses1_mask = (seq_dict['subject_ids'] == 1) & (seq_dict['session_nums'] == 1)
    sub_dict = {k: v[s1_ses1_mask] for k, v in seq_dict.items()}
    folds = get_stratified_session_trial_splits(sub_dict, n_splits=4, seed=42)
    print(f"Subject 1 Session 1: {len(sub_dict['labels'])} sequences across 24 trials.")
    for f in folds:
        print(f"  Fold {f['fold']}: Train trials={len(f['train_trials'])} ({np.sum(f['train_mask'])} seqs), Test trials={len(f['test_trials'])} ({np.sum(f['test_mask'])} seqs)")
        
    print("\n[PASS] Temporal sequence dataset and stratified trial splits verified cleanly!")
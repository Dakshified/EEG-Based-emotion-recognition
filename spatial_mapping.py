"""
Spatial Topology Mapping (9x9 2D Grid) for SEED-IV EEG
======================================================
Maps the 62 standard 10-20 EEG channels of SEED-IV into a structured (9, 9) 2D spatial grid.
Zero-pads unoccupied coordinates to preserve hemispheric symmetry and local cortical proximity.

Canonical SEED-IV Channel Order (BCMI Lab):
Fp1, Fpz, Fp2, AF3, AF4, F7, F5, F3, F1, Fz, F2, F4, F6, F8,
FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8,
T7, C5, C3, C1, Cz, C2, C4, C6, T8,
TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8,
P7, P5, P3, P1, Pz, P2, P4, P6, P8,
PO7, PO5, PO3, POz, PO4, PO6, PO8,
CB1, O1, Oz, O2, CB2
"""

import numpy as np
import torch

SEED_IV_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

GRID_COORDINATES_9X9 = {
    'Fp1': (0, 3), 'Fpz': (0, 4), 'Fp2': (0, 5),
    'AF3': (1, 3), 'AF4': (1, 5),
    'F7':  (2, 0), 'F5':  (2, 1), 'F3':  (2, 2), 'F1':  (2, 3), 'Fz':  (2, 4), 'F2':  (2, 5), 'F4':  (2, 6), 'F6':  (2, 7), 'F8':  (2, 8),
    'FT7': (3, 0), 'FC5': (3, 1), 'FC3': (3, 2), 'FC1': (3, 3), 'FCz': (3, 4), 'FC2': (3, 5), 'FC4': (3, 6), 'FC6': (3, 7), 'FT8': (3, 8),
    'T7':  (4, 0), 'C5':  (4, 1), 'C3':  (4, 2), 'C1':  (4, 3), 'Cz':  (4, 4), 'C2':  (4, 5), 'C4':  (4, 6), 'C6':  (4, 7), 'T8':  (4, 8),
    'TP7': (5, 0), 'CP5': (5, 1), 'CP3': (5, 2), 'CP1': (5, 3), 'CPz': (5, 4), 'CP2': (5, 5), 'CP4': (5, 6), 'CP6': (5, 7), 'TP8': (5, 8),
    'P7':  (6, 0), 'P5':  (6, 1), 'P3':  (6, 2), 'P1':  (6, 3), 'Pz':  (6, 4), 'P2':  (6, 5), 'P4':  (6, 6), 'P6':  (6, 7), 'P8':  (6, 8),
    'PO7': (7, 0), 'PO5': (7, 1), 'PO3': (7, 2), 'POz': (7, 4), 'PO4': (7, 6), 'PO6': (7, 7), 'PO8': (7, 8),
    'CB1': (8, 0), 'O1':  (8, 2), 'Oz':  (8, 4), 'O2':  (8, 6), 'CB2': (8, 8)
}

# Precompute index arrays for fast vectorized mapping
CHANNEL_ROWS = np.array([GRID_COORDINATES_9X9[ch][0] for ch in SEED_IV_CHANNELS], dtype=np.int64)
CHANNEL_COLS = np.array([GRID_COORDINATES_9X9[ch][1] for ch in SEED_IV_CHANNELS], dtype=np.int64)

def get_channel_grid_map():
    """Returns a (9, 9) string array showing the electrode placement."""
    grid = np.full((9, 9), '---', dtype='<U5')
    for ch, (r, c) in GRID_COORDINATES_9X9.items():
        grid[r, c] = ch
    return grid

def transform_to_2d_grid_numpy(features):
    """
    Transforms 1D/2D channel features into a (5, 9, 9) spatial grid representation.
    
    Parameters:
        features: numpy.ndarray
            - If 2D: (N, 310) where 310 = 62 channels * 5 frequency bands
            - If 3D: (N, 62, 5) where 62 = channels, 5 = frequency bands
            - If 4D: (N, T, 62, 5) temporal sequence of frames
            
    Returns:
        grid_features: numpy.ndarray
            - If 2D/3D input: (N, 5, 9, 9)
            - If 4D input: (N, T, 5, 9, 9)
    """
    orig_shape = features.shape
    
    if len(orig_shape) == 2 and orig_shape[1] == 310:
        N = orig_shape[0]
        feats_3d = features.reshape(N, 62, 5)
        # Transpose to (N, 5, 62)
        feats_5x62 = np.transpose(feats_3d, (0, 2, 1))
        grid = np.zeros((N, 5, 9, 9), dtype=np.float32)
        grid[:, :, CHANNEL_ROWS, CHANNEL_COLS] = feats_5x62
        return grid
        
    elif len(orig_shape) == 3 and orig_shape[1] == 62 and orig_shape[2] == 5:
        N = orig_shape[0]
        # Transpose from (N, 62, 5) to (N, 5, 62)
        feats_5x62 = np.transpose(features, (0, 2, 1))
        grid = np.zeros((N, 5, 9, 9), dtype=np.float32)
        grid[:, :, CHANNEL_ROWS, CHANNEL_COLS] = feats_5x62
        return grid
        
    elif len(orig_shape) == 4 and orig_shape[2] == 62 and orig_shape[3] == 5:
        N, T = orig_shape[0], orig_shape[1]
        # Transpose from (N, T, 62, 5) to (N, T, 5, 62)
        feats_5x62 = np.transpose(features, (0, 1, 3, 2))
        grid = np.zeros((N, T, 5, 9, 9), dtype=np.float32)
        grid[:, :, :, CHANNEL_ROWS, CHANNEL_COLS] = feats_5x62
        return grid
        
    else:
        raise ValueError(f"Unsupported input feature shape {orig_shape}. Expected (N, 310), (N, 62, 5), or (N, T, 62, 5).")

def transform_to_2d_grid_torch(features):
    """
    PyTorch tensor implementation of transform_to_2d_grid.
    
    Parameters:
        features: torch.Tensor of shape (N, 62, 5) or (N, T, 62, 5)
        
    Returns:
        torch.Tensor of shape (N, 5, 9, 9) or (N, T, 5, 9, 9)
    """
    device = features.device
    rows = torch.tensor(CHANNEL_ROWS, dtype=torch.long, device=device)
    cols = torch.tensor(CHANNEL_COLS, dtype=torch.long, device=device)
    
    if features.dim() == 3 and features.shape[1] == 62 and features.shape[2] == 5:
        N = features.shape[0]
        feats_5x62 = features.permute(0, 2, 1) # (N, 5, 62)
        grid = torch.zeros((N, 5, 9, 9), dtype=torch.float32, device=device)
        grid[:, :, rows, cols] = feats_5x62
        return grid
        
    elif features.dim() == 4 and features.shape[2] == 62 and features.shape[3] == 5:
        N, T = features.shape[0], features.shape[1]
        feats_5x62 = features.permute(0, 1, 3, 2) # (N, T, 5, 62)
        grid = torch.zeros((N, T, 5, 9, 9), dtype=torch.float32, device=device)
        grid[:, :, :, rows, cols] = feats_5x62
        return grid
        
    else:
        raise ValueError(f"Unsupported tensor shape {features.shape}. Expected (N, 62, 5) or (N, T, 62, 5).")

if __name__ == '__main__':
    print("Testing spatial mapping...")
    grid_layout = get_channel_grid_map()
    print("\n9x9 Electrode Grid Map:")
    for r in range(9):
        print(f"Row {r}: " + " ".join(f"{grid_layout[r, c]:>4}" for c in range(9)))
        
    test_data = np.random.randn(10, 62, 5).astype(np.float32)
    grid_out = transform_to_2d_grid_numpy(test_data)
    print(f"\nNumpy Transformation: (10, 62, 5) -> {grid_out.shape}")
    assert grid_out.shape == (10, 5, 9, 9)
    
    test_seq = np.random.randn(10, 8, 62, 5).astype(np.float32)
    grid_seq_out = transform_to_2d_grid_numpy(test_seq)
    print(f"Numpy Sequence Transformation: (10, 8, 62, 5) -> {grid_seq_out.shape}")
    assert grid_seq_out.shape == (10, 8, 5, 9, 9)
    
    t_data = torch.tensor(test_data)
    t_out = transform_to_2d_grid_torch(t_data)
    print(f"PyTorch Transformation: {t_data.shape} -> {t_out.shape}")
    assert t_out.shape == (10, 5, 9, 9)
    
    print("\n[PASS] All spatial mapping tests passed cleanly!")
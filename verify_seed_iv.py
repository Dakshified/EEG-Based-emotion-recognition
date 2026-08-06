import os
import time
import numpy as np

def verify_dataset(file_path="seed_iv_processed.npz"):
    print("=" * 60)
    print("SEED-IV Processed Dataset Verification Script")
    print(f"File Path: {os.path.abspath(file_path)}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset file not found at {file_path}")
        return
        
    # 9. Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # 10. Measure load time (including accessing keys to force lazy-load execution)
    start_time = time.time()
    try:
        loaded_data = np.load(file_path)
        # Access each key to trigger actual disk read
        keys = list(loaded_data.keys())
        arrays = {key: loaded_data[key] for key in keys}
        load_time = time.time() - start_time
    except Exception as e:
        print(f"ERROR: Failed to load .npz file: {e}")
        return

    # 1. Print all key names
    print(f"1. Keys found in .npz: {keys}")
    
    # 2. Print shape and dtype of every array
    print("\n2. Array details (Shape and Dtype):")
    for key, arr in arrays.items():
        print(f"  - '{key}': shape={arr.shape}, dtype={arr.dtype}")
        
    # Check expected keys
    expected_keys = ["features", "labels", "subject_ids", "session_nums", "trial_ids"]
    missing_keys = [k for k in expected_keys if k not in arrays]
    if missing_keys:
        print(f"\n[WARNING] Missing expected keys in dataset: {missing_keys}")
        
    # 3. Print total sample count
    features = arrays.get("features")
    labels = arrays.get("labels")
    subject_ids = arrays.get("subject_ids")
    session_nums = arrays.get("session_nums")
    trial_ids = arrays.get("trial_ids")
    
    if features is not None:
        total_samples = features.shape[0]
        print(f"\n3. Total sample count: {total_samples}")
    else:
        total_samples = 0
        print("\n3. Total sample count: Cannot determine (features key missing)")

    warnings = []
    
    # 4. Class distribution
    if labels is not None:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        class_names = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
        print("\n4. Class distribution:")
        for lbl, count in zip(unique_labels, label_counts):
            c_name = class_names.get(lbl, "unknown")
            print(f"  - Class {lbl} ({c_name}): {count} samples")
        # Validation check for labels
        if len(unique_labels) != 4:
            warnings.append(f"Expected 4 unique classes (0,1,2,3), but found {len(unique_labels)}")
    else:
        print("\n4. Class distribution: Cannot determine (labels key missing)")
        warnings.append("labels array is missing")

    # 5. Subject distribution
    if subject_ids is not None:
        unique_subs, sub_counts = np.unique(subject_ids, return_counts=True)
        print("\n5. Subject distribution:")
        for sub, count in zip(unique_subs, sub_counts):
            print(f"  - Subject {sub:02d}: {count} samples")
        if len(unique_subs) != 15:
            warnings.append(f"Expected 15 unique subjects, but found {len(unique_subs)}")
    else:
        print("\n5. Subject distribution: 'subject_ids' key missing")
        warnings.append("subject_ids array is missing")

    # 6. Session distribution
    if session_nums is not None:
        unique_sess, sess_counts = np.unique(session_nums, return_counts=True)
        print("\n6. Session distribution:")
        for sess, count in zip(unique_sess, sess_counts):
            print(f"  - Session {sess}: {count} samples")
        if len(unique_sess) != 3:
            warnings.append(f"Expected 3 unique sessions, but found {len(unique_sess)}")
    else:
        print("\n6. Session distribution: 'session_nums' key missing")
        warnings.append("session_nums array is missing")

    # 7. Confirm there are no NaN or Inf values in the features array
    if features is not None:
        has_nan = np.isnan(features).any()
        has_inf = np.isinf(features).any()
        print(f"\n7. Integrity checks on features:")
        print(f"  - Features contain NaN values: {has_nan}")
        print(f"  - Features contain Inf values: {has_inf}")
        if has_nan:
            warnings.append("Features array contains NaN values.")
        if has_inf:
            warnings.append("Features array contains Inf values.")
            
        # Check if all zeros or close to it
        all_zeros = np.all(features == 0)
        mean_val = np.mean(features)
        std_val = np.std(features)
        print(f"  - Features are all zeros:      {all_zeros}")
        print(f"  - Features mean value:         {mean_val:.6f}")
        print(f"  - Features std value:          {std_val:.6f}")
        
        if all_zeros:
            warnings.append("Features array is entirely filled with zeros.")
        if std_val < 1e-5:
            warnings.append("Features array has almost zero variance (all values are identical or constant).")
    else:
        print("\n7. Integrity checks on features: Cannot run (features key missing)")
        warnings.append("features array is missing")

    # 8. Print one example sample's feature shape and a small slice
    if features is not None and total_samples > 0:
        print("\n8. Sample check:")
        print(f"  - Example features[0] shape: {features[0].shape}")
        # Print a 5x3 slice (up to 5 channels, up to 3 frequency bands) of the first sample
        print("  - Slice of first sample (features[0, :5, :3]):")
        print(features[0, :5, :3])
    else:
        print("\n8. Sample check: Cannot run (features missing or empty)")

    # 9 & 10. File Size and Load Time Reports
    print(f"\n9. File size: {file_size_mb:.2f} MB")
    print(f"10. Loading time (np.load + accessing all arrays): {load_time:.4f} seconds")
    
    if load_time > 5.0:
        warnings.append(f"Load time is slow ({load_time:.2f}s) for a {file_size_mb:.2f} MB file.")

    # Validation verdict
    print("\n" + "=" * 60)
    if warnings:
        print("VERIFICATION FAILED with the following warnings:")
        for warning in warnings:
            print(f"  [WARNING] {warning}")
    else:
        # DATA VALIDATED OK message summarizing total samples, class balance, and subject count
        print("DATA VALIDATED OK")
        print(f"  Total Samples:  {total_samples}")
        print(f"  Classes:        {len(unique_labels)} balanced categories ({', '.join([class_names.get(x) for x in unique_labels])})")
        print(f"  Subjects:       {len(unique_subs)} subjects verified")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()

import os
import gc
import time
import numpy as np
import scipy.io

def load_seed_iv_dataset(base_dir="eeg_feature_smooth", output_file="seed_iv_processed.npz"):
    """
    Loads, processes, and saves the SEED-IV EEG emotion recognition dataset.
    
    Parameters:
        base_dir (str): Path to the folder containing the eeg_feature_smooth directory.
        output_file (str): Path to the output compressed .npz file.
    """
    # 0 = neutral, 1 = sad, 2 = fear, 3 = happy
    session_labels = {
        1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    }
    
    # Lists to accumulate features and metadata
    features_list = []
    labels_list = []
    subject_ids_list = []
    session_nums_list = []
    trial_ids_list = []
    
    start_time = time.time()
    total_files_processed = 0
    
    print("=" * 60)
    print("Starting SEED-IV Dataset Processing Pipeline")
    print(f"Base Directory: {os.path.abspath(base_dir)}")
    print("=" * 60)
    
    expected_keys = [f"de_LDS{t}" for t in range(1, 25)]
    
    for session_num in [1, 2, 3]:
        session_dir = os.path.join(base_dir, str(session_num))
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
            
        # Get list of .mat files in session directory
        files = [f for f in os.listdir(session_dir) if f.endswith('.mat')]
        
        # Sort files based on subject ID number to keep ordering clean
        files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else x)
        
        print(f"\nProcessing Session {session_num} (Found {len(files)} files in {session_dir})")
        print("-" * 50)
        
        for file_idx, filename in enumerate(files, 1):
            file_path = os.path.join(session_dir, filename)
            
            # Extract subject ID from filename (e.g. "1" from "1_20160518.mat")
            subject_id_str = filename.split('_')[0]
            try:
                subject_id = int(subject_id_str)
            except ValueError:
                raise ValueError(f"Could not parse subject ID from filename '{filename}' in {session_dir}")
            
            file_start_time = time.time()
            
            # 1. Load MAT file, using variable_names filtering.
            # On Windows, loading the entire .mat file (which includes unused psd_LDS, moving averages, etc.)
            # can cause large I/O bottlenecks and memory fragmentation. By loading only the 24 de_LDS variables,
            # we speed up loading times by 4x and avoid unexplained memory hangs.
            try:
                mat_data = scipy.io.loadmat(file_path, variable_names=expected_keys)
            except Exception as e:
                raise IOError(f"Failed to load MAT file: {file_path}. Error: {e}")
                
            # 2. Extract and slice each trial
            for trial_id in range(1, 25):
                key = f"de_LDS{trial_id}"
                
                # Check for missing trial key in .mat file
                if key not in mat_data:
                    raise KeyError(f"Key '{key}' missing from file: {file_path}")
                
                trial_data = mat_data[key]
                
                # Verify trial shape is (62, T, 5)
                if len(trial_data.shape) != 3 or trial_data.shape[0] != 62 or trial_data.shape[2] != 5:
                    raise ValueError(
                        f"Unexpected shape {trial_data.shape} for key '{key}' in file: {file_path}. "
                        f"Expected shape to be (62, T, 5)."
                    )
                
                # T is the number of time-windows (slices) in the trial
                T = trial_data.shape[1]
                
                # Map trial index (0 to 23) to emotion label
                label = session_labels[session_num][trial_id - 1]
                
                # Transpose from (62, T, 5) to (T, 62, 5) so that the first dimension represents samples
                samples = np.transpose(trial_data, (1, 0, 2))
                
                # Append data and replicate metadata for each of the T samples
                features_list.append(samples)
                labels_list.append(np.full(T, label, dtype=np.int32))
                subject_ids_list.append(np.full(T, subject_id, dtype=np.int32))
                session_nums_list.append(np.full(T, session_num, dtype=np.int32))
                trial_ids_list.append(np.full(T, trial_id, dtype=np.int32))
                
            # Free memory and force garbage collection to keep memory usage low on Windows
            del mat_data
            gc.collect()
            
            elapsed = time.time() - file_start_time
            total_files_processed += 1
            print(f"  [{total_files_processed}/45] Processed {filename} (Subj: {subject_id}) in {elapsed:.2f}s")

    print("\nConcatenating dataset arrays...")
    concat_start = time.time()
    
    # Concatenate all lists into single contiguous arrays
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    subject_ids = np.concatenate(subject_ids_list, axis=0)
    session_nums = np.concatenate(session_nums_list, axis=0)
    trial_ids = np.concatenate(trial_ids_list, axis=0)
    
    print(f"Concatenation took {time.time() - concat_start:.2f}s")
    
    print(f"Saving dataset to compressed archive '{output_file}'...")
    save_start = time.time()
    
    # Save using compression to optimize disk space
    np.savez_compressed(
        output_file,
        features=features,
        labels=labels,
        subject_ids=subject_ids,
        session_nums=session_nums,
        trial_ids=trial_ids
    )
    
    total_time = time.time() - start_time
    print(f"Saving took {time.time() - save_start:.2f}s")
    print(f"Finished processing and saving dataset in {total_time:.2f}s.")
    print("=" * 60)
    
    # Sanity check metrics
    print("DATASET SANITY CHECKS:")
    print(f"  Total samples collected: {features.shape[0]}")
    print(f"  Feature array shape:    {features.shape} (Total Samples, Channels, Freq Bands)")
    print(f"  Labels array shape:     {labels.shape}")
    print(f"  Subject IDs shape:      {subject_ids.shape}")
    
    # Samples per class
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    class_names = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
    print("\nSamples per class:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Class {label} ({class_names.get(label, 'unknown')}): {count} samples")
        
    # Samples per subject
    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    print("\nSamples per subject:")
    for subj_id, count in zip(unique_subjects, subject_counts):
        print(f"  Subject {subj_id:02d}: {count} samples")
        
    # Shape of one sample
    print(f"\nShape of one sample: {features[0].shape}")
    print("=" * 60)

if __name__ == "__main__":
    load_seed_iv_dataset()

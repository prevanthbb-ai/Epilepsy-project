
# Synthetic Seizure generation

def augment_seizures(X, y, augment_factor=1):
    """
    Augments the dataset by adding Gaussian-noise–based synthetic
    seizure segments if real seizures are insufficient.

    Parameters:
        X : EEG segments
        y : labels (0 = normal, 1 = seizure)
        augment_factor : how many noisy copies to generate per seizure

    Returns:
        X_aug, y_aug
    """

    seizure_data = X[y == 1]

    # Case 1 — Not enough real seizures, generate new ones from normal data
    if len(seizure_data) == 0:
        print("⚠️ No real seizures found. Generating synthetic seizures...")

        normal_data = X[y == 0]
        n_aug = max(10, int(len(normal_data) * 0.08))

        syn_seizures = normal_data[:n_aug] + np.random.normal(
            0, 0.5, normal_data[:n_aug].shape
        )

        X = np.vstack([X, syn_seizures])
        y = np.hstack([y, np.ones(len(syn_seizures), dtype=np.int32)])

        print(f"✅ Added {len(syn_seizures)} synthetic seizures.")
        return X, y

    # Case 2 — Real seizures exist, create noisy versions
    augmented = []
    for seg in seizure_data:
        for _ in range(augment_factor):
            noise = np.random.normal(0, 0.1, seg.shape)
            augmented.append(seg + noise)

    if augmented:
        X = np.vstack([X, np.array(augmented, dtype=np.float32)])
        y = np.hstack([y, np.ones(len(augmented), dtype=np.int32)])
        print(f"✅ Added {len(augmented)} synthetic seizures.")

    return X, y

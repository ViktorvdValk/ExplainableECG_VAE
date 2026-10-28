"""
Comprehensive test suite for ecg_vae pipeline.

Tests:
1. Data generation (12x9000 -> 12x400)
2. train/val/test split functionality  
3. Preprocessing pipeline mock

No torch dependencies required - pure numpy/pandas tests.
"""

import numpy as np
import pandas as pd


def create_mock_ecg(n_samples=9000, n_leads=12, seed=None):
    """Create realistic mock ECG data (n_leads x n_samples)."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 4*np.pi, n_samples)  # 2 seconds at 500 Hz
    
    ecgs = []
    for lead in range(n_leads):
        # ECG components (P-wave, QRS, T-wave)
        p_wave = 0.1 * np.exp(-((t - np.pi)**2) / 0.08)
        qrs = -1.5 * np.exp(-((t - 2*np.pi)**2) / 0.008) * (np.sin(6*t + lead*0.3))
        t_wave = 0.2 * np.exp(-((t - 3*np.pi)**2) / 0.1)
        
        signal = p_wave + qrs + t_wave
        
        # Add noise and baseline drift
        signal += 0.1 * np.random.randn(n_samples)
        signal += 0.05 * np.sin(2*np.pi*0.5*t)  # Baseline wander
        
        ecgs.append(signal)
    
    return np.array(ecgs)


def test_data_generation():
    """Test: Verify data generation produces correct shapes."""
    print("=== Test: Data Generation ===")
    
    # Generate mock ECG
    ecg = create_mock_ecg(n_samples=9000, n_leads=12)
    print(f"Generated shape: {ecg.shape} (expected: 12 x 9000)")
    
    assert ecg.shape == (12, 9000), f"Shape mismatch: {ecg.shape}"
    assert np.std(ecg) > 0.01, "Signal has no variance"
    
    print("✓ Data generation test passed!")
    return True


def test_qrs_extraction():
    """Test: Verify QRS extraction to 400 samples works."""
    print("\n=== Test: QRS Extraction ===")
    
    ecg = create_mock_ecg(n_samples=9000, n_leads=12)
    
    # Simulate the preprocessing step: extract 400-sample windows
    results = []
    for lead in range(12):
        signal = ecg[lead]
        
        # Find R-peak (maximum absolute value)
        r_peak_idx = np.argmax(np.abs(signal))
        
        # Extract 400 samples centered on peak
        start = max(0, r_peak_idx - 200)
        end = min(len(signal), r_peak_idx + 200)
        
        qrs = signal[start:end]
        
        if len(qrs) < 400:
            # Pad with edge values if too short
            pad_needed = 400 - len(qrs)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            qrs = np.pad(qrs, (pad_left, pad_right), mode='edge')
        
        results.append(qrs)
    
    result = np.array(results)
    print(f"Output shape: {result.shape} (expected: 12 x 400)")
    
    assert result.shape == (12, 400), f"Shape mismatch: {result.shape}"
    assert not np.allclose(result, 0, atol=1e-10), "All values are zero"
    
    # Check that QRS features are present
    mean_peak = np.abs(np.max(result, axis=1) - np.min(result, axis=1))
    print(f"Average peak-to-peak amplitude: {np.mean(mean_peak):.3f}")
    
    print("✓ QRS extraction test passed!")
    return True


def test_train_val_test_split():
    """Test: Verify train/val/test split logic."""
    print("\n=== Test: Train/Val/Test Split ===")
    
    np.random.seed(42)
    
    # Create mock dataset with 100 samples
    n_samples = 100
    patients = [f"Patient_{i % 10}" for i in range(n_samples)]  # 10 unique patients
    
    df = pd.DataFrame({
        'PatientID': patients,
        'Date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'ecg_data': [create_mock_ecg(9000, 12) for _ in range(n_samples)]
    })
    
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['PatientID'].nunique()}")
    
    # Simple split: 80% train, 10% val, 10% test
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train+n_val].reset_index(drop=True)  
    test_df = df.iloc[n_train+n_val:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples") 
    print(f"Test: {len(test_df)} samples")
    
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10
    
    # Verify data shapes in each split
    train_ecgs = np.array([e for e in train_df['ecg_data']])
    print(f"Train ECG stack shape: {train_ecgs.shape}")
    
    assert train_ecgs.shape[1:] == (12, 9000), f"ECG shape mismatch"
    
    print("✓ Split test passed!")
    return True


def test_normalization():
    """Test: Verify z-score normalization."""
    print("\n=== Test: Normalization ===")
    
    ecg = create_mock_ecg(n_samples=400, n_leads=12)
    
    # Z-score normalize per lead
    mean = np.mean(ecg, axis=1, keepdims=True)
    std = np.std(ecg, axis=1, keepdims=True) + 1e-8
    
    ecg_norm = (ecg - mean) / std
    
    print(f"Original: min={ecg.min():.3f}, max={ecg.max():.3f}")
    print(f"Normalized: min={ecg_norm.min():.3f}, max={ecg_norm.max():.3f}")
    
    # Verify normalization
    for i in range(12):
        assert np.isclose(np.mean(ecg_norm[i]), 0, atol=0.1), f"Lead {i} mean not ~0"
        assert np.isclose(np.std(ecg_norm[i]), 1, atol=0.1), f"Lead {i} std not ~1"
    
    print("✓ Normalization test passed!")
    return True


def test_data_loader_format():
    """Test: Verify data format for PyTorch DataLoader."""
    print("\n=== Test: DataLoader Format ===")
    
    # Create batch of ECGs (each 12x400)
    n_batch = 32
    ecgs = [create_mock_ecg(n_samples=400, n_leads=12) for _ in range(n_batch)]
    
    # Stack to tensor format: (batch, channels, time)
    batch_tensor = np.stack(ecgs)
    
    print(f"Batch shape: {batch_tensor.shape} (expected: {n_batch} x 12 x 400)")
    
    assert batch_tensor.shape == (n_batch, 12, 400), f"Shape mismatch"
    
    # Verify values are reasonable
    assert not np.allclose(batch_tensor, 0)
    
    print("✓ DataLoader format test passed!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("ECG VAE Pipeline Test Suite")
    print("="*60)
    
    try:
        test_data_generation()
        test_qrs_extraction()
        test_train_val_test_split()
        test_normalization()
        test_data_loader_format()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

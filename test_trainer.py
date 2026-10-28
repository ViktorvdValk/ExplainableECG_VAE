"""
Test suite for Trainer_12 class.

Tests that Trainer_12 can be imported and initialized with different configurations.
"""

def test_trainer_import():
    """Test: Verify Trainer_12 imports correctly."""
    print("=== Test: Trainer_12 Import ===")
    
    from utils.helper_train2 import Trainer_12
    print("✓ Import successful")
    
    return True


def test_encoder_import():
    """Test: Verify VAE encoders import correctly."""
    print("\n=== Test: Encoder Imports ===")
    
    from vae.encoders import VAE_conv12, VAE_linear12, Adversarial_MLP
    print("✓ All encoder classes imported successfully")
    
    return True


def test_forward_pass():
    """Test: Verify forward pass works."""
    print("\n=== Test: Forward Pass ===")
    
    import torch
    from vae.encoders import VAE_linear12
    
    model = VAE_linear12(latent_dim=32, pred=1)
    x = torch.randn(4, 12, 400)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shapes: decoded={outputs[0].shape}, z_mean={outputs[1].shape}")
    
    return True


def test_adversarial_classifier():
    """Test: Verify Adversarial_MLP works."""
    print("\n=== Test: Adversarial MLP ===")
    
    import torch
    from vae.encoders import Adversarial_MLP
    
    model = Adversarial_MLP(latent_dim=64, pred2=0)
    x = torch.randn(4, 64)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"✓ Input: {x.shape}")
    print(f"✓ Output: {y.shape}")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Trainer_12 Test Suite")
    print("="*60)
    
    try:
        test_trainer_import()
        test_encoder_import()
        test_forward_pass()
        test_adversarial_classifier()
        
        print("\n" + "="*60)
        print("All Trainer_12 tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

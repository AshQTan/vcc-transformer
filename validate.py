#!/usr/bin/env python3
"""
Validation script to test VCC Transformer installation and basic functionality.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    # Test optional dependencies
    optional_deps = {
        'flash_attn': 'Flash Attention',
        'wandb': 'Weights & Biases',
        'polars': 'Polars',
        'scanpy': 'Scanpy',
        'anndata': 'AnnData'
    }
    
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")
    
    # Test core project modules
    try:
        from vcc_transformer.utils.config import load_config
        print("✅ Configuration utilities")
    except ImportError as e:
        print(f"❌ Config utilities failed: {e}")
        return False
    
    try:
        from vcc_transformer.models.transformer import MultiTaskTransformer
        print("✅ Transformer model")
    except ImportError as e:
        print(f"❌ Transformer model failed: {e}")
        return False
    
    try:
        from vcc_transformer.training.losses import CombinedLoss
        print("✅ Loss functions")
    except ImportError as e:
        print(f"❌ Loss functions failed: {e}")
        return False
    
    try:
        from vcc_transformer.training.trainer import VCCTrainer
        print("✅ Trainer")
    except ImportError as e:
        print(f"❌ Trainer failed: {e}")
        return False
    
    try:
        from vcc_transformer.data.dataset import VCCDataset
        print("✅ Dataset")
    except ImportError as e:
        print(f"❌ Dataset failed: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from vcc_transformer.utils.config import load_config, validate_config
        
        # Test base config
        config = load_config('configs/base_config.yaml')
        validate_config(config)
        print("✅ Base configuration loaded and validated")
        
        # Test other configs
        for config_file in ['configs/large_config.yaml', 'configs/small_config.yaml']:
            if Path(config_file).exists():
                config = load_config(config_file)
                validate_config(config)
                print(f"✅ {config_file} loaded and validated")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and basic operations."""
    print("\n🤖 Testing model creation...")
    
    try:
        from vcc_transformer.utils.config import load_config
        from vcc_transformer.models.transformer import create_model
        
        # Load config
        config = load_config('configs/small_config.yaml')  # Use small config for testing
        
        # Create model
        model = create_model(config)
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        
        # Test forward pass
        import torch
        batch_size = 2
        seq_len = config.model.max_seq_length
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len)
        dummy_input[:, 0] = config.model.cls_token_id  # CLS token
        dummy_input[:, 1] = config.model.unk_pert_token_id  # PERT token
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Reconstruction output shape: {outputs['reconstruction'].shape}")
        print(f"   Classification output shape: {outputs['classification'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss function creation and computation."""
    print("\n📉 Testing loss functions...")
    
    try:
        from vcc_transformer.training.losses import CombinedLoss, compute_challenge_metrics
        import torch
        
        # Create loss function
        criterion = CombinedLoss(
            reconstruction_weight=1.0,
            classification_weight=0.5
        )
        
        # Create dummy data
        batch_size = 4
        n_genes = 100
        n_classes = 50
        
        recon_pred = torch.randn(batch_size, n_genes)
        recon_target = torch.randn(batch_size, n_genes)
        class_pred = torch.randn(batch_size, n_classes)
        class_target = torch.randint(0, n_classes, (batch_size,))
        
        # Compute loss
        loss_dict = criterion(recon_pred, recon_target, class_pred, class_target)
        
        print("✅ Loss computation successful")
        print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"   Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
        print(f"   Classification loss: {loss_dict['classification_loss'].item():.4f}")
        
        # Test challenge metrics
        metrics = compute_challenge_metrics(recon_pred, recon_target, class_target)
        print(f"✅ Challenge metrics computed: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🚀 VCC Transformer Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Model Creation Tests", test_model_creation),
        ("Loss Function Tests", test_loss_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! VCC Transformer is ready to use.")
        print("\nNext steps:")
        print("1. Place your data files in the 'data/' directory")
        print("2. Configure training parameters in 'configs/base_config.yaml'")
        print("3. Start training: python scripts/train.py --config configs/base_config.yaml")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("You may need to install missing dependencies or fix configuration issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

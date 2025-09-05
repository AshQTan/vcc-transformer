#!/bin/bash
"""
Setup script for VCC Transformer project.
This script handles the complete environment setup.
"""

set -e  # Exit on any error

echo "🚀 Setting up VCC Transformer project..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export CUDA_AVAILABLE=1
else
    echo "⚠️  No NVIDIA GPU detected. Training will use CPU."
    export CUDA_AVAILABLE=0
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA if available)
echo "🔥 Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = "1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# Install Flash Attention if CUDA is available
if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo "⚡ Installing Flash Attention..."
    pip install flash-attn --no-build-isolation || {
        echo "⚠️  Flash Attention installation failed. The model will fall back to standard attention."
    }
fi

# Install the package in development mode
echo "🔧 Installing VCC Transformer package..."
pip install -e .

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

# Run a quick test to verify installation
echo "🧪 Running installation test..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')

try:
    from src.vcc_transformer.utils.config import load_config
    config = load_config('configs/base_config.yaml')
    print('✅ Configuration loading test passed')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    exit(1)

try:
    from src.vcc_transformer.models.transformer import create_model
    model = create_model(config)
    print(f'✅ Model creation test passed. Parameters: {model.count_parameters():,}')
except Exception as e:
    print(f'❌ Model creation test failed: {e}')
    exit(1)

print('✅ All tests passed!')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Place your data files in the 'data/' directory"
echo "  3. Configure training in 'configs/base_config.yaml'"
echo "  4. Start training: python scripts/train.py --config configs/base_config.yaml"
echo ""
echo "For more information, see README.md"

"""
Setup script for VCC Transformer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        "torch>=2.1.0",
        "anndata>=0.9.0",
        "scanpy>=1.9.0",
        "polars>=0.19.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ]

setup(
    name="vcc-transformer",
    version="0.1.0",
    author="VCC Team",
    author_email="",
    description="High-Performance Multi-Task Transformer for Virtual Cell Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/vcc-transformer-project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "flash-attn": [
            "flash-attn>=2.3.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "full": [
            "flash-attn>=2.3.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "memory-profiler>=0.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vcc-train=vcc_transformer.scripts.train:main",
            "vcc-predict=vcc_transformer.scripts.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

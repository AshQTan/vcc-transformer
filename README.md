# VCC Transformer: High-Performance Multi-Task Transformer for Virtual Cell Challenge

A PyTorch implementation of a multi-task learning Transformer model for predicting gene expression profiles in response to single-gene knockout perturbations. This project is designed for the "A Cell is a Computer: The Virtual Cell Challenge".

## Key Features

- **Multi-Task Lea## Beautiful Training Experience

### Rea## Challenge Metrics

The model is evaluated using challenge-specific metrics:

- **Differential Expression Score (DES)**: Measures accuracy of up/down regulation predictions
- **Perturbation Discrimination Score (PDS)**: Assesses if predictions can identify perturbations
- **Mean Absolute Error (MAE)**: Direct prediction accuracy

## Advanced Usageogress Visualization

Rich console output includes:

- **Live Progress Bars**: Real-time epoch and step progress with ETA
- **Dynamic Metrics Table**: Live updating loss values and trends  
- **Best Metrics Tracking**: Automatic tracking of peak performance
- **Smart Time Estimates**: Accurate completion time predictions
- **Color-Coded Output**: Easy-to-read formatted console displaymultaneous reconstruction and classification for robust representations
- **Flash Attention**: Memory-efficient attention mechanism for handling large gene sequences
- **Beautiful Progress Tracking**: Real-time metrics and progress bars
- **Comprehensive Reporting**: Automatic generation of training plots and HTML reports
- **High Performance**: Optimized for multiple RTX 3090s with 64GB RAM
- **Distributed Training**: Built-in support for multi-GPU training with PyTorch DDP
- **Memory Efficient**: Automatic Mixed Precision (AMP) and gradient checkpointing
- **Production Ready**: Model compilation with `torch.compile()` for maximum speed

## Model Architecture and Logic

### Background: The Virtual Cell Challenge

The Virtual Cell Challenge represents a significant frontier in computational biology, where the goal is to predict how human cells respond to genetic perturbations at the molecular level. Specifically, researchers aim to understand how single-gene knockouts affect the expression patterns of thousands of other genes within the cell. This is fundamentally a systems biology problem that requires modeling complex gene regulatory networks and their responses to targeted interventions.

Traditional approaches to this problem have relied on mechanistic models or simple machine learning techniques that struggle to capture the high-dimensional, non-linear relationships between genes. The challenge becomes even more complex when considering that gene expression is inherently noisy, cells exist in heterogeneous populations, and perturbation effects can propagate through intricate regulatory cascades. Furthermore, the goal is not just to predict expression changes, but to do so for perturbations that were not seen during training, requiring models that can generalize across different types of genetic interventions.

### Why Transformers for Gene Expression?

The choice of transformer architecture for this biological problem is driven by several fundamental parallels between gene regulation and language modeling. Just as words in a sentence derive meaning from their context and relationships with other words, genes derive their expression levels from complex interactions with other genes in the cellular environment. Traditional neural networks process genes as independent features, missing the crucial relational information that drives biological systems.

Transformers excel at capturing long-range dependencies and contextual relationships through their attention mechanisms. In gene expression, a perturbation to one gene can affect other genes through multi-step regulatory cascades that span the entire transcriptome. The self-attention mechanism allows the model to directly connect any gene to any other gene, regardless of their physical distance in the input sequence, making it ideal for capturing these complex regulatory relationships.

Moreover, transformers can learn to attend to different aspects of the data at different layers. Early layers might focus on direct regulatory interactions, while deeper layers could capture higher-order network effects and emergent cellular responses. This hierarchical learning mirrors the actual organization of biological systems, where molecular interactions aggregate into pathway-level effects, which in turn determine cellular phenotypes.

The attention weights themselves provide interpretability that is crucial for biological applications. Unlike black-box models, transformers can show which genes are most important for predicting the response of other genes, potentially revealing novel regulatory relationships or validating known biological pathways.

### Architecture Overview and Information Flow

```
Input Data Processing:
Control Cell → [Gene Expression: 50,000 genes] → [HVG Selection: 5,000 genes] → [Normalization]
Perturbation → [Perturbation ID] → [Embedding Lookup]

Tokenization and Embedding:
[CLS] [PERT] [GENE_1] [GENE_2] ... [GENE_5000]
   ↓      ↓       ↓        ↓           ↓
 512d   512d    512d     512d       512d    ← Embedding Dimension

Positional Encoding (Learned):
   +      +       +        +           +
Position Embeddings (5002 positions)

Transformer Encoder Stack (8 layers):
Layer 1: MultiHead-Attention → LayerNorm → FeedForward → LayerNorm
Layer 2: MultiHead-Attention → LayerNorm → FeedForward → LayerNorm
   ⋮
Layer 8: MultiHead-Attention → LayerNorm → FeedForward → LayerNorm

Output Processing:
[CLS_final] [PERT_final] [GENE_1_final] ... [GENE_5000_final]
     ↓                        ↓                    ↓
Classification Head    Reconstruction Head     (5000 outputs)
     ↓                        ↓
Perturbation ID         Gene Expression Predictions

Loss Computation:
Classification Loss (CrossEntropy) + Reconstruction Loss (MSE)
           ↓
    Combined Loss (α * Recon + β * Class)
```

This architecture creates a unified information processing pipeline where perturbation context informs gene expression predictions, and predicted expression patterns must be consistent with perturbation identity.

### Why Multi-Task Learning?

The decision to implement multi-task learning with both reconstruction and classification objectives is driven by the complementary nature of these tasks and their ability to enforce biologically meaningful constraints on the learned representations.

The reconstruction task provides direct supervision on the quantitative accuracy of gene expression predictions. However, if used alone, the model might learn to minimize prediction error through shortcuts that don't capture true biological relationships. For instance, the model might learn that certain genes always have similar expression levels regardless of perturbation, leading to accurate but non-informative predictions.

The classification task serves as a crucial biological constraint. By requiring the model to identify which perturbation was applied based solely on the predicted expression changes, we ensure that the predictions contain perturbation-specific signatures. This creates a self-consistency requirement: the model's predictions must be accurate enough quantitatively (reconstruction) and distinctive enough qualitatively (classification) to be biologically meaningful.

This dual objective also improves generalization to unseen perturbations. The classification task forces the model to learn general principles about how different types of perturbations affect cellular systems, rather than memorizing specific gene expression patterns. When encountering a novel perturbation, the model can apply these learned principles to generate biologically plausible predictions.

The multi-task framework also provides built-in validation. If a model achieves high reconstruction accuracy but poor classification performance, it suggests that the predictions, while numerically close to true values, lack the biological specificity that would make them useful for understanding perturbation effects.

### Why Learned Positional Encodings?

Traditional transformer models use fixed sinusoidal positional encodings because position in text has a clear, universal meaning. However, gene expression data has fundamentally different positional characteristics that make learned encodings more appropriate.

Genes don't have an inherent sequential order like words in a sentence. The ordering of genes in our input sequence is arbitrary and based on practical considerations like alphabetical naming or chromosomal location. However, there are meaningful biological relationships that could be encoded positionally: genes in the same pathway might benefit from similar positional encodings, or genes that are frequently co-regulated might be positioned to facilitate attention patterns.

Learned positional encodings allow the model to discover and exploit these biological relationships. During training, the model can learn that certain positions should attend to each other more readily, effectively creating a learned gene organization that optimizes for the prediction task. This flexibility is crucial because optimal gene ordering for one type of analysis (e.g., pathway analysis) might be different from optimal ordering for another (e.g., chromosomal effects).

Additionally, learned positional encodings can capture domain-specific patterns in gene expression data. For example, if housekeeping genes tend to have stable expression patterns, they might develop similar positional encodings that facilitate their recognition as a group. This biological awareness emerges naturally from the data rather than being imposed by fixed mathematical functions.

### Why Flash Attention?

The implementation of Flash Attention is not just an optimization choice but a necessity for scaling to the dimensions required for comprehensive gene expression modeling. Standard attention mechanisms have quadratic memory complexity with respect to sequence length, making them prohibitively expensive for sequences of 5000+ genes.

Beyond the computational benefits, Flash Attention maintains the full accuracy of attention computations, ensuring that no biological information is lost in the optimization process. This is crucial because gene regulatory networks involve complex, potentially weak interactions that might be missed if attention computations were approximated or truncated.

The memory efficiency of Flash Attention also enables larger batch sizes during training, which is particularly important for biological data where noise and batch effects can be significant. Larger batches provide more stable gradient estimates and better normalization statistics, leading to more robust model training.

### Detailed Component Architecture

#### Input Processing and Tokenization Strategy

The transformation of gene expression data into transformer-compatible tokens involves several crucial design decisions. The selection of the top 5000 highly variable genes (HVGs) focuses computational resources on the most informative features while maintaining biological coverage. HVGs are genes that show the greatest variation across different conditions and cell types, making them most likely to respond to perturbations and carry information about cellular state.

The special token design serves specific biological purposes. The `[CLS]` token aggregates global information about the cellular response, similar to how it captures sentence-level meaning in language models. This token becomes the repository for perturbation-specific signatures that distinguish one intervention from another. The `[PERT]` token explicitly encodes perturbation information, ensuring that this crucial context is available to all layers of the transformer.

Gene expression values are embedded into a high-dimensional space (typically 512 dimensions) that allows the model to learn rich representations capturing not just expression magnitude but also contextual relationships. This embedding process is learned during training, enabling the model to discover optimal representations for the prediction task.

#### Attention Mechanism and Gene Interaction Discovery

Each attention head in the multi-head attention layers can specialize in different types of gene relationships. Some heads might focus on direct regulatory interactions, others on pathway-level relationships, and still others on cell-type-specific co-expression patterns. This specialization emerges naturally during training as different heads learn to attend to different aspects of the gene expression landscape.

The multi-head design is particularly powerful for biological data because gene regulation involves multiple, parallel regulatory mechanisms. Transcriptional regulation, post-transcriptional modification, epigenetic effects, and protein-protein interactions all contribute to gene expression patterns. Different attention heads can capture these different regulatory layers, providing a comprehensive view of cellular responses.

The self-attention mechanism enables the model to discover both known and novel gene interactions. Known pathways should produce recognizable attention patterns, serving as validation for the model's biological understanding. Novel attention patterns might reveal previously unknown regulatory relationships, making the model not just predictive but potentially discovery-enabling.

#### Task-Specific Output Architectures

The reconstruction head operates on individual gene tokens, taking their final hidden states and projecting them to expression values. This design ensures that each gene's prediction is informed by the full cellular context while maintaining gene-specific prediction capability. The head learns a linear transformation that maps from the transformer's learned representation space to biologically meaningful expression levels.

The classification head takes a fundamentally different approach, using the global information aggregated in the `[CLS]` token. This design reflects the biological reality that perturbation identity is determined by the overall pattern of expression changes rather than individual gene values. The classification head must learn to recognize perturbation-specific signatures in the aggregate cellular response.

#### Loss Function Design and Optimization Dynamics

The combined loss function creates a dynamic balance between quantitative accuracy and biological specificity. The reconstruction loss (typically Mean Squared Error) provides dense supervision signals for every gene, ensuring that the model learns accurate quantitative relationships. The classification loss (Cross-Entropy) provides sparse but crucial supervision that ensures predictions maintain biological meaning.

The relative weighting of these losses (α for reconstruction, β for classification) can be adjusted based on the specific requirements of the application. Higher reconstruction weight emphasizes numerical accuracy, while higher classification weight emphasizes biological interpretability. The framework includes adaptive weighting mechanisms that can automatically balance these objectives based on training dynamics.

### Integration and Emergent Properties

When all these components work together, several emergent properties arise that make the model particularly suitable for cellular perturbation prediction. The combination of attention mechanisms and multi-task learning creates representations that are both quantitatively accurate and biologically meaningful. The model learns to predict not just what gene expression levels will be, but why they should be those levels based on the perturbation applied.

The architecture naturally handles the variable effects of different perturbations. Some perturbations might have strong, focused effects on specific pathways, while others might have weaker, more distributed effects. The attention mechanism can adapt to these different perturbation styles, focusing attention narrowly for targeted effects or broadly for systemic changes.

The multi-task framework also provides natural uncertainty quantification. When the model is confident in its predictions, both reconstruction and classification performance should be high. When the model encounters novel or difficult perturbations, mismatches between reconstruction and classification performance can indicate uncertainty and guide further investigation.

This integrated architecture transforms the challenge of cellular perturbation prediction from a simple regression problem into a comprehensive modeling framework that captures the complexity, interpretability, and biological reality of gene regulatory systems.

## Technical Architecture Overview

The model implements a Transformer-based architecture with:

1. **Input Representation**: 
   - `[CLS]` token for global cell representation
   - `[PERT]` token for perturbation information
   - Gene expression tokens for top 5000 highly variable genes

2. **Multi-Task Outputs**:
   - **Reconstruction Head**: Predicts post-perturbation gene expression
   - **Classification Head**: Identifies which perturbation was applied

3. **Performance Optimizations**:
   - Flash Attention for O(N) memory complexity
   - Learned positional encodings
   - Gradient checkpointing for large models
   - Mixed precision training

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 64GB+ RAM recommended
- Multiple GPUs recommended for distributed training

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd vcc-transformer-project

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

### Development Install

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black isort flake8
```

## Data Preparation

### Expected Data Structure

```
data/
├── adata_Training.h5ad          # Training data with control and perturbed cells
├── pert_counts_Validation.csv   # Validation perturbations to predict
└── preprocessing_info.npz       # Saved preprocessing parameters (auto-generated)
```

### Data Format Requirements

The training `.h5ad` file should contain:
- `obs['perturbation']`: Perturbation names ('Non-Targeting Control' for controls)
- Gene expression matrix in `.X`
- Gene names in `.var_names`

## Quick Start

### 1. Configure Training

Edit `configs/base_config.yaml`:

```yaml
# Essential configurations
data:
  training_file: "data/adata_Training.h5ad"
  validation_file: "data/pert_counts_Validation.csv"
  n_highly_variable_genes: 5000

model:
  d_model: 512        # Hidden dimension
  n_layers: 8         # Number of transformer layers
  n_heads: 16         # Number of attention heads
  
training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 100
```

### 2. Train the Model

```bash
# Single GPU training
python scripts/train.py --config configs/base_config.yaml

# Multi-GPU training (auto-detects GPUs)
python scripts/train.py --config configs/base_config.yaml --world-size 4

# Resume from checkpoint
python scripts/train.py --config configs/base_config.yaml --resume-from checkpoints/best_model.pt
```

### 3. Generate Predictions

```bash
python scripts/predict.py \
    --config configs/base_config.yaml \
    --model-path checkpoints/best_model.pt \
    --validation-file data/pert_counts_Validation.csv \
    --training-file data/adata_Training.h5ad \
    --output-file predictions.h5ad \
    --run-cell-eval
```

### 4. View Training Progress (Demo)

```bash
# See beautiful training progress in action
python demo_progress.py
```

### 5. Generate Training Reports

```bash
# Generate comprehensive training report with plots
python scripts/generate_report.py \
    --history-file logs/training_history.json \
    --output-dir reports

# Generate plots only
python scripts/generate_report.py \
    --history-file logs/training_history.json \
    --plots-only
```

## Configuration

### Model Architecture

```yaml
model:
  d_model: 512                    # Hidden dimension
  n_layers: 8                     # Transformer layers
  n_heads: 16                     # Attention heads
  d_ff: 2048                      # Feed-forward dimension
  dropout: 0.1                    # Dropout probability
  use_flash_attention: true       # Enable Flash Attention
  use_learned_pe: true           # Learned positional encoding
```

### Training Optimization

```yaml
training:
  # Core settings
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 100
  
  # Loss weighting
  reconstruction_weight: 1.0      # Weight for MSE loss
  classification_weight: 0.5      # Weight for CE loss (beta)
  
  # Performance optimizations
  use_amp: true                   # Automatic Mixed Precision
  compile_model: true             # Model compilation
  use_gradient_checkpointing: false
  gradient_clip_norm: 1.0
```

### Hardware Configuration

```yaml
distributed:
  use_ddp: true                   # Distributed Data Parallel
  backend: "nccl"                 # Backend for multi-GPU

dataloader:
  num_workers: 8                  # Data loading workers
  pin_memory: true                # Pin memory for faster GPU transfer
  persistent_workers: true        # Keep workers alive

# GPU Power Management (Optional)
gpu:
  enable_undervolting: false      # Enable GPU undervolting for power efficiency
  undervolt_settings:
    core_offset: -100             # Core voltage offset in mV (negative = undervolt)
    memory_offset: -50            # Memory voltage offset in mV
    power_limit: 80               # Power limit as percentage of max TDP (e.g., 80%)
    temp_limit: 83                # Temperature limit in Celsius
    fan_curve_aggressive: true    # Use aggressive fan curve for better cooling
  auto_optimize: false            # Automatically find optimal undervolt settings
  safety_checks: true             # Enable safety checks and monitoring
```

## Monitoring and Logging

### Console Progress

The trainer provides detailed real-time progress tracking:

```
VCC Transformer Training
Model: 512d, 8 layers, 16 heads
Batch Size: 32 | Learning Rate: 0.0001

Epoch ████████████████████░░░░░░░░  75% • 2:30:45 • 0:50:15
Step  ████████████████████████████ 100% • 487/487

┌─────────────────── Training Metrics ───────────────────┐
│ Metric        │      Current │         Best │    Trend │
│ Loss          │       0.2341 │       0.2205 │        ↓ │
│ Recon Loss    │       0.1892 │       0.1756 │        ↓ │
│ Class Loss    │       0.0449 │       0.0421 │        ↓ │
│ Learning Rate │    8.45e-05  │    1.00e-04  │        → │
└─────────────────────────────────────────────────────────┘

Epoch 15 Complete
┌─── Epoch 15 Summary ────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Split                   │ Loss    │ Recon   │ Class   │ MAE     │ Corr    │
│ Train                   │ 0.2341  │ 0.1892  │ 0.0449  │ -       │ -       │
│ Validation              │ 0.2456  │ 0.1967  │ 0.0489  │ 0.1234  │ 0.7891  │
└─────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### Automatic Training Reports

After training, get comprehensive HTML reports with:

- **Interactive loss curves** showing training progression
- **Challenge metrics** (DES, PDS, MAE) over time  
- **Training speed** analysis and bottleneck identification
- **Model configuration** summary and hyperparameters
- **Best metrics** achieved during training

### Built-in Logging

The trainer provides comprehensive logging:
- Training/validation losses with beautiful formatting
- Learning rate schedules and optimization metrics
- Challenge-specific metrics (MAE, correlation)  
- GPU memory usage and training time per epoch
- Real-time progress bars and ETA estimates

### Weights & Biases Integration

```yaml
logging:
  use_wandb: true
  wandb_project: "vcc-transformer"
  wandb_entity: "your-entity"
```

Enable with:
```bash
pip install wandb
wandb login
```

## � Beautiful Training Experience

### Real-Time Progress Visualization

Experience training like never before with rich console output:

- **🎯 Live Progress Bars**: Real-time epoch and step progress with ETA
- **📊 Dynamic Metrics Table**: Live updating loss values and trends  
- **🏆 Best Metrics Tracking**: Automatic tracking of peak performance
- **⏱️ Smart Time Estimates**: Accurate completion time predictions
- **🎨 Color-Coded Output**: Easy-to-read formatted console display

### Comprehensive Training Reports

Automatically generated after training:

- **Loss Evolution Plots**: Detailed matplotlib visualizations
- **Challenge Metrics Dashboard**: DES, PDS, MAE trends over time
- **Performance Analysis**: Training speed and bottleneck identification  
- **HTML Summary Report**: Complete training overview in your browser
- **Exportable Data**: JSON training history for further analysis

### Demo the Experience

```bash
# See the beautiful progress tracking in action
python demo_progress.py
```

## �🏆 Challenge Metrics

The model is evaluated using challenge-specific metrics:

- **Differential Expression Score (DES)**: Measures accuracy of up/down regulation predictions
- **Perturbation Discrimination Score (PDS)**: Assesses if predictions can identify perturbations
- **Mean Absolute Error (MAE)**: Direct prediction accuracy

## 🔧 Advanced Usage

### Hyperparameter Tuning

Test different architectures:

```yaml
# Large model
model:
  d_model: 1024
  n_layers: 12
  n_heads: 16

# Efficient model
model:
  d_model: 256
  n_layers: 6
  n_heads: 8
```

### Custom Loss Functions

The framework supports multiple loss types:

```yaml
training:
  loss_type: "combined"     # Standard combined loss
  # loss_type: "adaptive"   # Adaptive weight adjustment
```

### Memory Optimization

For large models or limited memory:

```yaml
training:
  use_gradient_checkpointing: true
  batch_size: 16              # Reduce batch size

memory:
  empty_cache_every_n_steps: 50
  max_memory_usage: 0.9       # Use 90% of GPU memory
```

### GPU Power Management and Undervolting

For improved power efficiency and thermal management during long training runs:

```yaml
gpu:
  enable_undervolting: true
  undervolt_settings:
    core_offset: -100           # Reduce core voltage by 100mV
    memory_offset: -50          # Reduce memory voltage by 50mV
    power_limit: 80             # Limit power to 80% of TDP
    temp_limit: 83              # Keep temperatures under 83°C
    fan_curve_aggressive: true  # Use aggressive cooling
  auto_optimize: true           # Find optimal settings automatically
  safety_checks: true           # Enable monitoring and safety
```

**Benefits of GPU Undervolting:**
- **Lower Power Consumption**: Reduce electricity costs and heat generation
- **Better Thermal Performance**: Lower temperatures extend GPU lifespan
- **Quieter Operation**: Less heat means slower fan speeds
- **Stable Long Training**: Consistent performance during extended training sessions
- **Multi-GPU Efficiency**: Particularly beneficial when running multiple RTX 3090s

**Safety Features:**
- Automatic revert to stock settings if instability detected
- Real-time temperature and power monitoring
- Gradual voltage reduction with stability testing
- Emergency failsafe mechanisms

**Usage:**
```bash
# Train with undervolting enabled
python scripts/train.py --config configs/base_config.yaml --enable-gpu-optimization

# Auto-optimize GPU settings before training
python scripts/optimize_gpu.py --config configs/base_config.yaml --find-optimal

# Monitor GPU during training
python scripts/monitor_gpu.py --log-file gpu_monitoring.log
```

## Project Structure

```
vcc-transformer-project/
├── configs/
│   └── base_config.yaml         # Main configuration
├── data/                        # Data directory
├── scripts/
│   ├── train.py                # Training script
│   └── predict.py              # Prediction script
├── src/vcc_transformer/
│   ├── data/
│   │   └── dataset.py          # Dataset implementation
│   ├── models/
│   │   └── transformer.py      # Model architecture
│   ├── training/
│   │   ├── trainer.py          # Training logic
│   │   └── losses.py           # Loss functions
│   └── utils/
│       └── config.py           # Configuration utilities
├── requirements.txt
└── README.md
```

## Testing and Validation

### Unit Tests

```bash
pytest tests/ -v
```

### Model Validation

```bash
# Test model forward pass
python -c "
from src.vcc_transformer.models.transformer import create_model
from src.vcc_transformer.utils.config import load_config
config = load_config('configs/base_config.yaml')
model = create_model(config)
print(f'Model parameters: {model.count_parameters():,}')
"
```

## Troubleshooting

### Common Issues

1. **Flash Attention Installation**:
   ```bash
   pip install flash-attn --no-build-isolation
   # If fails, install without Flash Attention:
   # Set use_flash_attention: false in config
   ```

2. **CUDA Out of Memory**:
   - Reduce `batch_size`
   - Enable `use_gradient_checkpointing: true`
   - Reduce `d_model` or `n_layers`

3. **Distributed Training Issues**:
   ```bash
   export NCCL_DEBUG=INFO  # Enable NCCL debugging
   ```

4. **Data Loading Errors**:
   - Ensure `.h5ad` files are readable by scanpy
   - Check file paths in configuration
   - Verify gene names and perturbation labels

5. **GPU Undervolting Issues**:
   ```bash
   # Install GPU monitoring dependencies
   pip install nvidia-ml-py3 pynvml
   
   # Check if undervolting is supported
   python scripts/optimize_gpu.py --status
   
   # If undervolting fails, disable it in config:
   # Set enable_undervolting: false
   ```

6. **Permission Issues for GPU Optimization**:
   ```bash
   # Linux: Run with sudo for voltage changes
   sudo python scripts/optimize_gpu.py --apply-settings
   
   # Windows: Run as Administrator
   # Note: Some features may require MSI Afterburner or similar tools
   ```

### Performance Tips

1. **Optimal Batch Size**: Start with 32, adjust based on memory
2. **Learning Rate**: Use 1e-4 with warmup for stable training
3. **Multi-GPU**: Use DDP for best performance across GPUs
4. **Data Loading**: Set `num_workers` to 2x number of CPU cores
5. **Progress Tracking**: The rich console output is automatically enabled
6. **Report Generation**: Run after training for comprehensive analysis

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Multi-Task Learning in Deep Learning](https://arxiv.org/abs/1706.05098)
- [Virtual Cell Challenge](https://www.kaggle.com/competitions/virtual-cell-challenge)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review configuration examples

---

**Built for the Virtual Cell Challenge - Pushing the boundaries of cellular modeling with transformers!**

# Corn Leaf Disease Classification and Severity Quantification

An integrated deep learning pipeline for automated corn leaf disease detection and severity assessment using computer vision and deep learning techniques.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project presents a complete pipeline that not only **classifies** corn leaf diseases but also **quantifies** their severity by calculating the percentage of affected leaf area. The system bridges the gap between laboratory conditions and real-field applications through robust preprocessing and data augmentation strategies.

### Key Features

- **Multi-class Disease Classification**: Identifies 4 classes (3 diseases + healthy leaves) with 80%+ Dice coefficient
- **Zero-shot Leaf Segmentation**: Utilizes Segment Anything Model (SAM) with Convex Hull refinement for robust leaf isolation
- **Adaptive Severity Quantification**: Disease-specific HSV thresholding to measure affected area percentage
- **Lab-to-Field Generalization**: Aggressive data augmentation enables classification of field images despite training on laboratory data
- **End-to-End Pipeline**: From raw image input to complete diagnostic report with severity metrics

## 🌽 Supported Diseases

1. **Northern Leaf Blight** (Tizón Foliar del Norte)
2. **Gray Leaf Spot** (Mancha Gris de la Hoja)
3. **Common Rust** (Roya Común)
4. **Healthy Leaves** (Hojas Sanas)

## 🏗️ Pipeline Architecture

```
Input Image
    ↓
┌─────────────────────┐
│  Preprocessing      │
│  - Conditional HSV  │
│  - L*a*b* CLAHE    │
│  - AMF Filtering    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Classification     │
│  (ResNet50V2)       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Segmentation       │
│  - SAM Generation   │
│  - Convex Hull      │
│  - Morphological    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Quantification     │
│  - Adaptive HSV     │
│  - Area Calculation │
└─────────────────────┘
    ↓
Diagnosis + Severity (%)
```

## 📊 Dataset

The project uses a specialized subset of the **New PlantVillage** dataset, focusing exclusively on corn images:

- **Source**: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Total Images**: 3,852 original images
- **Augmented Dataset**: 6,000 images (1,500 per class)
- **Split**: 80% training (4,800 images) / 20% test (1,200 images)

### Dataset Composition

| Class | Number of Samples |
|-------|-------------------|
| Northern Leaf Blight | 985 |
| Gray Leaf Spot | 513 |
| Common Rust | 1,192 |
| Healthy Leaves | 1,162 |
| **Total** | **3,852** |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Corn-Leaf-Disease-Classification-and-Severity-Quantification.git
cd Corn-Leaf-Disease-Classification-and-Severity-Quantification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SAM model weights:
```bash
# Download vit_b checkpoint from Meta AI
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 📦 Project Structure

```
├── 1-Preprocesamiento.ipynb      # Data preprocessing pipeline
├── 2-Modelado.ipynb              # Model training and evaluation
├── 4-segmentacion.ipynb          # Base segmentation module
├── 5-clasif-loop.ipynb           # Classification pipeline
├── 6-deteccion-area.ipynb           # Area quantification module
├── IEEE_Maiz_ResNet.pdf             # Research paper
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Usage

### 1. Preprocessing

Run the preprocessing notebook to apply multi-space enhancement:

```bash
jupyter notebook 1-Preprocesamiento.ipynb
```

This applies:
- Conditional HSV enhancement (Gray Leaf Spot only)
- L*a*b* CLAHE for contrast improvement
- Adaptive Median Filter (AMF) for noise reduction

### 2. Model Training

Train the ResNet50V2 classifier:

```bash
jupyter notebook 2-Modelado.ipynb
```

**Training Configuration:**
- Architecture: ResNet50V2 with custom head
- Fine-tuning: Last 40 layers unfrozen
- Optimizer: Adam (lr=0.0007)
- Loss: 50% Categorical Crossentropy + 50% Dice Loss
- Epochs: 250 (with EarlyStopping patience=55)
- Batch size: 16

### 3. Segmentation

Process images through the SAM-based segmentation pipeline:

```bash
jupyter notebook 4-segmentacion.ipynb
```

The segmentation module:
1. Generates multiple masks using SAM
2. Filters masks by area and aspect ratio
3. Creates solid convex mask using Convex Hull
4. Applies morphological refinement
5. Crops and pads leaf to 256×256 pixels

### 4. Complete Pipeline

Run the full classification and quantification pipeline:

```bash
jupyter notebook 5-clasif-loop.ipynb
```

### 5. Severity Quantification

Calculate affected area percentage:

```bash
jupyter notebook 6-deteccion-area.ipynb
```

**Disease-Specific HSV Thresholds:**

| Disease | Hue (H) | Saturation (S) | Value (V) |
|---------|---------|----------------|-----------|
| Common Rust | 5–25 | 60–255 | 50–255 |
| Northern Leaf Blight | 8–30 | 30–255 | 30–255 |
| Gray Leaf Spot | L* channel analysis (L*a*b* space) |

## 📈 Results

### Classification Performance

| Metric | Value |
|--------|-------|
| **Dice Coefficient** | 0.8044 |
| **Accuracy** | 0.8183 |
| **Weighted Precision** | 0.87 |
| **Weighted Recall** | 0.82 |
| **Weighted F1-Score** | 0.83 |

### Per-Class Performance

| Class | Dice Coefficient | Precision | Recall | F1-Score |
|-------|-----------------|-----------|--------|----------|
| Northern Leaf Blight | 0.7628 | 0.94 | 0.68 | 0.79 |
| Gray Leaf Spot | 0.7422 | 0.60 | 0.96 | 0.74 |
| Common Rust | 0.8462 | 0.95 | 0.81 | 0.87 |
| Healthy | 0.8796 | 1.00 | 0.83 | 0.91 |

## 🔬 Methodology Highlights

### 1. Multi-Space Preprocessing
- **Conditional HSV Enhancement**: Applied only to Gray Leaf Spot class for subtle lesion detection
- **L*a*b* CLAHE**: Contrast enhancement independent of color channels
- **Adaptive Median Filter**: Noise reduction while preserving edges

### 2. Robust Data Augmentation
Simulates field conditions from lab images:
- Rotation (up to 45°)
- Zoom (up to 0.2)
- Horizontal flip
- Brightness variation (1.1–1.5)
- Width/height shift (0.2)
- Shear transformation (0.2)

### 3. SAM + Convex Hull Segmentation
Novel approach that:
- Generates solid, continuous leaf masks
- Eliminates internal discontinuities
- Works in zero-shot mode (no custom training required)
- Applies morphological refinement for clean edges

### 4. Adaptive Quantification
- Uses classification output to select disease-specific detection profile
- Calculates severity as: `Severity (%) = (Lesion Pixels / Total Leaf Pixels) × 100`

## 🎯 Key Innovations

1. **Complete Pipeline**: Unlike most works that stop at classification, this system provides both diagnosis and severity metrics

2. **Lab-to-Field Generalization**: Aggressive augmentation strategy enables model trained on laboratory images to classify field images with complex backgrounds

3. **Novel Segmentation Method**: Combination of SAM, Convex Hull filling, and morphological operations ensures solid, continuous masks without custom segmentation model training

4. **Adaptive Quantification**: Disease-specific HSV profiles selected based on classification output for accurate lesion detection

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{rodriguez2024corn,
  title={Sistema Integral para la Clasificación y Cuantificación de Severidad en Enfermedades Foliares del Maíz usando Deep Learning y Segmentación Avanzada},
  author={Rodriguez, Jose Emanuel},
  journal={Universidad Nacional de Luján},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Hughes, D.P. and Salathe, M. (2015)
- **Segment Anything Model (SAM)**: Meta AI Research
- **ResNet50V2**: He, K., Zhang, X., Ren, S., & Sun, J. (2016)
- Universidad Nacional de Luján, Argentina

## 📧 Contact

Jose Emanuel Rodriguez - joseemaro@hotmail.com

Project Link: [https://github.com/yourusername/Corn-Leaf-Disease-Classification-and-Severity-Quantification](https://github.com/yourusername/Corn-Leaf-Disease-Classification-and-Severity-Quantification)

---

## 🔮 Future Work

- [ ] Expand dataset with real field images
- [ ] Explore Vision Transformers architecture
- [ ] Implement mobile/web application
- [ ] Add quantitative validation against expert assessments
- [ ] Extend to other corn diseases and crops
- [ ] Optimize for edge device deployment

## 📚 References

1. Kirillov, A., et al. (2023). "Segment anything." arXiv preprint arXiv:2304.02643
2. Hughes, D.P., & Salathe, M. (2015). "An open access repository of images on plant health"
3. He, K., et al. (2016). "Deep residual learning for image recognition." CVPR
4. Mohanty, S.P., et al. (2016). "Using deep learning for image-based plant disease detection"

---

⭐ If you find this project useful, please consider giving it a star!

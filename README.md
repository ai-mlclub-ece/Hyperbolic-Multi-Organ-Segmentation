# Hyperbolic Multi Organ Segmentation

## Abstract

Medical image segmentation is important for diagnosing diseases and planning treatments. However, traditional methods in **Euclidean space** struggle with complex organ structures. In this project, we use **Hyperbolic** methods rather Euclidean to improve **multi-organ segmentation**. Hyperbolic space helps represent complex relationships between organs more effectively than Euclidean space. We are testing on **AMOS(Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation)** dataset and comparing segmentation performance and computational costs.


## Dataset details - (AMOS 22)

AMOS, a large-scale, diverse, clinical dataset for abdominal organ segmentation. AMOS provides 500 CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality, multi-phase, multi-disease patients, each with voxel-level annotations of 15 abdominal organs, providing challenging examples and test-bed for studying robust segmentation algorithms under diverse targets and scenarios

The dataset consists of 3 splits:
```
Training:
    - 240 scans
    - 35,524 image slices

Validation:
    - 120 scans
    - 18,537 image slices

Test:
    - 240 scans
    - 44,554 image slices
```

## Implementation

To train the architectures in this repo, firstly clone this repository.

```
git clone https://github.com/ai-mlclub-ece/Hyperbolic-Multi-Organ-Segmentation.git
```

### Installing libraries

To avoid conflicts between library versions, install these requirements:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r Hyperbolic-Multi-Organ-Segmentation/requirements.txt
```

### Training

To train using 2 GPUs use this :
```
torchrun --nnodes=1 --nproc-per-node=2 \
--rdzv-id=123 --rdzv-backend=c10d --max-restarts=0 \
Hyperbolic-Multi-Organ-Segmentation/training_scripts/train.py \
--mode train --data-dir "amos22/" \
--image-size 512 512 --window-preset "ct_abdomen" \
--model unet --loss dice \
--epochs 10 --batch-size 16 --visualize
```

**Note:** Make sure you extracted dataset(amos22.zip) in current working directory, from [here](https://zenodo.org/records/7262581)

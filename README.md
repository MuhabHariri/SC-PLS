# SC-PLS

**Source-Calibrated Pseudo-Label Selection (SC-PLS)** selects fish detections using a frozen **Faster R-CNN (torchvision v2 configuration) with a ResNet-50 FPN backbone** and pretrained reliability models. Accepted detections are exported as COCO-format pseudo-labels with their original bounding boxes and detector scores.

Detector weights and calibration parameters are provided. Supply your own images; no training or annotations are required for prediction. Datasets are not included.

## Installation

Install Git, Git LFS, and Conda. Run the following in Anaconda Prompt:

```bat
git clone https://github.com/MuhabHariri/SC-PLS.git
cd SC-PLS
git lfs install --local
git lfs pull

conda create -n scpls python=3.11 pip -y
conda activate scpls
python -m pip install -r requirements-cu128.txt
```

For CPU execution, install `requirements-cpu.txt` instead and use `--device cpu`.

## Select detections from your images

```bat
python run.py predict --images "C:/path/to/images" --output "runs/my_predictions" --device cuda:0 --methods SC_PLS
```

Replace the image path with your folder. Results are saved to:

```text
runs/my_predictions/SC_PLS_pseudo_labels.json
```

## Optional evaluation

If ground-truth fish annotations are available in COCO JSON format:

```bat
python run.py reproduce --dataset custom --images "C:/path/to/images" --annotations "C:/path/to/Annotations.json" --output "runs/my_evaluation" --device cuda:0 --methods SC_PLS
```

Annotations are used only to calculate evaluation metrics. Results are written to `table.csv` and `RESULTS.json` inside the output directory.

## Command options

| Option | Meaning |
|---|---|
| `--images` | Path to the folder containing your images. |
| `--output` | Destination for generated files. Use a new directory for each run. |
| `--device` | Processing device: `cuda:0` for the first NVIDIA GPU, or `cpu`. |
| `--methods SC_PLS` | Runs SC-PLS selection using the supplied calibration models and acceptance threshold. |
| `--annotations` | Path to ground-truth COCO JSON annotations; required only for evaluation. |
| `--dataset custom` | Selects evaluation on your own annotated dataset. |
| `--methods all` | Runs all six methods: Detector-only, Score-only, BOX_JITTER, CONSISTENCY, SC-PLS, and SC-PLS + consistency. |

Accepted detections are predictions, not guaranteed ground truth, and may require human review.


## Dataset acknowledgement

The detector was fine-tuned on [FishDet-M](https://huggingface.co/datasets/GE9X/FishDet-M), introduced by Muayad Abujabal, Lyes Saad Saoud, and Irfan Hussain ([2025](https://arxiv.org/abs/2507.17859)). The SC-PLS reliability models were fitted using FishDet-M validation data. The dataset distribution page lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as its license. FishDet-M images and annotations are not redistributed in this repository.

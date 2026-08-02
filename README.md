# CA-MoDE: Context-Aware Mixture of Domain Experts for Bodily Expression of Emotion

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

PyTorch research implementation of **CA-MoDE** (**Context-Aware Mixture of Domain Experts**) for bodily expression of emotion recognition in the wild.

CA-MoDE models the relationship between bodily emotion and surrounding context explicitly. Its full research pipeline uses frozen pretrained scene and object experts to generate contextual soft pseudo-labels, which are converted into emotion-context co-occurrence priors and fused with the emotion expert through max-endorsement probabilistic gating.

> **Research status**
>
> This repository contains a validated PyTorch implementation scaffold, including:
>
> - Shared feature encoding with a `14 × 14 × 528` feature map
> - Scene, object, and emotion expert branches
> - Context-prior fitting on the training partition
> - Available and anticipated unavailable context priors
> - Max-endorsement probabilistic fusion
> - MSE training objective for 26 categorical emotions and 3 VAD dimensions
> - CUDA dry-run, shape test, checkpoint-reload test, and stress test
>
> Reproduction of the reported CA-MoDE results requires pretrained **Places2/Places365** scene and **MS COCO** object experts, together with the official **BoLD** data and split protocol. Do not claim reproduction of paper results from the synthetic debugging dataset.

---

## Overview

Given an input image $x$, CA-MoDE obtains a shared feature map:

$$
\mathbf{F} = \mathcal{H}^{\mathrm{base}}(\mathbf{X}),
\qquad
\mathbf{F} \in \mathbb{R}^{14 \times 14 \times 528}.
$$

Three domain experts process this representation:

- **Scene context expert** $\mathcal{H}^{\mathrm{place}}$: produces a 365-dimensional Places-style soft pseudo-label vector
- **Object context expert** $\mathcal{H}^{\mathrm{object}}$: produces an 80-dimensional COCO-style soft pseudo-label vector
- **Emotion expert** $\mathcal{H}^{\mathrm{emotion}}$: predicts 26 discrete emotions and Valence, Arousal, and Dominance (VAD)

The context experts generate empirical priors on the **training split only**:

$$
\mathcal{P}^{+}_{i,j} = \Pr(\mathbb{B}_i \mid \mathbb{A}_j),
\qquad
\mathcal{P}^{-}_{i,j} = \Pr(\mathbb{B}_i \mid \neg\mathbb{A}_j).
$$

CA-MoDE uses per-emotion max endorsement across scene and object context:

$$
p_i^{+} = \max_j \mathcal{P}^{+}_{i,j},
\qquad
p_i^{-} = \max_j \mathcal{P}^{-}_{i,j}.
$$

The gate is:

$$
Q_i = \sigma\left(\alpha(p_i^{+} - \tau)\right),
$$

and the final contextual prior is:

$$
\hat{p}_i = Q_i p_i^{+} + (1 - Q_i)p_i^{-}.
$$

The final prediction is:

$$
\tilde{\mathbf{y}}
=
\frac{1}{\lambda}
\hat{\mathbf{p}}
\odot
\mathbf{y}^{\mathrm{emotion}}
$$

---

## Repository layout

```text
CA-MoDE/
├── src/
│   └── camode/
│       ├── config.py
│       ├── model/
│       │   ├── backbone.py
│       │   ├── blocks.py
│       │   ├── camode.py
│       │   ├── context_experts.py
│       │   └── fusion.py
│       ├── data/
│       │   └── synthetic_dataset.py
│       ├── training/
│       │   ├── losses.py
│       │   └── trainer.py
│       └── utils/
│           └── seed.py
├── tests/
│   └── test_shapes.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── count_parameters.py
├── dry_run.py
├── environment_windows.yml
├── requirements-pip.txt
└── stress_test.py
```

---

## Environment setup

### Conda

```bash
conda env create -f environment_windows.yml
conda activate camode
```

For CUDA-enabled PyTorch, install the command generated for your
operating system and CUDA configuration by the official PyTorch selector:

https://pytorch.org/get-started/locally/

### Pip

Alternatively, create an environment manually:

```bash
conda create -n camode python=3.11 -y
conda activate camode

pip install -r requirements-pip.txt
```

Install a PyTorch build compatible with your CUDA driver from the official PyTorch installation guide:

```text
https://pytorch.org/get-started/locally/
```

Verify that CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Quick validation

The following commands validate installation and core model behaviour. They use a synthetic dataset and do **not** measure scientific performance on BoLD.

### Dry run

```bash
python dry_run.py
```

Expected properties:

- CUDA is used when available
- Contextual priors are fitted
- One training pass and one evaluation pass complete
- Output shape is `[batch_size, 29]`

### Unit tests

```bash
python -m pytest -q
```

### Stress test

```bash
python stress_test.py
```

The stress test validates:

- Forward propagation
- Backward propagation and optimiser updates
- Finite predictions and gate values
- Batch sizes 1, 2, 4, and 8
- Checkpoint save/load equivalence
- Exact deterministic checkpoint recovery in evaluation mode

A successful run reports:

```json
{
  "all_passed": true
}
```

### Count parameters

```bash
python count_parameters.py
```

Report both:

- **Total parameters:** all learnable tensors stored in the model checkpoint
- **Trainable parameters:** tensors updated during CA-MoDE training
- **Frozen parameters:** pretrained scene/object expert weights retained but not updated

---

## Required datasets

A full CA-MoDE experiment has three data dependencies. Users are responsible for reviewing and complying with each dataset's license, terms of use, and access requirements.

| Resource | Role in CA-MoDE | Official source |
|---|---|---|
| **BoLD** | Bodily emotion recognition training, validation, and testing | [BoLD Challenge / Body Language Dataset](https://cydar.ist.psu.edu/emotionchallenge/index.php) |
| **Places2 / Places365** | Scene-expert pretraining or pretrained scene-classification weights | [Places2 / Places365 download page](http://places2.csail.mit.edu/download.html) |
| **Microsoft COCO** | Object-expert pretraining or pretrained COCO detection weights | [MS COCO download page](https://cocodataset.org/#download) |

### Important data-policy note

Do **not** commit raw BoLD, Places, COCO, derived video frames, annotations, or pretrained checkpoint files to GitHub unless their licenses explicitly permit redistribution. Add local data and checkpoint paths to `.gitignore`.

Recommended `.gitignore` entries:

```gitignore
data/
datasets/
checkpoints/
outputs/
runs/
wandb/
*.pt
*.pth
*.ckpt
*.onnx
```

---

## Data preparation

### 1. BoLD

Download BoLD from the official challenge portal and follow its access conditions.

CA-MoDE is evaluated using still images extracted from BoLD video clips. Split data **at the clip level**, rather than frame level, to avoid frames from the same source clip appearing in multiple partitions.

Use the following split policy:

| Partition | Proportion | Purpose |
|---|---:|---|
| Training | 60% | Fit model parameters and contextual priors |
| Validation | 10% | Hyperparameter selection and early model selection |
| Test | 30% | Final held-out evaluation only |

The dataset loader used for real training must return:

```python
{
    "image": image_tensor,    # torch.float32, shape [3, 224, 224]
    "emotion": target_tensor, # torch.float32, shape [29]
}
```

Target convention:

- Dimensions `0:26`: 26 discrete emotion labels in `[0, 1]`
- Dimensions `26:29`: Valence, Arousal, and Dominance (VAD), following the BoLD annotation convention

### 2. Places2 / Places365

Use a scene expert pretrained on Places2/Places365 that outputs a normalised 365-class probability vector:

```python
z_scene.shape == [batch_size, 365]
```

Freeze the scene-expert backbone during CA-MoDE training. Only its projection into the contextual latent space remains trainable.

### 3. Microsoft COCO

Use a COCO-pretrained object detector to produce an 80-dimensional object-confidence vector:

```python
z_object.shape == [batch_size, 80]
```

For each image:

1. Run the frozen detector
2. Accumulate confidence scores for detections belonging to each COCO category
3. Normalise the resulting 80-dimensional vector
4. Set entries for undetected categories to zero

Freeze the object detector during CA-MoDE training.

---

## Reproduction pipeline

A full paper-oriented pipeline should follow this order.

### Step 1: Prepare BoLD frames and labels

Extract still frames from BoLD clips and build clip-disjoint train/validation/test manifests.

Example target structure:

```text
data/
└── bold/
    ├── frames/
    │   ├── clip_000001/
    │   ├── clip_000002/
    │   └── ...
    ├── annotations/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── splits/
        ├── train_clips.txt
        ├── val_clips.txt
        └── test_clips.txt
```

Each manifest should include:

```text
image_path,clip_id,emotion_01,...,emotion_26,valence,arousal,dominance
```

### Step 2: Obtain pretrained contextual experts

Prepare:

- A frozen **Places2/Places365** scene classifier with 365-way softmax output
- A frozen **COCO** object detector with 80 COCO categories
- A trainable shared image encoder and emotion expert

The contextual classifiers must be in `eval()` mode during CA-MoDE training so that their pseudo-label distributions are stable.

### Step 3: Generate contextual pseudo-labels

For every training image, produce:

```text
scene pseudo-label:  [365]
object pseudo-label: [80]
```

Optionally cache these pseudo-labels locally to avoid repeating expensive Places/COCO inference.

### Step 4: Fit contextual priors

Fit context priors **only** on the BoLD training loader:

```python
trainer.fit_priors(train_loader)
```

Never use validation or test samples when fitting:

- Emotion marginal priors `pi`
- Available-context conditional priors `P`
- Anticipated-unavailable-context priors `P_neg`
- Context-event priors `q`

This avoids validation/test leakage.

### Step 5: Train CA-MoDE

The paper-oriented configuration is:

```python
TrainConfig(
    batch_size=8,
    lr=1e-2,
    momentum=0.9,
    weight_decay=5e-3,
    epochs=90,
)
```

Optimization:

- Optimizer: SGD
- Momentum: `0.9`
- Initial learning rate: `1e-2`
- Weight decay: `5e-3`
- LR schedule: multiply learning rate by `0.1` every 45 epochs
- Loss: mean squared error
- Maximum epochs: 90
- Batch size: 8

The selected CA-MoDE hyperparameters are:

```python
ModelConfig(
    kappa=56,
    lambda_scale=0.2,
    gate_threshold=0.5,  # Gate threshold
    gate_sharpness=10.0,  # Gate sharpness
)
```

Training loop outline:

```python
trainer.fit_priors(train_loader)

for epoch in range(train_config.epochs):
    train_loss = trainer.train_one_epoch(train_loader)
    val_loss = trainer.evaluate(val_loader)
    trainer.step_scheduler()

    print(
        f"Epoch {epoch + 1:03d} | "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={val_loss:.6f} | "
        f"lr={trainer.current_lr():.2e}"
    )
```

### Step 6: Evaluate once on test data

After selecting hyperparameters and the final checkpoint using the validation set, evaluate exactly once on the held-out test split.

Report:

- Mean Average Precision (m$AP$) for 26 discrete emotions
- Mean ROC area (m$RA$) for 26 discrete emotions
- Mean $R^{2}$ (m$R^{2}$) for VAD
- Emotion Recognition Score (ERS)

$$
\mathrm{ERS} = \frac{1}{2}\Big(\mathrm{m}R^2+\frac{1}{2}(\mathrm{m}AP + \mathrm{m}RA)\Big).
$$

---

## Checkpoints

Save both the model state and experiment metadata:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": trainer.optimizer.state_dict(),
    "scheduler_state_dict": trainer.scheduler.state_dict(),
    "model_config": vars(model_config),
    "train_config": vars(train_config),
    "epoch": epoch,
    "validation_loss": val_loss,
}

torch.save(checkpoint, "checkpoints/camode_best.pt")
```

The model `state_dict` includes fitted contextual-prior buffers. Therefore, checkpoints must be created **after** calling:

```python
trainer.fit_priors(train_loader)
```

When loading a checkpoint for inference:

```python
checkpoint = torch.load(
    "checkpoints/camode_best.pt",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
```

---

## Reproducibility checklist

Before reporting results, verify all items below:

- [ ] BoLD split is clip-disjoint
- [ ] Contextual priors were fitted on the training split only
- [ ] Validation and test data were not used for prior fitting
- [ ] Scene expert is initialised from Places2/Places365 weights and frozen
- [ ] Object expert is initialised from COCO detector weights and frozen
- [ ] Scene pseudo-labels have 365 dimensions
- [ ] Object pseudo-labels have 80 dimensions
- [ ] Context projectors remain trainable
- [ ] Shared encoder and emotion expert remain trainable
- [ ] Final checkpoint contains fitted-prior buffers
- [ ] Model is in `eval()` mode for validation, test, and checkpoint comparisons
- [ ] Hyperparameters are selected using validation data only
- [ ] Test data is evaluated only after selecting the final model

---

## Current limitations

This repository’s synthetic dataset and validation scripts are intended for implementation verification, debugging, and regression testing. They do not constitute a scientific benchmark.

In particular, paper-level reproduction requires:

1. Official BoLD access and clip-level splits
2. A pretrained Places2/Places365 scene expert
3. A pretrained COCO object detector
4. Conversion of detector outputs into normalised 80-category confidence vectors
5. Training and evaluation scripts for BoLD
6. Exact evaluation code for mAP, mRA, mR², and ERS
7. Reporting results across reproducible seeds

---

## Citation

If you use this code, model design, experimental protocol, or the BoLD evaluation setting, please cite both the CA-MoDE manuscript and the BoLD/ARBEE paper below.

### CA-MoDE

```bibtex
@unpublished{dehshibi_camode,
  author = {Mohammad Mahdi Dehshibi and David Masip},
  title  = {Context-Aware Mixture of Domain Experts for Bodily Expression of Emotion in the Wild},
  note   = {Manuscript under review},
  year   = {2026}
}
```

### BoLD

```bibtex
@article{luo2020arbee,
  title     = {{ARBEE: Towards Automated Recognition of Bodily Expression of Emotion in the Wild}},
  author    = {Luo, Y. and Ye, J. and Adams, R. B. and Li, J. and Newman, M. G. and Wang, J. Z.},
  journal   = {{International Journal of Computer Vision}},
  volume    = {128},
  number    = {1},
  pages     = {1--25},
  year      = {2020},
  doi       = {10.1007/s11263-019-01215-y},
  publisher = {Springer}
}
```

### Context datasets

If you train or distribute contextual experts, also cite the original Places and COCO work.

```bibtex
@article{zhou2018places,
  title     = {{Places: A 10 Million Image Database for Scene Recognition}},
  author    = {Zhou, B. and Lapedriza, A. and Khosla, A. and Oliva, A. and Torralba, A.},
  journal   = {IEEE Transactions on Pattern Analysis Machine Intelligence},
  volume    = {40},
  number    = {6},
  pages     = {1452--1464},
  year      = {2018},
  doi       = {10.1109/TPAMI.2017.2723009},
  publisher = {IEEE}
}

@inproceedings{lin2014microsoft,
  title     = {{Microsoft COCO: Common Objects in Context}},
  author    = {Lin, T. Y. and Maire, M. and Belongie, S. and Hays, J. and Perona, P. and Ramanan, D. and Doll{\'a}r, P. and Zitnick, C. L.},
  booktitle = {ECCV},
  pages     = {740--755},
  year      = {2014},
  doi       = {10.1007/978-3-319-10602-1_48},
  publisher = {Springer Cham}
}
```

---

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for the full license text.

The license applies only to the original source code in this repository.
BoLD, Places2/Places365, Microsoft COCO, third-party source code, and pretrained model weights remain subject to their own licenses, terms of use, and access conditions. Users are responsible for complying with those terms.

Copyright © 2026 Mohammad Mahdi Dehshibi and David Masip.

## Contact

For questions, bug reports, or collaboration enquiries, please open a GitHub issue or contact the authors through their institutional email addresses.

# AV-HuBERT With SU2SW Decoder

This repository implements a lip-reading system based on **AV-HuBERT** and **Fairseq**. The architecture leverages a two-stage training approach with a specialized decoder (**SU2SW**) to enhance decoding capabilities using large-scale speech data.

## Overview

The pipeline consists of three main processes:

1.  **Data Preprocessing**: Feature extraction and clustering to generate pseudo-labels (Speech Units).
2.  **SU2SW Training (Stage 1)**: Training a Transformer decoder to map Speech Units (SU) to Subwords (SW). This stage utilizes rich speech data to improve the decoder's language modeling capabilities.
3.  **Integrated Training (Stage 2)**: Fine-tuning the pre-trained AV-HuBERT encoder with the pre-trained SU2SW decoder on the Audio-Visual (AV) dataset.

## Dependencies

*   Python 3.x
*   Fairseq
*   AV-HuBERT dependencies

## Workflow

### Process 1: Data Preparation

Before training, prepare your datasets (e.g., LRS3, VoxCeleb) following the standard AV-HuBERT preparation and clustering protocols.

#### Feature Extraction & Clustering
Follow the standard AV-HuBERT clustering recipe to generate speech unit pseudo-labels.

**Important Note for Audio-Only Data:**
When processing ASR (speech-only) datasets to augment the SU2SW training, use the provided script dump_audio_feature.py to extract features. This script automatically pads the missing video modality with zero features to maintain compatibility.


*Ensure your data directories are organized similarly to `labels_unit` (for Speech Units) and `labels_vsr` (for Visual Speech Recognition data).*

---

### Process 2: Train SU2SW Model

The first training stage involves training the **Speech Unit to Subword (SU2SW)** model. This model takes the discrete speech units (generated in Process 1) as input and predicts subword tokens.

#### Configuration
*   **Script:** `pt_decoder/scripts/su2sw.sh`
*   **Config:** `pt_decoder/src/conf/su2sw.yaml`
*   **Tokenizer:** Uses `spm_unigram1000.model`

#### Run Training
Make sure the `data_pth` in the script points to your `labels_unit/audio_visual` directory.

```bash
./pt_decoder/scripts/su2sw.sh
```

---

### Process 3: Integrated Training (VSR + PTD)

In the final stage, we combine the pre-trained AV-HuBERT encoder and the pre-trained SU2SW decoder. The two modules are fine-tuned together on the lip-reading (visual-only) dataset.

#### Configuration
*   **Script:** `pt_decoder/scripts/vsr_ptd.sh`
*   **Config:** `pt_decoder/src/conf/vsr_ptd.yaml`

#### Update Checkpoint Path
**Crucial Step:** Before running, open `pt_decoder/scripts/vsr_ptd.sh` and update the `+model.su2sw_pth` argument to point to the `checkpoint_best.pt` generated in Process 2 and open `pt_decoder/src/model.py` and update the `avhubert_pth` argument to point to the pretrained avhubert checkpoint.

```bash
# Example inside scripts/vsr_ptd.sh
+model.su2sw_pth=/path/to/your/exp/su2sw_pretraining/audio_visual/checkpoints/checkpoint_best.pt
```

#### Run Training
Start the integrated video speech recognition training:

```bash
./pt_decoder/scripts/vsr_ptd.sh
```

## Acknowledgements

This code is based on [AV-HuBERT](https://github.com/facebookresearch/av_hubert) and [Fairseq](https://github.com/facebookresearch/fairseq).
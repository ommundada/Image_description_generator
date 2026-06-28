
# Image Caption Generator using CNN and LSTM

A deep learning project that automatically generates natural-language captions for images, combining a **CNN (VGG16)** for visual feature extraction with an **LSTM** for sequence generation. Trained and evaluated on the **Flickr8k** dataset.

## Overview

Given an input image, the model produces a descriptive caption in plain English — e.g. *"a dog is running through the grass"*. It works in two stages:

1. **Image understanding (CNN):** A pretrained VGG16 network extracts a 4096-dimensional feature vector summarizing the visual content of each image.
2. **Caption generation (LSTM):** An LSTM-based decoder takes the image feature vector plus the words generated so far and predicts the next word, one at a time, until an end-of-sequence token is produced.

This is the classic **encoder–decoder image captioning architecture**: CNN as encoder, LSTM as decoder, merged through a dense layer.

## Dataset

- **Flickr8k** — 8,000 images, each paired with 5 human-written captions (~40,000 captions total)
- Download from Kaggle: [adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (~1 GB)
- Expected structure:
  ```
  flickr8k/
  ├── Images/            # all .jpg images
  └── captions.txt       # image_id, caption pairs (CSV-style)
  ```

> Flickr30k and MS-COCO are larger alternatives, but Flickr8k is used here to keep training time manageable.

## Architecture

### Feature Extraction
- **Model:** VGG16 (pretrained on ImageNet), with the final classification layer removed
- **Output:** 4096-dimensional feature vector per image (from the second-to-last layer)
- Features are extracted once for all images and cached to disk (`features.pkl`) so they don't need to be recomputed on every run.

### Caption Generation Model

| Branch | Layers |
|---|---|
| **Image input** | `Input(4096,)` → `Dropout(0.4)` → `Dense(256, relu)` |
| **Text input** | `Input(max_length,)` → `Embedding(vocab_size, 256)` → `Dropout(0.4)` → `LSTM(256)` |
| **Decoder** | `add([image_branch, text_branch])` → `Dense(256, relu)` → `Dense(vocab_size, softmax)` |

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- Captions are wrapped with `startseq` / `endseq` tokens to mark sequence boundaries during generation.

## Project Workflow

1. **Import modules** — TensorFlow/Keras, NumPy, tqdm, pickle, NLTK
2. **Extract image features** with VGG16 → cache to `features.pkl`
3. **Load and parse captions** from `captions.txt` into an `{image_id: [captions]}` mapping
4. **Clean text** — lowercase, strip special characters/extra whitespace, add `startseq`/`endseq` tags
5. **Tokenize** captions with Keras `Tokenizer`; compute vocabulary size and max caption length
6. **Train/test split** — 90% / 10% split by image ID
7. **Data generator** — yields `(image_features, padded_input_sequence) → next_word` training pairs in batches, avoiding loading everything into memory at once
8. **Build and train the model** — 20 epochs, batch size 32
9. **Save the trained model** to `best_model.h5`
10. **Generate captions** — greedy word-by-word decoding starting from `startseq` until `endseq` or `max_length` is reached
11. **Evaluate with BLEU score** (BLEU-1, BLEU-2) on the held-out test set
12. **Visualize results** — display test images alongside their actual vs. predicted captions

## Requirements

```
tensorflow
numpy
tqdm
nltk
pillow
matplotlib
pydot        # for plot_model architecture visualization
graphviz     # system dependency for pydot
```

Install with:
```bash
pip install tensorflow numpy tqdm nltk pillow matplotlib pydot
```
(`graphviz` must also be installed at the OS level for `plot_model` to render.)

## Setup

The notebook expects data at:
```python
BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/working'
```

To run outside Kaggle, update these paths to point to your local Flickr8k folder and a writable output directory, e.g.:
```python
BASE_DIR = './flickr8k'
WORKING_DIR = './working'
```

## Usage

```bash
jupyter notebook image-caption-generator-using-cnn-and-lstm.ipynb
```

Run all cells in order. Key stages to be aware of:
- **Feature extraction is slow** the first time (runs VGG16 over all 8,000 images) but only needs to run once — subsequent runs can load straight from `features.pkl`.
- **Training** runs for 20 epochs by default; increase this for better caption quality.
- To caption a specific image after training, call:
  ```python
  generate_caption("your_image_name.jpg")
  ```

## Outputs

| File | Description |
|---|---|
| `features.pkl` | Cached VGG16 feature vectors for all images |
| `best_model.h5` | Trained caption generation model |

## Evaluation

Model quality is measured with **BLEU score** (Bilingual Evaluation Understudy), comparing generated captions against the human-written reference captions:
- BLEU-1 (unigram overlap)
- BLEU-2 (bigram overlap)

A BLEU score above **0.4** is generally considered a good result for this task; scores improve with more training epochs.

## Notes & Tips

- VGG16 already performs the role of the "CNN" in this pipeline — no additional convolutional layers are trained from scratch, only the LSTM decoder.
- Increasing epochs, using a larger dataset (e.g. Flickr30k), or adding more model layers can improve caption quality at the cost of training time and resources.
- The data generator pattern (yielding batches on the fly) is specifically used to avoid memory/session crashes when working with the full set of image-caption pairs.

## Acknowledgments

- Dataset: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Pretrained model: VGG16 (ImageNet weights, via `tensorflow.keras.applications`)

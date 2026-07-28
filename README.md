<div align="center">

<img src="app/assets/logo.png" alt="FaceGuard" width="220">

**Detection of AI-generated faces**

A local web app powered by a ResNet50V2 fine-tuned with transfer learning.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-89.6%25-16A34A)
![AUC](https://img.shields.io/badge/AUC-96.3%25-16A34A)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## What it does

You drop a face photo, and the model answers **REAL** or **AI-GENERATED** with a confidence score. Everything runs locally: no image leaves your machine, and nothing is written to disk.

## Results

Measured on the test set: 5,714 images never seen during training, balanced equally between the two classes.

| Metric | Value |
|---|---|
| Accuracy | **89.64%** |
| Precision | 87.32% |
| Recall | 92.67% |
| AUC | **96.30%** |

Confusion matrix:

| | predicted real | predicted fake |
|---|---|---|
| **real image** | 2,647 | 210 |
| **generated image** | 382 | 2,479 |

Errors are not symmetric: 382 generated images pass as real, versus 210 genuine photos wrongly flagged. This is the more problematic error direction, and it can be tuned by shifting the decision threshold without retraining.

The model was trained on the **140k Real and Fake Faces** dataset (Kaggle).

## Installation

You need **Python 3.10 or 3.11** (TensorFlow does not support newer versions yet).

```bash
git clone https://github.com/WafaaRahmoune/Fake-face-image-detection.git
cd Fake-face-image-detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

That's all: the model is included in the repository (`models/resnet50v2_faceguard.keras`, 97 MB), so there is nothing else to download. The clone takes a bit longer, but the project works right away.

## Run

```bash
# Windows
run.bat

# macOS / Linux
chmod +x run.sh && ./run.sh
```

The browser opens on <http://127.0.0.1:8000>. **The first start takes 30 to 60 seconds** (importing TensorFlow and building the network). Subsequent predictions are instant. Press `Ctrl+C` to stop.

## Testing guide

1. **Drop a face photo** into the dashed zone, or click to browse. Accepted formats: JPG, PNG, JPEG.
2. **Click "Analyze Image".** The first analysis may take a second or two; the next ones are instant.
3. **Read the result.** The ring and bar show the confidence; a green banner means an authentic photo, a purple one a generated image.

For a representative test:

- Use **centered, tightly cropped faces**: the model was trained on this kind of image and degrades on wide scenes or profile faces.
- Test **both classes**. Testing only on real photos says nothing about the ability to detect fakes.
- For generated images, sites like *this-person-does-not-exist* provide instant examples.
- **A single mistake is not a failure**: the model is wrong about one time in ten. You need around twenty images to form a fair opinion.

The server also exposes a small API if you want to automate:

```bash
curl http://127.0.0.1:8000/api/status
# {"modele": "resnet50v2_faceguard.keras", "taille_entree": 224}

curl -X POST http://127.0.0.1:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"image": "data:image/jpeg;base64,..."}'
# {"label": "real", "score": 0.9214, "confiance": 92.1}
```

## How it works

### The model

A **ResNet50V2** pre-trained on ImageNet, with its original classification layer removed and a binary head grafted on:

```
Input 224x224x3
  |- ResNet50V2 (ImageNet pre-trained, global average pooling)  ->  2048 features
  |- Dense 256, ReLU
  |- BatchNormalization
  |- Dropout 50%
  |- Dense 128, ReLU
  |- Dropout 30%
  |- Dense 1, sigmoid  ->  probability that the image is real
```

### Training, in two stages

| | Phase 1 | Phase 2 |
|---|---|---|
| Pre-trained base | frozen | last 100 layers unfrozen |
| Epochs | 10 | 5 |
| Learning rate | 1e-3 | 1e-5 |
| Stopping | (none) | early stopping |

The base is frozen at first because the head starts with random weights: its gradients are huge on the first pass and would destroy the pre-trained weights. Once the head is stable, the top of the network is unfrozen with a learning rate a hundred times smaller, to fine-tune it without damaging it.

### Preprocessing

```
Undersampling  ->  Resize 224x224  ->  Normalize /255
  ->  Augmentation (rotation, zoom, shift, fill_mode='nearest')
  ->  Shuffle  ->  Batches of 32
```

Augmentation and shuffling apply **only to training**. Validation and test receive the images untouched, otherwise the metrics would not be reproducible.

### Two details not to miss

**Normalization is not optional.** The model expects pixels in `[0, 1]`, obtained by dividing by 255. Feeding it raw `0-255` pixels saturates it: it returns the same value for every image and accuracy drops to 50%. Verified on 600 images.

| Input | Accuracy | AUC |
|---|---|---|
| `x / 255` | **89.2%** | **97.1%** |
| raw pixels `0-255` | 50.0% | 10.9% |

**The meaning of the output.** `flow_from_directory()` sorts folders alphabetically, so `fake` is 0 and `real` is 1. A sigmoid output **close to 1 means a real image**.

## Repository structure

```
FaceGuard/
├── app/
│   ├── server.py       HTTP server, standard library only
│   ├── model.py        model loading and prediction
│   ├── page.html       full interface: structure, styles, behavior
│   └── assets/         logo and illustrations
├── models/             the weights file (resnet50v2_faceguard.keras, 97 MB, included in the repo)
├── notebooks/          training notebook
├── requirements.txt
├── run.bat / run.sh
└── README.md
```

Only three files for the application, with no web framework and no build step. `page.html` is re-read on every request, so you can tweak the design and refresh the browser without restarting the server, and therefore without reloading the model.

## Technologies

- **Deep learning:** TensorFlow, Keras, ResNet50V2 pre-trained on ImageNet
- **Data:** NumPy, Pillow, OpenCV, imagehash for deduplication
- **Evaluation:** scikit-learn, Matplotlib, Seaborn
- **Interface:** Python (standard library), HTML, CSS, JavaScript, no framework
- **Training:** Kaggle, GPU

## License

Released under the MIT License. See [LICENSE](LICENSE).

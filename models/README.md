# Model

`resnet50v2_faceguard.keras` (97 MB, included in the repository).

The server automatically loads the first `.keras` or `.h5` file found in this folder, so there is nothing to configure.

## Characteristics

| | |
|---|---|
| Architecture | ResNet50V2 pre-trained on ImageNet + binary head |
| Input | 224x224x3, pixels normalized to **[0, 1]** |
| Output | 1 sigmoid neuron, close to **1 = real image** |
| Accuracy | 89.64% on 5,714 test images |
| AUC | 96.30% |

The file was re-exported without the optimizer state: 97 MB instead of 277, for strictly identical predictions. The 180 MB removed were the Adam optimizer moments, useful only to resume training.

## Using another file

To point to a model located elsewhere:

```bash
# Windows
set FACEGUARD_MODEL=C:\path\to\my_model.keras && run.bat

# macOS / Linux
FACEGUARD_MODEL=/path/to/my_model.keras ./run.sh
```

It must follow the same convention: input 224x224x3 normalized to [0, 1], output of a single sigmoid neuron. See `app/model.py`.

## Note

The `.h5`, `.pt`, `.pth` and `.onnx` formats are excluded from version control. The original `.h5` is 277 MB, which exceeds GitHub's 100 MB per-file limit, so its push would be rejected.

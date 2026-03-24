# ComputerVision_LocalFeatures_BayesianClassification_and_KMeansClustering

## 3.1. Bayesian pixel classification

Run these steps from the project root.

1. Create and activate the virtual environment (if it does not exist yet):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies in the virtual environment:

```bash
pip install -r requirements.txt
```

3. Visualize and save color components (RGB, HSV, YCrCb, cbcr_grad):

```bash
python3 src/Display_Components.py img/Essex_Faces/94/asamma.19.jpg --space all --save-dir outputs/components
```

4. Train and test Bayesian pixel classification:

```bash
python3 src/Bayes_Model_Training.py img/Essex_Faces/94/asamma.19.jpg --features cbcr_grad --model qda --decision map --test-image img/Essex_Faces/94/asheal.18.jpg
```
--model qda : correlations entre features
--model gnb : es Gaussian Naive Bayes

5. Optional multi-image training in one run:

```bash
python3 src/Bayes_Model_Training.py img/Essex_Faces/94/asamma.19.jpg img/Essex_Faces/94/ajones.19.jpg --features cbcr_grad --model qda --decision map --test-image img/Essex_Faces/94/asheal.18.jpg
```

6. Reuse a previously trained model without relabeling:

```bash
python3 src/Bayes_Model_Training.py --trained outputs/bayes/asheal.18/cbcr_grad_qda_map --test-image img/Essex_Faces/94/ajones.19.jpg
```

Notes:
- During training, draw rectangular RoIs with the mouse.
- Press `p` for skin, `n` for non-skin, `r` to reset the current RoI, and `q` to move to the next training image or finish on the last one.
- Each new training run now saves a `trained_model.pkl` bundle inside the output directory so it can be reused with `--trained`.

## 3.2. K-Means pixel clustering

Run these commands from the project root.

1. Train K-Means on one image and save cluster labels and centers:

```bash
python3 src/KMeans_Clustering.py img/Essex_Faces/94/asamma.19.jpg --no-show
```

Default values:
- `--features ycrcb`
- `--k 6`
- `--test-image` defaults to the training image
- if `--skin-clusters` is omitted, the script saves labels and centers only

2. Train on `asamma.19` and predict labels on another image:

```bash
python3 src/KMeans_Clustering.py img/Essex_Faces/94/asamma.19.jpg --test-image img/Essex_Faces/94/ajones.19.jpg --no-show
```

3. After inspecting `train_labels.png` and `centers.json`, rerun with the cluster IDs interpreted as skin:

```bash
python3 src/KMeans_Clustering.py img/Essex_Faces/94/asamma.19.jpg --test-image img/Essex_Faces/94/ajones.19.jpg --skin-clusters 1 4 --no-show
```

4. Try a different observation space or number of clusters:

```bash
python3 src/KMeans_Clustering.py img/Essex_Faces/94/asamma.19.jpg --test-image img/Essex_Faces/96/acatsa.17.jpg --features cbcr_grad --k 8 --no-show
```

Available options:
- `train_image`: image used to fit K-Means
- `--test-image <path>`: image where labels are predicted with the same cluster centers
- `--features {rgb,hsv,ycrcb,cbcr_grad}`: feature space used for clustering
- `--k <int>`: number of clusters
- `--skin-clusters <id1> <id2> ...`: cluster IDs interpreted as skin
- `--output-dir <dir>`: base directory for saved outputs, default `outputs/kmeans`
- `--no-show`: disables OpenCV windows

Outputs are saved in:

```text
outputs/kmeans/<test_stem>/<features>_k<k>_train_<train_stem>/
```

Saved files:
- `train_labels.png`
- `test_labels.png`
- `centers.json`
- `metadata.json`
- `train_skin_mask.png`, `train_skin_overlay.png`, `test_skin_mask.png`, `test_skin_overlay.png` when `--skin-clusters` is provided

Notes:
- `KMeans_Clustering.py` trains on a single image per run and reuses the same cluster centers on the test image.
- For `cbcr_grad`, the script applies an internal z-score normalization based only on the training image before clustering and prediction.
- K-Means is unsupervised: it does not know what skin is by itself. The skin class appears only after manually selecting which clusters correspond to skin.

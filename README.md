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

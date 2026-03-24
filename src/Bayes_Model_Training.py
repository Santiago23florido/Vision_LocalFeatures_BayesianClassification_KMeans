import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from features import extract_feature_image, flatten_feature_image, list_feature_spaces


LABEL_SKIN = 1
LABEL_NON_SKIN = -1
TRAINED_MODEL_FILENAME = "trained_model.pkl"


class ROISelector:
    def __init__(self):
        self.start = None
        self.end = None
        self.current_roi = None
        self.is_drawing = False

    def clear(self):
        self.start = None
        self.end = None
        self.current_roi = None
        self.is_drawing = False

    def callback(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.current_roi = None
            self.is_drawing = True
            return

        if event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.end = (x, y)
            self.is_drawing = False
            self.current_roi = self._normalized_roi()

    def preview_roi(self):
        if self.start is None or self.end is None:
            return None
        return self._normalized_roi()

    def _normalized_roi(self):
        if self.start is None or self.end is None:
            return None
        x0 = min(self.start[0], self.end[0])
        x1 = max(self.start[0], self.end[0])
        y0 = min(self.start[1], self.end[1])
        y1 = max(self.start[1], self.end[1])
        if x1 == x0 or y1 == y0:
            return None
        return x0, y0, x1, y1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Bayesian training for skin vs non-skin pixel classification."
    )
    parser.add_argument(
        "train_images",
        nargs="*",
        help="One or more training image paths.",
    )
    parser.add_argument(
        "--test-image",
        default=None,
        help="Optional test image path. Default: first training image.",
    )
    parser.add_argument(
        "--features",
        choices=list_feature_spaces(),
        default="cbcr_grad",
        help="Feature space used for training and prediction.",
    )
    parser.add_argument(
        "--model",
        choices=["qda", "gnb"],
        default="qda",
        help="Classifier model.",
    )
    parser.add_argument(
        "--decision",
        choices=["map", "ml"],
        default="map",
        help="Decision criterion. map uses empirical priors, ml uses uniform priors.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bayes",
        help="Base output directory for predictions and metadata.",
    )
    parser.add_argument(
        "--trained",
        default=None,
        help=(
            "Path to a previously saved trained model bundle or to a run directory "
            f"containing {TRAINED_MODEL_FILENAME}."
        ),
    )
    args = parser.parse_args()
    if args.trained is None and not args.train_images:
        parser.error("Provide at least one training image or pass --trained.")
    if args.trained is not None and args.train_images:
        parser.error("Do not pass training images when using --trained.")
    return args


def load_color_image(image_path):
    img_bgr = cv2.imread(str(image_path), -1)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image.")
    return img_bgr


def annotate_single_image_samples(train_path, train_bgr, train_features, image_index, total_images):
    selector = ROISelector()
    window_name = f"Training image ({image_index}/{total_images})"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, selector.callback)
    exit_label = "next" if image_index < total_images else "done"

    feature_batches = []
    label_batches = []
    roi_count = {"skin": 0, "non_skin": 0}
    pixel_count = {"skin": 0, "non_skin": 0}
    label_key_map = {
        ord("p"): ("skin", LABEL_SKIN),
        ord("n"): ("non_skin", LABEL_NON_SKIN),
    }

    print(f"Annotating image {image_index}/{total_images}: {train_path}")
    print("Instructions: draw ROI with mouse, press 'p' for skin, 'n' for non-skin.")
    if image_index < total_images:
        print("Press 'r' to reset current ROI and 'q' to continue to the next image.")
    else:
        print("Press 'r' to reset current ROI and 'q' when annotation is done.")

    while True:
        display = train_bgr.copy()
        roi_to_draw = selector.preview_roi()
        if roi_to_draw is not None:
            x0, y0, x1, y1 = roi_to_draw
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Image {image_index}/{total_images} | p:skin n:non_skin r:reset q:{exit_label}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selector.clear()
            continue

        if key not in label_key_map:
            continue

        if selector.current_roi is None:
            print("No ROI selected. Draw a non-empty rectangle first.")
            continue

        class_name, label_value = label_key_map[key]
        x0, y0, x1, y1 = selector.current_roi
        roi_features = train_features[y0:y1, x0:x1]
        if roi_features.size == 0:
            print("Empty ROI ignored.")
            selector.clear()
            continue

        batch_features = roi_features.reshape(-1, roi_features.shape[2])
        batch_labels = np.full((batch_features.shape[0],), label_value, dtype=np.int32)
        feature_batches.append(batch_features)
        label_batches.append(batch_labels)
        roi_count[class_name] += 1
        pixel_count[class_name] += batch_features.shape[0]
        print(
            f"Recorded {class_name} ROI: ({x0},{y0})-({x1},{y1})"
            f" with {batch_features.shape[0]} pixels."
        )
        selector.clear()

    cv2.destroyWindow(window_name)
    if not feature_batches:
        return None, None, roi_count, pixel_count

    X_train = np.concatenate(feature_batches, axis=0)
    y_train = np.concatenate(label_batches, axis=0)
    return X_train, y_train, roi_count, pixel_count


def annotate_training_samples(train_paths, feature_space):
    feature_batches = []
    label_batches = []
    total_roi_count = {"skin": 0, "non_skin": 0}
    total_pixel_count = {"skin": 0, "non_skin": 0}
    feature_names = None

    for image_index, train_path in enumerate(train_paths, start=1):
        train_bgr = load_color_image(train_path)
        print(
            "Training image:",
            train_bgr.shape[0],
            "rows x",
            train_bgr.shape[1],
            "cols x",
            train_bgr.shape[2],
            "channels",
        )
        train_feature_image, current_feature_names = extract_feature_image(
            train_bgr, feature_space
        )
        if feature_names is None:
            feature_names = current_feature_names

        X_image, y_image, roi_count, pixel_count = annotate_single_image_samples(
            train_path,
            train_bgr,
            train_feature_image,
            image_index,
            len(train_paths),
        )
        if X_image is not None:
            feature_batches.append(X_image)
            label_batches.append(y_image)
        for class_name in total_roi_count:
            total_roi_count[class_name] += roi_count[class_name]
            total_pixel_count[class_name] += pixel_count[class_name]

    if not feature_batches:
        raise RuntimeError("No training samples were annotated across the selected training images.")

    X_train = np.concatenate(feature_batches, axis=0)
    y_train = np.concatenate(label_batches, axis=0)
    if LABEL_SKIN not in y_train or LABEL_NON_SKIN not in y_train:
        raise RuntimeError(
            "Need both classes before training. Add at least one 'p' ROI and one 'n' ROI."
        )
    return X_train, y_train, feature_names, total_roi_count, total_pixel_count


def make_classifier(model_name, decision):
    priors = None if decision == "map" else [0.5, 0.5]
    if model_name == "qda":
        return QuadraticDiscriminantAnalysis(priors=priors, reg_param=1e-6)
    return GaussianNB(priors=priors)


def build_training_info(
    train_paths,
    feature_names,
    features,
    model_name,
    decision,
    roi_count,
    pixel_count,
    X_train,
    y_train,
):
    training_info = {
        "train_images": [str(train_path) for train_path in train_paths],
        "features": features,
        "feature_names": list(feature_names),
        "model": model_name,
        "decision": decision,
        "class_labels": {"skin": LABEL_SKIN, "non_skin": LABEL_NON_SKIN},
        "roi_count": roi_count,
        "pixel_count": pixel_count,
        "train_samples": int(X_train.shape[0]),
        "feature_dimension": int(X_train.shape[1]),
        "train_label_distribution": {
            "skin": int(np.sum(y_train == LABEL_SKIN)),
            "non_skin": int(np.sum(y_train == LABEL_NON_SKIN)),
        },
    }
    if len(train_paths) == 1:
        training_info["train_image"] = str(train_paths[0])
    return training_info


def resolve_trained_model_path(trained_path):
    trained_path = Path(trained_path)
    if trained_path.is_dir():
        trained_path = trained_path / TRAINED_MODEL_FILENAME
    return trained_path


def save_trained_model_bundle(run_dir, classifier, training_info):
    bundle = {
        "classifier": classifier,
        "training_info": training_info,
    }
    model_path = run_dir / TRAINED_MODEL_FILENAME
    with open(model_path, "wb") as model_file:
        pickle.dump(bundle, model_file)
    return model_path


def load_trained_model_bundle(trained_path):
    model_path = resolve_trained_model_path(trained_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find trained model bundle at '{model_path}'.")
    with open(model_path, "rb") as model_file:
        bundle = pickle.load(model_file)
    if "classifier" not in bundle or "training_info" not in bundle:
        raise ValueError(f"Invalid trained model bundle '{model_path}'.")
    return model_path, bundle["classifier"], bundle["training_info"]


def predict_mask_and_probability(classifier, feature_image):
    height, width = feature_image.shape[:2]
    flat_features = flatten_feature_image(feature_image)
    flat_labels = classifier.predict(flat_features)
    mask = np.where(flat_labels.reshape(height, width) == LABEL_SKIN, 255, 0).astype(np.uint8)

    prob_map_u8 = None
    if hasattr(classifier, "predict_proba"):
        class_list = list(classifier.classes_)
        if LABEL_SKIN in class_list:
            skin_index = class_list.index(LABEL_SKIN)
            probabilities = classifier.predict_proba(flat_features)[:, skin_index]
            prob_map_u8 = np.clip(probabilities.reshape(height, width) * 255.0, 0, 255).astype(
                np.uint8
            )
    return mask, prob_map_u8


def build_overlay(image_bgr, mask):
    color_layer = image_bgr.copy()
    color_layer[mask == 255] = (40, 180, 255)
    return cv2.addWeighted(image_bgr, 0.7, color_layer, 0.3, 0.0)


def save_outputs(
    run_dir,
    test_path,
    mask,
    overlay,
    probability_map,
    training_info,
    trained_model_path=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "mask.png"), mask)
    cv2.imwrite(str(run_dir / "overlay.png"), overlay)
    if probability_map is not None:
        cv2.imwrite(str(run_dir / "skin_probability.png"), probability_map)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_image": str(test_path),
        **training_info,
    }
    if trained_model_path is not None:
        metadata["trained_model_path"] = str(trained_model_path)
        metadata["loaded_from_trained"] = True
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()
    trained_model_path = None

    if args.trained is None:
        train_paths = [Path(train_image) for train_image in args.train_images]
        test_path = Path(args.test_image) if args.test_image is not None else train_paths[0]

        X_train, y_train, feature_names, roi_count, pixel_count = annotate_training_samples(
            train_paths, args.features
        )

        classifier = make_classifier(args.model, args.decision)
        classifier.fit(X_train, y_train)
        training_info = build_training_info(
            train_paths,
            feature_names,
            args.features,
            args.model,
            args.decision,
            roi_count,
            pixel_count,
            X_train,
            y_train,
        )
    else:
        trained_model_path, classifier, training_info = load_trained_model_bundle(args.trained)
        train_paths = [Path(train_image) for train_image in training_info["train_images"]]
        default_test_path = train_paths[0]
        test_path = Path(args.test_image) if args.test_image is not None else default_test_path

    test_bgr = load_color_image(test_path)
    test_feature_image, _ = extract_feature_image(test_bgr, training_info["features"])
    mask, probability_map = predict_mask_and_probability(classifier, test_feature_image)
    overlay = build_overlay(test_bgr, mask)

    run_dir = (
        Path(args.output_dir)
        / test_path.stem
        / (
            f"{training_info['features']}_"
            f"{training_info['model']}_"
            f"{training_info['decision']}"
        )
    )
    save_outputs(
        run_dir,
        test_path,
        mask,
        overlay,
        probability_map,
        training_info,
        trained_model_path=trained_model_path,
    )
    if args.trained is None:
        trained_model_path = save_trained_model_bundle(run_dir, classifier, training_info)
    print(f"Outputs saved in: {run_dir}")
    if trained_model_path is not None:
        print(f"Trained model bundle: {trained_model_path}")

    cv2.imshow("Predicted mask", mask)
    cv2.imshow("Skin overlay", overlay)
    if probability_map is not None:
        cv2.imshow("Skin probability", probability_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

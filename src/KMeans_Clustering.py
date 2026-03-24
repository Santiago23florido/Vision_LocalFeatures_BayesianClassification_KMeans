import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from features import extract_feature_image, flatten_feature_image, list_feature_spaces


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-Means clustering for pixel grouping and manual skin-cluster interpretation."
    )
    parser.add_argument("train_image", help="Training image used to fit K-Means.")
    parser.add_argument(
        "--test-image",
        default=None,
        help="Optional test image. Default: reuse the training image.",
    )
    parser.add_argument(
        "--features",
        choices=list_feature_spaces(),
        default="ycrcb",
        help="Feature space used for clustering and prediction.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--skin-clusters",
        nargs="+",
        type=int,
        default=None,
        help="One or more cluster IDs interpreted as skin.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kmeans",
        help="Base output directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open OpenCV windows.",
    )
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be at least 2.")
    return args


def load_color_image(image_path):
    img_bgr = cv2.imread(str(image_path), -1)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image.")
    return img_bgr


def extract_flat_features(img_bgr, feature_space):
    feature_image, feature_names = extract_feature_image(img_bgr, feature_space)
    flat_features = flatten_feature_image(feature_image)
    return feature_image, flat_features, feature_names


def fit_feature_normalizer(flat_features, feature_space):
    if feature_space != "cbcr_grad":
        return None

    mean = flat_features.mean(axis=0)
    std = flat_features.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize_features(flat_features, normalizer):
    if normalizer is None:
        return flat_features
    return (flat_features - normalizer["mean"]) / normalizer["std"]


def denormalize_centers(cluster_centers, normalizer):
    if normalizer is None:
        return cluster_centers
    return cluster_centers * normalizer["std"] + normalizer["mean"]


def reshape_labels(flat_labels, image_shape):
    height, width = image_shape[:2]
    return flat_labels.reshape(height, width)


def make_cluster_palette(num_clusters):
    palette = []
    for cluster_id in range(num_clusters):
        hue = int(round((179 * cluster_id) / max(num_clusters, 1))) % 180
        hsv_color = np.uint8([[[hue, 220, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append([int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])])
    return palette


def render_label_image(label_image, palette):
    color_labels = np.zeros((*label_image.shape, 3), dtype=np.uint8)
    for cluster_id, color in enumerate(palette):
        color_labels[label_image == cluster_id] = color
    return color_labels


def count_pixels_per_cluster(label_image, num_clusters):
    return np.bincount(label_image.reshape(-1), minlength=num_clusters).astype(int).tolist()


def build_skin_mask(label_image, skin_clusters):
    return np.where(np.isin(label_image, skin_clusters), 255, 0).astype(np.uint8)


def build_overlay(image_bgr, mask):
    color_layer = image_bgr.copy()
    color_layer[mask == 255] = (40, 180, 255)
    return cv2.addWeighted(image_bgr, 0.7, color_layer, 0.3, 0.0)


def validate_skin_clusters(skin_clusters, num_clusters):
    if skin_clusters is None:
        return None

    unique_clusters = sorted(set(skin_clusters))
    invalid_clusters = [cluster_id for cluster_id in unique_clusters if cluster_id < 0 or cluster_id >= num_clusters]
    if invalid_clusters:
        raise ValueError(
            f"Invalid cluster IDs {invalid_clusters}. Valid range is 0 to {num_clusters - 1}."
        )
    return unique_clusters


def print_cluster_summary(cluster_centers, feature_names, train_counts, test_counts, palette):
    print("Cluster centers:")
    for cluster_id, center in enumerate(cluster_centers):
        center_values = ", ".join(
            f"{feature_name}={center_value:.3f}"
            for feature_name, center_value in zip(feature_names, center)
        )
        print(
            f"  [{cluster_id}] {center_values} | "
            f"train_pixels={train_counts[cluster_id]} | "
            f"test_pixels={test_counts[cluster_id]} | "
            f"display_bgr={palette[cluster_id]}"
        )


def build_centers_payload(cluster_centers, feature_names, train_counts, test_counts, palette):
    payload = []
    for cluster_id, center in enumerate(cluster_centers):
        payload.append(
            {
                "cluster_id": cluster_id,
                "center": {
                    feature_name: float(center_value)
                    for feature_name, center_value in zip(feature_names, center)
                },
                "train_pixel_count": int(train_counts[cluster_id]),
                "test_pixel_count": int(test_counts[cluster_id]),
                "display_bgr": palette[cluster_id],
            }
        )
    return payload


def save_outputs(
    run_dir,
    train_path,
    test_path,
    train_label_image,
    test_label_image,
    centers_payload,
    metadata,
    train_skin_mask=None,
    train_skin_overlay=None,
    test_skin_mask=None,
    test_skin_overlay=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "train_labels.png"), train_label_image)
    cv2.imwrite(str(run_dir / "test_labels.png"), test_label_image)

    if train_skin_mask is not None:
        cv2.imwrite(str(run_dir / "train_skin_mask.png"), train_skin_mask)
        cv2.imwrite(str(run_dir / "train_skin_overlay.png"), train_skin_overlay)
        cv2.imwrite(str(run_dir / "test_skin_mask.png"), test_skin_mask)
        cv2.imwrite(str(run_dir / "test_skin_overlay.png"), test_skin_overlay)

    with open(run_dir / "centers.json", "w", encoding="utf-8") as centers_file:
        json.dump(centers_payload, centers_file, indent=2)

    metadata_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_image": str(train_path),
        "test_image": str(test_path),
        **metadata,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata_payload, metadata_file, indent=2)


def maybe_show_results(
    no_show,
    train_label_image,
    test_label_image,
    train_skin_mask=None,
    train_skin_overlay=None,
    test_skin_mask=None,
    test_skin_overlay=None,
):
    if no_show:
        return

    cv2.imshow("Train labels", train_label_image)
    cv2.imshow("Test labels", test_label_image)
    if train_skin_mask is not None:
        cv2.imshow("Train skin mask", train_skin_mask)
        cv2.imshow("Train skin overlay", train_skin_overlay)
        cv2.imshow("Test skin mask", test_skin_mask)
        cv2.imshow("Test skin overlay", test_skin_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    train_path = Path(args.train_image)
    test_path = Path(args.test_image) if args.test_image is not None else train_path

    train_bgr = load_color_image(train_path)
    test_bgr = load_color_image(test_path)

    print(
        "Training image:",
        train_bgr.shape[0],
        "rows x",
        train_bgr.shape[1],
        "cols x",
        train_bgr.shape[2],
        "channels",
    )
    print(
        "Test image:",
        test_bgr.shape[0],
        "rows x",
        test_bgr.shape[1],
        "cols x",
        test_bgr.shape[2],
        "channels",
    )

    _, train_flat_features, feature_names = extract_flat_features(train_bgr, args.features)
    _, test_flat_features, _ = extract_flat_features(test_bgr, args.features)

    normalizer = fit_feature_normalizer(train_flat_features, args.features)
    train_fit_features = normalize_features(train_flat_features, normalizer)
    test_fit_features = normalize_features(test_flat_features, normalizer)

    kmeans = KMeans(n_clusters=args.k, random_state=0, n_init=10)
    kmeans.fit(train_fit_features)

    train_labels = reshape_labels(kmeans.labels_, train_bgr.shape)
    test_labels = reshape_labels(kmeans.predict(test_fit_features), test_bgr.shape)
    palette = make_cluster_palette(args.k)
    train_label_image = render_label_image(train_labels, palette)
    test_label_image = render_label_image(test_labels, palette)

    cluster_centers = denormalize_centers(kmeans.cluster_centers_, normalizer)
    train_counts = count_pixels_per_cluster(train_labels, args.k)
    test_counts = count_pixels_per_cluster(test_labels, args.k)
    centers_payload = build_centers_payload(
        cluster_centers, feature_names, train_counts, test_counts, palette
    )
    print_cluster_summary(cluster_centers, feature_names, train_counts, test_counts, palette)

    skin_clusters = validate_skin_clusters(args.skin_clusters, args.k)
    train_skin_mask = None
    train_skin_overlay = None
    test_skin_mask = None
    test_skin_overlay = None
    if skin_clusters is not None:
        train_skin_mask = build_skin_mask(train_labels, skin_clusters)
        train_skin_overlay = build_overlay(train_bgr, train_skin_mask)
        test_skin_mask = build_skin_mask(test_labels, skin_clusters)
        test_skin_overlay = build_overlay(test_bgr, test_skin_mask)
        print(f"Skin clusters: {skin_clusters}")
    else:
        print("No skin clusters selected. Labels and centers were saved for manual inspection.")

    metadata = {
        "features": args.features,
        "feature_names": list(feature_names),
        "k": args.k,
        "skin_clusters": skin_clusters,
        "train_image_shape": list(train_bgr.shape),
        "test_image_shape": list(test_bgr.shape),
        "cluster_centers_file": "centers.json",
        "normalization": None,
    }
    if normalizer is not None:
        metadata["normalization"] = {
            "type": "zscore_from_train_image",
            "mean": [float(value) for value in normalizer["mean"]],
            "std": [float(value) for value in normalizer["std"]],
        }

    run_dir = (
        Path(args.output_dir)
        / test_path.stem
        / f"{args.features}_k{args.k}_train_{train_path.stem}"
    )
    save_outputs(
        run_dir,
        train_path,
        test_path,
        train_label_image,
        test_label_image,
        centers_payload,
        metadata,
        train_skin_mask=train_skin_mask,
        train_skin_overlay=train_skin_overlay,
        test_skin_mask=test_skin_mask,
        test_skin_overlay=test_skin_overlay,
    )
    print(f"Outputs saved in: {run_dir}")

    maybe_show_results(
        args.no_show,
        train_label_image,
        test_label_image,
        train_skin_mask=train_skin_mask,
        train_skin_overlay=train_skin_overlay,
        test_skin_mask=test_skin_mask,
        test_skin_overlay=test_skin_overlay,
    )


if __name__ == "__main__":
    main()

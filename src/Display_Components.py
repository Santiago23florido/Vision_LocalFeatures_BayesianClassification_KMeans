import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from features import extract_feature_image


SPACE_CHOICES = ("rgb", "hsv", "ycrcb", "cbcr_grad")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display and optionally save RGB/HSV/YCrCb/cbcr_grad component images."
    )
    parser.add_argument("image_path", help="Input image path.")
    parser.add_argument(
        "--space",
        choices=[*SPACE_CHOICES, "all"],
        default="all",
        help="Color space to visualize.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory where components are saved. Disabled if omitted.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows (useful for non-GUI runs).",
    )
    return parser.parse_args()


def load_image(image_path):
    img_bgr = cv2.imread(str(image_path), -1)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image.")
    return img_bgr


def to_uint8_channel(channel_img, normalize=False):
    channel = channel_img.astype(np.float32)
    if normalize:
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(channel, 0, 255).astype(np.uint8)


def build_space_data(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    cbcr_grad, _ = extract_feature_image(img_bgr, "cbcr_grad")
    cb_grad = to_uint8_channel(cbcr_grad[:, :, 0])
    cr_grad = to_uint8_channel(cbcr_grad[:, :, 1])
    grad_y = to_uint8_channel(cbcr_grad[:, :, 2], normalize=True)

    return {
        "rgb": {
            "original": img_rgb,
            "channels": [
                {"title": "R", "filename": "R", "image": r, "cmap": "gray"},
                {"title": "G", "filename": "G", "image": g, "cmap": "gray"},
                {"title": "B", "filename": "B", "image": b, "cmap": "gray"},
            ],
        },
        "hsv": {
            "original": img_rgb,
            "channels": [
                {"title": "H", "filename": "H", "image": h, "cmap": "hsv"},
                {"title": "S", "filename": "S", "image": s, "cmap": "gray"},
                {"title": "V", "filename": "V", "image": v, "cmap": "gray"},
            ],
        },
        "ycrcb": {
            "original": img_rgb,
            "channels": [
                {"title": "Y", "filename": "Y", "image": y, "cmap": "gray"},
                {"title": "Cr", "filename": "Cr", "image": cr, "cmap": "gray"},
                {"title": "Cb", "filename": "Cb", "image": cb, "cmap": "gray"},
            ],
        },
        "cbcr_grad": {
            "original": img_rgb,
            "channels": [
                {"title": "Cb", "filename": "Cb", "image": cb_grad, "cmap": "gray"},
                {"title": "Cr", "filename": "Cr", "image": cr_grad, "cmap": "gray"},
                {
                    "title": "|grad(Y)|",
                    "filename": "gradY",
                    "image": grad_y,
                    "cmap": "gray",
                },
            ],
        },
    }


def render_space(space_name, space_data):
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax_list = axes.ravel()
    ax_list[0].imshow(space_data["original"])
    ax_list[0].set_title("Original")
    ax_list[0].axis("off")

    for index, channel_data in enumerate(space_data["channels"], start=1):
        ax_list[index].imshow(channel_data["image"], cmap=channel_data["cmap"])
        ax_list[index].set_title(channel_data["title"])
        ax_list[index].axis("off")

    figure.suptitle(space_name.upper(), fontsize=12)
    figure.tight_layout()
    return figure


def save_space_outputs(base_output_dir, image_stem, space_name, space_data, figure):
    output_dir = Path(base_output_dir) / image_stem / space_name
    output_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = cv2.cvtColor(space_data["original"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "original.png"), original_bgr)
    for channel_data in space_data["channels"]:
        cv2.imwrite(str(output_dir / f"{channel_data['filename']}.png"), channel_data["image"])
    figure.savefig(output_dir / "figure.png", dpi=150)


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    img_bgr = load_image(image_path)
    height, width, channels = img_bgr.shape
    print(
        "Image size:",
        height,
        "rows x",
        width,
        "cols x",
        channels,
        "channels",
    )

    spaces = build_space_data(img_bgr)
    selected_spaces = list(SPACE_CHOICES) if args.space == "all" else [args.space]
    figures = []
    for space_name in selected_spaces:
        figure = render_space(space_name, spaces[space_name])
        figures.append(figure)
        if args.save_dir is not None:
            save_space_outputs(args.save_dir, image_path.stem, space_name, spaces[space_name], figure)

    if not args.no_show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()

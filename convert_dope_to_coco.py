# Converts data from DOPE format into COCO format

# Debugging Tool to Visualize Synthetic Data Projected Points Accuracy

import json

import sys
from tqdm import tqdm

sys.path.append("../dope_training")
# from utils import loadimages

import argparse
import os


def append_dot(extensions):
    res = []

    for ext in extensions:
        if not ext.startswith("."):
            res.append(f".{ext}")
        else:
            res.append(ext)

    return res

def loadimages(root, extensions=["png"]):
    imgs = []
    extensions = append_dot(extensions)

    def add_json_files(
        path,
    ):
        for ext in extensions:
            for file in os.listdir(path):
                imgpath = os.path.join(path, file)
                if (
                    imgpath.endswith(ext)
                    and os.path.isfile(imgpath)
                    and os.path.isfile(imgpath.replace(ext, ".json"))
                ):
                    imgs.append(
                        (
                            imgpath,
                            imgpath.replace(path, "").replace("/", ""),
                            imgpath.replace(ext, ".json"),
                        )
                    )

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isdir(os.path.join(path, o))
        ]

        for path_entry in folders:
            explore(path_entry)

        add_json_files(path)

    explore(root)

    return imgs


def merge_json(path_output, path_root):

    detections_gt = []
    keypoints_gt = []
    instances_gt = []

    # Find all json files recursively
    imgs = loadimages(path_root, extensions=["jpg", "png"])

    for i, (img_path, _, json_path) in enumerate(
        tqdm(imgs, desc="Merge Progress", colour="green")
    ):
        img_rel_path = img_path.replace(opt.data, "")

        with open(json_path) as f:
            data_json = json.load(f)

        for object in data_json["objects"]:

            x_min = min(object["projected_cuboid"], key=lambda point: point[0])[0]
            x_max = max(object["projected_cuboid"], key=lambda point: point[0])[0]
            y_min = min(object["projected_cuboid"], key=lambda point: point[1])[1]
            y_max = max(object["projected_cuboid"], key=lambda point: point[1])[1]

            detections_gt.append(
                {
                    "image_id": img_rel_path,
                    "category_id": 1,
                    "bbox": [  # Coco bounding boxes are [x_min, y_min, width, height]
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                    ],
                    "score": 1.0,
                }
            )
            projected_cuboids = object["projected_cuboid"]

            if len(projected_cuboids) == 8 and "projected_cuboid_centroid" in object:
                projected_cuboids.append(object["projected_cuboid_centroid"])

            keypoints_gt.append(
                {
                    "num_keypoints": len(projected_cuboids),
                    "iscrowd": 0,
                    "keypoints": [  # Flatten projected_cuboids list
                        point for pair in projected_cuboids for point in pair
                    ],
                    "image_id": img_rel_path,
                    "category_id": 1,
                    "bbox": [  # Coco bounding boxes are [x_min, y_min, width, height]
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                    ],
                    "id": i,
                    "full_file_path": str(os.path.join(os.getcwd(), img_path)),
                }
            )

            instances_gt.append(
                {
                    "iscrowd": 0,
                    "image_id": img_rel_path,
                    "category_id": 1,
                    "bbox": [  # Coco bounding boxes are [x_min, y_min, width, height]
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                    ],
                }
            )

    print("Writing to groundtruth files...")
    os.makedirs(path_output, exist_ok=True)

    with open(os.path.join(path_output, "detections.json"), "w") as output:
        json.dump(detections_gt, output, indent=4)

    with open(os.path.join(path_output, "keypoints.json"), "w") as output:
        json.dump(keypoints_gt, output, indent=4)

    # with open(os.path.join(path_output, "val_keypoints.json"), "w") as output:
    #     json.dump({"annotations" : keypoints_gt}, output, indent=4)

    with open(os.path.join(path_output, "instances.json"), "w") as output:
        json.dump(instances_gt, output, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outf",
        default="output/debug",
        help="Where to store the debug output images.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Folder containing groundtruth and images.",
    )

    opt = parser.parse_args()

    merge_json(opt.outf, opt.data)

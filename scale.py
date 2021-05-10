import csv
import os
import re
from statistics import mean

import click
import cv2

from append import assemble_new_filepath
from lib.hazelnuts import (
    rotate_if_you_must,
    crop_hazelnuts,
    sort_nuts,
    read_qr_code,
    get_attributes_from_filename,
)
from lib.utils import get_kv_pairs


def process_image(image_path: str) -> list:
    image_stats = []
    columns = 5
    rows = 6

    image = cv2.imread(image_path)
    image = rotate_if_you_must(image)

    qr_code_content = read_qr_code(image)
    qr_attributes = get_attributes_from_filename(qr_code_content)
    # print(qr_attributes)

    expected_nuts = columns * rows
    cropped_nuts = crop_hazelnuts(image, expected_nuts)

    sorted_nuts = sort_nuts(cropped_nuts, rows, columns)

    for i, nut in enumerate(sorted_nuts):
        width = nut["right"][0] - nut["left"][0]
        height = nut["bottom"][1] - nut["top"][1]

        # add UID
        image_stats.append(
            {
                "index": i + 1,
                "UID": qr_attributes["UID"],
                "rect_width": width,
                "rect_height": height,
                "ext_width": width,
                "ext_height": height,
            }
        )
    return image_stats


def write_to_csv(stats: list, dest: str):
    header = ["filename"]
    for i in range(1, 31):
        header += [
            f"rect_width_{i}",
            f"rect_height_{i}",
            f"ext_width_{i}",
            f"ext_height_{i}",
        ]
    csv_filename = os.path.join(dest, "nutscale.csv")
    with open(csv_filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header)
        for entry in stats:
            row = [entry["filename"]]
            for stat in entry["stats"]:
                row.append(stat["rect_width"])
                row.append(stat["rect_height"])
                row.append(stat["ext_width"])
                row.append(stat["ext_height"])
            writer.writerow(row)


def generate_dest_file_mapping(dest: str) -> dict:
    mapping = {}
    for cur_path, _, files in os.walk(dest):
        for file in files:
            if file.endswith(".png"):
                abspath = os.path.join(cur_path, file)
                uuid = re.findall(r"\{UID_(.*?)\}", file)
                nut = re.findall(r"\{Nut_(.*?)\}", file)
                if uuid and nut:
                    key = f"{uuid[0]}---{nut[0]}"
                    if key not in mapping.keys():
                        mapping[key] = [abspath]
                    else:
                        mapping[key].append(abspath)
    return mapping


def append_dimensions_to_filenames(
    dimensions: list, filenames: list, height: int, width: int
):
    # get avg scale
    # scale: px \ width, avg. with height

    measurements = []
    for d in dimensions:
        measurements.append(d["height"] / height)
        measurements.append(d["width"] / width)
    mean_scale = mean(measurements)
    mean_scale = round(mean_scale, 3)

    # append that scale to the filenames in question
    for filename in filenames:
        kv_pairs = get_kv_pairs(filename)
        kv_pairs.append(f"Scale_{mean_scale}")
        new_filename = assemble_new_filepath(filename, kv_pairs)
        os.rename(filename, new_filename)


# -s, --src: path to source images
# -d, --dest: path to already-existing processed images
# -w, --width: width of the squares in mm
# -h, --height: height of the squares in mm
# -a, --append: add a {Key_Value} pair to processed images as described below
# -c, --csv: write out a csv file like it already does


@click.command()
@click.option("--dest", "-d", type=click.Path(exists=True), help="dest")
@click.option("--src", "-s", type=click.Path(exists=True), help="source file")
@click.option("--width", "-w", type=click.FLOAT, help="width in mm")
@click.option("--height", "-h", type=click.FLOAT, help="height in mm")
@click.option("--csv", "-c", is_flag=True, help="write out csv file")
@click.option("--append", "-a", is_flag=True, help="append to files")
def main(dest: str, src: str, width: int, height: int, csv: bool, append: bool):
    if not src:
        return
    # 1. based on dest, create a mapping
    # key: what's in the qr code
    # value: list of processed files that belong to that qr code (masks, in shell, kernel etc.)
    existing_files_mapping = generate_dest_file_mapping(dest)

    # 2. go and process the images
    dimension_mapping = {}

    single_image = src.lower().endswith("jpg")
    stats = []
    if single_image:
        stat = process_image(src)
        stats.append({"filename": src.split("/")[-1], "stats": stat})
        dest = os.path.join("/", *src.split("/")[:-1])
    else:
        for filename in os.listdir(src):
            image_path = os.path.join(src, filename)
            print(image_path)
            try:
                stat = process_image(image_path)
                if csv:
                    stats.append({"filename": filename, "stats": stat})
                for s in stat:
                    index = s["index"]
                    uid = s["UID"]
                    key = f"{uid}---{index}"
                    key_stats = {"height": s["rect_height"], "width": s["rect_width"]}
                    if key not in dimension_mapping.keys():
                        dimension_mapping[key] = [key_stats]
                    else:
                        dimension_mapping[key].append(key_stats)

            except Exception as e:
                print(e)
    if csv:
        write_to_csv(stats, src)

    if append and width is not None and height is not None:
        for key in dimension_mapping.keys():
            files_to_change = existing_files_mapping.get(key, None)
            if files_to_change:
                append_dimensions_to_filenames(
                    dimension_mapping[key], files_to_change, height, width
                )


if __name__ == "__main__":
    main()

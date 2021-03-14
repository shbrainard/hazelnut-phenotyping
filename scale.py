import csv
import os

import click
import cv2

from lib.hazelnuts import rotate_if_you_must, crop_hazelnuts, sort_nuts


def process_image(image_path: str) -> list:
    image_stats = []
    columns = 5
    rows = 6

    image = cv2.imread(image_path)
    image = rotate_if_you_must(image)
    expected_nuts = columns * rows
    cropped_nuts = crop_hazelnuts(image, expected_nuts)

    sorted_nuts = sort_nuts(cropped_nuts, rows, columns)

    for i, nut in enumerate(sorted_nuts):
        width = nut["right"][0] - nut["left"][0]
        height = nut["bottom"][1] - nut["top"][1]
        image_stats.append(
            {
                "index": i,
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
    # print(header)
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


@click.command()
@click.option("--src", "-s", type=click.Path(exists=True), help="source file")
def main(src: str):
    single_image = src.lower().endswith("jpg")
    stats = []
    if single_image:
        stat = process_image(src)
        stats.append({"filename": src.split("/")[-1], "stats": stat})
        dest = os.path.join("/", *src.split("/")[:-1])
    else:
        for filename in os.listdir(src):
            image_path = os.path.join(src, filename)
            try:
                stat = process_image(image_path)
                stats.append({"filename": filename, "stats": stat})
            except Exception as e:
                print(e)
        dest = src
    write_to_csv(stats, dest)


if __name__ == "__main__":
    main()

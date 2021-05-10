import csv

import click
import cv2
import imutils
import numpy as np

from lib.constants import BINARY_MASKS_DIR
from lib.crop import get_carrot_contour
from lib.utils import get_masks_to_process, get_attributes_from_filename


def assemble_instance(file):
    instance = get_attributes_from_filename(file)

    if "kernel" in file:
        image_type = "kernel"
    elif "in-shell" in file:
        image_type = "in-shell"
    instance["type"] = image_type

    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    white_pixels = cv2.countNonZero(image)
    instance["white_pixels"] = white_pixels

    scale = instance.get("Scale", None)
    if scale:
        instance["area"] = round(white_pixels / float(scale) ** 2, 3)
    else:
        instance["area"] = "no scale"

    contour = get_carrot_contour(image)

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    # cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # print(type(cnts))
    # c = max(cnts, key=cv2.contourArea)
    # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(color_image, [contour], -1, (0, 0, 255), 3)

    x, y, w, h = cv2.boundingRect(contour)
    # color_image = cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    instance["width"] = w
    instance["height"] = h

    # https://stackoverflow.com/questions/31281235/anomaly-with-ellipse-fitting-when-using-cv2-ellipse-with-different-parameters
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html#drawing-ellipse
    ellipse = cv2.fitEllipse(contour)
    ellipse_major_axis, ellipse_minor_axis = ellipse[1]
    ellipse_angle = ellipse[2]
    instance["ellipse_major_axis"] = round(ellipse_major_axis, 2)
    instance["ellipse_minor_axis"] = round(ellipse_minor_axis, 2)
    instance["ellipse_angle"] = round(ellipse_angle, 2)
    # cv2.ellipse(color_image, ellipse, (0, 255, 0), 2)

    # print(ellipse[1], ellipse[2])
    rect = cv2.minAreaRect(contour)
    _, width_height, _ = rect
    min_area_width, min_area_height = width_height
    instance["min_area_width"] = round(min_area_width, 2)
    instance["min_area_height"] = round(min_area_height, 2)
    # print(min_area_width, min_area_height)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(color_image, [box], 0, (255, 0, 0), 2)
    # print(len(cnts))
    # cv2.imwrite("/Users/creimers/Downloads/glibba.png", color_image)
    return instance


def spit_out_csv(instances_dict: dict, dest: str):
    if instances_dict:
        header = [
            "UID",
            "Location",
            "Year",
            "Row",
            "Plant",
            "Nut",
            "Scale",
            "inshell_white_pixels",
            "inshell_area",
            "inshell_width",
            "inshell_height",
            "inshell_ellipse_major_axis",
            "inshell_ellipse_minor_axis",
            "inshell_ellipse_angle",
            "inshell_min_area_width",
            "inshell_min_area_height",
            "kernel_white_pixels",
            "kernel_area",
            "kernel_width",
            "kernel_height",
            "kernel_ellipse_major_axis",
            "kernel_ellipse_minor_axis",
            "kernel_ellipse_angle",
            "kernel_min_area_width",
            "kernel_min_area_height",
        ]

        with open(dest, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(list(header))
            for _, instance in instances_dict.items():
                first_instance = instance[list(instance.keys())[0]]
                common_properties = [
                    first_instance["UID"],
                    first_instance["Location"],
                    first_instance["Year"],
                    first_instance["Row"],
                    first_instance["Plant"],
                    first_instance["Nut"],
                    first_instance["Scale"],
                ]
                inshell_properties = [None] * 9
                kernel_properties = [None] * 9
                if instance.get("in-shell", None):
                    inshell_properties = [
                        instance["in-shell"]["white_pixels"],
                        instance["in-shell"]["area"],
                        instance["in-shell"]["width"],
                        instance["in-shell"]["height"],
                        instance["in-shell"]["ellipse_major_axis"],
                        instance["in-shell"]["ellipse_minor_axis"],
                        instance["in-shell"]["ellipse_angle"],
                        instance["in-shell"]["min_area_width"],
                        instance["in-shell"]["min_area_height"],
                    ]
                if instance.get("kernel", None):
                    kernel_properties = [
                        instance["kernel"]["white_pixels"],
                        instance["kernel"]["area"],
                        instance["kernel"]["width"],
                        instance["kernel"]["height"],
                        instance["kernel"]["ellipse_major_axis"],
                        instance["kernel"]["ellipse_minor_axis"],
                        instance["kernel"]["ellipse_angle"],
                        instance["kernel"]["min_area_width"],
                        instance["kernel"]["min_area_height"],
                    ]
                row = common_properties + inshell_properties + kernel_properties
                writer.writerow(row)


@click.command()
@click.option("--dest", "-d", type=click.Path(), help="destination directory of csv")
@click.option("--dry", "-d", is_flag=True, help="don't touch the database")
@click.option(
    "--src",
    "-s",
    type=click.Path(exists=True),
    help="source directory of images to process",
)
def run(dest, dry, src):
    instances_dict = {}
    subdirs = get_masks_to_process(src, BINARY_MASKS_DIR)
    for dir in subdirs:
        for file in dir["files"]:
            try:
                instance = assemble_instance(file)
                instance_key = f"{instance['UID']}___{instance['Nut']}"
                if instance_key not in instances_dict.keys():
                    instances_dict[instance_key] = {f"{instance['type']}": instance}
                else:
                    instances_dict[instance_key][instance["type"]] = instance
                if dry:
                    print(instance)
            except Exception as e:
                print(e)
    if not dry:
        spit_out_csv(instances_dict, dest)


if __name__ == "__main__":
    run()

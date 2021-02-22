import os
import shutil

import cv2
import imutils
import numpy as np
from pyzbar import pyzbar

from lib.crop import get_carrot_contour
from lib.utils import get_attributes_from_filename


def create_nut_mask_threshold(nut_image):
    gray = cv2.cvtColor(nut_image, cv2.COLOR_BGR2GRAY)
    #  gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # return gray
    # white backdrop
    inv = cv2.THRESH_BINARY_INV

    # black backdrop
    inv = cv2.THRESH_BINARY

    # the lower the first value, the more grey is detected

    # white backdrop
    # threshold = 150

    # black backdrop
    threshold = 5
    mask = cv2.threshold(gray, threshold, 255, inv)[1]

    # if smoothen > 0:
    #     thresh = cv2.erode(thresh, None, iterations=smoothen)
    #     thresh = cv2.dilate(thresh, None, iterations=smoothen)

    # binary = thresh

    #  return binary
    # black_row = np.zeros((12, binary.shape[1]), dtype=np.uint8)
    # buffered_binary = np.vstack([black_row, binary, black_row])

    # return buffered_binary
    return mask


def create_nut_mask_blue_hsv(nut_image):
    """
    create a binary mask of a hazelnut or its kernal
    """
    hsv_img = cv2.cvtColor(nut_image, cv2.COLOR_BGR2HSV)

    # https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv
    blue_min = np.array([100, 75, 75], np.uint8)
    blue_max = np.array([120, 255, 255], np.uint8)

    mask = cv2.inRange(hsv_img, blue_min, blue_max)
    mask = cv2.bitwise_not(mask)

    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    ## final mask and masked
    # target = cv2.bitwise_and(nut_image, nut_image, mask=mask)
    # return target

    return mask


def create_nut_mask_brown_hsv(nut_image):
    """
    create a binary mask of a hazelnut or its kernal
    """
    hsv_img = cv2.cvtColor(nut_image, cv2.COLOR_BGR2HSV)

    # BROWN_MIN = np.array([0, 30, 5], np.uint8)
    # BROWN_MAX = np.array([35, 255, 255], np.uint8)

    # TODO: blue backdrop and then select everything that's not blue

    # values from here: https://stackoverflow.com/a/46113436/1909378
    brown_min = np.array([5, 0, 0], np.uint8)
    brown_max = np.array([50, 255, 255], np.uint8)

    frame_threshed = cv2.inRange(hsv_img, brown_min, brown_max)

    # cnts = cv2.findContours(
    #     frame_threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sorted_contours = sorted(cnts, key=cv2.contourArea)[::-1]

    # cv2.drawContours(frame_threshed, sorted_contours, -1, (0, 0, 255), -1)
    return frame_threshed


def create_nut_mask_overlay(nut_image, masking_method):
    overlay = nut_image.copy()
    output = nut_image.copy()

    mask = masking_method(nut_image)
    contour = get_carrot_contour(mask)
    overlay_color = (0, 255, 0)

    alpha = 0.5

    cv2.drawContours(overlay, [contour], -1, overlay_color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def create_nut_mask(nut_image, masking_method):
    """
    create a binary mask of the nut
    """
    output = np.zeros(nut_image.shape, dtype=np.uint8)

    mask = masking_method(nut_image)
    contour = get_carrot_contour(mask)
    if contour is not None and contour.shape[0] > 100:
        cv2.drawContours(output, [contour], -1, (255, 255, 255), -1)
        return output


def crop_hazelnuts(image: np.ndarray, expected_nuts: int):
    """
    extract nut images
    """
    if expected_nuts == 1:
        return [image]

    # fuck the qr code on top of the image (~ 15%)
    image = image[int(image.shape[0] * 0.15) :, :]

    # cv2.imwrite("/Users/creimers/Downloads/affe.png", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # lots of light
    thresh = cv2.threshold(gray, 200, 250, cv2.THRESH_BINARY)[1]

    # little light
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    # thresh = cv2.erode(thresh, None, iterations=2)
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imwrite("/Users/creimers/Downloads/jiggi.png", thresh)

    # qr_codes = pyzbar.decode(thresh)
    # max_codes = expected_carrots

    # find contours
    _, contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    ignore = 1  # ignore the x largest contours
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[
        ignore : expected_nuts + ignore
    ]

    nuts = []
    for c in cnts:
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        mask = image[ext_top[1] : ext_bot[1], ext_left[0] : ext_right[0]]
        mask = crop_away_grid_artifacts(mask)
        mask_dict = {"mask": mask, "top": ext_top, "left": ext_left}
        nuts.append(mask_dict)

    return nuts


def sort_nuts(cropped_nuts: list, rows: int, columns: int) -> list:
    """
    sort cropped nuts from top left to bottom right
    """
    sorted_nuts = []
    sorted_by_top = sorted(cropped_nuts, key=lambda x: x["top"][1])

    for row in range(rows):
        that_row = sorted_by_top[row * columns : (row + 1) * columns]
        sorted_row = sorted(that_row, key=lambda x: x["left"][0])
        sorted_nuts += sorted_row
    return sorted_nuts


def crop_away_grid_artifacts(cropped_nut):
    """
    crop away grid artifacts from the cropped nut
    """
    buffer = 50  # px
    crop_from_top = buffer  # px (white border)
    crop_from_bottom = cropped_nut.shape[1] - buffer  # px (white border)
    crop_from_left = buffer
    crop_from_right = cropped_nut.shape[0] - buffer

    return cropped_nut[crop_from_top:crop_from_bottom, crop_from_left:crop_from_right]


def read_qr_code(image: np.ndarray) -> str:
    # fuck everything but the qr code on top of the image (~ 20%)
    image = image[: int(image.shape[0] * 0.2), :]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    qr_codes = pyzbar.decode(thresh)
    decoded = qr_codes[0].data.decode("utf-8")
    return decoded


def rotate_if_you_must(image):
    """
    rotate the image such that the qr code is on top
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # less light
    # thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    # more light
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    qr_codes = pyzbar.decode(thresh)

    code_rect = qr_codes[0].rect

    image_height = image.shape[0]
    image_width = image.shape[1]

    # image 'on side'
    if image_height < image_width:
        # left
        if code_rect.left < image_width / 2:
            print("ℹ️  qr code on the left")
            return np.rot90(image, 3)  # three times counter clockwise
        # right
        else:
            print("ℹ️  qr code on the right")
            return np.rot90(image, 1)
    # top or bottom
    else:
        if code_rect.top < image_height / 2:
            print("ℹ️  qr code on top")
            return image
        else:
            print("ℹ️  qr code at bottom")
            return np.rot90(image, 2)


def process_image(src: str, dest: str, columns: int, rows: int, scan_type: str):
    image = cv2.imread(src)
    image = rotate_if_you_must(image)
    expected_nuts = columns * rows
    cropped_nuts = crop_hazelnuts(image, expected_nuts)

    qr_code_content = read_qr_code(image)
    qr_attributes = get_attributes_from_filename(qr_code_content)

    masking_method = create_nut_mask_blue_hsv

    sorted_nuts = sort_nuts(cropped_nuts, rows, columns)
    cells_with_nuts = 0
    for i, cropped_nut_dict in enumerate(sorted_nuts):
        i = i + 1
        cropped_nut = cropped_nut_dict["mask"]

        location = qr_attributes["Location"]
        year = qr_attributes["Year"]
        row = qr_attributes["Row"]
        plant = qr_attributes["Plant"]

        dest_dir = os.path.join(
            dest, location, year, f"Row_{row}-Plant_{plant}", scan_type
        )

        mask = create_nut_mask(cropped_nut, masking_method)
        if mask is not None:
            cells_with_nuts += 1
            # write crop to disk
            os.makedirs(dest_dir, exist_ok=True)
            file_name = f"{qr_code_content}{{Nut_{i}}}.png"
            dest_path_crop = os.path.join(dest_dir, file_name)
            cv2.imwrite(dest_path_crop, cropped_nut)

            # write mask to disk
            dest_dir_mask = os.path.join(dest_dir, "binary-masks")
            os.makedirs(dest_dir_mask, exist_ok=True)
            dest_path_mask = os.path.join(dest_dir_mask, file_name)
            cv2.imwrite(dest_path_mask, mask)

            # write mask overlay to disk
            mask_overlay = create_nut_mask_overlay(cropped_nut, masking_method)
            if mask_overlay is not None:
                dest_dir_mask_overlay = os.path.join(dest_dir, "binary-masks-overlay")
                os.makedirs(dest_dir_mask_overlay, exist_ok=True)
                dest_path_mask_overlay = os.path.join(dest_dir_mask_overlay, file_name)
                cv2.imwrite(dest_path_mask_overlay, mask_overlay)

    print("ℹ️  total nuts found:", cells_with_nuts, "🌰 " * cells_with_nuts)

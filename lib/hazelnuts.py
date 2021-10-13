import os
import click

import cv2
import numpy as np
from pyzbar import pyzbar

from lib.crop import get_carrot_contour
from lib.constants import config
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


def create_ellipse_overlay(nut_image, masking_method):
    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    image = nut_image.copy()
    mask = masking_method(nut_image)
    contour = get_carrot_contour(mask)

    if contour is not None:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)
        _, cols = image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(image, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
        return image


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

        # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        x, y, w, h = cv2.boundingRect(c)

        mask_dict = {
            "mask": mask,
            "top": ext_top,
            "left": ext_left,
            "right": ext_right,
            "bottom": ext_bot,
            "width": w,
            "height": h,
        }
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
    if qr_codes:
        decoded = qr_codes[0].data.decode("utf-8")
        return decoded


def get_position_of_qr_code(image, manual_mode):
    """
    get position of qr code
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

    if qr_codes:
        code_rect = qr_codes[0].rect

        image_height = image.shape[0]
        image_width = image.shape[1]

        # image 'landscape'
        if image_height < image_width:
            # left
            if code_rect.left < image_width / 2:
                print("ℹ️  qr code on the left")
                return "left"
            # right
            else:
                print("ℹ️  qr code on the right")
                return "right"
        # top or bottom
        else:
            if code_rect.top < image_height / 2:
                print("ℹ️  qr code on top")
                return "top"
                # return image
            else:
                print("ℹ️  qr code at bottom")
                return "bottom"
    else:
        if manual_mode is True:
            manual_position = click.prompt(
                "🚨 No qr code found. Is it top [t], right [r], bottom [b] or left [l]?",
                type=str,
                default="t",
            )
            return {"t": "top", "r": "right", "b": "bottom", "l": "left"}.get(
                manual_position, None
            )
        print("🚨 no qr code found...")


def rotate_if_you_must(image, qr_code_position):
    """
    rotate the image such that the qr code is on top
    """

    if qr_code_position == "left":
        return np.rot90(image, 3)  # three times counter clockwise
    elif qr_code_position == "right":
        return np.rot90(image, 1)
    elif qr_code_position == "top":
        return image
    elif qr_code_position == "bottom":
        return np.rot90(image, 2)
    return image


def hazelnut_double_check(dest: str, scan_type: str, qr_attributes: dict) -> str:
    """
    extra line of defense against user
    """

    location = qr_attributes["Location"]
    year = qr_attributes["Year"]
    row = qr_attributes["Row"]
    plant = qr_attributes["Plant"]
    dest_dir = os.path.join(dest, location, year, f"Row_{row}-Plant_{plant}", scan_type)

    # if the user says that something is a "kernel" photo, but the directory does not yet
    # contain an "in-shell" directory, they are prompted to confirm that they indeed are
    # taking photos of kernels
    if scan_type == "kernel":
        in_shell_dir = os.path.join(
            dest, location, year, f"Row_{row}-Plant_{plant}", "in-shell"
        )
        if not os.path.exists(in_shell_dir):
            return "🚨 An in-shell directory does not exist yet. Are you sure you're processing an image of kernels here?"
    # if the user says that a photo is an "in-shell" photo or a "kernel" photo, but there is
    # already a directory called, respectively, "in-shell" or "kernel", you are again prompted
    # to confirm that this is what you intended.
    if os.path.exists(dest_dir):
        return f"🚨 The directory '{scan_type}' already exists. Are you sure you want to proceed?"


def get_qr_code_content(src: str, manual_mode: bool):
    image = cv2.imread(src)
    qr_code_position = get_position_of_qr_code(image, manual_mode)
    if qr_code_position is None:
        return None, image
    image = rotate_if_you_must(image, qr_code_position)

    qr_code_content = read_qr_code(image)
    if qr_code_content is None and manual_mode is False:
        return None, image
    if qr_code_content is None and manual_mode is True:
        enter_manual_code_content = click.prompt(
            "do you want to enter K/V pairs manually [y/N]?",
            type=str,
            default="N",
        )
        if enter_manual_code_content == "N":
            return
        else:
            qr_code_content = click.prompt(
                "Please proceed with the following format: '{key_value}{anotherKey_anotherValue}'.",
                type=str,
                default="",
            )
            if not qr_code_content:
                return None, image
            else:
                return qr_code_content, image
    return qr_code_content, image


def process_image(
    image, dest: str, columns: int, rows: int, scan_type: str, qr_code_content: str
):
    qr_attributes = get_attributes_from_filename(qr_code_content)
    expected_nuts = columns * rows
    cropped_nuts = crop_hazelnuts(image, expected_nuts)

    masking_method = create_nut_mask_blue_hsv

    sorted_nuts = sort_nuts(cropped_nuts, rows, columns)
    cells_with_nuts = 0
    for i, cropped_nut_dict in enumerate(sorted_nuts):
        i = i + 1
        cropped_nut = cropped_nut_dict["mask"]
        width_px = cropped_nut_dict["width"]
        height_px = cropped_nut_dict["height"]
        scale = round(
            (width_px / config["square_width"] + height_px / config["square_height"])
            / 2,
            3,
        )

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
            file_name = f"{qr_code_content}{{Scale_{scale}}}{{Nut_{i}}}.png"
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
            try:
                ellipse_overlay = create_ellipse_overlay(cropped_nut, masking_method)
                if ellipse_overlay is not None:
                    dest_dir_ellipse_overlay = os.path.join(dest_dir, "ellipse-overlay")
                    os.makedirs(dest_dir_ellipse_overlay, exist_ok=True)
                    dest_path_ellipse_overlay = os.path.join(
                        dest_dir_ellipse_overlay, file_name
                    )
                    cv2.imwrite(dest_path_ellipse_overlay, ellipse_overlay)
            except Exception as e:
                print(e)

    print("ℹ️  total nuts found:", cells_with_nuts, "🌰 " * cells_with_nuts)

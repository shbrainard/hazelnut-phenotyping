import os
import re
import time
import click
import cv2
import numpy as np
from pyzbar import pyzbar
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from lib.constants import config
from lib.hazelnuts import process_image, hazelnut_double_check, get_qr_code_content
from lib.utils import get_kv_pairs, get_kv_pairs_dict, get_attributes_from_filename

UID_RE = r"{uid_(?P<uid>.+?)}"


def crop_boxes(image, expected_carrots):
    """
    takes the path of an image that comes out of the camara and
    has already been unskewed.

    Args:
        image (np.array): the image
        expected_carrots (int): the number of carrots to be expected in the image

    Returns:
        boxes (list of np.arrays ): list of the found boxes
    """
    # put a white buffer around the image
    buffer_width = 100

    white_cols = np.ones((image.shape[0], buffer_width, 3), dtype=np.uint8) * 255
    image = np.hstack([white_cols, image, white_cols])

    white_rows = np.ones((buffer_width, image.shape[1], 3), dtype=np.uint8) * 255
    image = np.vstack([white_rows, image, white_rows])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    qr_codes = pyzbar.decode(thresh)
    max_codes = expected_carrots

    # find contours
    _, contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[1 : max_codes + 1]

    # debugging stuff
    # cv2.drawContours(image, cnts, -1, (0, 255, 255), 5)
    # cv2.imwrite("/Users/creimers/Downloads/gates.png", image)

    # find extreme points in contours
    boxes = []
    for i, c in enumerate(cnts):
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        mask = image[ext_top[1] : ext_bot[1], ext_left[0] : ext_right[0]]
        mask_tresh = thresh[ext_top[1] : ext_bot[1], ext_left[0] : ext_right[0]]

        box_code = None
        kode = pyzbar.decode(mask_tresh)
        if len(kode) > 0:
            box_code = kode[0]

        if box_code:
            decoded = box_code.data.decode("utf-8")

            mask_height = mask.shape[0]
            mask_width = mask.shape[1]

            code_rect = qr_codes[0].rect

            if mask_height > mask_width:
                # barcode on bottom
                if code_rect.top > ext_top[1] + mask_height / 2:
                    mask = np.rot90(mask, 1)

                # barcode on top
                if code_rect.top < ext_bot[1] - mask_height / 2:
                    mask = np.rot90(mask, 3)

            else:
                # barcode on left hand side
                if code_rect.left < ext_left[0] + mask_width / 2:
                    mask = np.rot90(mask, 2)

            boxes.append({"mask": mask, "qr": decoded})

    return boxes


def save_box_as_image(box, dest, destdir="Genotype", destsub=""):
    """
    Args:
        box - object
        dest - destination directory as string
        destdir: (str) key to be used to name directory
        subdir: (str) key to be used to name sub directory
    """
    attributes = box["qr"]

    if attributes:
        uid_match = re.search(UID_RE, attributes.lower())
        uid = uid_match.groupdict().get("uid", None)

        kv_dict = get_kv_pairs_dict(attributes)

        # default
        dest_dir = dest

        dirname = kv_dict.get(destdir, None)
        if dirname:
            dest_dir = os.path.join(dest, dirname)

            if destsub:
                subdirname = kv_dict.get(destsub, None)
                if subdirname:
                    dest_dir = os.path.join(dest, dirname, subdirname)

        os.makedirs(dest_dir, exist_ok=True)

        existing_files = {}
        for file in os.listdir(dest_dir):
            uid_match = re.search(UID_RE, file.lower())
            if uid_match:
                existing_uid = uid_match.groupdict().get("uid", None)
                if existing_uid == uid:
                    if uid not in existing_files.keys():
                        existing_files[uid] = []
                    existing_files[uid].append(file)

        count = len(existing_files.get(uid, []))

        kv_pairs = get_kv_pairs(attributes)

        # try:
        #     scalebar_length = measure_scalebar_new(box["mask"])
        #     kv_pairs.append("Scale_%s" % scalebar_length)
        # except:
        #     click.secho("Could not read scalebar!", fg="red")

        kv_pairs.append("Photo_%s" % (count + 1))

        attributes = "".join(["{%s}" % pair for pair in kv_pairs])

        filename = attributes + config["file_format"]
        dest_path = os.path.join(dest_dir, filename)
        msg = " ".join(["writing to disk: ", dest_path])
        click.secho(msg, fg="green")
        cv2.imwrite(dest_path, box["mask"])


def do_acquisition(
    image_path, dest, tmp, destdir, destsub, discard, columns, rows, manual_mode
):
    """
    Args:
        image_path - path to the image as string
        dest - destination directory as string
        destdir: (str) key to be used to name directory
        destsub: (str) key to be used to name sub directory
        discard: a directory to copy the photo to if not processed (str)
        columns: (int) number of columns in grid
        rows: (int) number of rows in grid
        manual_mode: (bool) enter content of missing qr code
    """

    temp_dest = tmp
    if not os.path.exists(temp_dest):
        os.makedirs(temp_dest)

    scan_type_map = {"s": "in-shell", "k": "kernel", "b": "blanched"}
    scan_type = click.prompt(
        "in-shell [s], kernels [k] or blanched [b]?", type=str, default="s"
    )

    qr_code_content, image = get_qr_code_content(image_path, manual_mode)
    qr_code_attributes = get_attributes_from_filename(qr_code_content)

    # do the checks here
    double_check_message = hazelnut_double_check(
        dest, scan_type_map[scan_type.lower()], qr_code_attributes
    )
    if double_check_message:
        proceed = click.prompt(f"{double_check_message} [N|y]", default="N")
        if proceed == "y":
            process_image(
                image,
                dest,
                columns,
                rows,
                scan_type_map[scan_type.lower()],
                qr_code_content,
            )
        else:
            do_acquisition(
                image_path,
                dest,
                tmp,
                destdir,
                destsub,
                discard,
                columns,
                rows,
                manual_mode,
            )
    else:
        process_image(
            image,
            dest,
            columns,
            rows,
            scan_type_map[scan_type.lower()],
            qr_code_content,
        )


class MyHandler(FileSystemEventHandler):
    def __init__(self, *args, **kwargs):
        super(MyHandler, self).__init__()
        self.dest = kwargs["dest"]
        self.tmp = kwargs["tmp"]
        self.destdir = kwargs.get("destdir", None)
        self.destsub = kwargs.get("destsub", None)
        self.discard = kwargs.get("discard", None)
        self.columns = kwargs.get("columns", None)
        self.rows = kwargs.get("rows", None)
        self.manual_mode = kwargs.get("manual_mode", None)

    def on_created(self, event):
        msg = "New file %s detected." % event.src_path
        click.secho(msg, fg="magenta")

        if event.src_path.lower().endswith(".jpg") or event.src_path.lower().endswith(
            ".jpeg"
        ):

            do_acquisition(
                event.src_path,
                self.dest,
                self.tmp,
                self.destdir,
                self.destsub,
                self.discard,
                self.columns,
                self.rows,
                self.manual_mode,
            )

            msg = "🛎   Bing bong! "
            click.secho(msg, fg="green")
            click.secho("Ready for the next one.", fg="white")


@click.command()
@click.option(
    "--columns", "-c", type=click.INT, help="number of columns in grid", default=5
)
@click.option(
    "--dest", "-d", type=click.Path(exists=True), help="destination of croped boxes"
)
@click.option(
    "--destdir",
    default="Genotype",
    help="The key to be used to name the destination directory",
)
@click.option(
    "--destsub",
    default="",
    help="The key to be used to name the destination sub directory",
)
@click.option(
    "--discard",
    type=click.Path(exists=True),
    help="a directory to copy the photo to if not processed",
)
@click.option(
    "--manual", "-m", is_flag=True, help="Manually enter them key/value pairs."
)
@click.option("--rows", "-r", type=click.INT, help="number of rows in grid", default=6)
@click.option(
    "--src", "-s", type=click.Path(exists=True), help="source folder to watch"
)
@click.option(
    "--tmp",
    "-t",
    type=click.Path(exists=True),
    help="destination of tmp dir. Non dropbox dir is advised.",
    required=True,
)
def run(columns, dest, destdir, destsub, discard, manual, rows, src, tmp):
    if src is None:
        click.secho("No source specified. Use --src option.", fg="red")
        return
    if dest is None:
        click.secho("No destination specified. Use --dest option.", fg="red")
        return

    msg = "I'm observing %s" % src
    click.secho(msg, fg="green")

    msg = f"Expecting a {columns} (columns) x {rows} (rows) grid."
    click.secho(msg, fg="magenta")

    msg = "I'm writing to %s" % dest
    click.secho(msg, fg="blue")

    event_handler = MyHandler(
        dest=dest,
        discard=discard,
        destdir=destdir,
        destsub=destsub,
        columns=columns,
        rows=rows,
        tmp=tmp,
        manual_mode=manual,
    )
    observer = Observer()
    observer.schedule(event_handler, src, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.secho("   I'm shutting down... see you! 👋", fg="white")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run()

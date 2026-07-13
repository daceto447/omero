import os
import sys
import time
import random
import numpy as np
import glob
import json
import re
import logging
import multiprocessing
import argparse

import pyvips as pv
from tqdm import tqdm
from PIL import Image
import cv2
from skimage import draw

import omero.sys
from omero.gateway import BlitzGateway, RoiWrapper
from omero_model_PolygonI import PolygonI
from omero_model_EllipseI import EllipseI
from omero_model_PolygonI import PolygonI
from omero_model_RectangleI import RectangleI


# ── Configuration ──────────────────────────────────────────────────────────────
OMERO_HOST = "wss://wsi.lavlab.mcw.edu/omero-wss"
OMERO_PORT = 443
OMERO_USER = "mjbarrett"
OMERO_PASS = """"

# Default values

PROJECT_ID = 51
BASE_PATH = "/Volumes/Siren/Prostate_data/"
NUM_WORKERS = 12

# Set True to regenerate ROI masks and replace JP2 files and annotations.
OVERWRITE = False

# Scale to downsample image for ROI mask generation.
DOWNSAMPLE = 10

# If True, overwrite existing _annot.jp2 file.
NEW_ANNOTS = True

# Text appended to the ROI mask annotation. Default: "_annot"
SUFFIX = "_annot"

# List of text annotations to include. If empty, or --all is added, all are included.
TEXT_FILTER = []

# Print completed image IDs and paths to stdout when True.
PRINT_COMPLETED = True

# Namespace tag applied to the file annotation so it can be found later
FILE_ANN_NS = "LargeRecon.10.roi"

NAME_RE = re.compile(r"N(\d+)_S(\d+)(_Deeper\d*)?_(\w+)\.ome\.tiff")


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.ERROR,
    format="%(processName)-20s %(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _connect() -> BlitzGateway:
    retries = 5
    base_delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            conn = BlitzGateway(
                OMERO_USER, OMERO_PASS,
                host=OMERO_HOST, port=OMERO_PORT, secure=True,
            )
            if conn.connect():
                return conn
            log.info("Failed to create session. Attempt %d/%d", attempt, retries)
            try:
                conn.close()
            except Exception:
                pass
        except Exception as exc:
            log.info("Failed to create session. Attempt %d/%d: %s", attempt, retries, exc)

        if attempt < retries:
            sleep_time = base_delay * (2 ** (attempt - 1)) + random.random()
            time.sleep(sleep_time)

    raise RuntimeError("Failed to connect to OMERO after %d attempts" % retries)


# ── Per-worker persistent connection ──────────────────────────────────────────
_worker_conn: "BlitzGateway | None" = None


def _init_worker() -> None:
    """Pool initializer: open one OMERO connection per worker process."""
    global _worker_conn
    # Disable pyvips operation/tile cache — without this, pyvips accumulates
    # decoded tiles across images and worker RAM grows without bound.
    pv.cache_set_max(0)
    pv.cache_set_max_mem(0)
    pv.cache_set_max_files(0)
    _worker_conn = _connect()


def _get_worker_conn() -> BlitzGateway:
    """Return the worker-local connection, reconnecting if the session dropped."""
    global _worker_conn
    if _worker_conn is None or not _worker_conn.isConnected():
        if _worker_conn is not None:
            try:
                _worker_conn.close()
            except Exception:
                pass
        _worker_conn = _connect()
    return _worker_conn


def uint_to_rgba(uint: int) -> int:
    """
    Return the color as an Integer in RGBA encoding.

    Parameters
    ----------
    int
        Integer encoding rgba value.

    Returns
    -------
    red: int
        Red color val (0-255)
    green: int
        Green color val (0-255)
    blue: int
        Blue color val (0-255)
    alpha: int
        Alpha opacity val (0-255)"""
    if uint < 0:  # convert from signed 32-bit int
        uint = uint + 2**32

    red = (uint >> 24) & 0xFF
    green = (uint >> 16) & 0xFF
    blue = (uint >> 8) & 0xFF
    alpha = uint & 0xFF

    return red, green, blue, alpha


def _resolve_paths(image_id: int, name: str, base_path: str) -> tuple[str, str] | None:
    """Return (jp2_path, omero_id_path) for an image name, or None if unparseable."""
    match = NAME_RE.match(name)
    if not match:
        log.warning("Image %d: name '%s' does not match expected pattern, skipping.", image_id, name)
        return None

    subject = match.group(1)
    slide_raw = match.group(2)
    deeper = match.group(3) or ""  # e.g. "_Deeper[2]" or ""
    stain = match.group(4)

    # Strip leading zeros but keep at least one digit
    slide = slide_raw.lstrip("0") or "0"

    subject_dirs = glob.glob(f"{BASE_PATH}*{subject}")
    if not subject_dirs:
        log.warning("Image %d: no directory found for subject %s, skipping.", image_id, subject)
        return None

    subject_dir = subject_dirs[0]

    # Slide folder incorporates optional _Deeper[N] suffix (e.g. "1" or "1_Deeper[2]")
    slide_folder = slide + deeper

    # Non-HE stains get their own subdirectory under the slide folder
    if stain == "HE":
        output_dir = os.path.join(subject_dir, "Hist", slide_folder, "Huron")
    else:
        output_dir = os.path.join(subject_dir, "Hist", slide_folder, "Huron", stain)

    # JP2 filename: LR10_ prepended to the stem of the original OMERO name
    stem = name.split(".")[0]
    jp2_path = os.path.join(output_dir, f"LR10_{stem}.jp2")
    omero_id_path = os.path.join(output_dir, f"{image_id}.omero.id")

    return jp2_path, omero_id_path


def _get_source_file_path(conn: BlitzGateway, image_id: int) -> str | None:
    """Return the absolute server-side path of the primary OME-TIFF for an image.

    Uses the fileset → usedFiles → originalFile relationship so the path is
    always what OMERO recorded on import — the same path visible from any pod
    that mounts the same storage.
    """
    qs = conn.getQueryService()
    params = omero.sys.ParametersI()
    params.addId(image_id)
    files = qs.findAllByQuery(
        "select f from Image i "
        "join i.fileset fs "
        "join fs.usedFiles fe "
        "join fe.originalFile f "
        "where i.id = :id",
        params,
        conn.SERVICE_OPTS,
    )
    if not files:
        return None

    # Prefer the largest file — that's the primary OME-TIFF in a multi-file set.
    files.sort(key=lambda f: f.size.val if f.size is not None else 0, reverse=True)
    f = files[0]
    return "/OMERO/ManagedRepository/" + f.path.val + f.name.val


# ── Per-image worker ───────────────────────────────────────────────────────────
def _is_conn_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient OMERO/Ice connection failure."""
    name = type(exc).__name__
    msg = str(exc)
    return (
        "Ice" in name
        or "omero" in name.lower()
        or "connect" in msg.lower()
        or "timeout" in msg.lower()
        or "session" in msg.lower()
    )


def get_rois(img, roi_service=None):
    """
    Gathers OMERO RoiI objects.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects()
    roi_service: omero.RoiService, optional
        Allows roiservice passthrough for performance
    """
    if roi_service is None:
        roi_service = img._conn.getRoiService()
        close_roi = True
    else:
        close_roi = False

    rois = roi_service.findByImage(img.getId(), None, img._conn.SERVICE_OPTS).rois

    if close_roi:
        roi_service.close()

    return rois


def get_shapes_as_points(
    img, point_downsample=4, img_downsample=1, roi_service=None, new_annots=False, text_filter=[]
) -> list[tuple[int, tuple[int, int, int], list[tuple[float, float]]]]:
    """
    Gathers Rectangles, Polygons, and Ellipses as a tuple containing the shapeId, its rgb val, and a tuple of yx points of its bounds.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    point_downsample: int, Default: 4
        Grabs every nth point for faster computation.
    img_downsample: int, Default: 1
        How much to scale roi points.
    roi_service: omero.RoiService, optional
        Allows roiservice passthrough for performance.

    Returns
    -------
    returns: list[ shape.id, (r,g,b), list[tuple(x,y)] ]
        list of tuples containing a shape's id, rgb value, and a tuple of row and column points
    """

    sizeX = img.getSizeX() / img_downsample
    sizeY = img.getSizeY() / img_downsample
    yx_shape = (sizeY, sizeX)

    shapes = []
    for roi in get_rois(img, roi_service, text_filter):

        points = None

        for shape in roi.copyShapes():
            if shape.getTextValue() is not None and text_filter != []:
                if (shape.getTextValue().getValue() not in text_filter):
                    continue

                # TODO
                # if (
                #     shape.getTextValue().getValue() == "Transferred Annotation"
                #     and new_annots is True
                # ):
                #     continue
                # elif (
                #     new_annots is False
                #     and shape.getTextValue().getValue() != "Transferred Annotation"
                # ):
                #     continue
                # elif (
                #     shape.getTextValue().getValue() == "Exclusion ROI"
                # ):
                #     continue
                # elif (
                #         shape.getTextValue().getValue() == "Vessel"
                #     ):
                #         continue
                # elif (
                #         shape.getTextValue().getValue() == "Urethra"
                #     ):
                #         continue
                # elif (
                #         shape.getTextValue().getValue() == "mask_use"
                #     ):
                #         continue
                # elif (
                #     "TMA" in shape.getTextValue().getValue()
                # ):
                #     continue
                
            elif new_annots is False:
                continue
            if type(shape) == RectangleI:
                x = float(shape.getX().getValue()) / img_downsample
                y = float(shape.getY().getValue()) / img_downsample
                w = float(shape.getWidth().getValue()) / img_downsample
                h = float(shape.getHeight().getValue()) / img_downsample
                # points = [(x, y),(x+w, y), (x+w, y+h), (x, y+h), (x, y)]
                points = draw.rectangle_perimeter(
                    (y, x), (y + h, x + w), shape=yx_shape
                )
                points = [
                    (points[1][i], points[0][i]) for i in range(0, len(points[0]))
                ]

            if type(shape) == EllipseI:
                points = draw.ellipse_perimeter(
                    float(shape._y._val / img_downsample),
                    float(shape._x._val / img_downsample),
                    float(shape._radiusY._val / img_downsample),
                    float(shape._radiusX._val / img_downsample),
                    shape=yx_shape,
                )
                points = [
                    (points[1][i], points[0][i]) for i in range(0, len(points[0]))
                ]

            if type(shape) == PolygonI:
                pointStrArr = shape.getPoints()._val.split(" ")

                xy = []
                for i in range(0, len(pointStrArr)):
                    coordList = pointStrArr[i].split(",")
                    xy.append(
                        (
                            float(coordList[0]) / img_downsample,
                            float(coordList[1]) / img_downsample,
                        )
                    )
                if xy:
                    points = xy

            if points is not None:
                color_val = shape.getStrokeColor()._val
                rgb = uint_to_rgba(color_val)[:-1]  # ignore alpha value for computation
                points = points[::point_downsample]

                shapes.append((shape.getId()._val, rgb, points))

    if not shapes:  # if no shapes in shapes return none
        return None

    # make sure is in correct order
    return sorted(shapes)


def get_roi_mask(image, downsample: int, new_annots: bool, text_filter: list) -> np.array:
    rgb_mask = np.zeros(
        (
            int(image.getSizeY() / downsample),
            int(image.getSizeX() / downsample),
            image.getSizeC(),
        ),
        dtype=np.uint8,
    )
    rgb_mask[:] = 255
    for id, rgb, xy in get_shapes_as_points(image, img_downsample=downsample, new_annots=new_annots, text_filter=text_filter):
        yx = np.array(xy, np.int32)  # Ensure the points are of integer type
        yx = yx.reshape((-1, 1, 2))  # Reshape to (-1, 1, 2)
        cv2.fillPoly(rgb_mask, [yx], color=rgb)
    return rgb_mask


def create_roi_image(kwargs) -> tuple[int, str] | None:
    image_id = kwargs.get("image_id")
    suffix = kwargs.get("suffix", SUFFIX)
    text_filter = kwargs.get("text_filter", TEXT_FILTER) #TODO
    base_path = kwargs.get("base_path", BASE_PATH)
    overwrite = kwargs.get("overwrite", OVERWRITE)
    downsample = kwargs.get("downsample", DOWNSAMPLE)
    print_completed = kwargs.get("print_completed", PRINT_COMPLETED)
    new_annots = kwargs.get("new_annots", NEW_ANNOTS)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            conn = _get_worker_conn()
            image = conn.getObject("Image", image_id)
            conn.c.sf.setSecurityContext(image.details.group)
            if image is None:
                log.warning("Image %d not found in OMERO, skipping.", image_id)
                return None
        
            name = image.getName()
            paths = _resolve_paths(image_id, name, base_path)
            if paths is None:
                return None
        
            jp2_path, omero_id_path = paths
            roi_path = jp2_path.replace(".jp2", f"{suffix}.jp2")
            print(roi_path)

            already_local = os.path.exists(roi_path) and os.path.exists(omero_id_path)
            if already_local and not overwrite:
                # Sidecar is written after upload, so its presence guarantees completion.
                log.info("ROI %d already created and uploaded, skipping.", image_id)
                if print_completed:
                    print(f"Completed ROI for image {image_id}: {roi_path}")
                return (image_id, roi_path)
            else:
                src_path = _get_source_file_path(conn, image_id)
                if src_path is None:
                    log.error("Image %d: no source file found in fileset, skipping.", image_id)
                    return None
                if not os.path.exists(src_path):
                    log.error(
                        "Image %d: source file '%s' is not accessible from this pod, skipping.",
                        image_id, src_path,
                    )
                    return None
                
                os.makedirs(os.path.dirname(roi_path), exist_ok=True)

                # Create ROI mask and save to roi_path.
                roi_mask = get_roi_mask(image, downsample, new_annots)
                roi_img = pv.Image.new_from_array(roi_mask)
                del roi_mask
                roi_img.write_to_file(roi_path)
                log.info("Image %d: saved ROI → %s", image_id, roi_path)

            # Remove any existing annotation in this namespace before uploading.
            # In OVERWRITE mode this includes existing JP2s (the whole point);
            # otherwise only remove stale non-JP2 annotations.
            for ann in image.listAnnotations(ns=FILE_ANN_NS):
                if hasattr(ann, "getFile"):
                    orig_file = ann.getFile()
                    if orig_file is not None and (overwrite or not orig_file.getName().endswith(".jp2")):
                        log.info(
                            "Image %d: removing old annotation '%s' (id=%d).",
                            image_id, orig_file.getName(), ann.getId(),
                        )
                        image.removeAnnotations([ann])
                        conn.deleteObject(ann._obj)

            file_ann = conn.createFileAnnfromLocalFile(
                roi_path,
                mimetype="image/jp2",
                ns=FILE_ANN_NS,
            )
            image.linkAnnotation(file_ann)
            log.info("ROI %d: uploaded ROI annotation to OMERO.", image_id)

            # Sidecar written after successful upload so its presence = upload done.
            with open(omero_id_path, "w") as f:
                f.write("")
            log.info("ROI %d: wrote sidecar → %s", image_id, omero_id_path)

            if print_completed:
                print(f"Completed ROI for image {image_id}: {roi_path}")

            return (image_id, roi_path)

        except Exception as exc:
            if _is_conn_error(exc) and attempt < max_attempts:
                global _worker_conn
                _worker_conn = None
                log.warning(
                    "Image %d: connection error on attempt %d/%d, reconnecting: %s",
                    image_id, attempt, max_attempts, exc,
                )
                time.sleep(1.0 * attempt)
                continue
            log.exception("Image %d: unhandled error.", image_id)
            return None
    return None


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ROI mask generation and upload to OMERO.")
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Get all annotations regardless of text value."
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str, 
        default="_annot", 
        help=f"Change the text appended to the ROI mask annotation (default: '{SUFFIX}')."
    )
    parser.add_argument(
        "-f", "--text_filter",
        type=str,
        nargs="*",
        default=[],
        help="List of text values to include, e.g. --text 'Transferred Annotation' 'mask_use'."
    )
    parser.add_argument(
        "-i", "--project-id",
        type=int,
        default=PROJECT_ID,
        help="OMERO project ID to process."
    )
    parser.add_argument(
        "-b", "--base-path",
        type=str,
        default=BASE_PATH,
        help="Base path for output directories."
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="Overwrite existing ROI masks and annotations."
    )
    parser.add_argument(
        "-d", "--downsample",
        type=int, 
        default=DOWNSAMPLE, 
        help="Downsample factor for ROI mask generation."
    )
    parser.add_argument(
        "-p", "--print-completed",
        action="store_true",
        default=PRINT_COMPLETED,
        help="Print completed image IDs and paths to stdout."
    )
    parser.add_argument(
        "-n", "--new-annots",
        action="store_true",
        default=NEW_ANNOTS,
        help="Use new annotations (default: False)."
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    conn = _connect()
    conn.SERVICE_OPTS.setOmeroGroup(-1)
    try:
        project = conn.getObject("Project", args.project_id)
        if project is None:
            raise RuntimeError(f"Project {args.project_id} not found.")

        image_ids = [
            image.getId()
            for dataset in project.listChildren()
            for image in dataset.listChildren()
        ]
    finally:
        conn.close()
            
    log.info("Found %d images in project %d. Starting pool of %d workers.",
             len(image_ids), args.project_id, args.num_workers)
    
    image_ids = image_ids[:2]
    kwargs_list = [
        {
            "image_id": image_id,
            "suffix": args.suffix,
            "text_filter": args.text_filter,
            "base_path": args.base_path,
            "overwrite": args.overwrite,
            "downsample": args.downsample,
            "print_completed": args.print_completed,
            "new_annots": args.new_annots,
        }
        for image_id in image_ids
    ]
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(args.num_workers, initializer=_init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(create_roi_image, kwargs_list),
            total=len(image_ids),
            desc="Images",
            unit="img",
        ))

    log.info("Batch complete.")


if __name__ == "__main__":
    main()
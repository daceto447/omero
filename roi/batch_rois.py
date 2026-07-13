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
OMERO_PASS = "gzyxby01"

PROJECT_ID = 51
BASE_PATH = "/Volumes/Siren/Prostate_data/"
NUM_WORKERS = 1

# Set True to regenerate ROI masks and replace JP2 files and annotations.
OVERWRITE = False

# Scale to downsample image for ROI mask generation.
DOWNSAMPLE = 10

# If True, overwrite existing _annot.jp2 file.
NEW_ANNOTS = True

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
    #TODO don't need pyvips?
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


def _resolve_paths(image_id: int, name: str) -> tuple[str, str] | None:
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


def getRois(img, roi_service=None):
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


def getShapesAsPoints(
    img, point_downsample=4, img_downsample=1, roi_service=None
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
    for roi in getRois(img, roi_service):

        points = None

        for shape in roi.copyShapes():
            if shape.getTextValue() is not None:
                if (
                    shape.getTextValue().getValue() == "Transferred Annotation"
                    and NEW_ANNOTS is True
                ):
                    continue
                elif (
                    NEW_ANNOTS is False
                    and shape.getTextValue().getValue() != "Transferred Annotation"
                ):
                    continue
                elif (
                    shape.getTextValue().getValue() == "Exclusion ROI"
                ):
                    continue
                elif (
                        shape.getTextValue().getValue() == "Vessel"
                    ):
                        continue
                elif (
                        shape.getTextValue().getValue() == "Urethra"
                    ):
                        continue
                elif (
                        shape.getTextValue().getValue() == "mask_use"
                    ):
                        continue
                elif (
                    "TMA" in shape.getTextValue().getValue()
                ):
                    continue
                
            elif NEW_ANNOTS is False:
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


def get_roi_mask(image):
    rgb_mask = np.zeros(
        (
            int(image.getSizeY() / DOWNSAMPLE),
            int(image.getSizeX() / DOWNSAMPLE),
            image.getSizeC(),
        ),
        dtype=np.uint8,
    )
    rgb_mask[:] = 255
    for id, rgb, xy in getShapesAsPoints(image, img_downsample=DOWNSAMPLE):
        yx = np.array(xy, np.int32)  # Ensure the points are of integer type
        yx = yx.reshape((-1, 1, 2))  # Reshape to (-1, 1, 2)
        cv2.fillPoly(rgb_mask, [yx], color=rgb)
    mask = Image.fromarray(rgb_mask)
    return mask


def create_roi_image(image_id: int) -> tuple[int, str] | None:
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
            paths = _resolve_paths(image_id, name)
            if paths is None:
                return None
        
            jp2_path, omero_id_path = paths
            #ext = "_annot.jp2" if NEW_ANNOTS else "_annot_old.jp2"
            roi_path = jp2_path.replace(".jp2", "_annot.jp2")
            print(roi_path)

            already_local = os.path.exists(roi_path) and os.path.exists(omero_id_path)
            if already_local and not OVERWRITE:
                # Sidecar is written after upload, so its presence guarantees completion.
                log.info("ROI %d already created and uploaded, skipping.", image_id)
                if PRINT_COMPLETED:
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
                mask = get_roi_mask(image)
                mask.save(roi_path)
                log.info("Image %d: saved ROI → %s", image_id, roi_path)

            # Remove any existing annotation in this namespace before uploading.
            # In OVERWRITE mode this includes existing JP2s (the whole point);
            # otherwise only remove stale non-JP2 annotations.
            for ann in image.listAnnotations(ns=FILE_ANN_NS):
                if hasattr(ann, "getFile"):
                    orig_file = ann.getFile()
                    if orig_file is not None and (OVERWRITE or not orig_file.getName().endswith(".jp2")):
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

            if PRINT_COMPLETED:
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
def main() -> None:
    conn = _connect()
    conn.SERVICE_OPTS.setOmeroGroup(-1)
    try:
        project = conn.getObject("Project", PROJECT_ID)
        if project is None:
            raise RuntimeError(f"Project {PROJECT_ID} not found.")

        image_ids = [
            image.getId()
            for dataset in project.listChildren()
            for image in dataset.listChildren()
        ]
    finally:
        conn.close()
            
    log.info("Found %d images in project %d. Starting pool of %d workers.",
             len(image_ids), PROJECT_ID, NUM_WORKERS)
    
    image_ids = image_ids[:1]
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(create_roi_image, image_ids),
            total=len(image_ids),
            desc="Images",
            unit="img",
        ))

    # Merge new mappings into any existing JSON file.
    mapping_path = os.path.join(BASE_PATH, "omero_id_mappings.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mappings: dict[str, str] = json.load(f)
    else:
        mappings = {}

    for result in results:
        if result is not None:
            omero_id, roi_path = result
            if omero_id in mappings:
                #TODO list for key?
                mappings[str(omero_id)] = list(mappings[str(omero_id)] + [roi_path])
            else:
                mappings[str(omero_id)] = [roi_path]

    # Upload mapping file to OMERO
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2)
    log.info("Wrote %d mappings to %s", len(mappings), mapping_path)

    log.info("Batch complete.")


if __name__ == "__main__":
    main()
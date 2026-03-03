import os
import time
import random
import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import concurrent.futures
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# ========================================================================
# Configuration & Parameters
# ========================================================================

CHUNK_WIDTH = 320      # Target chip width (pixels)
CHUNK_HEIGHT = 320     # Target chip height (pixels)
MIN_CHUNK_HEIGHT = 320 # Minimum valid chip height (pixels)
MIN_CHUNK_WIDTH = 320  # Minimum valid chip width (pixels)
IMAGE_WRITING = True   # Whether to write chunk images to disk
JPEG_COMPRESSION = 95  # JPEG export quality (0-100)
RANDOM_SEED = 42       # Random seed for reproducibility
DEBUG = False          # Debug mode: restrict number of images

INPUT_DATASET_PATH = Path("../data/raw/xview-dataset")
OUTPUT_DATASET_PATH = Path("../data/processed")
LABELS_JSON_PATH = INPUT_DATASET_PATH / "train_labels" / "xView_train.geojson"
IMAGE_FOLDER_PATH = INPUT_DATASET_PATH / "train_images"
SAVE_IMAGES_FOLDER_PATH = OUTPUT_DATASET_PATH / "images"
OUTPUT_DATA_PARQUET_PATH = OUTPUT_DATASET_PATH / "xview_labels.parquet"
OUTPUT_CLASS_MAP_PATH = OUTPUT_DATASET_PATH / "xView_class_map.json"
COCO_JSON_PATH = OUTPUT_DATASET_PATH / "coco_annotations.json"

random.seed(RANDOM_SEED)

# ========================================================================
# Utility Functions
# ========================================================================

def make_empty_dir(directory):
    """
    Delete the directory (if it exists) and recreate it empty.

    Args:
        directory (Path): Directory to reset.
    """
    if directory.is_dir():
        shutil.rmtree(directory)
    os.makedirs(directory)

def check_path(p, name, expect_file=False):
    """
    Print basic existence and metadata for a file/directory.

    Args:
        p (Path): Path to check.
        name (str): Label for printing.
        expect_file (bool): If True, expects a file.
    """
    print(f"\n{name}: {p.resolve()} exists={p.exists()}")

def print_first_n_lines(file_path, n):
    """
    Print first n lines of a text file.

    Args:
        file_path (str or Path): Path of file to read.
        n (int): Number of lines to print.
    """
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if line_num > n:
                    break
                print(line.strip())
    except FileNotFoundError:
        print('Unable to open file')

def load_image(file_pth):
    """
    Load an image (OpenCV) and convert to RGB for visualization.

    Args:
        file_pth (str): Path to image.
    Returns:
        np.ndarray: Image as array (RGB).
    """
    image_obj = cv2.imread(file_pth)
    if image_obj is None:
        print(f'Unable to load image at {file_pth}')
        return None
    image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
    return image_obj

def load_bgr_image(file_pth):
    """
    Load an image for processing (OpenCV, BGR).

    Args:
        file_pth (str): Path to image.
    Returns:
        np.ndarray: Image as array (BGR).
    """
    image_obj = cv2.imread(file_pth)
    return image_obj

# ========================================================================
# Bounding Box Utilities
# ========================================================================

def get_boxes(in_df, class_list=[]):
    """
    Extract all bounding boxes for each image from DataFrame.

    Args:
        in_df (pd.DataFrame): Annotations DataFrame.
        class_list (list, optional): Filter for specific class IDs.
    Returns:
        dict: {image_name: [list of boxes]}
    """
    if class_list:
        in_df = in_df[in_df["TYPE_ID"].isin(class_list)]
    unique_images = in_df.IMAGE_ID.unique().tolist()
    boxes = {}
    for image in tqdm(unique_images):
        mask = in_df["IMAGE_ID"] == image
        masked = in_df[mask][["TYPE_ID", "X_MIN", "Y_MIN", "X_MAX", "Y_MAX"]]
        boxes[image] = masked.values.tolist()
    return boxes

def get_corners(x_center, y_center, annotation_width, annotation_height, image_width, image_height):
    """
    Convert YOLO box (normalized) to pixel corner coordinates.

    Args:
        x_center, y_center, annotation_width, annotation_height: Normalized box params (YOLO format)
        image_width, image_height: Pixel dimensions of image.
    Returns:
        tuple: (left, top, right, bottom) pixel coordinates
    """
    x_center, y_center, annotation_width, annotation_height = float(x_center), float(y_center), float(annotation_width), float(annotation_height)
    left = (x_center - annotation_width/2)*image_width
    top = (y_center - annotation_height/2)*image_height
    right = (x_center + annotation_width/2)*image_width
    bottom = (y_center + annotation_height/2)*image_height
    return int(left), int(top), int(right), int(bottom)

def match_boxes(box_list, chunk_limits):
    """
    Find bounding boxes inside an image chunk and convert them to YOLO format.

    Args:
        box_list (list): List of original bounding boxes for the source image.
        chunk_limits (list): [chunk_left, chunk_top, chunk_width, chunk_height]:
            - chunk_left: Horizontal position (pixels) of chunk on original image (left).
            - chunk_top: Vertical position (pixels) of chunk on original image (top).
            - chunk_width: Width of chunk (pixels).
            - chunk_height: Height of chunk (pixels).
    Returns:
        list: YOLO boxes, each formatted [class_id, x_center, y_center, width, height] (normalized).
    """
    boxes_lists = []
    chunk_left, chunk_top = chunk_limits[0], chunk_limits[1]
    width, height = chunk_limits[2], chunk_limits[3]
    for box in box_list:
        original_left, original_top, original_right, original_bottom = box[1], box[2], box[3], box[4]
        left, right = (original_left - chunk_left)/width, (original_right - chunk_left)/width
        top, bottom = (original_top - chunk_top)/height, (original_bottom - chunk_top)/height
        horizontal_match = (0 <= left < 1) or (0 < right <= 1)
        vertical_match = (0 <= top < 1) or (0 < bottom <= 1)
        if vertical_match and horizontal_match:
            clipped = np.clip([left, top, right, bottom], a_min=0, a_max=1)
            left, top, right, bottom = clipped[0], clipped[1], clipped[2], clipped[3]
            bounding_box = [
                str(box[0]),
                str(round((left + right)/2, 5)),
                str(round((top + bottom)/2, 5)),
                str(round(right - left, 5)),
                str(round(bottom - top, 5))
            ]
            boxes_lists.append(bounding_box)
    return boxes_lists

# =======================================================================
# Image Processing
# =======================================================================

def process_image(image_filename, dir_path, boxes, out_dir, chunk_height, chunk_width, jpg_quality, min_height, min_width, writing):
    """
    Split an image into chunks, save each chunk as JPG, and convert bounding boxes to YOLO format within each chunk.

    Args:
        image_filename (str): Name of the image file to process.
        dir_path (Path): Path to input images directory.
        boxes (dict): Mapping from image filename to list of bounding boxes.
        out_dir (Path): Directory to save chunk images.
        chunk_height (int): Chunk height (pixels).
        chunk_width (int): Chunk width (pixels).
        jpg_quality (int): JPEG quality for saving.
        min_height (int): Minimum chunk height (pixels).
        min_width (int): Minimum chunk width (pixels).
        writing (bool): Whether to save chunk images to disk.

    Returns:
        tuple: (list of chunk filenames, list of widths, list of heights, dict of YOLO labels per chunk)
    """
    labels_list = boxes[image_filename]
    image_path = str(dir_path / image_filename)
    image = load_bgr_image(image_path)
    full_height, full_width, _ = image.shape
    y_boxes = {}
    file_names, widths, heights = [], [], []
    # Iterate over the image grid in steps of chunk_height/chunk_width
    for row in range(0, full_height, chunk_height):
        for col in range(0, full_width, chunk_width):
            stem = image_filename.split('.')[0]
            filenames = str(f"img_{stem}_{row}_{col}.jpg")
            out_pth = str(out_dir / filenames)
            width = chunk_width
            height = chunk_height
            if row + height > full_height:
                height = full_height - row
            if col + width > full_width:
                width = full_width - col
            big_enough = (height >= min_height) and (width >= min_width)
            if big_enough:
                # Save chunk image if required
                if writing:
                    cv2.imwrite(out_pth, image[row:row+height, col:col+width,:], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                chunk_limits = [col, row, width, height]
                # Compute YOLO-format boxes for this chunk only          
                y_boxes[filenames] = match_boxes(labels_list, chunk_limits)
                file_names.append(filenames)
                widths.append(width)
                heights.append(height)
    return file_names, widths, heights, y_boxes

def remove_empty(image_folder, yolo_boxes, image_data, file_names, widths, heights):
    """
    Remove 66% of empty annotation files and their corresponding images, filtering labels and metadata accordingly.

    Args:
        image_folder (str): Path to the folder of chunk images and .txt labels.
        yolo_boxes (dict): YOLO annotations per chunk.
        image_data (dict): Metadata for chunk images.
        file_names (list): List of chunk image filenames.
        widths (list): List of image widths.
        heights (list): List of image heights.

    Returns:
        tuple: (filtered yolo_boxes, image_data, file_names, widths, heights)
    """
    all_image_files = set(os.listdir(image_folder))
    empty_files = []
    # Identify empty .txt files (labels)
    for txt_file in all_image_files:
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(image_folder, txt_file)
            with open(txt_path, 'r') as file:
                content = file.read().strip()
            if not content:
                image_file = txt_file.replace('.txt', '.jpg')
                empty_files.append(image_file)
    num_to_remove = int(len(empty_files) * 0.66)
    files_to_remove = random.sample(empty_files, num_to_remove)
    # Remove files from disk and from label/metadata dicts
    for image_file in files_to_remove:
        txt_file = image_file.replace('.jpg', '.txt')
        txt_path = os.path.join(image_folder, txt_file)
        image_path = os.path.join(image_folder, image_file)
        if os.path.exists(txt_path): os.remove(txt_path)
        if os.path.exists(image_path): os.remove(image_path)
    yolo_boxes = {key: value for key, value in yolo_boxes.items() if key not in files_to_remove}
    image_data = {key: value for key, value in image_data.items() if key not in files_to_remove}
    filtered_file_names = [name for name in file_names if name not in files_to_remove]
    filtered_widths = [widths[i] for i in range(len(file_names)) if file_names[i] not in files_to_remove]
    filtered_heights = [heights[i] for i in range(len(file_names)) if file_names[i] not in files_to_remove]
    return yolo_boxes, image_data, filtered_file_names, filtered_widths, filtered_heights

# =======================================================================
# Main pipeline
# =======================================================================

def preprocessing_main():
    """
    Complete preprocessing pipeline for xView:
    - Load and clean annotation data.
    - Class remapping.
    - Split images into chips, save chips & YOLO labels.
    - Remove empty chips/images.
    - Save labels as Parquet.
    - Create and save COCO-format JSON.
    """
    # Ensure output folder is fresh for this run
    make_empty_dir(SAVE_IMAGES_FOLDER_PATH)
    random.seed(RANDOM_SEED)

    # Load & clean annotations
    with open(LABELS_JSON_PATH, 'r') as infile:
        data = json.load(infile)
    feature_list = data['features']
    COLUMNS = ['IMAGE_ID', 'TYPE_ID', 'X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX', 'LONG', 'LAT']
    rows = []
    for feature in tqdm(feature_list):
        props = feature['properties']
        image_id = props['image_id']
        type_id = props['type_id']
        bbox = props['bounds_imcoords'].split(",")
        geom = feature['geometry']
        coordinates = geom['coordinates'][0]
        # Compute approximate geographic center
        longitude = coordinates[0][0] / 2 + coordinates[2][0] / 2
        latitude = coordinates[0][1] / 2 + coordinates[1][1] / 2
        rows.append([image_id, type_id, bbox[0], bbox[1], bbox[2], bbox[3], longitude, latitude])
    df = pd.DataFrame(rows, columns = COLUMNS)
    df[['X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']] = df[['X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']].apply(pd.to_numeric)
    # Remove erroneous labels and missing images
    # Erroneous object classes (75, 82) and images with missing files (e.g. '1395.tif') are filtered from the dataframe
    df = df[(df.TYPE_ID != 75) & (df.TYPE_ID != 82)]
    df = df[df.IMAGE_ID != '1395.tif']

    # Class remapping
    old_dict = {
        11:'Fixed-wing Aircraft', 12:'Small Aircraft', 13:'Passenger/Cargo Plane', 15:'Helicopter',
        17:'Passenger Vehicle', 18:'Small Car', 19:'Bus', 20:'Pickup Truck', 21:'Utility Truck',
        23:'Truck', 24:'Cargo Truck', 25:'Truck Tractor w/ Box Trailer', 26:'Truck Tractor',27:'Trailer',
        28:'Truck Tractor w/ Flatbed Trailer', 29:'Truck Tractor w/ Liquid Tank', 32:'Crane Truck',
        33:'Railway Vehicle', 34:'Passenger Car', 35:'Cargo/Container Car', 36:'Flat Car', 37:'Tank car',
        38:'Locomotive', 40:'Maritime Vessel', 41:'Motorboat', 42:'Sailboat', 44:'Tugboat', 45:'Barge',
        47:'Fishing Vessel', 49:'Ferry', 50:'Yacht', 51:'Container Ship', 52:'Oil Tanker',
        53:'Engineering Vehicle', 54:'Tower crane', 55:'Container Crane', 56:'Reach Stacker',
        57:'Straddle Carrier', 59:'Mobile Crane', 60:'Dump Truck', 61:'Haul Truck', 62:'Scraper/Tractor',
        63:'Front loader/Bulldozer', 64:'Excavator', 65:'Cement Mixer', 66:'Ground Grader', 71:'Hut/Tent',
        72:'Shed', 73:'Building', 74:'Aircraft Hangar', 76:'Damaged Building', 77:'Facility', 79:'Construction Site',
        83:'Vehicle Lot', 84:'Helipad', 86:'Storage Tank', 89:'Shipping container lot', 91:'Shipping Container',
        93:'Pylon', 94:'Tower'
    }
    old_keys = sorted(list(old_dict.keys()))
    new_dict = {old_dict[x]: y for y, x in enumerate(old_keys)}
    class_map_dict = {y: old_dict[x] for y, x in enumerate(old_keys)}
    with open(OUTPUT_CLASS_MAP_PATH, "w") as json_file:
        json.dump(class_map_dict, json_file)
    df['TYPE_ID'] = df['TYPE_ID'].apply(lambda x: new_dict[old_dict[x]])

    # Chips & YOLO label generation
    boxes_dict = get_boxes(df)
    image_filenames = df.IMAGE_ID.unique().tolist()
    if DEBUG:
        image_filenames = image_filenames[:len(image_filenames)//120]
        df = df[df['IMAGE_ID'].isin(image_filenames)]

    # Parallel generation of image chips and YOLO bounding boxes
    start_time = time.time()
    num_threads = mp.cpu_count()
    overall_progress = tqdm(total=len(image_filenames), desc="Creating and saving image tiles")
    yolo_boxes = {}
    file_names, widths, heights = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for f_names, c_widths, c_heights, y_boxes in executor.map(
                lambda fname: process_image(
                    fname, IMAGE_FOLDER_PATH, boxes_dict, SAVE_IMAGES_FOLDER_PATH,
                    CHUNK_HEIGHT, CHUNK_WIDTH, JPEG_COMPRESSION,
                    MIN_CHUNK_HEIGHT, MIN_CHUNK_WIDTH, IMAGE_WRITING
                ),
                image_filenames):
            file_names.extend(f_names)
            widths.extend(c_widths)
            heights.extend(c_heights)
            yolo_boxes.update(y_boxes)
            overall_progress.update(1)
    overall_progress.close()
    image_data = {file_names[i]: [widths[i], heights[i]] for i in range(len(file_names))}
    print(f"Chips generation completed in {time.time() - start_time:.2f}s.")

    # Write YOLO bounding box label files (.txt) for each chip
    all_image_files = os.listdir(SAVE_IMAGES_FOLDER_PATH)
    for image_filename in tqdm(all_image_files, desc="Write YOLO label files"):
        stem = image_filename.split('.')[0]
        file_name = str(stem) + '.txt'
        txt_path = str(SAVE_IMAGES_FOLDER_PATH / file_name)
        separator = ' '
        with open(txt_path, 'a') as f:
            if image_filename in yolo_boxes:
                for bbox in yolo_boxes[image_filename]:
                    txt = separator.join(bbox) + '\n'
                    f.write(txt)

    # Remove empty chips/images and associated .txt files
    yolo_boxes, image_data, filtered_file_names, filtered_widths, filtered_heights = remove_empty(
        SAVE_IMAGES_FOLDER_PATH, yolo_boxes, image_data, file_names, widths, heights
    )

    # Save YOLO chip labels as a parquet dataframe for inspection and statistical analysis
    text_paths = [SAVE_IMAGES_FOLDER_PATH / x for x in os.listdir(SAVE_IMAGES_FOLDER_PATH) if x.endswith(".txt")]
    column_names = ['Class_ID', 'x_center', 'y_center', 'width', 'height']
    data_out = []
    for file_path in text_paths:
        with open(file_path, 'r') as file:
            for line in file:
                values = line.strip().split(' ')
                row_data = {col: val for col, val in zip(column_names, values)}
                row_data['File_Name'] = file_path.name
                data_out.append(row_data)
    out_df = pd.DataFrame(data_out)
    out_df['Class_ID'] = out_df['Class_ID'].astype(int)
    out_df['Class_Name'] = out_df['Class_ID'].map(class_map_dict).fillna('unknown')
    out_df = out_df[['File_Name', 'Class_Name', 'Class_ID', 'x_center', 'y_center', 'width', 'height']]
    out_df.to_parquet(OUTPUT_DATA_PARQUET_PATH, index=False)

    # Convert YOLO chip labels to COCO format and save as JSON
    image_data_final = {'width': filtered_widths, 'height': filtered_heights, 'file_name': filtered_file_names}
    im_df = pd.DataFrame(image_data_final)
    im_df['id'] = im_df['file_name'].str.replace(r'\D', '', regex=True).astype(int)
    annotations_df = out_df.copy()
    annotations_df['image_id'] = annotations_df['File_Name'].str.replace(r'\D', '', regex=True).astype(int)
    annotations_df = annotations_df.rename(columns={'height': 'h', 'width': 'w'})
    an_df = annotations_df.merge(im_df, left_on='image_id', right_on='id', how='left')
    an_df['x_center'] = (an_df['x_center'].astype(np.float64)*an_df['width']).round(decimals=0)
    an_df['y_center'] = (an_df['y_center'].astype(np.float64)*an_df['height']).round(decimals=0)
    an_df['w'] = (an_df['w'].astype(np.float64)*an_df['width']).round(decimals=0)
    an_df['h'] = (an_df['h'].astype(np.float64)*an_df['height']).round(decimals=0)
    an_df['Class_ID'] = an_df['Class_ID'].astype(int)
    an_df = an_df.drop(columns=['File_Name', 'file_name', 'width', 'height', 'id'])
    an_df['left'] = (an_df['x_center'] - an_df['w']/2).round(decimals=0)
    an_df['top'] = (an_df['y_center'] - an_df['h']/2).round(decimals=0)
    an_df['bbox'] = ('[' + an_df['left'].astype(str) + ', '
                    + an_df['top'].astype(str) + ', '
                    + an_df['w'].astype(str) + ', '
                    + an_df['h'].astype(str) + ']')
    an_df['area'] = an_df['w'] * an_df['h']
    an_df = an_df.drop(columns=['x_center', 'y_center', 'w', 'h', 'left', 'top', 'Class_Name'])
    an_df.reset_index(inplace=True)
    an_df.rename(columns={'index': 'id'}, inplace=True)

    # Serialize final output to COCO JSON
    # COCO format consists of 'images', 'annotations', and 'categories' lists
    def row_to_dict(row):
        return {
            'id': row['id'],
            'image_id': row['image_id'],
            'category_id': row['Class_ID'],
            'area': row['area'],
            'bbox': row['bbox']
        }
    images_list = im_df.apply(lambda row: {
        'id': row['id'],
        'width': row['width'],
        'height': row['height'],
        'file_name': row['file_name']
    }, axis=1).tolist()
    annotations_list = an_df.apply(row_to_dict, axis=1).tolist()
    categories_list = [{key: val} for key, val in class_map_dict.items()]
    out_json_data = {'images': images_list, 'annotations': annotations_list, 'categories': categories_list}
    with open(COCO_JSON_PATH, 'w') as json_file:
        json.dump(out_json_data, json_file, indent=4)

    print(f"Preprocessing sequence complete. Output:")
    print(f"  YOLO chips & labels: {SAVE_IMAGES_FOLDER_PATH}")
    print(f"  COCO JSON: {COCO_JSON_PATH}")
    print(f"  Class map: {OUTPUT_CLASS_MAP_PATH}")
    print(f"  Parquet labels: {OUTPUT_DATA_PARQUET_PATH}")

if __name__ == "__main__":
    preprocessing_main()
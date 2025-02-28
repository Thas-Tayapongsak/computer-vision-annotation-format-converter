from pathlib import Path
from shutil import copy
import json, yaml, cv2

#this needs reorganizing
#maybe move process functions back to converter

def validate_options(opt, verbose=True):
    """
    Validate selected options

    Parameters
    ----------
    opt : dict
        A dictionary containing the selected options
    verbose : bool, optional
        Print progress messages, True by default
    
    Returns
    -------
    Path
        Source path for the dataset
    Path
        Destination path for the dataset
    str
        Name of the source dataset
    str
        Task to perform

    """
    # Define source and destination paths, dataset name, and task
    src_path = Path(opt.get('src_path', ''))
    dst_path = Path(opt.get('dst_path', ''))
    src_dataset = opt.get('src_dataset', '')
    src_format = opt.get('src_format', '')
    dst_format = opt.get('dst_format', '')
    task = opt.get('task', '')

    # Print initial information if verbose is True
    if verbose:
        print(f"Starting conversion from {src_format} to {dst_format} format.")
        print(f"Source path: {src_path}")
        print(f"Destination path: {dst_path}")
        print(f"Dataset: {src_dataset}")
        print(f"Source format: {src_format}")
        print(f"Task: {task}")

    # Check if source and destination paths exist
    if not src_path.exists():
        raise FileNotFoundError(f"Source path {src_path} does not exist.")
    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created destination path: {dst_path}")
    
    # Check task and format compatibility
    if task not in ['detect', 'segment']:
        raise ValueError(f"Invalid task {task}. Task must be 'detect' or 'segment'.")
    elif src_format == 'bin' and task != 'segment':
        raise ValueError(f"Invalid task {task} for source format 'bin'. Task must be 'segment'.")
     
    return opt

def copy_images(src_path, dst_path, images, verbose=True):
    for image in images:
        try:
            src_image_path = src_path / image['file_name']
            copy(src_image_path, dst_path)
            if verbose:
                print(f"\r...Copying image file #{image['id']}: {image['file_name']}    ", end='')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file {image['file_name']} not found in {src_path}.") from e
        except Exception as e:
            print(f"Error copying image file {image['file_name']} to {dst_path}: {e}")
            return
    if verbose:
            print()

def write_yolo_yaml(dst_path, src_dataset, src_split, categories):
    """
    Write dataset information to YAML file

    Parameters
    ----------
    dst_path : Path
        Destination path to write the YAML file
    src_dataset : str
        Name of the source dataset
    src_split : bool
        True if the dataset is split into train, validation, and test sets
    categories : list
        List of categories in the dataset

    """
    # Write categories and split paths to YAML file
    try:
        with open(dst_path / f'{src_dataset}.yaml', 'w') as f:
            if src_split:
                yaml.dump({
                        'train': '../train/images',
                        'val': '../val/images',
                        'test': '../test/images',
                        'names': {cat['id'] - 1: cat['name'] for cat in categories}
                    }, f)
            else:
                yaml.dump({
                        'names': {cat['id'] - 1: cat['name'] for cat in categories}
                    }, f)
    except Exception as e:
        print(f"Error creating YAML file for dataset {src_dataset}: {e}")

def initialize_yolo_labels(dst_path, images, verbose=True):
    """
    Create empty YOLO label files for each image

    Parameters
    ----------
    dst_path : Path
        Destination path for YOLO label files
    src_split : bool
        True if the dataset is split into train, validation, and test sets
    images : list
        List of images in the dataset
    verbose : bool, optional
        Print progress messages, True by default

    """
    # Create empty YOLO txt files for each image
    yolo_txt_path = ''
    for image in images:
        try:
            image_name = image['file_name'].stem
            image_id = image['id']
            yolo_txt_path = dst_path / 'labels' / f'{image_name}.txt'
            yolo_txt_path.touch(exist_ok=True)
            if verbose:
                print(f"\r...Creating YOLO label file for image #{image_id}: {image_name}    ", end='')
        except Exception as e:
            print(f"Error creating YOLO label file for image #{image_id}: {e}")
    if verbose:
        print()

def process_coco(dst_path, src_dataset, json_path, verbose=True):
    """
    Load COCO JSON data and create directory structure

    Parameters
    ----------
    dst_path : Path
        Destination path for directory structure
    src_dataset : str
        Name of the source dataset
    json_path : Path
        Path to the COCO JSON file
    verbose : bool, optional
        Print progress messages, True by default

    Returns
    -------
    bool
        True if the dataset is split into train, validation, and test sets
    Path
        Destination path for split datasets
    list
        List of images in the dataset
    list
        List of categories in the dataset
    list
        List of annotations in the dataset

    """
    # Create directory structure based on the dataset

    if json_path.parent.name != src_dataset:
        split_name = json_path.parent.name
        key = split_name
    else: 
        split_name = ''
        key = 'all'

    if verbose:
        print(f"Processing JSON file: {json_path.name} in {json_path.parent.name}")

    # Load COCO JSON data
    with open(json_path, 'r') as f:
        coco_json = json.load(f)
        missing_keys = [key for key in ['categories', 'images', 'annotations'] if key not in coco_json]
        if missing_keys:
            raise ValueError(f"Invalid COCO JSON format in {json_path}. Missing keys: {', '.join(missing_keys)}")
        categories = coco_json['categories']
        images = coco_json['images']
        annotations = coco_json['annotations']

    return key, images, categories, annotations

def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

def process_bin(dst_path, images_path, masks_path, split, verbose=True):
    if split.is_dir():
        split_name = split.name
        key = split_name
    else:
        split_name = ''
        key = 'all'

    images = []
    for image in (images_path / split_name).iterdir():
        if verbose:
            print(f"\rProcessing image file: {image.name} in {key}", end='')
        height, width, _ = cv2.imread(image).shape
        images.append({
            'id': len(images),
            'file_name': image.name,
            'width' : width,
            'height' : height
        })

    categories = []
    annotations = []
    for category in (masks_path).iterdir():
        if category.is_dir():
            category_id = len(categories) + 1

            categories.append({
                'id': category_id,
                'name': category.name,
                'supercategory': category.name
            })

            for mask in (masks_path / category / split_name).iterdir():
                mask_image_name = mask.name

                for image in images:
                    if image.get('file_name') == mask_image_name:
                        image_id = image.get('id')

                contours = find_contours(cv2.imread(mask))
                for contour in contours:
                    segmentation = contour.flatten().tolist()

                    if not segmentation:
                        raise ValueError(f"Segmentation data missing for {category} binary mask {mask_image_name}")
                    elif len(segmentation) % 2 != 0:
                        segmentation.pop()

                    x_points = segmentation[::2]
                    y_points = segmentation[1::2]
                    x = int(min(x_points))
                    y = int(min(y_points))
                    width = int(max(x_points) - min(x_points))
                    height = int(max(y_points) - min(y_points))

                    annotations.append({
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "bbox": [
                            x,
                            y,
                            width,
                            height
                        ],
                        "area": width*height,
                        "iscrowd": 0,
                        "category_id": category_id,
                        "segmentation": [
                            segmentation
                        ]
                    })

    if verbose:
        print()

    return key, images, categories, annotations

def process_yolo():
    pass
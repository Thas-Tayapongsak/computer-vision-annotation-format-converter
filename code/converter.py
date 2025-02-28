from random import randint
import json

from converter_utils import (
    copy_images, 
    write_yolo_yaml, 
    initialize_yolo_labels,
    process_coco,
    validate_options,
    process_bin,
)

# TODO: ***VERY IMPORTANT***
#       REFACTOR SOME MORE AND CLEAN UP THE CODE
#       REDO CONVERTER 
#           YOLO11 -> COCO
#       WRITE LOGGING 
#       WRITE EXCEPTION ONLY FOR USER'S ERROR
#       WRITE COMMENTS AND DOCSTRING
#       WRITE TEST

def from_bin(opt, verbose=True):

    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')

    # find images and masks directory
    images_path = src_path / 'images'
    masks_path = src_path / 'masks'
    if not images_path.is_dir():
        raise FileNotFoundError(f"No images directory found in {src_path}")
    if not masks_path.is_dir():
        raise FileNotFoundError(f"No masks directory found in {src_path}")

    # check if images are split into train, validation, and test sets, and create directory structure
    splits = []
    for split in images_path.iterdir(): 
        key, images, categories, annotations = process_bin(dst_path, images_path, masks_path, split, verbose)
        splits.append({key: {'images': images, 'categories': categories, 'annotations': annotations}})
        if not split.is_dir():
            break

    coco_dict = {
        'options': opt,
        'splits': splits
    }

    return coco_dict

def from_coco(opt, verbose=True):
    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')
    src_dataset = opt.get('src_dataset')

    #find COCO json files
    json_paths = list(src_path.rglob('*.json')) 
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {src_path}")

    splits = []
    for json_path in json_paths:
        key, images, categories, annotations = process_coco(dst_path, src_dataset, json_path, verbose)
        splits.append({key: {'images': images, 'categories': categories, 'annotations': annotations}})

    coco_dict = {
        'options': opt,
        'splits': splits
    }

    return coco_dict

def from_yolo(opt, verbose=True):
    pass

def convert_to_yolo(annotation, image_dict, dst_path, split_name, task, verbose):
    
    image_id = annotation.get('image_id')
    cat_id = annotation.get('category_id') - 1

    image = image_dict.get(image_id, None)
    if image is None:
        raise ValueError(f"Image with ID {image_id} not found.")
    
    img_width = image['width']
    img_height = image['height']
    image_name = image['file_name'].stem

    yolo_txt_path = dst_path / split_name / 'labels' / f'{image_name}.txt'

    if verbose:
        print(f"\r...Converting annotation #{annotation['id']} for image #{image_id}: {image_name}    ", end='')

    if task == 'detect':
        # Convert bounding box annotations
        min_x, min_y, ann_width, ann_height = annotation['bbox']
        width = min(ann_width, img_width - min_x) / img_width
        height = min(ann_height, img_height - min_y) / img_height
        x_center = min_x / img_width + width / 2
        y_center = min_y / img_height + height / 2

        yolo_ann = f"{cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    elif task == 'segment':
        # Convert segmentation annotations
        segmentation = annotation['segmentation']
        if len(segmentation) == 1:
            segmentation = segmentation[0]
        if not segmentation:
            raise ValueError(f"Segmentation data missing for annotation {annotation['id']}")
        
        yolo_ann = f"{cat_id} "
        x_points = [x/img_width for x in segmentation[::2]]
        y_points = [y/img_height for y in segmentation[1::2]]
        for x, y in zip(x_points, y_points):
            yolo_ann += f"{x:.6f} {y:.6f} "
        yolo_ann += "\n"

    return yolo_ann, yolo_txt_path

def convert_to_cira(ann, categories, colors, task, verbose):
    if verbose:
        print("\rProcessing annotations " + str(ann.get('id')) + " ...", end='')

    bbox_coords = ann.get('bbox')
    x = int(bbox_coords[0])
    y = int(bbox_coords[1])
    w = int(bbox_coords[2])
    h = int(bbox_coords[3])
    bbox = f"{x}, {y}, {w}, {h}"
    x_center = x + w // 2
    y_center = y + h // 2
    center = f"{x_center}, {y_center}"
    label_index = ann.get('category_id') - 1
    label = categories[label_index].get('name')
    rgb = colors[label_index]
    color = f"{rgb[0]}, {rgb[1]}, {rgb[2]}"
    if task == 'segment':
        segmentation = ann.get('segmentation')
        if len(segmentation) == 1:
            segmentation = segmentation[0]
        if not segmentation:
            raise ValueError(f"Segmentation data missing for annotation {ann['id']}")
        
        if len(segmentation) % 2 != 0:
            landmark.append(landmark[-2])
        landmark_len = len(segmentation)//2
        landmark = ",".join([f"{segmentation[i]}:{segmentation[i+1]}" for i in range(0, len(segmentation), 2)]) + ","
    else:
        landmark = ""
        landmark_len = 0

    return {
        'bbox': bbox,
        'center': center,
        'color': color,
        'label': label,
        'label_index': label_index,
        'landmark': landmark,
        'landmark_len': landmark_len
    }

def to_yolo(coco_dict, verbose=True):
    opt = coco_dict.get('options')
    splits = coco_dict.get('splits')

    src_format = opt.get('src_format')
    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')
    src_dataset = opt.get('src_dataset')
    task = opt.get('task')

    src_split = True

    for split in splits:
        for key, data in split.items():
            if key == 'all':
                split_name = ''
                src_split = False
            else: 
                split_name = key

            if src_format in ['bin', 'yolo']:
                images_path = src_path / split_name / 'images'
            else:
                images_path = src_path / split_name

            (dst_path / split_name).mkdir(parents=True, exist_ok=True)
            (dst_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
            (dst_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)

            images = data.get('images')
            categories = data.get('categories')
            annotations = data.get('annotations')

            initialize_yolo_labels(dst_path / split_name, images, verbose)
            
            copy_images(images_path, dst_path / split_name / 'images', images, verbose)

            image_dict = {image['id']: image for image in images} # Create a dictionary to map image IDs to image data

            # Convert COCO annotations to YOLO format
            for ann in annotations:
                yolo_ann, yolo_txt_path = convert_to_yolo(ann, image_dict, dst_path, split_name, task, verbose)

                # Write YOLO annotation to file
                with open(yolo_txt_path, 'a') as f:
                    f.write(yolo_ann)
            if verbose:
                print()

    write_yolo_yaml(dst_path, src_dataset, src_split, categories)
    if verbose:
        print("Conversion completed successfully.")

def to_cira(coco_dict, verbose=True):
    opt = coco_dict.get('options')
    splits = coco_dict.get('splits')

    src_format = opt.get('src_format')
    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')
    src_dataset = opt.get('src_dataset')
    task = opt.get('task')

    for split in splits:
        for key, data in split.items():
            if key == 'all':
                split_name = ''
            else: 
                split_name = key

            if src_format in ['bin', 'yolo']:
                images_path = src_path / split_name / 'images'
            else:
                images_path = src_path / split_name

            (dst_path / split_name).mkdir(parents=True, exist_ok=True)
            (dst_path / split_name / 'images').mkdir(parents=True, exist_ok=True)

            images = data.get('images')
            categories = data.get('categories')
            annotations = data.get('annotations')

            colors = [[randint(0, 255), randint(0, 255), randint(0, 255)] for i in range(len(categories))]

            cira_list = []
            for image in images:
                cira_list.append({
                    'filename': image.get('file_name'),
                    'obj_array': []
                })

            copy_images(images_path, dst_path / split_name / 'images', images, verbose)

            image_dict = {image['id']: image for image in images} # Create a dictionary to map image IDs to image data

            for ann in annotations:
                image_id = ann.get('image_id')

                image = image_dict.get(image_id, None)
                if image is None:
                    raise ValueError(f"Image with ID {image_id} not found.")
                
                image_name = image['file_name']

                for file in cira_list:
                    if file['filename'] == image_name:
                        file['obj_array'].append(convert_to_cira(ann, categories, colors, task, verbose))
                
            if verbose:
                print()
            with open(dst_path / split_name / f'{src_dataset}.gt', "w") as outfile:
                json.dump(cira_list, outfile, indent=4)

def to_coco(coco_dict, verbose=True):
    opt = coco_dict.get('options')
    splits = coco_dict.get('splits')

    src_format = opt.get('src_format')
    src_path = opt.get('src_path')
    dst_path = opt.get('dst_path')

    for split in splits:
        for key, data in split.items():
            if key == 'all':
                split_name = ''
            else: 
                split_name = key

            if src_format in ['bin', 'yolo']:
                images_path = src_path / split_name / 'images'
            else:
                images_path = src_path / split_name

            (dst_path / split_name).mkdir(parents=True, exist_ok=True)

            images = data.get('images')

            copy_images(images_path, dst_path / split_name, images, verbose)

            with open(dst_path / split_name / f'_annotations.coco.json', "w") as outfile:
                json.dump(data, outfile, indent=4)

def convert(opt, verbose=True):

    opt = validate_options(opt, verbose)

    converters = {
        'from': {
            'bin': from_bin,
            'coco': from_coco,
            'yolo': from_yolo
        },
        'to': {
            'cira': to_cira,
            'coco': to_coco,
            'yolo': to_yolo,
        }
    }
        
    converters['to'][opt['dst_format']](converters['from'][opt['src_format']](opt, verbose))


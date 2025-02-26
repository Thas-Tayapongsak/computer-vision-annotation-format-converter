from shutil import rmtree
from typing import Union
from utils import get_user_input, get_root_path
from converter import convert

def get_options() -> dict[str, Union[float, str]]:
    """
    Get user input for dataset and task options

    Returns
    -------
    dict
        A dictionary containing the selected options

    """
    # Get the root path of the project
    root_path = get_root_path()
    datasets_path = root_path / 'datasets'
    output_path = root_path / 'output'

    # List available datasets and define valid options for tasks, modes, and formats
    DATASETS_LIST = [str(dataset.name) for dataset in datasets_path.iterdir()]
    TASKS_LIST = ['classify', 'detect', 'segment']
    MODES_LIST = ['convert', 'split']
    SRC_FORMATS_LIST = ['bin', 'coco', 'yolo']
    DST_FORMATS_LIST = ['coco', 'yolo', 'cira']

    # Raise an error if no datasets are found
    if DATASETS_LIST == []:
        raise FileNotFoundError("No datasets found in the 'datasets' directory.")

    # Get user input for source dataset and task
    src_dataset = get_user_input('Enter source dataset: ', DATASETS_LIST)
    task = get_user_input('Enter task: ', TASKS_LIST)

    # Set mode and formats based on the selected task
    if task == 'classify':
        mode = 'split'
        src_format = 'none'
        dst_format = 'none'
    else:
        mode = get_user_input('Enter mode: ', MODES_LIST)
        src_format = get_user_input('Enter source format: ', SRC_FORMATS_LIST)
        if mode == 'convert':
            dst_format = get_user_input('Enter destination format: ', DST_FORMATS_LIST)
        else:
            dst_format = src_format

    # Define source and destination paths
    src_path = datasets_path / src_dataset
    dst_path = output_path / f'{src_dataset}_{task[0]}{mode[0]}{src_format[0]}{dst_format[2]}'

    # Create or overwrite the destination directory
    if not dst_path.exists():
        dst_path.mkdir()
        print(f"Created new directory at {dst_path}")
    else:
        print(f"Directory {dst_path} already exists, it will be overwritten.")
        rmtree(dst_path)
        dst_path.mkdir()
        print(f"Created new directory at {dst_path}")

    # Store the options in a dictionary and return it
    options = {
        'src_dataset': src_dataset,
        'task': task,
        'mode': mode,
        'src_format': src_format,
        'dst_format': dst_format,
        'root_path': root_path,
        'src_path': src_path,
        'dst_path': dst_path,
    }

    return options

def print_options(options: dict[str, Union[float, str]]):
    """
    Print the selected options

    Parameters
    ----------
    options : dict
        A dictionary containing the selected options

    """
    print("\nSelected options:")
    for key, value in options.items():
        print(f"{key}: {value}")

def main(options: dict[str, Union[float, str]]):
    """
    Main function to execute the selected task
    """
    verbose = True if get_user_input('Verbose mode: ', ['yes', 'no']) == 'yes' else False
    if options['mode'] == 'convert':
        print(f"Converting {options['src_dataset']} dataset from {options['src_format']} to {options['dst_format']} format...")
        if verbose:
            print_options(options)
        
        convert(options, verbose)
                 
    elif options['mode'] == 'split':
        test_train_ratio = get_user_input('Enter test/train ratio: ', (0.0, 1.0))
        val_ratio = get_user_input('Enter validation ratio, enter 0 for no validation set: ', (0.0, 1.0)) 
        seed = get_user_input('Enter random seed: ', (0, 1000))
        split_names = ['train', 'val', 'test'] 

        print(f"Splitting {options['src_dataset']} dataset into {', '.join(split_names)} sets with ratios: test/train = {test_train_ratio}, val/test = {val_ratio}...")

        options['test_train_ratio'] = test_train_ratio
        options['val_ratio'] = val_ratio
        options['seed'] = seed
        options['split_names'] = split_names
        print_options(options)

if __name__ == '__main__':
    opt = get_options()
    main(opt)
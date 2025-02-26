from typing import Union
from pathlib import Path

def get_user_input(prompt: str, valid_range: Union[tuple[float, float], list[str]]) -> Union[float, str]:
    """
    Prompt the user for input

    Parameters
    ----------
    prompt : str
        The message to display to the user.
    valid_range : tuple or list
        The range of valid inputs. If a tuple, it should contain two elements (min, max).
        If a list, it should contain the valid options.

    Returns
    -------
    float or str
        The validated user input.

    """
    # Check if valid_range is a tuple
    if isinstance(valid_range, tuple):
        # Ensure the tuple has exactly two elements and both are int or float
        if len(valid_range) != 2 or not all(isinstance(i, (int, float)) for i in valid_range):
            raise ValueError("valid_range tuple must contain exactly two elements of type int or float.")
    # Check if valid_range is a list
    elif isinstance(valid_range, list):
        # Ensure all elements in the list are strings
        if not all(isinstance(i, (Path, str)) for i in valid_range):
            raise ValueError("valid_range list must contain elements of type str.")
    else:
        # Raise an error if valid_range is neither a tuple nor a list
        raise ValueError("valid_range must be a tuple with two elements or a list of valid options.")

    while True:
        # Display valid options to the user
        if isinstance(valid_range, tuple):
            print(f"Please enter a number between {valid_range[0]} and {valid_range[1]}.")
        elif isinstance(valid_range, list):
            print(f"Please choose one of the following options: {', '.join(valid_range)}.")

        # Prompt the user for input
        ans = input(prompt)
        
        # If valid_range is a tuple, validate the input as a number within the range
        if isinstance(valid_range, tuple):
            try:
                ans = float(ans)
                if valid_range[0] <= ans <= valid_range[1]:
                    return ans
                else:
                    print(f"Invalid input. Please enter a number between {valid_range[0]} and {valid_range[1]}.")
            except ValueError:
                print("Please enter a valid number.")
        # If valid_range is a list, validate the input as one of the valid options
        elif isinstance(valid_range, list):
            if ans in valid_range:
                return ans
            else:
                print(f"Invalid input. Please choose one of the following options: {', '.join(valid_range)}.")

def get_root_path() -> Path:
    """
    Find the root path of the project directory

    Returns
    -------
    Path
        The path to the root directory of the project

    """
    print("Searching for the root path...")
    root_path = None
    max_depth = 3  # Maximum depth to traverse up the directory tree
    depth = 0
    current_path = Path.cwd()  # Get the current working directory

    # Traverse up to the root directory
    while current_path != current_path.parent and depth < max_depth:
        potential_root_path = current_path / 'annotation_converter'
        print(f"\rChecking {potential_root_path} for 'annotation_converter' directory...", end="")
        if potential_root_path.is_dir():
            root_path = potential_root_path
            print(f"\nFound 'annotation_converter' directory at {root_path}")
            break
        current_path = current_path.parent  # Move up one directory level
        depth += 1

    # If not found, traverse down the directory tree
    if root_path is None:
        print("Traversing down the directory tree to find 'annotation_converter' directory...")
        for path in Path.cwd().rglob('annotation_converter'):
            if path.is_dir():
                root_path = path
                print(f"Found 'annotation_converter' directory at {root_path}")
                break
            
    # Raise an error if the directory is not found
    if root_path is None:
        raise FileNotFoundError("annotation_converter directory not found in any parent directories.")
    
    return root_path
import os
from box.exceptions import BoxValueError
import yaml
from logging import Logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from TextSummarizer.logging import logger 



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): The path to the YAML file.
        
    Raises:
        FileNotFoundError: If the YAML file is not found.
        ValueError: If the YAML file is empty.
        yaml.YAMLError: If there's an error while parsing the YAML file.
        
    Returns:
        ConfigBox: The content of the YAML file wrapped in a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")  # Use the logger instance
            return ConfigBox(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"{path_to_yaml} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error while parsing YAML file: {e}")
    except ValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    Args:
        path (Path): path of the file
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

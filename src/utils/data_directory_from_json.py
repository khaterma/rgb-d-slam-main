import json
from pathlib import Path


def get_data_directory_name():
    """
    Returns the name of the directory in the data folder where the data pertaining to the bag and threshold as specified
    in config.json is located.
    """
    path_to_json = Path(__file__).parent.parent / 'config.json'
    f = open(path_to_json)
    data = json.load(f)
    transl_int, transl_dec = str(data["transl_threshold"]).split('.')
    rot_int, rot_dec = str(data["rot_threshold"]).split('.')

    name = data["bag_name"] + "_" + transl_int + "-" + transl_dec + "_" + rot_int + "-" + rot_dec

    return name


def get_data_directory_path():
    path_to_data = Path(__file__).parent.parent / 'data' / get_data_directory_name()

    return path_to_data

def get_bag_name():
    """
    Utility function that returns name of bag, including .bag ending as specified in .json file.
    """
    path_to_json = Path(__file__).parent.parent / 'config.json'
    f = open(path_to_json)
    data = json.load(f)

    return data["bag_name"] + ".bag"

def get_thresholds():
    """
    Utility function that returns the thresholds for translation and rotation after which to sample a new pose.
    """
    path_to_json = Path(__file__).parent.parent / 'config.json'
    f = open(path_to_json)
    data = json.load(f)

    return data["transl_threshold"], data["rot_threshold"]
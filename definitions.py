import os

ROOT_DIR = os.path.dirname(__file__)

PATH = {
    "IMAGE_IN": os.path.join(ROOT_DIR, 'data/z_stack_images/'),
    "BINARY_MAPS": os.path.join(ROOT_DIR, 'data/binary_maps/'),
    "IMAGE_OUT": os.path.join(ROOT_DIR, 'data/z_stack_processed/'),
    "CSV": os.path.join(ROOT_DIR, 'data/z_stack_images.csv'),
}

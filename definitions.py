import os

ROOT_DIR = os.path.dirname(__file__)

PATH = {
    "IMAGE_IN": os.path.join(ROOT_DIR, 'data/z_stack_images/'),
    "BINARY_MAPS": os.path.join(ROOT_DIR, 'data/binary_maps/'),
    "SIMCEP": os.path.join(ROOT_DIR, 'data/simcep/'),
    "SIMCEP_CSV": os.path.join(ROOT_DIR, 'data/simcep/simcep.csv'),
    "IMAGE_OUT": os.path.join(ROOT_DIR, 'data/z_stack_processed/'),
    "CSV": os.path.join(ROOT_DIR, 'data/z_stack_images.csv'),
    "MODEL_OUT": os.path.join(ROOT_DIR, 'model/out')
}

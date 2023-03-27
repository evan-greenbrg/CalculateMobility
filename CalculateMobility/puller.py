import glob
import os
from natsort import natsorted


def get_paths(root):
    # Get the rivers
    fps = glob.glob(os.path.join(root, 'mask', '*.tif'))
    out_paths = {root: fps}

    return out_paths


def get_paths_dswe(root):
    fp_roots = natsorted(glob.glob(os.path.join(root, 'WaterLevel*')))
    out_paths = {}
    for root in fp_roots:
        # Get the rivers
        water_level = root.split('/')[-1]
        fps = glob.glob(os.path.join(root, 'mask', '*.tif'))
        out_paths[root] = fps

    return out_paths

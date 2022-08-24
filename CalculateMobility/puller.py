import glob
import os


def get_paths(poly, root, river):
    # Get the rivers
    fps = glob.glob(os.path.join(root, river, 'mask', '*'))
    blocks = {}
    for fp in fps:
        block = fp.split('_')[-1].split('.')[0]
        if not blocks.get(block):
            blocks[block] = [fp]
        if blocks.get(block):
            blocks[block].append(fp)

    out_paths = []
    for block in sorted(blocks.keys()):
        out_paths.append(blocks[block])

    return out_paths

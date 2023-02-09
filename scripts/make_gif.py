import argparse
import glob
import os
import io
import re
from natsort import natsorted
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def make_gif(root, out):
    # Find files
    fps = natsorted(glob.glob(os.path.join(root, '*.tif')))

    agrs = []
    years = []
    for i, fp in enumerate(fps):
        year = re.findall(r"[0-9]{4,7}", fp)[-1]
        ds = rasterio.open(fp).read(1).astype(int)
        if not np.sum(ds):
            skip_flag = True
            continue

        if (not i) or (skip_flag):
            agr = ds
            skip_flag = False
        else:
            agr += ds

        agr[np.where(ds)] = 2
        ag_save = np.copy(agr)
        agr[np.where(agr)] = 1
        agrs.append(ag_save)
        years.append(year)

    images = []
    # legend_elements = [
    #     Patch(color='#ad2437', label='Visited Pixels'),
    #     Patch(color='#6b2e10', label='Unvisted Pixels'),
    #     Patch(color='#9eb4f0', label='Yearly Water'),
    # ]
    for i, ag in enumerate(agrs):
        year = years[i]

        img_buf = io.BytesIO()
        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        # ax.imshow(ag, cmap='Paired_r')
        ax.imshow(ag, cmap='Greys')
        ax.text(
            0.95,
            0.95,
            f'Year: {year}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            color='red'
        )

        # ax.legend(
        #     handles=legend_elements,
        #     loc='lower left',
        #     prop={'size': 10}
        # )
        ax.axis('off')

        plt.savefig(img_buf, format='png')
        images.append(Image.open(img_buf))
        plt.close('all')

    img, *imgs = images
    print(out)
    img.save(
        fp=out,
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=400,
        loop=30
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make gif')
    parser.add_argument(
        '--root', 
        metavar='root', 
        type=str,
        help='root folder with tif files to make gif'
    )

    parser.add_argument(
        '--out', 
        metavar='out', 
        type=str,
        help='path to save the file'
    )

    args = parser.parse_args()
    make_gif(args.root, args.out)

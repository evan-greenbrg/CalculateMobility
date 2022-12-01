import os
import numpy as np
import pandas

from mobility_helpers import clean
from mobility_helpers import create_mask_shape
from mobility_helpers import get_scale 


def get_mobility_rivers(poly, paths, out, river):
    print(river)
    for block, path_list in enumerate(paths):
        path_list = sorted(path_list)

        mask = create_mask_shape(
            poly,
            river,
            path_list
        )

        images, metas = clean(
            poly,
            river,
            path_list
        )

        scale = get_scale(path_list[-1])

        river_dfs = get_mobility_yearly(
            images,
            mask,
            scale=scale
        )

        full_df = pandas.DataFrame()
        for year, df in river_dfs.items():
            rnge = f"{year}_{df.iloc[-1]['year']}"
            df['dt'] = pandas.to_datetime(
                df['year'],
                format='%Y'
            )
            df['range'] = rnge

            full_df = full_df.append(df)

        out_path = os.path.join(
            out,
            f'{river}_yearly_mobility_block_{block}.csv'
        )
        full_df.to_csv(out_path)

    return river


def get_mobility_yearly(images, mask, scale=30):

    A = len(np.where(mask == 1)[1])

    year_range = list(images.keys())
    ranges = [year_range[i:] for i, yr in enumerate(year_range)]
    river_dfs = {}
    for yrange in ranges:
        data = {
            'year': [],
            'i': [],
            'O_avg': [],
            'O_wd': [],
            'O_dw': [],
            'O_wick': [],
            'fR': [],
            'fR_wick': [],
            'w_b': [],
            'd_b': [],
        }
        length = images[yrange[0]].shape[0]
        width = images[yrange[0]].shape[1]
        long = len(yrange)
        all_images = np.empty((length, width, long))
        years = []
        for j, year in enumerate(yrange):
            years.append(year)
            im = images[str(year)].astype(int)
            filt = np.where(~np.array(mask) + 2)
            im[filt] = 0
            all_images[:, :, j] = im

        baseline = all_images[:, :, 0]
        w_b = len(np.where(baseline == 1)[0])
        fb = mask - baseline
        fw_b = w_b / A
        fd_b = np.sum(fb) / A
        Na = A * fd_b

        for j in range(all_images.shape[2]):
            im = all_images[:, :, j]

            kb = (
                np.sum(all_images[:, :, :j + 1], axis=(2))
                + mask
            )
            kb[np.where(kb != 1)] = 0
            Nb = np.sum(kb)
            # fR = (Na / w_b) - (Nb / w_b)
            fR = (Na - Nb)
            fR_wick = 1 - (Nb / Na)

            # Calculate D - EQ. (1)
            D = np.subtract(baseline, im)
            # 1 - wet -> dry
            d_wd = len((np.where(D == 1))[0])
            # -1 - dry -> wet
            d_dw = len((np.where(D == -1))[0])

            # Calculate Phi
            w_t = len(np.where(im == 1)[0])
            fw_t = w_t / A
            fd_t = (A - w_t) / A

            # Calculate O_Phi
            PHI = (fw_b * fd_t) + (fd_b * fw_t)
            o_wick = 1 - (np.sum(np.abs(D)) / (A * PHI))
            o_avg = w_b - np.mean([d_wd, d_dw])
            o_wd = w_b - d_wd
            o_dw = w_b - d_dw

            data['i'].append(j)
            data['O_avg'].append(o_avg * (scale**2))
            data['O_wd'].append(o_wd * (scale**2))
            data['O_dw'].append(o_dw * (scale**2))
            data['O_wick'].append(o_wick)
            data['fR'].append(fR * (scale**2))
            data['fR_wick'].append(fR_wick)
            data['w_b'].append(w_b * (scale**2))
            data['d_b'].append(Na * (scale**2))

        data['year'] = years
        river_dfs[yrange[0]] = pandas.DataFrame(data=data)

    return river_dfs

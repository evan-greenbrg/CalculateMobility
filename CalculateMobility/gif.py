import re
import os
import io
import glob
import rasterio
import numpy as np
import pandas
from PIL import Image
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit


def func_3_param(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p

def func_r_param(x, r, p):
    return (-p * np.exp(-r * x)) + p

def func_m_param(x, m, p, aw):
    return ((aw - p) * np.exp(-m * x)) + p

def m_wrapper(aw):
    def tempfunc(x, m, p, aw=aw):
        return func_m_param(x, m, p, aw)
    return tempfunc

def func_2_param(x, m, p):
    return (p * np.exp(-m * x)) + p


def fit_curve(x, y, fun, p0):
    # Fitting
    popt, pcov = curve_fit(
        fun, x, y, 
        p0=p0, maxfev=1000000
    )
    # R-squared
    residuals = y - fun(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return (*popt), r_squared


def make_gif(fps, fp_in, fp_out, stat_out):

    # Handle mobility dataframes
    full_dfs = [pandas.read_csv(fp) for fp in fps]
    full_dfs_clean = []
    for full_df in full_dfs:
        full_df_clean = pandas.DataFrame()
        for group, df in full_df.groupby('range'):
            df['x'] = df['year'] - df.iloc[0]['year']
            full_df_clean = full_df_clean.append(df)
        full_dfs_clean.append(full_df_clean)

    # Stack all blocks
    full_df = pandas.DataFrame()
    for df in full_dfs_clean:
        full_df = full_df.append(df)

    # Handle images
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    years = {}
    for im in imgs:
        year = re.findall(r"[0-9]{4,7}", im)[-1]
        print(year)
        years[str(year)] = im

    year_keys = list(full_df['year'].unique())
    years_filt = {}
    for key in year_keys:
        year_im = years.get(str(key), None)

        if not year_im:
            continue

        years_filt[key] = year_im

    years = years_filt
    year_keys = list(years.keys())
    imgs = []
    agrs = []
    combos = []
    for year, file in years.items():
        ds = rasterio.open(file).read(1)

        image = ds
        if year == year_keys[0]:
            agr = ds
        else:
            agr += ds

        ag_save = np.copy(agr)

        combo = agr + image

        imgs.append(image)
        agrs.append(ag_save)
        combos.append(combo)

    # Make avg_df
    avg_df = full_df.groupby('x').median().reset_index(drop=False).iloc[:25]
    avg_df = avg_df.dropna(how='any')

    # make max and min dfs
    max_df = full_df.groupby('x').quantile(0.85).reset_index(drop=False).iloc[:25]
    max_df = max_df.dropna(how='any')

    min_df = full_df.groupby('x').quantile(0.15).reset_index(drop=False).iloc[:25]
    min_df = min_df.dropna(how='any')

    aw = max_df['w_b'].mean()
    m_avg, pm_avg, m_r2_avg = fit_curve(
        max_df['x'],
        max_df['O_avg'].to_numpy(),
        m_wrapper(aw),
        [1, 1]
    )

    m_wd, pm_wd, m_r2_wd = fit_curve(
        max_df['x'],
        max_df['O_wd'].to_numpy(),
        m_wrapper(aw),
        [1, 1]
    )

    m_dw, pm_dw, m_r2_dw = fit_curve(
        max_df['x'],
        max_df['O_dw'].to_numpy(),
        m_wrapper(aw),
        [1, 1]
    )

    r, pr, f_r2 = fit_curve(
        min_df['x'],
        min_df['fR'].to_numpy(),
        func_r_param,
        [.001, 1]
    )

    am_wick, cm_wick, pm_wick, m_r2_wick = fit_curve(
        max_df['i'],
        max_df['O_wick'].to_numpy(),
        func_3_param,
        [1, .01, 1]
    )

    ar_wick, cr_wick, pr_wick, r_r2_wick = fit_curve(
        max_df['i'],
        (1 - max_df['fR_wick']).to_numpy(),
        func_3_param,
        [1, .01, 1]
    )

    # Get average w_b
    w_b = avg_df['w_b'].median()
    w_b_max = max_df['w_b'].median()

    stats = pandas.DataFrame(data={
        'Type': ['Value', 'Rsquared'],
        'CM_avg': [round(m_avg, 8), round(m_r2_avg, 8)],
        'PM_avg': [round(pm_avg, 8), None],
        'CM_wd': [round(m_wd, 8), round(m_r2_wd, 8)],
        'PM_wd': [round(pm_wd, 8), None],
        'CM_dw': [round(m_dw, 8), round(m_r2_dw, 8)],
        'PM_dw': [round(pm_dw, 8), None],
        'CR': [round(r, 8), round(f_r2, 8)],
        'PR': [round(pr, 8), None],
        'Aw': [round(w_b, 8), None],
        'Aw_max': [round(w_b_max, 8), None],
        'CM_wick': [round(cm_wick, 8), round(m_r2_wick, 8)],
        'AM_wick': [round(am_wick, 8), None],
        'PM_wick': [round(pm_wick, 8), None],
        'CR_wick': [round(cr_wick, 8), round(r_r2_wick, 8)],
        'AR_wick': [round(ar_wick, 8), None],
        'PR_wick': [round(pr_wick, 8), None],
    })
    stats.to_csv(stat_out)

    m_pred = func_m_param(avg_df['x'], m_avg, pm_avg, aw)
    r_pred = func_r_param(avg_df['x'], r, pr)

    # METHOD 2
    images = []
    legend_elements = [
        Patch(color='#ad2437', label='Visited Pixels'),
        Patch(color='#6b2e10', label='Unvisted Pixels'),
        Patch(color='#9eb4f0', label='Yearly Water'),
    ]
    # for i, ag in enumerate(agrs):
    for i, ag in enumerate(combos):
        year = list(years.keys())[i]
        if i < len(avg_df):
            data = avg_df.iloc[i]

        img_buf = io.BytesIO()

        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.imshow(ag, cmap='Paired_r')
        ax1.text(
            0.05,
            0.95,
            f'Year: {year}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax1.transAxes
        )

        ax1.legend(
            handles=legend_elements,
            loc='lower left',
            prop={'size': 10}
        )

        ax2.plot(
            min_df['x'],
            r_pred,
            zorder=5,
            color='green',
            label='3 Parameter'
        )
        ax2.scatter(
            min_df['x'],
            min_df['fR'],
            zorder=4,
            s=70,
            facecolor='black',
            edgecolor='black'
        )
        ax2.scatter(
            full_df['x'],
            full_df['fR'],
            zorder=2,
            s=50,
            facecolor='white',
            edgecolor='black'
        )
        if i < len(avg_df):
            ax2.scatter(
                data['x'],
                data['fR'],
                s=200,
                zorder=3,
                color='red'
            )
        ax2.set_ylabel('Remaining Rework Fraction')
        ax2.legend(
            loc='upper left',
            frameon=True
        )

        ax3.plot(
            max_df['x'],
            m_pred,
            zorder=5,
            color='blue'
        )
        ax3.scatter(
            max_df['x'],
            max_df['O_avg'],
            zorder=4,
            s=70,
            facecolor='black',
            edgecolor='black'
        )
        ax3.scatter(
            full_df['x'],
            full_df['O_avg'],
            zorder=2,
            s=25,
            facecolor='white',
            edgecolor='black'
        )
        ax3.scatter(data['x'], data['O_avg'], s=200, zorder=3, color='red')
        ax3.set_ylabel('Normalized Channel Overlap')
        # ax3.set_ylim([0, 1])

        plt.savefig(img_buf, format='png')
        images.append(Image.open(img_buf))
        plt.close('all')

    img, *imgs = images
    img.save(
        fp=fp_out,
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=400,
        loop=30
    )

    root = '/'.join(fps[0].split('/')[:-1])
    years = [i for i in range(1985, 2023)]
    for year in years:
        dire = os.path.join(root, str(year))
        if os.path.isdir(dire):
            os.rmdir(dire)


def make_gifs(river, root):
    print(river)
    fps = sorted(
        glob.glob(os.path.join(root, f'{river}/*mobility_block_0.csv'))
    )
    fp_in = os.path.join(
        root, f'{river}/mask/*_mask*.tif'
    )

    fp_out = os.path.join(
        root, f'{river}/{river}_cumulative.gif'
    )
    stat_out = os.path.join(
        root, f'{river}/{river}_mobility_stats.csv'
    )
    make_gif(fps, fp_in, fp_out, stat_out)


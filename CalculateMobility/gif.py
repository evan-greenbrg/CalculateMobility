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


def get_stats(fps, stat_out):
    # Handle mobility dataframes
    if not fps:
        print(f'Number of files found: {len(fps)}')
        raise ValueError('No files found')

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

    # Pick when to stop - this is when there are only 3 images left
    stop = len(np.unique(full_df['x'])) - 3
    # Make avg_df
    df50 = full_df.groupby('x').quantile(0.5).reset_index(drop=False).iloc[:stop]
    df50 = df50.dropna(how='any')

    df75 = full_df.groupby('x').quantile(0.75).reset_index(drop=False).iloc[:stop]
    df75 = df75.dropna(how='any')

    df25 = full_df.groupby('x').quantile(0.25).reset_index(drop=False).iloc[:stop]
    df25 = df25.dropna(how='any')

    # OAvg
    aw25 = df25['w_b'].mean()
    m25, pm25, m25_r2 = fit_curve(
        df25['x'],
        df25['O_avg'].to_numpy(),
        m_wrapper(aw25),
        [.01, 100]
    )

    aw50 = df50['w_b'].mean()
    m50, pm50, m50_r2 = fit_curve(
        df50['x'],
        df50['O_avg'].to_numpy(),
        m_wrapper(aw50),
        [.01, 100]
    )

    aw75 = df75['w_b'].mean()
    m75, pm75, m75_r2 = fit_curve(
        df75['x'],
        df75['O_avg'].to_numpy(),
        m_wrapper(aw75),
        [.01, 100]
    )

    # Owet-dry
    m25wd, pm25wd, m25wd_r2 = fit_curve(
        df25['x'],
        df25['O_wd'].to_numpy(),
        m_wrapper(aw25),
        [.01, 100]
    )

    m50wd, pm50wd, m50wd_r2 = fit_curve(
        df50['x'],
        df50['O_wd'].to_numpy(),
        m_wrapper(aw50),
        [.01, 100]
    )

    m75wd, pm75wd, m75wd_r2 = fit_curve(
        df75['x'],
        df75['O_wd'].to_numpy(),
        m_wrapper(aw75),
        [.01, 100]
    )

    # Odry-wed
    m25dw, pm25dw, m25dw_r2 = fit_curve(
        df25['x'],
        df25['O_dw'].to_numpy(),
        m_wrapper(aw25),
        [.01, 100]
    )

    m50dw, pm50dw, m50dw_r2 = fit_curve(
        df50['x'],
        df50['O_dw'].to_numpy(),
        m_wrapper(aw50),
        [.01, 100]
    )

    m75dw, pm75dw, m75dw_r2 = fit_curve(
        df75['x'],
        df75['O_dw'].to_numpy(),
        m_wrapper(aw75),
        [.01, 100]
    )

    r25, pr25, r25_r2 = fit_curve(
        df25['x'],
        df25['fR'].to_numpy(),
        func_r_param,
        [.001, 100]
    )
    r50, pr50, r50_r2 = fit_curve(
        df50['x'],
        df50['fR'].to_numpy(),
        func_r_param,
        [.001, 100]
    )
    r75, pr75, r75_r2 = fit_curve(
        df75['x'],
        df75['fR'].to_numpy(),
        func_r_param,
        [.001, 100]
    )

    stats = pandas.DataFrame(data={
        'Quantile': [25, 50, 75],
        'Aw': [aw25, aw50, aw75],
        'CM': [m25, m50, m75],
        'PM': [pm25, pm50, pm75],
        'M_r2': [m25_r2, m50_r2, m75_r2],
        'CMwd': [m25wd, m50wd, m75wd],
        'PMwd': [pm25wd, pm50wd, pm75wd],
        'Mwd_r2': [m25wd_r2, m50wd_r2, m75wd_r2],
        'CMdw': [m25dw, m50dw, m75dw],
        'PMdw': [pm25dw, pm50dw, pm75dw],
        'Mdw_r2': [m25dw_r2, m50dw_r2, m75dw_r2],
        'CR': [r25, r50, r75],
        'PR': [pr25, pr50, pr75],
        'R_r2': [r25_r2, r50_r2, r75_r2],
    })
    stats.to_csv(stat_out)

    return stats


def get_mobility(stats, mobility_out):
    """
    HAVE THIS SPIT OUT REAL DATA
    """
    quantile = [25, 50, 75]
    M = (stats['CM'] * (1 - (stats['PM'] / stats['Aw']))).values
    T_M = 3 / M

    Mwd = (stats['CMwd'] * (1 - (stats['PMwd'] / stats['Aw']))).values
    T_Mwd = 3 / Mwd

    Mdw = (stats['CMdw'] * (1 - (stats['PMdw'] / stats['Aw']))).values
    T_Mdw = 3 / Mdw

    R = (stats['CR'] * (stats['PR'] / stats['Aw'])).values
    T_R = 3 / R

    mobility = pandas.DataFrame(data={
        'Quantile': quantile,
        'M': M,
        'T_M': T_M,
        'Mwd': Mwd,
        'T_Mwd': T_Mwd,
        'Mdw': Mdw,
        'T_Mdw': T_Mdw,
        'R': R,
        'T_R': T_R,
        'Aw': stats['Aw'].values
    })
    mobility.to_csv(mobility_out)


def make_gif(fps, fp_in, fp_out, dswe=False):
    """
    HAVE TO FIX SOME OF THE REFERENCES TO THE OLD DF
    """

    # Handle mobility dataframes
    if dswe:
        full_dfs = []
        for fp in fps:
            level = fp.split('/')[-2]
            df = pandas.read_csv(fp)
            df['DSWE_level'] = level
            full_dfs.append(df)
    else:
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

    # Make avg_df
    if dswe:
        avgs = {}
        for name, group in full_df.groupby('DSWE_level'):
            avgs[name] = group.groupby('x').quantile(0.5).reset_index(drop=False).iloc[:]
    else:
        df50 = full_df.groupby('x').quantile(0.5).reset_index(drop=False).iloc[:]
        df50 = df50.dropna(how='any')

    # Handle images
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    years = {}
    for im in imgs:
        year = re.findall(r"[0-9]{4,7}", im)[-1]
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
    for year, file in years.items():
        ds = rasterio.open(file).read(1)

        image = ds
        if year == year_keys[0]:
            agr = ds
        else:
            agr += ds

        agr[np.where(ds)] = 2
        ag_save = np.copy(agr)
        agr[np.where(agr)] = 1

        imgs.append(image)
        agrs.append(ag_save)

    # METHOD 2
    images = []
    legend_elements = [
        Patch(color='#ad2437', label='Visited Pixels'),
        Patch(color='#6b2e10', label='Unvisted Pixels'),
        Patch(color='#9eb4f0', label='Yearly Water'),
    ]
    for i, ag in enumerate(agrs):
        year = list(years.keys())[i]

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
        if dswe:
            for level, df50 in avgs.items():
                ax2.scatter(
                    df50['x'],
                    df50['fR'],
                    zorder=4,
                    s=70,
                    facecolor='#8B0000',
                    edgecolor='black'
                )
                ax2.scatter(
                    df50.iloc[i]['x'],
                    df50.iloc[i]['fR'],
                    s=200,
                    zorder=3,
                    color='red'
                )
            for name, group in full_df.groupby('DSWE_level'):
                ax2.scatter(
                    group['x'],
                    group['fR'],
                    zorder=2,
                    s=30,
                )
        else:
            ax2.scatter(
                df50['x'],
                df50['fR'],
                zorder=4,
                s=70,
                facecolor='#8B0000',
                edgecolor='black'
            )
            ax2.scatter(
                df50.iloc[i]['x'],
                df50.iloc[i]['fR'],
                s=200,
                zorder=3,
                color='red'
            )
            ax2.scatter(
                full_df['x'],
                full_df['fR'],
                zorder=2,
                s=30,
                facecolor='black',
                edgecolor='black'
            )

        ax2.set_ylabel('Remaining Rework Fraction')
        ax2.legend(
            loc='upper left',
            frameon=True
        )

        if dswe:
            for level, df50 in avgs.items():
                ax3.scatter(
                    df50['x'],
                    df50['O_avg'],
                    zorder=4,
                    s=70,
                    facecolor='#8B0000',
                    edgecolor='black'
                )
                ax3.scatter(
                    df50.iloc[i]['x'],
                    df50.iloc[i]['O_avg'],
                    s=200,
                    zorder=3,
                    color='red'
                )
            for name, group in full_df.groupby('DSWE_level'):
                ax3.scatter(
                    group['x'],
                    group['O_avg'],
                    zorder=2,
                    s=30,
                )
        else:
            ax3.scatter(
                df50['x'],
                df50['O_avg'],
                zorder=4,
                s=70,
                facecolor='#8B0000',
                edgecolor='black'
            )
            ax3.scatter(
                df50.iloc[i]['x'],
                df50.iloc[i]['O_avg'],
                s=200,
                zorder=3,
                color='red'
            )
            ax3.scatter(
                full_df['x'],
                full_df['O_avg'],
                zorder=2,
                s=30,
                facecolor='black',
                edgecolor='black'
            )

        ax3.set_ylabel('Normalized Channel Overlap')
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


def filter_dswe_stats(stats, r2_thresh):
    stats.at[
        np.where(stats['M_r2'] < r2_thresh)[0], 
        ['CM', 'PM']
    ] = None
    stats.at[
        np.where(stats['Mwd_r2'] < r2_thresh)[0], 
        ['CMwd', 'PMwd']
    ] = None
    stats.at[
        np.where(stats['Mdw_r2'] < r2_thresh)[0], 
        ['CMdw', 'PMdw']
    ] = None
    stats.at[
        np.where(stats['R_r2'] < r2_thresh)[0], 
        ['R', 'PR']
    ] = None
    stats.at[
        np.where(
            (stats['M_r2'] < r2_thresh)
            & (stats['R_r2'] < r2_thresh)
        )[0],
        ['Aw']
    ] = None
    
    return stats.groupby('Quantile').median()


def make_gifs(river, root, dswe=False, r2_thresh=.76):
    print(river)

    fp_out = os.path.join(
        root, f'{river}_cumulative.gif'
    )
    stat_out = os.path.join(
        root, f'{river}_pixel_values.csv'
    )
    mobility_out = os.path.join(
        root, f'{river}_mobility_metrics.csv'
    )
    print('Finding Stats')
    if dswe:
        fp_in = os.path.join(
            root, 'WaterLevel2/mask/*_mask*.tif'
        )
        # Iterate through all the water levels in the folder
        fp_roots = glob.glob(os.path.join(root, 'WaterLevel*'))
        stats = pandas.DataFrame()
        fps = []
        for root in fp_roots:
            level = root.split('/')[-1]
            print(level)
            water_stat_out = os.path.join(root, f'{river}_pixel_values.csv')
            water_fps = glob.glob(
                os.path.join(root, '*yearly_mobility*.csv')
            )
            # Get individual water level stats and save those csv
            level_stats = get_stats(water_fps, water_stat_out)
            level_stats ['WaterLevel'] = level

            # Concatenate all the water level stats and save
            stats = stats.append(level_stats)
            fps.append(water_fps[0])
        stats = stats.reset_index(drop=True)
        stats.to_csv(stat_out)

        # Reduce the stats df according to the given r2 threhold
        stats = filter_dswe_stats(stats, r2_thresh=r2_thresh)

    else:
        fps = sorted(
            glob.glob(os.path.join(root, '*yearly_mobility.csv'))
        )
        fp_in = os.path.join(
            root, 'mask/*_mask*.tif'
        )
        stats = get_stats(fps, stat_out)

    print('Calculating Mobility')
    get_mobility(stats, mobility_out)
    print('Making Gif')
    make_gif(fps, fp_in, fp_out, dswe=dswe)



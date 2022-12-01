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
    print(stop)
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
        'Type': ['Value', 'Rsquared'],
        'Aw25': [round(aw25, 8), None],
        'Aw50': [round(aw50, 8), None],
        'Aw75': [round(aw75, 8), None],
        'CM25': [round(m25, 8), round(m25_r2, 8)],
        'PM25': [round(pm25, 8), None],
        'CM50': [round(m50, 8), round(m50_r2, 8)],
        'PM50': [round(pm50, 8), None],
        'CM75': [round(m75, 8), round(m75_r2, 8)],
        'PM75': [round(pm75, 8), None],
        'CM25wd': [round(m25wd, 8), round(m25wd_r2, 8)],
        'PM25wd': [round(pm25wd, 8), None],
        'CM50wd': [round(m50wd, 8), round(m50wd_r2, 8)],
        'PM50wd': [round(pm50wd, 8), None],
        'CM75wd': [round(m75wd, 8), round(m75wd_r2, 8)],
        'PM75wd': [round(pm75wd, 8), None],
        'CM25dw': [round(m25dw, 8), round(m25dw_r2, 8)],
        'PM25dw': [round(pm25dw, 8), None],
        'CM50dw': [round(m50dw, 8), round(m50dw_r2, 8)],
        'PM50dw': [round(pm50dw, 8), None],
        'CM75dw': [round(m75dw, 8), round(m75dw_r2, 8)],
        'PM75dw': [round(pm75dw, 8), None],
        'CR25': [round(r25, 8), round(r25_r2, 8)],
        'PR25': [round(pr25, 8), None],
        'CR50': [round(r50, 8), round(r50_r2, 8)],
        'PR50': [round(pr50, 8), None],
        'CR75': [round(r75, 8), round(r75_r2, 8)],
        'PR75': [round(pr75, 8), None],
    })
    stats.to_csv(stat_out)

    return stats


def get_mobility(stats, mobility_out):
    """
    HAVE THIS SPIT OUT REAL DATA
    """
    stats = stats.iloc[0]

    M25 = stats['CM25'] * (1 - (stats['PM25'] / stats['Aw25']))
    T_M25 = 3 / M25

    M50 = stats['CM50'] * (1 - (stats['PM50'] / stats['Aw50']))
    T_M50 = 3 / M50

    M75 = stats['CM75'] * (1 - (stats['PM75'] / stats['Aw75']))
    T_M75 = 3 / M75

    M25wd = stats['CM25wd'] * (1 - (stats['PM25wd'] / stats['Aw25']))
    T_M25wd = 3 / M25wd

    M50wd = stats['CM50wd'] * (1 - (stats['PM50wd'] / stats['Aw50']))
    T_M50wd = 3 / M50wd

    M75wd = stats['CM75wd'] * (1 - (stats['PM75wd'] / stats['Aw75']))
    T_M75wd = 3 / M75wd

    M25dw = stats['CM25dw'] * (1 - (stats['PM25dw'] / stats['Aw25']))
    T_M25dw = 3 / M25dw

    M50dw = stats['CM50dw'] * (1 - (stats['PM50dw'] / stats['Aw50']))
    T_M50dw = 3 / M50dw

    M75dw = stats['CM75dw'] * (1 - (stats['PM75dw'] / stats['Aw75']))
    T_M75dw = 3 / M75dw

    R25 = stats['CR25'] * ((stats['PR25'] / stats['Aw25']))
    T_R25 = 3 / R25

    R50 = stats['CR50'] * ((stats['PR50'] / stats['Aw50']))
    T_R50 = 3 / R50

    R75 = stats['CR75'] * ((stats['PR75'] / stats['Aw75']))
    T_R75 = 3 / R75

    mobility = pandas.DataFrame(data={
        'M25': [M25],
        'T_M25': [T_M25],
        'M50': [M50],
        'T_M50': [T_M50],
        'M75': [M75],
        'T_M75': [T_M75],
        'M25wd': [M25wd],
        'T_M25wd': [T_M25wd],
        'M50wd': [M50wd],
        'T_M50wd': [T_M50wd],
        'M75wd': [M75wd],
        'T_M75wd': [T_M75wd],
        'M25dw': [M25dw],
        'T_M25dw': [T_M25dw],
        'M50dw': [M50dw],
        'T_M50dw': [T_M50dw],
        'M75dw': [M75dw],
        'T_M75dw': [T_M75dw],
        'R25': [R25],
        'T_R25': [T_R25],
        'R50': [R50],
        'T_R50': [T_R50],
        'R75': [R75],
        'T_R75': [T_R75],
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
    df50 = full_df.groupby('x').quantile(0.5).reset_index(drop=False).iloc[:]
    df50 = df50.dropna(how='any')

    df75 = full_df.groupby('x').quantile(0.75).reset_index(drop=False).iloc[:]
    df75 = df75.dropna(how='any')

    df25 = full_df.groupby('x').quantile(0.25).reset_index(drop=False).iloc[:]
    df25 = df25.dropna(how='any')

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
        if i < len(df50):
            data = df50.iloc[i]

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

        ax2.scatter(
            df25['x'],
            df25['fR'],
            zorder=4,
            s=50,
            facecolor='#FFCCCB',
            edgecolor='black'
        )
        ax2.scatter(
            df50['x'],
            df50['fR'],
            zorder=4,
            s=70,
            facecolor='#8B0000',
            edgecolor='black'
        )
        ax2.scatter(
            df75['x'],
            df75['fR'],
            zorder=4,
            s=50,
            facecolor='#FFCCCB',
            edgecolor='black'
        )

        if dswe:
            for name, group in full_df.groupby('DSWE_level'):
                ax2.scatter(
                    group['x'],
                    group['fR'],
                    zorder=2,
                    s=30,
                )
        else:
            ax2.scatter(
                full_df['x'],
                full_df['fR'],
                zorder=2,
                s=30,
                facecolor='black',
                edgecolor='black'
            )

        if i < len(df50):
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

        ax3.scatter(
            df25['x'],
            df25['O_avg'],
            zorder=4,
            s=50,
            facecolor='#FFCCCB',
            edgecolor='black'
        )
        ax3.scatter(
            df50['x'],
            df50['O_avg'],
            zorder=4,
            s=70,
            facecolor='#8B0000',
            edgecolor='black'
        )
        ax3.scatter(
            df75['x'],
            df75['O_avg'],
            zorder=4,
            s=50,
            facecolor='#FFCCCB',
            edgecolor='black'
        )
        if dswe:
            for name, group in full_df.groupby('DSWE_level'):
                ax3.scatter(
                    group['x'],
                    group['O_avg'],
                    zorder=2,
                    s=30,
                )
        else:
            ax3.scatter(
                full_df['x'],
                full_df['O_avg'],
                zorder=2,
                s=30,
                facecolor='black',
                edgecolor='black'
            )
        ax3.scatter(
            data['x'], 
            data['O_avg'], 
            s=200, zorder=3, color='red'
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


def make_gifs(river, root, dswe=False):
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
    if dswe:
        fp_in = os.path.join(
            root, 'WaterLevel2/mask/*_mask*.tif'
        )
        fps = []
        fp_roots = glob.glob(os.path.join(root, 'WaterLevel*'))
        for root in fp_roots:
            level = root.split('/')[-1]
            fps.append(glob.glob(
                os.path.join(root, '*yearly_mobility.csv')
            )[0])
    else:
        fps = sorted(
            glob.glob(os.path.join(root, '*yearly_mobility.csv'))
        )
        fp_in = os.path.join(
            root, 'mask/*_mask*.tif'
        )

    print('Finding Stats')
    stats = get_stats(fps, stat_out)
    print('Calculating Mobility')
    get_mobility(stats, mobility_out)
    print('Making Gif')
    make_gif(fps, fp_in, fp_out, dswe=dswe)


if __name__ == '__main__':
    pass
    # root = '/Volumes/Samsung_T5/Mac/PhD/Projects/Mobility/MethodsPaper/RiverData/MeanderingRivers/Data/Indus'
    # river='PearlUpstream'
    # poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/River_Shapes/$river.gpkg"
    # gif="true"
    # out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/River_Files"
    # ocale=30

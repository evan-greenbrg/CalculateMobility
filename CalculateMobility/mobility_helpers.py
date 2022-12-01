import re
import numpy as np
import fiona
import rasterio
from rasterio.mask import mask
from rasterio import warp
from pyproj import CRS


def create_mask_shape(polygon_path, river, fps):
    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            image = fps[0]
            ds = rasterio.open(image)

            geom = rasterio.warp.transform_geom(
                src_crs=layer.crs,
                dst_crs=ds.crs,
                geom=feature['geometry'],
            )

            out_image, out_transform = mask(
                ds, [geom],
                crop=False, filled=False
            )
            out_image = out_image.astype('int64')
            out_image += 11
            out_image[np.where(out_image < 10)] = 0
            out_image[np.where(out_image > 10)] = 1

            return out_image[0, :, :]


def clean(poly, river, fps):
    polygon_name = poly.split('/')[-1].split('.')[0]
    with fiona.open(poly, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']

    images = {}
    metas = {}
    for fp in fps:
        year = re.findall(r"[0-9]{4,7}", fp)[-1]
        ds = rasterio.open(fp)

        image, tf = mask(
            ds, [geom],
            crop=False, filled=False
        )

        # Threshold
        water = image.data[0, :, :] > 0
        if not np.sum(water):
            continue

        meta = ds.meta
        meta.update(
            width=water.shape[1],
            height=water.shape[0],
            count=1,
            dtype=rasterio.int8
        )

        images[year] = water
        metas[year] = meta

        with rasterio.open(fp, "w", **meta) as dest:
            dest.write(water.astype(rasterio.int8), 1)

    return images, metas


def find_epsg(lat, long):
    '''
    Based on: https://stackoverflow.com/questions/9186496/determining-utm-zone-to-convert-from-longitude-latitude
    '''

    # Svalbard
    if (lat >= 72.0) and (lat <= 84.0):
        if (long >= 0.0)  and (long<  9.0):
            utm_number = 31
        if (long >= 9.0)  and (long < 21.0):
            utm_number = 33
        if (long >= 21.0) and (long < 33.0):
            utm_number = 35
        if (long >= 33.0) and (long < 42.0):
            utm_number = 37
    
    # Special zones for Norway
    elif (lat >= 56.0) and (lat < 64.0):
        if (long >= 0.0)  and (long <  3.0):
            utm_number = 31
        if (long >= 3.0)  and (long < 12.0):
            utm_number = 32

    if (lat > -80.0) and (lat <= 84.0):
        utm_number = int((np.floor((long + 180) / 6) % 60) + 1)

    if lat > 0:
        utm_letter = False
    else:
        utm_letter = True

    utm_zone = str(utm_number) + str(utm_letter)
    
    crs = CRS.from_dict({
        'proj': 'utm',
        'zone': utm_number,
        'south': utm_letter
    })

    return crs 


def get_scale(fp):
    # fp = '/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/files/Beni/mask/Beni_2021_01-01_12-31_mask.tif'
    ds = rasterio.open(fp)
    lon, lat = ds.xy(0, 0)
    print(lat, lon)
    if ds.crs.to_epsg() == 4326:
        epsg = find_epsg(lat, lon)
        gt, width, height = warp.calculate_default_transform(
            ds.crs,
            epsg,
            ds.width,
            ds.height,
            *ds.bounds
        )

        return gt[0]

    # This is such a crap check
    elif str(ds.crs.to_epsg())[0] == '3':
        return ds.transform[0]
    
    else:
        raise ValueError('Unrecognized EPSG')

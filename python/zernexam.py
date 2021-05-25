"""Tools for studying Zernike sky approximations
"""
from os import path
from argparse import ArgumentParser
from collections import OrderedDict
from collections import namedtuple
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import EarthLocation

from lsst.sims.skybrightness_pre import zernike

SITE = EarthLocation.of_site('Cerro Pachon')
SKY_DATA_PATH = '/data/des70.a/data/neilsen/sims_skybrightness_pre/data'
NEW_MOON_MJD = 60025.2
FULL_MOON_MJD = 60010.2
DIFFICULT_MJD = 60010.4
BANDS = ('u', 'g', 'r', 'i', 'z', 'y')
BAND_COLOR = {'u': '#56b4e9',
              'g': '#008060',
              'r': '#ff4000',
              'i': '#850000',
              'z': '#6600cc',
              'y': '#000000'}

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
# logger.info("Starting")

def compute_instant_residuals(zernike_sky, pre_sky, mjd, band):
    sky = pre_sky.sky.loc[(band, mjd)].copy()
    sky['zsky'] = zernike_sky.compute_sky(sky.alt, sky.az, mjd)
    sky.loc[sky['sky'] == 0, 'sky'] = np.nan
    sky.loc[sky['zsky'] == 0, 'zsky'] = np.nan
    sky['resid'] = sky.eval('sky-zsky')
    return sky


def compute_band_residuals(zernike_sky, pre_sky, band):
    def compute_zsky(df):
        mjd = df['mjd'].values[0]
        df['zsky'] = zernike_sky.compute_sky(df.alt, df.az, mjd)
        return df

    sky = (pre_sky
           .sky
           .query('sky>0')
           .loc[band]
           .groupby(level='mjd')
           .apply(compute_zsky))
    sky.loc[sky['sky'] == 0, 'sky'] = np.nan
    sky.loc[sky['zsky'] == 0, 'zsky'] = np.nan
    sky['resid'] = sky.eval('sky-zsky')
    return sky


def map_sky(fig, nrows, ncols, subplot_index, sky, column, title, radial_column='laea_r', colorbar=True, **kwargs):
    ax = fig.add_subplot(nrows, ncols, subplot_index, projection='polar')
    p = ax.scatter(sky.az_rad, sky[radial_column], c=sky[column], **kwargs)
    ax.set_title(title, pad=20)
    ax.set_ylim([0, np.max(sky[radial_column])])
    ax.set_ylabel('')
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(p, orientation='horizontal', ax=ax)
    return ax


#def resid_map(zernike_sky, pre_sky, mjd, band, fig=None):
#    sky = compute_instant_residuals(
#        zernike_sky, pre_sky, mjd, band)
def resid_map(sky, mjd, band, fig=None):
    sky['az_rad'] = np.radians(sky['az'])
    sky['zd'] = 90-sky['az']
    sky.query('sky>0 and zsky>0', inplace=True)
    sky['laea_r'] = 2*np.cos(0.5*(np.pi/2 + np.radians(sky.alt)))

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    vmax = np.max([sky.sky.max(), sky.zsky.max()])
    vmin = np.min([sky.sky.min(), sky.zsky.min()])

    axes = OrderedDict()

    ax = map_sky(fig, 2, 2, 1, sky, 'sky', 'pre',
                 cmap='viridis_r', vmin=vmin, vmax=vmax)
    axes['pre'] = ax

    ax = map_sky(fig, 2, 2, 2, sky, 'zsky', 'Zernike sky',
                 cmap='viridis_r', vmin=vmin, vmax=vmax)
    axes['zsky'] = ax

    vmax = np.max(np.abs(sky.resid))
    vmin = -1*vmax
    ax = map_sky(fig, 2, 2, 3, sky, 'resid', 'pre-Zernike',
                 cmap='coolwarm_r', vmin=vmin, vmax=vmax)
    axes['resid'] = ax

    ax = fig.add_subplot(2, 2, 4)
    ax.hist(sky.query('(zd < 66) and (moon_sep>10)').resid, bins=30)
    ax.set_title("pre-Zernike (masking moon)")
    axes['histogram'] = ax

    fig.suptitle(f"MJD {mjd:.3f} in {band} band", y=1)
    plt.tight_layout()
    return fig, axes


Resid2dHistReturn = namedtuple(
    'Resid2dHistReturn',
    ('fig', 'ax', 'counts', 'xedges', 'yedges', 'im'))


def resid_2dhist(sky, column, bins=50, cmap='viridis_r', norm=mpl.colors.LogNorm(), **kwargs):
    fig, ax = plt.subplots()
    counts, xedges, yedges, im = ax.hist2d(
        sky.resid, sky[column], bins=bins, cmap=cmap, norm=norm, **kwargs)
    fig.colorbar(im, ax=ax)
    return Resid2dHistReturn(fig, ax, counts, xedges, yedges, im)

def resid_2dhist2(sky, column, bins=50, cmap='viridis_r', norm=mpl.colors.LogNorm(), **kwargs):
    subskies = {'dark':  sky.query('moon_alt<0'),
                'bright': sky.query('moon_alt>=0')}

    fig, axes = plt.subplots(1, 2, figsize=(2*8, 5))

    results = []
    for subsky_name, ax in zip(subskies.keys(), axes):
        subsky = subskies[subsky_name]
        counts, xedges, yedges, im = ax.hist2d(
            subsky.resid, subsky[column], bins=bins, cmap=cmap, norm=norm, **kwargs)
        ax.set_title(subsky_name)
        fig.colorbar(im, ax=ax)
        results.append(Resid2dHistReturn(fig, ax, counts, xedges, yedges, im))

    return results

def compute_residuals_for_file_band(zern_fname,
                                    pre_npz_fname,
                                    resid_fname,
                                    band,
                                    mjd_range=[0,99999]):

    logging.info(f'Starting {band} band')

    # Find the parameters we need to instantiate a ZernikeSky,
    # and do it
    zernike_metadata = pd.read_hdf(zern_fname, "zernike_metadata")
    order = int(np.round(zernike_metadata['order']))
    max_zd = zernike_metadata['max_zd']
    logger.info(f'Instantiating ZernikeSky with order={order}, max_zd={max_zd}')
    zern_sky = zernike.ZernikeSky(order=order, max_zd=max_zd)
    logger.info('loading Zernike coefficients')
    zern_sky.load_coeffs(zern_fname, band)
    
    # Load the pre-computed healpix maps
    pre_dir, pre_npz_base = path.split(pre_npz_fname)
    pre_base = path.splitext(pre_npz_base)[0]
    logging.info('Loading pre-computed data')
    pre_sky = zernike.SkyBrightnessPreData(
        pre_base, band, pre_dir)

    logging.info('Computing residuals')
    sky_residuals = compute_band_residuals(
        zern_sky, pre_sky, band)

    logging.info('Saving residuals')
    sky_residuals.to_hdf(resid_fname, band)

    logging.info(f'Finished with {band} band')

    
def compute_full_residuals_for_file(zern_fname,
                                    pre_npz_fname,
                                    resid_fname,
                                    bands=None,
                                    **kwargs):
    if bands is None:
        bands = BANDS
        
    for band in bands:
        compute_residuals_for_file_band(zern_fname, pre_npz_fname, resid_fname, band, **kwargs)

def main(args):
    compute_full_residuals_for_file(
        args.zern_fname,
        args.pre_npz_fname,
        args.resid_fname,
        args.bands)

if __name__ == '__main__':
    parser = ArgumentParser(description="Calculate residuals")
    parser.add_argument("zern_fname", type=str, help="hdf5 file with Zernike coefficients")
    parser.add_argument("pre_npz_fname", type=str, help="npz file with pre-computed map")
    parser.add_argument("resid_fname", type=str, help="hdf5 file in which to store residuals")
    parser.add_argument("bands", type=str, nargs="?", default='ugrizy', help="bands for whiich to calculate residuals")
    args = parser.parse_args()

    main(args)
    

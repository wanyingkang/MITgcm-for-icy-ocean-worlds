import facets as fa
import sparsemap as sm
import exchange
import MITgcmutils as mu
import mymitgcmutils as mymu
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr
import string
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter


def get_dudr(expt):
    uv = expt.get_horz_velocities_at_C(expt.tstart, expt.tstop)
    u = np.mean(uv.u.values, axis=2)
    u = np.concatenate(
        (np.zeros_like(u[[0], :]), u, np.zeros_like(u[[-1], :])), axis=0)
    du = np.diff(u, axis=0)
    drc = mymu.mu.rdmds(expt.gdir + 'DRC').squeeze()
    dudr = du / drc[:, np.newaxis]
    dudr = 0.5 * (dudr[:-1, :] + dudr[1:, :])
    return dudr


def get_1byrdudtheta(expt):
    uv = expt.get_horz_velocities_at_C(expt.tstart, expt.tstop)
    u = np.mean(uv.u.values, axis=2)
    u = np.concatenate((u[:, [-1]], u, u[:, [0]]), axis=1)
    theta = uv.y.values
    dtheta = np.radians(np.diff(theta)[0])
    rdtheta = (expt.rsphere + expt.rc) * dtheta
    durdtheta = np.diff(u, axis=1) / rdtheta[:, np.newaxis]
    durdtheta = 0.5 * (durdtheta[:, :-1] + durdtheta[:, 1:])
    return durdtheta


def get_dudz(expt):
    dudr = get_dudr(expt)
    durdtheta = get_1byrdudtheta(expt)

    uv = expt.get_horz_velocities_at_C(expt.tstart, expt.tstop)
    theta = np.radians(uv.y.values)
    dudz = -np.sin(theta) * dudr + np.cos(theta) * durdtheta
    return xr.DataArray(dudz, dims=('z', 'y'), coords=dict(z=uv.z, y=uv.y))


def get_db_rdtheta(expt):
    drf = mymu.mu.rdmds(expt.gdir + 'DRF').squeeze()
    T = expt.get_temp(expt.tstart, expt.tstop)
    T = np.mean(T, axis=2)
    b = expt.g * expt.alpha * T
    b = b.values
    b = np.concatenate((b[:, [-1]], b, b[:, [0]]), axis=1)
    dtheta = np.radians(np.diff(T.y.values)[0])
    dy = (expt.rsphere + expt.rc) * dtheta
    dbdy = np.diff(b, axis=1) / dy[:, np.newaxis]
    dbdy = 0.5 * (dbdy[:, :-1] + dbdy[:, 1:])
    dbdy = dbdy / 2 / expt.omega
    return xr.DataArray(dbdy, dims=('z', 'y'), coords=dict(z=T.z, y=T.y))


def get_2omegaucostheta_g(expt):
    uv = expt.get_horz_velocities_at_C(expt.tstart, expt.tstop)
    u = np.mean(uv.u.values, axis=2)
    retval = 2 * expt.omega * u * np.cos(np.radians(uv.y.values)) / expt.g
    return xr.DataArray(retval, dims=('z', 'y'), coords=dict(z=uv.z, y=uv.y))


def get_thermal_wind_balance(expt):
    pass

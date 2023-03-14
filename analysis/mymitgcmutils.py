import facets as fa
import sparsemap as sm
import exchange
import MITgcmutils as mu
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr
import string
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter


def get_iters_from_tstamps(filebase, tstart, tstop):
    """
    Returns a list of iterations that span the
    timestamps `tstart` and `tstop` which should always be specified
    in seconds.
    """
    iters_to_read = []
    iters = mu.mds.scanforfiles(filebase)
    for i in iters:
        fname = glob.glob(filebase + ".{:010d}.meta".format(i))[0]
        file_t_start, file_t_stop = mu.mds.readmeta(fname)[4]
        if file_t_start < tstop and file_t_stop > tstart:
            iters_to_read.append(i)
    return iters_to_read


def rdmds_by_tstamps(filebase, tstart, tstop, **kwargs):
    """
    Returns the MITgcm output data between timestamps `tstart` and
    `tstop`.
    """
    iters = get_iters_from_tstamps(filebase, tstart, tstop)
    return mu.rdmds(filebase, iters, **kwargs)


class Experiment():
    def __init__(self, **kwargs):
        self.cs510 = kwargs.get('cs510', False)
        self.facedims = kwargs.get("facedims", 6 * [96, 96])
        self.gdir = kwargs.get("gdir")
        self.ddir = kwargs.get("ddir")
        self.rdir = kwargs.get("rdir")
        self.idir = kwargs.get("idir")
        self.grid = fa.MITGrid(self.rdir, dims=self.facedims)
        self.rc = (mu.rdmds(self.gdir + 'RC')).squeeze()
        self.rf = (mu.rdmds(self.gdir + 'RF')).squeeze()

        self.rsphere = kwargs.get("rsphere", 1550000)
        self.radius = kwargs.get("radius", 1500000)
        self.tstart = kwargs.get("tstart", 20)
        self.tstop = kwargs.get("tstop", 30)
        self.Q = kwargs.get('Q', 10)
        self.g = kwargs.get('g', 1.3)
        self.alpha = kwargs.get('alpha', 3.e-4)
        self.cp = kwargs.get('cp', 4000)
        self.D = kwargs.get('D', 100000)
        self.rho = kwargs.get('rho', 1000)
        self.omega = kwargs.get('omega', 2 * np.pi / 3e5)
        self.av = kwargs.get('av', 9)
        if self.Q > 1:
            Qstr = '{}W'.format(int(self.Q))
        else:
            Qstr = '{}mW'.format(int(self.Q * 1000))
        self.name = 'H = {:3d}km, Q = {}'.format(int(self.D / 1000), Qstr)

        self.B = self.g * self.alpha * self.Q / self.rho / self.cp
        self.rostar = (self.B * (2 * self.omega)**-3)**(1 / 2) / self.D
        self.E = self.av / (2 * self.omega * self.D**2)
        self.lrot = (self.B * (2 * self.omega)**-3)**(1 / 2)
        self.urot = (self.B * (2 * self.omega)**-1)**(1 / 2)
        self.unonrot = (self.B * self.D)**(1 / 3)
        self.ucone = 2 * self.omega * self.D * self.rostar**(1 / 2)
        self.lp = self.D * np.sqrt(self.rostar)
        self.gprime = np.sqrt(self.B * 2 * self.omega) / self.alpha / self.g
        self.gprimenonrot = (self.B**2 / self.D)**(1 / 3) / self.alpha / self.g

        self.wmax = kwargs.get('wmax', None)
        self.robustw = kwargs.get('robustw', False)

        self.c = {'100mW': 'r', '10W': 'b', '1000W': 'k', '10000W': 'g'}[Qstr]
        self.style = {29000: '--', 100000: '-.', 500000: '-'}[int(self.D)]
        self.mstyle = {29000: 'o', 100000: 's', 500000: '*'}[int(self.D)]

    def save_netcdf(self, istart, istop, dt, dtmodel):
        itrs = list(np.arange(istart, istop, int(dt / dtmodel)))
        print(itrs)
        u = mu.rdmds(self.idir + 'U', itrs)
        v = mu.rdmds(self.idir + 'V', itrs)
        if len(itrs) == 1:
            uv = self.get_uv_1deg(u, v)
        else:
            uv = self.get_uv_1deg_with_time(u, v)
        w = mu.rdmds(self.idir + 'W', itrs)
        w = self.get_T_1deg(w)
        T = mu.rdmds(self.idir + 'T', itrs)
        T = self.get_T_1deg(T)
        zeta = mu.rdmds(self.idir + 'vortDiag', itrs)
        zeta = self.get_T_1deg(zeta)
        ds = xr.Dataset(dict(u=uv.u, v=uv.v, w=w, T=T, zeta=zeta))
        if len(itrs) > 1:
            ds['t'].values *= dt
        ds.to_netcdf(
            self.idir + 'uvwT_{}'.format(istart) + '_{}.nc'.format(istop))

    def get_tprime(self, lats, latn):
        T = mu.rdmds(self.idir + "T", np.Inf)
        T = self.get_T_1deg(T)
        T = T.sel(y=slice(lats, latn))
        self.Tprime = np.sqrt(
            np.mean(T**2, axis=(1, 2)) - np.mean(T, axis=(
                1, 2)) * np.mean(T, axis=(1, 2)))
        return self.Tprime.values

    def get_uprime(self, lats, latn):
        u = mu.rdmds(self.idir + "U", np.Inf)
        v = mu.rdmds(self.idir + "V", np.Inf)
        uv = self.get_uv_1deg_with_time(u, v).sel(y=slice(lats, latn))
        u_meanofsquares = np.mean(uv.u**2 + uv.v**2, axis=(1, 2))
        u_mean = np.mean(uv.u, axis=(1, 2))
        v_mean = np.mean(uv.v, axis=(1, 2))
        u_squareofmeans = u_mean**2 + v_mean**2
        self.Uprime = np.sqrt(u_meanofsquares - u_squareofmeans)
        return self.Uprime.values

    def scaling_params(self, func=np.mean, lats=-15, latn=15):
        return [
            self.name, self.D / 1000, self.Q, self.B, self.lrot / 1000,
            self.urot, self.ucone, self.lp / 1000, self.rostar,
            np.sqrt(self.rostar), self.gprime,
            func(self.get_uprime(lats, latn)),
            func(self.get_tprime(lats, latn))
        ]

    def midlatitude_scales(self, iteration, lats=35, latn=45, z=10):
        u = mu.rdmds(self.idir + "U", iteration)
        v = mu.rdmds(self.idir + "V", iteration)
        uv = self.get_uv_1deg(u, v)
        u = uv.u.sel(y=slice(lats, latn)).isel(z=z).mean('y')
        n = len(u)
        ps = np.fft.rfft(u.values)
        # sample_rate = np.radians(1) * self.radius / 1000 * np.cos(
        #     np.radians((lats + latn) / 2))
        wl = np.fft.rfftfreq(n, 1)
        return (1 / wl, ps)

    def get_expt_resolution(self):
        if self.cs510:
            M = sm.frommap(
                '/home/bire/europa_analysis_scripts/cs510_to_llsixthdeg_conservative_stackx.i4i4f8_2160x1080x1560600.map'
            )
        else:
            M = sm.frommap(
                '/home/bire/europa_analysis_scripts/cs96_to_ll1deg_conservative_oldmap.i4i4f8_360x180x55296.map'
            )
        return M

    def get_uv_1deg(self, u, v, cs510=False):
        exch = exchange.cs()
        u = fa.fromglobal(u, dims=self.facedims, extrau=1)
        v = fa.fromglobal(v, dims=self.facedims, extrav=1)
        exch.uv(u, v)
        v = 0.5 * (v[:, :-1] + v[:, 1:])
        u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        anglecs = self.grid.anglecs
        anglesn = self.grid.anglesn
        ull = anglecs * u - anglesn * v
        vll = anglesn * u + anglecs * v
        M = self.get_expt_resolution()
        u1deg = M(ull.toglobal())
        v1deg = M(vll.toglobal())
        if cs510 or self.cs510:
            x1deg = np.arange(-180 * 6, 180 * 6) / 6
            y1deg = np.arange(-90 * 6, 90 * 6) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        u1deg = xr.DataArray(
            u1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))
        v1deg = xr.DataArray(
            v1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))
        return xr.Dataset(dict(u=u1deg, v=v1deg))

    def get_uv_1deg_with_time(self, u, v):
        exch = exchange.cs()
        u = fa.fromglobal(u, dims=self.facedims, extrau=1)
        v = fa.fromglobal(v, dims=self.facedims, extrav=1)
        exch.uv(u, v)
        v = 0.5 * (v[:, :, :-1] + v[:, :, 1:])
        u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
        anglecs = self.grid.anglecs
        anglesn = self.grid.anglesn
        ull = anglecs * u - anglesn * v
        vll = anglesn * u + anglecs * v
        M = self.get_expt_resolution()
        u1deg = M(ull.toglobal())
        v1deg = M(vll.toglobal())
        if self.cs510:
            x1deg = np.arange(-180 * 6, 180 * 6) / 6
            y1deg = np.arange(-90 * 6, 90 * 6) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        t = np.arange(np.shape(u1deg)[0])
        u1deg = xr.DataArray(
            u1deg,
            dims=('t', 'z', 'y', 'x'),
            coords=dict(t=t, z=z, y=y1deg, x=x1deg))
        v1deg = xr.DataArray(
            v1deg,
            dims=('t', 'z', 'y', 'x'),
            coords=dict(t=t, z=z, y=y1deg, x=x1deg))
        return xr.Dataset(dict(u=u1deg, v=v1deg))

    def get_horz_velocities_at_C(self, tstart, tstop, **kwargs):
        u = rdmds_by_tstamps(
            self.ddir + "dynDiag", tstart, tstop, rec=0, **kwargs)
        v = rdmds_by_tstamps(
            self.ddir + "dynDiag", tstart, tstop, rec=1, **kwargs)
        if u.ndim == 4 and v.ndim == 4:
            tmean = kwargs.get('tmean', True)
            if tmean:
                u = np.mean(u, axis=0)
                v = np.mean(v, axis=0)

        return self.get_uv_1deg(u, v)

    def get_vert_vel(self, tstart, tstop, **kwargs):
        w = rdmds_by_tstamps(self.ddir + "dynDiag", tstart, tstop, rec=2)
        if w.ndim == 4:
            tmean = kwargs.get('tmean', True)
            if tmean:
                w = np.mean(w, axis=0)
        M = self.get_expt_resolution()
        w1deg = M(w)
        if self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        return xr.DataArray(
            w1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))

    def get_T_1deg(self, T, cs510=False):
        M = self.get_expt_resolution()
        T1deg = M(T)
        if cs510 or self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        if T1deg.ndim == 4:
            t = np.arange(np.shape(T1deg)[0])
            return xr.DataArray(
                T1deg,
                dims=('t', 'z', 'y', 'x'),
                coords=dict(t=t, z=z, y=y1deg, x=x1deg))
        else:
            return xr.DataArray(
                T1deg,
                dims=('z', 'y', 'x'),
                coords=dict(z=z, y=y1deg, x=x1deg))

    def get_vert_vel(self, tstart, tstop, **kwargs):
        w = rdmds_by_tstamps(self.ddir + "dynDiag", tstart, tstop, rec=2)
        if w.ndim == 4:
            tmean = kwargs.get('tmean', True)
            if tmean:
                w = np.mean(w, axis=0)
        M = self.get_expt_resolution()
        w1deg = M(w)
        if self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        return xr.DataArray(
            w1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))

    def get_vort_1deg(self, vort, cs510=False):
        M = self.get_expt_resolution()
        vort1deg = M(vort)
        if cs510 or self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        return xr.DataArray(
            vort1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))

    def get_eta_1deg(self, T):
        M = self.get_expt_resolution()
        T1deg = M(T)
        if self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        return xr.DataArray(
            T1deg, dims=('T', 'y', 'x'), coords=dict(y=y1deg, x=x1deg))

    def get_temp(self, tstart, tstop, **kwargs):
        T = rdmds_by_tstamps(self.ddir + "dynDiag", tstart, tstop, rec=4)
        if T.ndim == 4:
            tmean = kwargs.get('tmean', True)
            if tmean:
                T = np.mean(T, axis=0)
        M = self.get_expt_resolution()
        T1deg = M(T)
        if self.cs510:
            x1deg = np.arange(-6 * 180, 6 * 180) / 6
            y1deg = np.arange(-6 * 90, 6 * 90) / 6
        else:
            x1deg = np.arange(-180, 180)
            y1deg = np.arange(-90, 90)
        z = self.rc
        return xr.DataArray(
            T1deg, dims=('z', 'y', 'x'), coords=dict(z=z, y=y1deg, x=x1deg))

    def get_psi(self, tstart, tstop, **kwargs):
        uv = self.get_horz_velocities_at_C(tstart, tstop, **kwargs)
        xx, yy = np.meshgrid(uv.x.values, uv.y.values)
        R = self.radius
        rf = self.rf
        dz = np.diff(rf, axis=0)[:, np.newaxis, np.newaxis]
        dx = np.radians(np.diff(xx)[:, 0]) * R * np.cos(
            np.radians(uv.y.values))
        dx = dx[:, np.newaxis]
        psi = np.zeros((rf.size, uv.y.size))
        psi[1:, :] = np.sum(np.cumsum(uv.v.values * dz, axis=0) * dx, axis=2)
        psi = xr.DataArray(
            psi, dims=('z', 'y'), coords=dict(z=rf, y=uv.y.values))
        return psi

    def monitor(self, variable, lines_after_each):
        files = glob.glob(self.ddir + "dynStDiag.*.txt")
        monitor_dict = dict()
        for file_ in files:
            with open(file_, 'r') as f:
                line = f.readline()
                while line != '':
                    if "field : " + variable in line:
                        iter_ = float(line.split()[6])
                        line = f.readline()
                        monitor_dict[iter_] = []
                        for i in range(lines_after_each):
                            line = [
                                float(sanitize(f))
                                for f in f.readline().split()
                            ]
                            monitor_dict[iter_].append(line)
                    line = f.readline()
        return monitor_dict


def sanitize(string):
    string = string[::-1]
    if '-' in string:
        locE = string.find('-') + 1
        if string[locE] is not "E":
            string = string[:locE] + 'E' + string[locE:]
    return string[::-1]


def get_mean_N2(expt, tstart, tstop):
    T = expt.get_temp(tstart, tstop)
    alpha = expt.alpha
    rho = -alpha * T
    g = expt.g
    rho0 = expt.rho
    drc = np.diff(expt.rc)
    N2 = -g / rho0 * rho.diff('z') / drc[:, np.newaxis, np.newaxis]
    return N2


def get_omega_components(omega, y, z, yskip, zskip):
    y = y[::yskip]
    z = z[::zskip]
    yy, zz = np.meshgrid(y, z)
    omega = omega * np.ones_like(yy)
    omegay = omega * np.cos(np.radians(y))
    omegaz = omega * np.sin(np.radians(y))
    return y, z, omegay, omegaz


f23 = Experiment(
    name='F23',
    gdir='/net/fs09/d2/bire/mitgcm/europa/run_13/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/run_13/',
    idir='/net/fs09/d2/bire/mitgcm/europa/run_13/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/run_13/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=50,
    tstop=60)

f26 = Experiment(
    name='F26',
    gdir='/net/fs09/d2/bire/mitgcm/europa/run_17/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/run_17/',
    idir='/net/fs09/d2/bire/mitgcm/europa/run_17/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/run_17/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    Q=10000,
    wmax=5e-3)

f27 = Experiment(
    name='F27',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_18/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_18/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_18/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_18/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    Q=1000,
    wmax=2.5e-3)

f28 = Experiment(
    name='F28',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_19/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_19/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_19/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_19/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=50,
    tstop=60,
    Q=0.1)

f29 = Experiment(
    name='F29',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_20/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_20/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_20/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_20/cs96_dxC3_dXYa.face{0:03d}.bin',
    D=500000,
    rsphere=1750000)

f30 = Experiment(
    name='F30',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_21/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_21/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_21/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_21/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    D=500000,
    Q=10000,
    rsphere=1750000)

f31 = Experiment(
    name='F31',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_22/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_22/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_22/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_22/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    D=500000,
    Q=1000,
    rsphere=1750000)

f32 = Experiment(
    name='F32',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_23/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_23/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_23/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_23/cs96_dxC3_dXYa.face{0:03d}.bin',
    D=500000,
    Q=0.1,
    rsphere=1750000)

f33 = Experiment(
    name='F33',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_24/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_24/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_24/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_24/cs96_dxC3_dXYa.face{0:03d}.bin',
    D=29000,
    rsphere=1514500)

f34 = Experiment(
    name='F34',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_25/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_25/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_25/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_25/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    D=29000,
    Q=10000,
    wmax=5e-3,
    robustw=True,
    rsphere=1514500)

f35 = Experiment(
    name='F35',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_26/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_26/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_26/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_26/cs96_dxC3_dXYa.face{0:03d}.bin',
    tstart=0,
    tstop=6,
    D=29000,
    Q=1000,
    wmax=5e-3,
    rsphere=1514500)

f36 = Experiment(
    name='F36',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/cs96_dxC3_dXYa.face{0:03d}.bin',
    D=29000,
    Q=0.1,
    rsphere=1514500)

f37 = Experiment(
    name='F37',
    gdir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    idir='/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/',
    rdir=
    '/net/fs09/d2/bire/mitgcm/europa/finalcs96/run_27/cs96_dxC3_dXYa.face{0:03d}.bin',
    D=29000)

f39 = Experiment(
    gdir='/net/fs09/d2/bire/mitgcm/europa/run_30/',
    ddir='/net/fs09/d2/bire/mitgcm/europa/run_30/',
    idir='/net/fs09/d2/bire/mitgcm/europa/run_30/',
    rdir='/net/fs09/d2/bire/mitgcm/europa/run_30/cs96_dxC3_dXYa.face{0:03d}.bin'
)

expts = [f36, f33, f35, f34, f28, f23, f27, f26, f32, f29, f31, f30]
exptsno100mw = [f33, f35, f34, f23, f27, f26, f29, f31, f30]


def expt_grid(expts, plot_this_variable, ncols=4, z=10, figsize=(12, 8)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols),
        ncols,
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.Orthographic(central_latitude=23)))
    for axc, expt in zip(ax.ravel(), expts):
        read_variable = dict(
            U=expt.get_horz_velocities_at_C,
            W=expt.get_vert_vel,
            T=expt.get_temp)
        cmap = dict(U='RdBu_r', W='RdBu_r', T='viridis')[plot_this_variable]
        var = read_variable[plot_this_variable](expt.tstart, expt.tstop)
        if plot_this_variable == "U":
            var = var.u
        elif plot_this_variable == "W":
            var.values = gaussian_filter(
                var.values, [0, 1.5, 1.5],
                order=[0, 0, 0],
                mode=['reflect', 'mirror', 'mirror'])
        if plot_this_variable == "W" and expt.wmax is not None:
            im = var.isel(z=z).plot(
                vmax=expt.wmax,
                vmin=-expt.wmax,
                add_labels=False,
                ax=axc,
                cmap=cmap,
                transform=ccrs.PlateCarree())
        else:
            im = var.isel(z=z).plot(
                add_labels=False,
                ax=axc,
                cmap=cmap,
                transform=ccrs.PlateCarree())
        im.colorbar.formatter.set_powerlimits((-1, 1))
        im.colorbar.update_ticks()
        gl = axc.gridlines(
            crs=ccrs.PlateCarree(), linewidth=0.5, color='k', alpha=0.5)
        axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.1,
            -0.1,
            'RoC = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
        axc.text(
            0.1,
            -0.2,
            'Ek = {:1.2e}'.format(expt.E),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
    return fig


def expt_grid_u_polar(expts, ncols=4, figsize=(12, 9), polar=True):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols), ncols, figsize=figsize, subplot_kw=dict(polar=polar))
    rorigin = {100000: -500, 29000: -200, 500000: -1750}
    for axc, expt in zip(ax.ravel(), expts):
        uv = expt.get_horz_velocities_at_C(expt.tstart, expt.tstop)
        if polar:
            axc.set_thetamin(-90)
            axc.set_thetamax(90)
            axc.set_rorigin(rorigin[expt.D])
            axc.set_rticks([])
            y1 = np.radians(uv.y)
        u = np.mean(uv.u.values, axis=2)
        vmax = np.amax(np.fabs(u))
        im = axc.pcolormesh(
            y1, uv.z / 1000, u, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()

        #axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.0,
            -0.15,
            r'Ro$^{*}$' + ' = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=18,
            family='monospace')
        # axc.text(
        #     0.1,
        #     -0.2,
        #     'Ek = {:1.2e}'.format(expt.E),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
    return fig


def expt_grid_psi_polar(expts, ncols=4, figsize=(12, 9), polar=True):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols), ncols, figsize=figsize, subplot_kw=dict(polar=polar))
    rorigin = {100000: -500, 29000: -200, 500000: -1750}
    for axc, expt in zip(ax.ravel(), expts):
        psi = expt.get_psi(expt.tstart, expt.tstop)
        if polar:
            axc.set_thetamin(-90)
            axc.set_thetamax(90)
            axc.set_rorigin(rorigin[expt.D])
            axc.set_rticks([])
            y2 = np.radians(psi.y)
        vmax = np.amax(np.fabs(psi)) * 1e-6
        im = axc.pcolormesh(
            y2,
            psi.z / 1000,
            -psi.values * 1e-6,
            vmin=-vmax,
            vmax=vmax,
            cmap='RdBu_r')
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()

        axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.1,
            -0.1,
            'RoC = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
        axc.text(
            0.1,
            -0.2,
            'Ek = {:1.2e}'.format(expt.E),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
    return fig


def expt_grid_w_polar(expts, ncols=4, figsize=(12, 9), polar=True):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols), ncols, figsize=figsize, subplot_kw=dict(polar=polar))
    rorigin = {100000: -500, 29000: -200, 500000: -1750}
    for axc, expt in zip(ax.ravel(), expts):
        wda = expt.get_vert_vel(expt.tstart, expt.tstop)
        if polar:
            axc.set_thetamin(-90)
            axc.set_thetamax(90)
            axc.set_rorigin(rorigin[expt.D])
            axc.set_rticks([])
            y1 = np.radians(wda.y)
        w = np.mean(wda.values, axis=2)
        if expt.robustw:
            vmax = np.nanpercentile(np.fabs(w), 95)
        else:
            vmax = np.amax(np.fabs(w))
        im = axc.pcolormesh(
            y1, wda.z / 1000, w, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()

        axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.1,
            -0.1,
            'RoC = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
        axc.text(
            0.1,
            -0.2,
            'Ek = {:1.2e}'.format(expt.E),
            transform=axc.transAxes,
            fontsize=12,
            family='monospace')
    return fig


def expt_grid_T_polar(expts, ncols=4, figsize=(12, 9), polar=True, perc=90):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols), ncols, figsize=figsize, subplot_kw=dict(polar=polar))
    rorigin = {100000: -500, 29000: -200, 500000: -1750}
    for axc, expt in zip(ax.ravel(), expts):
        Tda = expt.get_temp(expt.tstart, expt.tstop)
        if polar:
            axc.set_thetamin(-90)
            axc.set_thetamax(90)
            axc.set_rorigin(rorigin[expt.D])
            axc.set_rticks([])
            y1 = np.radians(Tda.y)
        T = np.mean(Tda.values, axis=2)
        vmin, vmax = np.nanpercentile(T, (100 - perc, perc))
        im = axc.pcolormesh(
            y1, Tda.z / 1000, T, vmin=vmin, vmax=vmax, cmap='viridis')
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()

        # axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.0,
            -0.15,
            r'Ro$^{*}$' + ' = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=18,
            family='monospace')
        # axc.text(
        #     0.1,
        #     -0.2,
        #     'Ek = {:1.2e}'.format(expt.E),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
    return fig


def myfmt(x, pos):
    return '{0:.2e}'.format(x)


def expt_grid_instantaneous(expts,
                            plot_this_variable,
                            iteration,
                            ncols=4,
                            z=10,
                            figsize=(12, 9),
                            ax=None):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    if ax is None:
        fig, ax = plt.subplots(
            int(nax / ncols),
            ncols,
            figsize=figsize,
            subplot_kw=dict(projection=ccrs.Orthographic(central_latitude=23)))
        ax = ax.ravel()
    else:
        assert len(ax) == len(expts)
    for axc, expt in zip(ax, expts):
        if plot_this_variable is not "U":
            var = mu.rdmds(expt.idir + plot_this_variable, iteration)
            var = expt.get_T_1deg(var)
        else:
            u = mu.rdmds(expt.idir + "U", iteration)
            v = mu.rdmds(expt.idir + "V", iteration)
            uv = expt.get_uv_1deg(u, v)
            var = uv.u
        vmax = max(1e-8, np.amax(np.fabs(var)))
        cmap = dict(U='RdBu_r', W='RdBu_r', T='viridis')[plot_this_variable]
        if plot_this_variable is not "T":
            im = var.isel(z=z).plot(
                add_labels=False,
                ax=axc,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                vmax=vmax,
                vmin=-vmax)
        else:
            vmin, vmax = np.percentile(var.isel(z=z), (0.2, 99.8))
            vmax = max(1e-4, vmax)
            vmin = max(1e-4, vmin)
            im = var.isel(z=z).plot(
                add_labels=False,
                ax=axc,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                transform=ccrs.PlateCarree())
        im.colorbar.formatter.set_powerlimits((0, 0))
        im.colorbar.update_ticks()
        gl = axc.gridlines(
            crs=ccrs.PlateCarree(), linewidth=0.5, color='k', alpha=0.5)
        #axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.0,
            -0.15,
            r'Ro$^{*}$' + ' = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=18,
            family='monospace')
        # axc.text(
        #     0.1,
        #     -0.2,
        #     'Ek = {:1.2e}'.format(expt.E),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
    return axc.figure


def expt_grid_powerspectrums(expts,
                             iteration,
                             lats=35,
                             latn=45,
                             ncols=4,
                             z=10,
                             figsize=(12, 9)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(int(nax / ncols), ncols, figsize=figsize)
    for axc, expt in zip(ax.ravel(), expts):
        wl, ps = expt.midlatitude_scales(iteration, lats=lats, latn=latn, z=z)
        axc.semilogx(wl, np.abs(ps)**2, 'k-')
        axc.grid()
        axc.set_title('(' + next(labs) + r') ' + expt.name)
        # axc.text(
        #     0.1,
        #     -0.1,
        #     'RoC = {:1.2e}'.format(expt.rostar),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
        # axc.text(
        #     0.1,
        #     -0.2,
        #     'Ek = {:1.2e}'.format(expt.E),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
    return fig


def expt_grid_meanvelocity(
        expts,
        iteration,
        # ncols=4,
        lats=45,
        latn=50,
        func=np.mean,
        sharex=False,
        figsize=(9, 4)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    # fig, ax = plt.subplots(int(nax / ncols), ncols, figsize=figsize)
    fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=sharex, sharey=True)
    meanvels = []
    ucones = []
    urots = []
    cols = []
    styledict = {29000: '--', 100000: '-.', 500000: '-'}
    coldict = {0.1: 'r', 10: 'b', 1000: 'k', 10000: 'g'}
    for expt in expts:
        u = mu.rdmds(expt.idir + "U", iteration)
        v = mu.rdmds(expt.idir + "V", iteration)
        uv = expt.get_uv_1deg(u, v).sel(y=slice(lats, latn))
        u_meanofsquares = np.mean(uv.u**2 + uv.v**2, axis=(1, 2))
        u_mean = np.mean(uv.u, axis=(1, 2))
        v_mean = np.mean(uv.v, axis=(1, 2))
        u_squareofmeans = u_mean**2 + v_mean**2
        u_prime = np.sqrt(u_meanofsquares - u_squareofmeans)

        z = u_prime.z / expt.D
        ls = styledict[expt.D]
        cs = coldict[expt.Q]
        ax[0].plot(
            u_prime.values, u_prime.z / expt.D, ls=ls, c=cs, label=expt.name)
        ax[1].plot(
            u_prime.values / expt.ucone,
            u_prime.z / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[2].plot(
            u_prime.values / expt.unonrot,
            u_prime.z / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[3].plot(
            u_prime.values / expt.urot,
            u_prime.z / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        meanvels.append(func(u_prime))
        urots.append(expt.urot)
        ucones.append(expt.ucone)
        #cols.append(coldict[expt.D])
        cols.append(coldict[expt.Q])
    return fig, meanvels, urots, ucones, cols


def expt_grid_rmst(
        expts,
        iteration,
        # ncols=4,
        lats=45,
        latn=50,
        func=np.mean,
        figsize=(9, 4)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    # fig, ax = plt.subplots(int(nax / ncols), ncols, figsize=figsize)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    meanTs = []
    gprimes = []
    cols = []
    styledict = {29000: '--', 100000: '-.', 500000: '-'}
    coldict = {0.1: 'r', 10: 'b', 1000: 'k', 10000: 'g'}
    for expt in expts:
        T = mu.rdmds(expt.idir + "T", iteration)
        T = expt.get_T_1deg(T)
        T = T.sel(y=slice(lats, latn))
        Trms = np.sqrt(
            np.mean(T**2, axis=(1, 2)) - np.mean(T, axis=(
                1, 2)) * np.mean(T, axis=(1, 2)))
        ls = styledict[expt.D]
        cs = coldict[expt.Q]
        z = Trms.z / expt.D
        zlims = (z > -0.9) & (z < -0.1)
        ax[0].plot(
            Trms.values[zlims],
            Trms.z[zlims] / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        #ax[1].plot(urms.values / expt.urot, urms.z / expt.D,ls=ls,c=cs, label=expt.name)
        ax[1].plot(
            Trms.values[zlims] / expt.gprime,
            Trms.z[zlims] / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[2].plot(
            Trms.values[zlims] / expt.gprimenonrot,
            Trms.z[zlims] / expt.D,
            ls=ls,
            c=cs,
            label=expt.name)
        meanTs.append(func(Trms))
        gprimes.append(expt.gprime)
        #cols.append(coldict[expt.D])
        cols.append(coldict[expt.Q])
    return fig, meanTs, gprimes, cols


def expt_grid_vort(expts, iteration, ncols=4, perc=99, z=10, figsize=(12, 9)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(
        int(nax / ncols),
        ncols,
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.Orthographic(central_latitude=23)))
    for axc, expt in zip(ax.ravel(), expts):
        vort = mu.rdmds(expt.idir + 'vortDiag', iteration)
        vort = expt.get_vort_1deg(vort)
        vmax = np.nanpercentile(np.fabs(vort), perc)
        vmax = max(1e-7, vmax)
        im = axc.pcolormesh(
            vort.x,
            vort.y,
            vort.isel(z=z).values.squeeze(),
            vmin=-vmax,
            vmax=vmax,
            cmap='RdBu_r',
            transform=ccrs.PlateCarree())
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()

        # axc.set_title('(' + next(labs) + r') ' + expt.name)
        axc.text(
            0.0,
            -0.15,
            r'Ro$^{*}$' + ' = {:1.2e}'.format(expt.rostar),
            transform=axc.transAxes,
            fontsize=18,
            family='monospace')
        # axc.text(
        #     0.1,
        #     -0.2,
        #     'Ek = {:1.2e}'.format(expt.E),
        #     transform=axc.transAxes,
        #     fontsize=12,
        #     family='monospace')
    return fig

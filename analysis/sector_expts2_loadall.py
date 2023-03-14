import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mu
import cartopy.crs as ccrs
import string
import xarray as xr
from scipy.interpolate import griddata
from os import path
import re

class Experiment():
    def __init__(self, **kwargs):
        self.base_dir = kwargs.get('base_dir',
                                   "./data_")
        base_dir = self.base_dir
        self.name = kwargs.get("name",None)
        self.path = base_dir + self.name + "/"
        runpath=''
        if not path.exists(self.path+"XC.data"):
            runpath='run0/'
        self.xc = mu.rdmds(self.path+runpath + "XC").squeeze()
        self.yc = mu.rdmds(self.path+runpath + "YC").squeeze()
        self.xg = mu.rdmds(self.path+runpath + "XG").squeeze()
        self.yg = mu.rdmds(self.path+runpath + "YG").squeeze()
        self.zc = mu.rdmds(self.path+runpath + "RC").squeeze()
        self.zf = mu.rdmds(self.path+runpath + "RF").squeeze()
        self.dxc = mu.rdmds(self.path+runpath + "DXC").squeeze()
        self.dyc = mu.rdmds(self.path+runpath + "DYC").squeeze()
        self.dxg = mu.rdmds(self.path+runpath + "DXG").squeeze()
        self.dyg = mu.rdmds(self.path+runpath + "DYG").squeeze()
        self.rac = mu.rdmds(self.path+runpath + "RAC").squeeze()
        self.ras = mu.rdmds(self.path+runpath + "RAS").squeeze()
        self.raw = mu.rdmds(self.path+runpath + "RAW").squeeze()
        self.raz = mu.rdmds(self.path+runpath + "RAZ").squeeze()
        hfacc = mu.rdmds(self.path+runpath + "hFacC").squeeze()
        self.hfacc = xr.DataArray( hfacc, dims=('z', 'lat', 'lon'),
            coords=dict(z=self.zc, lat=self.yc[:, 0], lon=self.xc[0, :]))
        drf = mu.rdmds(self.path+runpath + "DRF").squeeze()
        self.drf=xr.DataArray(drf,dims=('z'),coords=dict(z=self.zc))
        self.mask=self.hfacc*0+1
        self.mask=self.mask.where(self.hfacc>0.0)

        self.cp = 4000
        self.beta=0.0
        self.topo=False
        self.Dfig = kwargs.get('Dfig', 500000)  # Used only for polar plots
        with open(self.path + "data") as namelist:
            for line in namelist:
                if 'dumpFreq=' in line:
                    self.dumpfreq = float(re.search('=(.*),',line).group(1))
                if 'gravity=' in line:
                    self.g = float(re.search('=(.*),',line).group(1))
                if 'rSphere=' in line:
                    self.a = float(re.search('=(.*),',line).group(1))
                if 'tAlpha=' in line:
                    self.alpha = float(re.search('=(.*),',line).group(1))
                if 'sBeta=' in line:
                    self.beta = float(re.search('=(.*),',line).group(1))
                if 'HeatCapacity_Cp=' in line:
                    try:
                        self.cp = float(re.search('=(.*),',line).group(1))
                    except:
                        self.cp=4000
                if 'rSphere=' in line:
                    self.rsphere=float(re.search('=(.*),',line).group(1))
                if 'rhoNil=' in line:
                    self.rhoref=float(re.search('=(.*),',line).group(1))
                if 'viscAh=' in line:
                    self.ah=float(re.search('=(.*),',line).group(1))
                if 'viscAr=' in line:
                    self.av=float(re.search('=(.*),',line).group(1))
                if 'diffKhT=' in line:
                    self.difh=float(re.search('=(.*),',line).group(1))
                if 'diffKrT=' in line:
                    self.difv=float(re.search('=(.*),',line).group(1))
                if 'rotationPeriod=' in line:
                    self.omega=2*np.pi/float(re.search('=(.*),',line).group(1))
                if 'tRef=' in line:
                    try:
                        self.Tref=float(re.search('\*(.*),',line).group(1))
                    except:
                        tmp=re.search('\*(.*),',line).group(1)
                        self.Tref=float(re.search('(.*),',tmp).group(1))
                if 'sRef=' in line:
                    self.Sref=float(re.search('\*(.*),',line).group(1))
                if 'deltaT=' in line:
                    self.dt=float(re.search('=(.*),',line).group(1))
                if 'nIter0=' in line:
                    self.itermax=int(re.search('=(.*),',line).group(1))
                if 'nTimeSteps=' in line:
                    self.eachiter=int(re.search('=(.*),',line).group(1))
                if 'dumpFreq=' in line:
                    self.dtoutput=round(float(re.search('=(.*),',line).group(1)))
                    self.iteroutput=self.dtoutput/self.dt
                if 'geothermalFile' in line:
                    self.Q=float(re.search('bottom_(.*)mW',line).group(1))/1e3


        if path.exists(self.path + "data.shelfice"): 
            with open(self.path + "data.shelfice") as shelficedata:
                for line in shelficedata:
                    if 'rhoShelfIce=' in line:
                        self.rhoice=float(re.search('=(.*),',line).group(1))
                    if 'SHELFICEthetaSurface=' in line:
                        self.Ts=float(re.search('=(.*),',line).group(1))

        if path.exists(self.path + "gendata.m"):
            with open(self.path + "gendata.m") as gendata:
                for line in gendata:
                    if 'Htot=' in line:
                        self.Htot=float(re.search('=(.*);',line).group(1))
                    if 'kappa0=' in line:
                        self.kappa0=float(re.search('=(.*);',line).group(1))
                        self.fluxfac=self.kappa0/651
                    if 'Hice0=' in line:
                        self.Hice=float(line[line.find('Hice0=')+len('Hice0='):line.find(';')])
                    if 'Htide0_portion=' in line:
                        self.tideportion=float(re.search('=(.*);',line).group(1))
                    if 'Hice_P2=' in line and '==' not in line:
                        icm=line.find('%')
                        icl=line.find(';')
                        if not (icm>=0 and icm<icl):
                            tmp=line[line.find('Hice_P2=')+len('Hice_P2='):]
                            self.HiceP2=float(tmp[:tmp.find(';')])
                    if 'Hice_P3=' in line and '==' not in line:
                        icm=line.find('%')
                        icl=line.find(';')
                        if not (icm>=0 and icm<icl):
                            tmp=line[line.find('Hice_P3=')+len('Hice_P3='):]
                            self.HiceP3=float(tmp[:tmp.find(';')])
            if self.HiceP2!=0 or self.HiceP3!=0:
                self.topo=True

            self.runmax=int(self.itermax/self.eachiter)-1
            self.Hcond=self.kappa0*np.log((self.Tref+273.15)/(self.Ts+273.15))/self.Hice
            self.Q=self.Hcond*(1-self.tideportion)
            self.Htide=self.Hcond*self.tideportion
            self.showname='core:{}W, shell:{}W, visr:{}, vish:{}'.format(self.Q,self.Htide,self.av,self.ah)
        else:
            self.Htot=100e3
            self.Hice=0
        self.D=self.Htot-self.Hice
        self.ro=self.rsphere-self.Hice
        self.ri=self.rsphere-self.Htot
        self.geomratio = self.ri / self.ro
        self.thetatc = np.degrees(np.arccos(self.geomratio))


        # nondimensional numbers
        self.B = self.g * self.alpha * (self.Q) / self.rhoref / self.cp
        self.rostar = (self.B * (2 * self.omega)**-3)**(1 / 2) / (self.D)
        self.E = self.av / (2 * self.omega * (self.D)**2)
        self.taylor = 4 * self.omega**2 * (self.rsphere)**4 / self.av**2
        self.lrot = (self.B * (2 * self.omega)**-3)**(1 / 2)
        self.urot = (self.B * (2 * self.omega)**-1)**(1 / 2)
        self.unonrot = (self.B * (self.D))**(1 / 3)
        self.ucone = 2 * self.omega * (self.D) * self.rostar**(1 / 2)
        self.lcone = (self.D) * np.sqrt(self.rostar)
        self.gprime = np.sqrt(self.B * 2 * self.omega) / self.alpha / self.g
        self.gprimenonrot = (self.B**2 / (self.D))**(1 / 3) / self.alpha / self.g
        self.rastarq = self.alpha * self.g * self.Q / self.rhoref / self.cp / (
            self.omega**3) / (self.D**2)

        self.deepfacc = 1 + self.zc / self.rsphere
        self.deepfacf = 1 + self.zf / self.rsphere
        self.umax = kwargs.get('umax', None)
        self.wmax = kwargs.get('wmax', None)
        self.Tmax = kwargs.get('Tmax', None)
        self.Tmin = kwargs.get('Tmin', None)
        self.ummax = kwargs.get('ummax', None)
        self.Tmmax = kwargs.get('Tmmax', None)

    def get_non_dims(self, final_tsteps=50):
        return [
            self.D / 1000, self.Q,
            np.mean(self.ra().values[-final_tsteps:]),
            np.mean(self.rastar().values[-final_tsteps:]), self.rastarq,
            self.E, self.Ev,
            np.mean(self.nustar().values[-final_tsteps:]),
            np.mean(self.nu().values[-final_tsteps:]),
            np.mean(self.roloc().values[-final_tsteps:])
        ]

    def get_deltaT(self):
        T = self.T(iteration=np.Inf)
        vmeanT = (T * self.rac).sum(('lat', 'lon')) / np.sum(self.rac)
        deltaT = vmeanT.isel(z=-1) - vmeanT.isel(z=4)
        return deltaT

    def rastar(self):
        return self.alpha * self.g / (self.omega**2 *
                                      self.D) * self.get_deltaT()

    def ra(self):
        return self.alpha * self.g * self.D**3 / (
            self.av**2) * self.get_deltaT()  # assumed Pr = 1

    def nustar(self):
        return self.rastarq / self.rastar()

    def nu(self):
        return self.rastarq / self.ra() / self.E**3  # assumed Pr = 1

    def roloc(self):
        return self.ra()**(5 / 4) * self.E**2

    def get_runpath(self,iteration):
        it=iteration
        if self.itermax>0:
            if it>=0 and it<=self.itermax:
                runnum=np.mod(it,self.eachiter)
                runnum=it-runnum-int(runnum==0 and it!=0)*self.eachiter
            else:
                runnum=self.itermax-self.eachiter
                if it>=0:
                    it=self.itermax
            runpath='run{}/'.format(int(runnum))
        else:
            if not np.isnan(it):
                it=np.Inf
            runpath=''
        return it,runpath;

    def get_var(self,var,iteration=np.NaN,sample=1,dimnum=3):
        #it,runpath=self.get_runpath(iteration)
        it=iteration
        Vardata,its,meta = mu.rdmds(self.path + var, it,returnmeta=True)
        its=np.array(its)
        if (sample>1 and (((type(iteration)==list or type(iteration)==tuple) and len(iteration)>1) or np.isnan(iteration).all() )):
            itsid=np.arange(len(its))
            if (type(iteration)==list or type(iteration)==tuple) and len(iteration)>1:
                ibeg=iteration[0]
                iend=iteration[1]
            else:
                ibeg=0
                iend=np.Inf
            ikeep=np.array(np.where((np.mod(itsid,sample)==0) & (its>=ibeg) & (its<=iend))).squeeze()
            if dimnum==3:
                Vardata=Vardata[ikeep,:,:,:]
            if dimnum==2:
                Vardata=Vardata[ikeep,:,:]
                
            its=its[ikeep]
        Time = np.array(its)*self.dt/86400

        if np.ndim(Vardata)==dimnum:
            Vardata=np.reshape(Vardata,np.concatenate([[1],np.array(Vardata.shape)]))

        if dimnum==3:
            #Vardata=np.multiply(Vardata,np.tile(self.mask,[np.size(its),1,1,1]))
            Varxdata= xr.DataArray(
                Vardata,
                dims=('Time', 'z', 'lat', 'lon'),
                coords=dict(
                    Time=Time, z=self.zc, lat=self.yc[:, 0], lon=self.xc[0, :]))
            Varxdata=Varxdata.where(self.hfacc>0.0)

        if dimnum==2:
            Varxdata= xr.DataArray(
                Vardata,
                dims=('Time', 'lat', 'lon'),
                coords=dict(
                    Time=Time, lat=self.yc[:, 0], lon=self.xc[0, :]))

        return Varxdata
    
    def U(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        Udata=self.get_var(runpath+'U',iteration=iteration,sample=sample)
        Udata.name='U'
        Udata.attrs['units']='m/s'
        Udata.attrs['showname']='U (m/s)'
        Udata.attrs['longshowname']='Zonal flow speed (m/s)'
        return Udata

    def T(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        Tdata=self.get_var(runpath+'T',iteration=iteration,sample=sample)
        Tdata.name='T'
        Tdata.attrs['units']='degC'
        Tdata.attrs['showname']='T (degC)'
        Tdata.attrs['longshowname']='Temperature (degC)'
        return Tdata

    def V(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        Vdata=self.get_var(runpath+'V',iteration=iteration,sample=sample)
        Vdata.name='V'
        Vdata.attrs['units']='m/s'
        Vdata.attrs['showname']='V (m/s)'
        Vdata.attrs['longshowname']='Meridional flow speed (m/s)'
        return Vdata

    def W(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        Wdata=self.get_var(runpath+'W',iteration=iteration,sample=sample)
        Wdata.name='W'
        Wdata.attrs['units']='m/s'
        Wdata.attrs['showname']='W (m/s)'
        Wdata.attrs['longshowname']='Vertical flow speed (m/s)'
        return Wdata


    def q(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        qdata=self.get_var(runpath+'SHICE_fwFlux',iteration=iteration,sample=sample,dimnum=2)
        qdata=qdata/1000*(86400*360*1e6)/1000
        qdata.name='q'
        qdata.attrs['units']='km/Myr'
        qdata.attrs['showname']='q (km/Myr)'
        qdata.attrs['longshowname']='freezing rate (km/Myr)'
        return qdata


    def S(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        Sdata=self.get_var(runpath+'S',iteration=iteration,sample=sample)
        Sdata.name='S'
        Sdata.attrs['units']='psu'
        Sdata.attrs['showname']='S (psu)'
        Sdata.attrs['longshowname']='Salinity (psu)'
        return Sdata

    def Rho(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,S=None,T=None):
        if runpath==None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        if not 'rray' in type(S).__name__:
            S=self.get_var(runpath+'S',iteration=iteration,sample=sample)
        if not 'rray' in type(T).__name__:
            T=self.get_var(runpath+'T',iteration=iteration,sample=sample)
        T['Time']=S.Time
        density=self.beta*(S-self.Sref)-self.alpha*(T-self.Tref)
        density.name='Rho'
        density.attrs['units']='kg/m3'
        density.attrs['showname']='Rho (kg/m3)'
        density.attrs['longshowname']='density (kg/m3)'
        return density
    
    def Psi(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,V=None):
        if not 'rray' in type(V).__name__:
            if runpath==None:
                if np.isnan(runnum):
                    runpath='run*/'
                elif runnum<0:
                    runpath=''
                else:
                    if self.runmax>=0:
                        runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                        runpath='run{}/'.format(runid)
                    else:
                        runpath=''
            V=self.get_var(runpath+'V',iteration=iteration,sample=sample)
        if not 'lon' in V.dims:
            vdr=V*self.hfacc.mean(dim='lon')*self.drf*2*np.pi*self.a*np.cos(np.radians(V.lat))*self.rhoref
        else:
            vdr=V*self.hfacc*self.drf*2*np.pi*self.a*np.cos(np.radians(V.lat))*self.rhoref
        Psidata=(vdr.reindex(z=vdr.z[::-1])).cumsum(dim='z').reindex(z=vdr.z)
        Psidata=Psidata.where(self.hfacc>0.0)
        Psidata.name='Psi'
        Psidata.attrs['units']='kg/s'
        Psidata.attrs['showname']='Psi (kg/s)'
        Psidata.attrs['longshowname']='Meridional stream function (kg/s)'
        return Psidata

    def get(self,varlist,iteration=np.NaN,runnum=np.Inf,runpathstr=None,sample=1):
        dic={}
        codetemplate1="""dic['varname']=self.varname(iteration=it_,runnum=rn_,runpath=rp_,sample=sp_)"""
        for var in varlist:
            print('reading {}'.format(var))
            code=codetemplate1.replace('varname',var).replace('it_',str(iteration)).replace('rn_',str(runnum)).replace('nan','np.NaN').replace('inf','np.Inf').replace('sp_',str(sample)).replace('rp_',str(runpathstr))
            if var=='Rho' and 'T' in dic:
                code=code.replace(")",",T=dic['T'])")
            if var=='Rho' and 'S' in dic:
                code=code.replace(")",",S=dic['S'])")
            if var=='Psi' and 'V' in dic:
                code=code.replace(")",",V=dic['V'])")

            exec(code)
        return dic

    def monitor(self,iteration=np.Inf,pltarray=(3,2),wdirs=[0,0,0,1,1,1],pltvar=['T','S','Rho','U','W','Psi'],dimmeths={'Time':'mean','lon':'mean','lat':None,'z':None},figsize=(12,12),projection=None,pltcontour={},flip=True,pltdir='F',labelorientation=None,xlabelpad=0,ylabelpad=20,labelpad=0.04,labelshrink=1,labelaspect=20,linetype=['k-']*20,sharex=True,sharey=True,xlims=None,ylims=None,vminmax={},cms={},savefig=0,sample=3):
        d=self.get(pltvar,runpathstr="''",iteration=iteration,sample=sample)
        its=d[pltvar[0]].Time/self.dt*86400.0
        its=its.values
        print('its={}-{}'.format(its[0],its[-1]))
        fig,axf,da,figname=my2dplt(self,d,pltarray,wdirs=wdirs,pltvar=pltvar,dimmeths=dimmeths,figsize=figsize,projection=projection,pltcontour=pltcontour,flip=flip,pltdir=pltdir,labelorientation=labelorientation,xlabelpad=xlabelpad,ylabelpad=ylabelpad,labelpad=labelpad,labelshrink=labelshrink,labelaspect=labelaspect,linetype=linetype,sharex=sharex,sharey=sharey,xlims=xlims,ylims=ylims,vminmax=vminmax,cms=cms,savefig=savefig)
        return fig,axf,da

    def bulk_heatflux_y(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None):
        if runpath is None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''

        print('reading ADVy_TH...')
        Vardata,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=0)
        print('its={}-{}'.format(its[0],its[-1]))
        if len(its)==1:
            Vardata=Vardata[np.newaxis,:,:,:]
        Time = np.array(its)*self.dt/86400
        ADVy_TH= xr.DataArray(
            Vardata,
            dims=('Time', 'z', 'lat', 'lon'),
            coords=dict(
                Time=Time, z=self.zc, lat=self.yc[:, 0], lon=self.xc[0, :]))
        if layers is None:
            layers=range(len(self.zc))
        Heatflux_y=ADVy_TH.mean(dim='Time').isel(z=layers).sum(dim=['lon','z']).squeeze()*self.rhoref*self.cp/1e6
        Heatflux_y.name='Heatflux_y'
        Heatflux_y.attrs['units']='MW'
        Heatflux_y.attrs['showname']='F_Heat_y (MW)'
        Heatflux_y.attrs['longshowname']='Meridional Heat Flux (MW)'

        if doplt:
            with plt.style.context(('labelsize15')):
                fig, ax = plt.subplots(1,1)
                im=ax.plot(Heatflux_y.lat,Heatflux_y,color='black')
                ax.plot(Heatflux_y.lat,Heatflux_y*0.,'k--')
                ax.set_title(Heatflux_y.longshowname,loc='right')
                plt.xlabel('lat')
        
        return Heatflux_y
        

    def bulk_saltflux_y(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None):
        if runpath is None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''

        print('reading ADVy_SLT...')
        Vardata,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', it,returnmeta=True,rec=5)
        print('its={}-{}'.format(its[0],its[-1]))
        if len(its)==1:
            Vardata=Vardata[np.newaxis,:,:,:]
        Time = np.array(its)*self.dt/86400
        ADVy_SLT= xr.DataArray(
            Vardata,
            dims=('Time', 'z', 'lat', 'lon'),
            coords=dict(
                Time=Time, z=self.zc, lat=self.yc[:, 0], lon=self.xc[0, :]))
        if layers is None:
            layers=range(len(self.zc))
        Saltflux_y=ADVy_SLT.mean(dim='Time').isel(z=layers).sum(dim=['lon','z']).squeeze()/1e3
        Saltflux_y.name='Saltflux_y'
        Saltflux_y.attrs['units']='kg/s'
        Saltflux_y.attrs['showname']='F_Salt_y (kg/s)'
        Saltflux_y.attrs['longshowname']='Meridional Salt Flux (kg/s)'

        if doplt:
            with plt.style.context(('labelsize15')):
                fig, ax = plt.subplots(1,1)
                im=ax.plot(Saltflux_y.lat,Saltflux_y,color='black')
                ax.set_title(Saltflux_y.longshowname,loc='right')
                plt.xlabel('lat')
        return Saltflux_y

    def get_tprime(self, lats, latn, runpath=None, runnum=np.Inf, iteration=np.Inf,sample=3):
        T = self.T(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath)
        T=T.mean(dim='Time')
        T = T.sel(lat=slice(lats, latn))
        self.Tprime = np.sqrt(
            np.mean(T**2, axis=(1, 2)) - np.mean(T, axis=(
                1, 2)) * np.mean(T, axis=(1, 2)))
        return self.Tprime


    def get_uprime(self, lats, latn, runpath=None, runnum=np.Inf, iteration=np.Inf,sample=3):
        u = self.U(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath)
        v = self.V(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath)
        u=u.mean(dim='Time')
        v=v.mean(dim='Time')
        u = u.sel(lat=slice(lats, latn))
        v = v.sel(lat=slice(lats, latn))
        u_meanofsquares = np.mean(u**2 + v**2, axis=(1, 2))
        u_mean = np.mean(u, axis=(1, 2))
        v_mean = np.mean(v, axis=(1, 2))
        u_squareofmeans = u_mean**2 + v_mean**2
        self.Uprime = np.sqrt(u_meanofsquares - u_squareofmeans)
        return self.Uprime

    def umom(self, iteration=np.NaN, runnum=np.Inf, runpath=None):
        if runpath is None:
            if np.isnan(runnum):
                runpath='run*/'
            elif runnum<0:
                runpath=''
            else:
                if self.runmax>=0:
                    runid=int(np.minimum(runnum,self.runmax)*self.eachiter)
                    runpath='run{}/'.format(runid)
                else:
                    runpath=''
        umcori,its,meta = mu.rdmds(self.path + runpath + "momDiag", iteration, rec=0,returnmeta=True)
        if len(its)==1:
            Vardata=Vardata[np.newaxis,:,:,:]
        Time = np.array(its)*self.dt/86400
        dims = ('Time', 'z', 'lat', 'loni')
        coords = dict(Time=Time,
                      z=self.zc,
                      lat=self.yc[:, 0],
                      loni=self.xg[0, :])
        umcori = xr.DataArray(umcori, dims=dims, coords=coords)
        umadvec = mu.rdmds(self.path + "momDiag", iteration, rec=1)
        umadvec = xr.DataArray(umadvec, dims=dims, coords=coords)
        umadvre = mu.rdmds(self.path + "momDiag", iteration, rec=2)
        umadvre = xr.DataArray(umadvre, dims=dims, coords=coords)
        umdiss = mu.rdmds(self.path + "momDiag", iteration, rec=3)
        umdiss = xr.DataArray(umdiss, dims=dims, coords=coords)
        umimpld = mu.rdmds(self.path + "momDiag", iteration, rec=4)
        umimpld = xr.DataArray(umimpld, dims=dims, coords=coords)
        umdphix = mu.rdmds(self.path + "momDiag", iteration, rec=5)
        umdphix = xr.DataArray(umdphix, dims=dims, coords=coords)
        return xr.Dataset(
            dict(umcori=umcori,
                 umadvec=umadvec,
                 umadvre=umadvre,
                 umdiss=umdiss,
                 umimpld=umimpld,
                 umdphix=umdphix))

    def midlatitude_scales(self, iteration=np.Inf, lats=35, latn=45, z=10):
        it,runpath=self.get_runpath(iteration)
        u = mu.rdmds(self.path +runpath+ "U", it)
        u = xr.DataArray(
            u,
            dims=('z', 'lat', 'lon'),
            coords=dict(z=self.zc, lat=self.yc[:, 0], lon=self.xc[0, :]))
        u = u.sel(y=slice(lats, latn)).isel(z=z).mean('lat')
        n = len(u)
        ps = np.fft.rfft(u.values)
        # sample_rate = np.radians(1) * self.radius / 1000 * np.cos(
        #     np.radians((lats + latn) / 2))
        wl = np.fft.rfftfreq(n, 1)
        return (1 / wl, ps)

def cal_density(expt,T,S):
    density=expt.rhoref+expt.beta*(S-expt.Sref)-expt.alpha*(T-expt.Tref)
    density.name='Rho'
    density.attrs['units']='kg/m3'
    density.attrs['showname']='Rho (kg/m3)'
    density.attrs['longshowname']='Density (kg/m3)'
    return density

def cal_psi(expt,V):
    if not 'lon' in V.dims:
        vdr=V*expt.hfacc.mean(dim='lon')*expt.drf
    else:
        vdr=V*expt.hfacc*expt.drf
    Psi=(vdr.reindex(z=vdr.z[::-1])).cumsum(dim='z').reindex(z=vdr.z)
    Psi.name='Psi'
    Psi.attrs['units']='m/s'
    Psi.attrs['showname']='Psi (m/s)'
    Psi.attrs['longshowname']='Meridional flow speed (m/s)'
    return Psi


def my2dplt(expt,d,pltarray,wdirs=np.zeros(30),pltvar=None,dimmeths={'Time':'mean','lon':'mean','lat':None,'z':None},figsize=(12,12),projection=None,pltcontour={},flip=True,pltdir='F',labelorientation=None,xlabelpad=0,ylabelpad=20,labelaspect=20,labelshrink=1,labelpad=0.04,linetype=['k-']*20,sharex=True,sharey=True,xlims=None,ylims=None,vminmax={},cms={},savefig=0):
    nplt=pltarray[0]*pltarray[1]
    dout={}
    if pltvar==None:
        pltvar=d.keys()
    if labelorientation==None:
        if pltarray[0]==1 and pltarray[1]!=1:
            labelorientation='horizontal'
        else:
            labelorientation='vertical'

    # create figure
    with plt.style.context(('labelsize15')):
        if projection=="sphere":
            fig, ax = plt.subplots(pltarray[0],pltarray[1], sharex=sharex, sharey=sharey, figsize=figsize, subplot_kw=dict(projection=ccrs.Orthographic(central_latitude=0)))
        else:
            fig, ax = plt.subplots(pltarray[0],pltarray[1], sharex=sharex, sharey=sharey, figsize=figsize)

        if nplt>1:
            axf=ax.flatten(pltdir)
        else:
            axf=[ax]
        iplt=0
        nds=[]
        # loop over all vars
        for key in pltvar:
            if wdirs[iplt]:
                cm='RdBu_r'
            else:
                cm='hot' 
            if key=="S" or key=='salinity' or key=='q' or 'SHICE' in key:
                cm='viridis'
            if key in cms:
                cm=cms[key]

            var=d[key]
            allvardim=var.dims
            # deal with dimensions
            for dim in allvardim:
                if not dim in dimmeths:
                    continue
                dimmeth=dimmeths[dim]
                if dimmeth==None:
                    continue
                elif dimmeth=='mean':
                    var=var.mean(dim=dim,keep_attrs=True)
                elif dimmeth=='anom':
                    var=var-var.mean(dim=dim,keep_attrs=True)
                elif type(dimmeth) is not string:
                    if type(dimmeth) is float:
                        if dimmeth is np.Inf:
                            var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)==len(var[dim])-1),drop=True).mean(dim=dim,keep_attrs=True)
                        else:
                            var=var.where(abs(var[dim]-dimmeth)==min(abs(var[dim]-dimmeth)),drop=True).mean(dim=dim,keep_attrs=True)
                    elif type(dimmeth) is int:
                        var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)==dimmeth),drop=True).mean(dim=dim,keep_attrs=True)
                    elif type(dimmeth) is tuple or type(dimmeth) is list:
                        if type(dimmeth[0]) is int:
                            var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)>=dimmeth[0])&(xr.DataArray(range(len(var[dim])),dims=dim)<=dimmeth[1]),drop=True)
                        elif type(dimmeth[0]) is float:
                            var=var.where((var[dim]>=dimmeth[0])&(var[dim]<=dimmeth[1]),drop=True)
                        else:
                            raise NameError('dimmeths format wrong.')
                        if type(dimmeth) is list:
                            var=var.mean(dim=dim,keep_attrs=True)
                    else:
                        raise NameError('dimmeths format wrong.')
                else:
                    raise NameError('dimmeths format wrong.')

            newvardim=var.squeeze(drop=True).dims  
            if len(newvardim)!=2 and len(newvardim)!=1:
                raise NameError('{} dimension doesn''t equal to 1 or 2'.format(key))

            if len(newvardim)==1:
                if newvardim[0]=='lat':
                    var[0]=np.NaN
                    var[-1]=np.NaN
                im=axf[iplt].plot(var[newvardim[0]],var,linetype[iplt])
                if wdirs[iplt]:
                    axf[iplt].plot(var[newvardim[0]],var*0.,'k--')
                axf[iplt].set_title(var.longshowname,loc='right')
                plt.xlabel(newvardim[0])
                if not ylims==None:
                    axf[iplt].set_ylim(ylims)
                if not xlims==None:
                    axf[iplt].set_xlim(xlims)
                nds=nds+[1]
                iplt=iplt+1

            else: # dim==2
                # flip direction if required
                if flip:
                    xdim=newvardim[1]
                    ydim=newvardim[0]
                else:
                    xdim=newvardim[0]
                    ydim=newvardim[1]
                    var=var.transpose()

                dout[key]=var

                xcoord=var[xdim]
                ycoord=var[ydim]
                if xdim=='z':
                    xcoord=xcoord/1e3
                    xcoord=xcoord+expt.Hice/1e3
                    if xlims==None and (not expt.topo):
                        xlims=[(-expt.Htot+expt.Hice)/1e3,(expt.Hice-expt.Hice)/1e3]
                if ydim=='z':
                    ycoord=ycoord/1e3
                    ycoord=ycoord+expt.Hice/1e3
                    if ylims==None and (not expt.topo):
                        ylims=[(-expt.Htot+expt.Hice)/1e3,(expt.Hice-expt.Hice)/1e3]
                if xdim=='Time':
                    xcoord=xcoord/360
                    if xlims==None:
                        xlims=[var.Time.min()/360,var.Time.max()/360]
                if ydim=='Time':
                    ycoord=ycoord/360
                    if ylims==None:
                        ylims=[var.Time.min()/360,var.Time.max()/360]
                if not xlims==None:
                    xcoord=xcoord.where((xcoord>=xlims[0])&(xcoord<=xlims[1]),drop=True)
                    var=var.where((xcoord>=xlims[0])&(xcoord<=xlims[1]),drop=True)
                if not ylims==None:
                    ycoord=ycoord.where((ycoord>=ylims[0])&(ycoord<=ylims[1]),drop=True)
                    var=var.where((ycoord>=ylims[0])&(ycoord<=ylims[1]),drop=True)

                # offset
                varmean=var.mean(dim=[xdim,ydim])
                print('{} mean:{}'.format(key,varmean.values))
                if np.isnan(varmean):
                    print(var)
                if wdirs[iplt]==0 and (not np.isnan(varmean)) and varmean!=0:
                    offset=np.round(varmean,int(5-np.log10(np.abs(varmean))))
                else:
                    offset=0

                # vmin vmax
                if key in vminmax:
                    v_min=vminmax[key][0]
                    v_max=vminmax[key][1]
                elif wdirs[iplt]==1:
                    vamp=abs(var).max()
                    v_max=vamp
                    v_min=-vamp
                else:
                    v_max=None
                    v_min=None
                
                # plot
                if projection=='sphere':
                    im=axf[iplt].pcolormesh(xcoord,ycoord, var-offset,transform=ccrs.PlateCarree(),cmap=cm,vmin=v_min,vmax=v_max)
                    gl = axf[iplt].gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='k', alpha=0.5)
                    if key in pltcontour:
                        if wdirs[iplt]==0:
                            linestyles='solid'
                        else:
                            linestyles=None
                        axf[iplt].contour(xcoord,ycoord, var-offset,pltcontour[key],linestyles=linestyles,colors='gray',transform=ccrs.PlateCarree(),linewidths=2)
                else:
                    im=axf[iplt].pcolormesh(xcoord,ycoord, var-offset,cmap=cm,vmin=v_min,vmax=v_max)
                    if key in pltcontour:
                        if wdirs[iplt]==0:
                            linestyles='solid'
                        else:
                            linestyles=None
                        axf[iplt].contour(xcoord,ycoord, var-offset,pltcontour[key],linestyles=linestyles,colors='gray',linewidths=2)

                if wdirs[iplt]==0 and (not np.isnan(varmean)) and varmean!=0:
                    if hasattr(var,'showname'):
                        title=var.showname.replace(')',',ref:{:g})'.format(offset.values))
                    else:
                        title=key.replace(')',',ref:{:g})'.format(offset.values))

                else:
                    if hasattr(var,'showname'):
                        title=var.showname
                    else:
                        title=key

                axf[iplt].set_title(title,loc='left')
                cb = fig.colorbar(im, ax=axf[iplt],orientation=labelorientation, shrink=labelshrink,aspect=labelaspect, pad=labelpad)
               # cb.set_label(key)
                cb.formatter.set_powerlimits((-2,2))
                cb.formatter.set_useOffset(False)
                cb.update_ticks()
                #if not ylims==None:
                #    axf[iplt].set_ylim(ylims)
                #if not xlims==None:
                #    axf[iplt].set_xlim(xlims)

                nds=nds+[2]
                iplt=iplt+1
        
            # common axis label
            if all(np.array(nds)==2):
                xlab=xdim
                ylab=ydim
                if xdim=='z':
                    xlab=xlab+' (km)'
                if ydim=='z':
                    ylab=ylab+' (km)'
                if xdim=='Time':
                    xlab=xlab+' (yr)'
                if ydim=='Time':
                    ylab=ylab+' (yr)'
                if projection==None:
                    fig.add_subplot(111, frameon=False)
                    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    plt.xlabel(xlab,labelpad=xlabelpad)
                    plt.ylabel(ylab,labelpad=ylabelpad)

    # save figure
    figname=None
    if savefig:
        figname='{}_clim.png'.format(expt.name)
        fig.savefig(figname, bbox_inches='tight', dpi=150)
    return fig,axf,dout,figname
        



def get_dudr(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(expt,iteration)
    u = mu.rdmds(expt.path +runpath+ "dynDiag", it, rec=0)
    if np.isnan(it):
        u = np.mean(u, axis=0)
    udy = u * expt.dyg[np.newaxis, :, :] * expt.deepfacc[:, np.newaxis,
                                                         np.newaxis]
    udy = np.concatenate((-udy[[0]], udy, -udy[[-1]]), axis=0)
    du = np.diff(udy, axis=0)
    drc = mu.rdmds(expt.path + 'DRC').squeeze()
    dudr = du / (drc[:, np.newaxis, np.newaxis] * expt.dyg[np.newaxis, :, :] *
                 expt.deepfacf[:, np.newaxis, np.newaxis])
    dudr = 0.5 * (dudr[:-1] + dudr[1:])
    dudr = np.mean(dudr, axis=2)
    return dudr


def get_dudr_old(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    u = mu.rdmds(expt.path + "dynDiag", it, rec=0)
    u = np.mean(u, axis=-1)
    if np.isnan(it):
        u = np.mean(u, axis=0)
    u = np.concatenate((-u[[0], :], u, -u[[-1], :]), axis=0)
    du = np.diff(u, axis=0)
    drc = mu.rdmds(expt.path + 'DRC').squeeze()
    dudr = du / drc[:, np.newaxis]
    dudr = 0.5 * (dudr[:-1] + dudr[1:])
    return dudr


def get_1byrdudtheta(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    u = mu.rdmds(expt.path + "dynDiag", it, rec=0)
    if np.isnan(it):
        u = np.mean(u, axis=0)
    u = np.concatenate(
        (-u[:, [0]], u),
        axis=1)  # Assumption of no-slip at southern and northen edge
    dudrdtheta = np.diff(
        u, axis=1
    ) / (expt.dyc[np.newaxis, :, :] * expt.deepfacc[:, np.newaxis, np.newaxis])
    dudrdtheta = np.mean(dudrdtheta, axis=2)
    return dudrdtheta


def get_1byrdudtheta_old(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    u = mu.rdmds(expt.path + "dynDiag", it, rec=0)
    if np.isnan(it):
        u = np.mean(u, axis=0)
    u = np.concatenate(
        (-u[:, [0]], u),
        axis=1)  # Assumption of no-slip at southern and northen edge
    dtheta = np.radians(np.diff(expt.yc[:, 0])[0])
    dy = (expt.rsphere + expt.zc) * dtheta
    durdtheta = np.diff(u, axis=1) / dy[:, np.newaxis, np.newaxis]
    durdtheta = np.mean(durdtheta, axis=2)
    return durdtheta


def get_dudz(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    dudr = get_dudr_old(expt, it)
    durdtheta = get_1byrdudtheta_old(expt, it)

    theta = np.radians(expt.yc[:, 0])
    dudz = -np.sin(theta) * dudr + np.cos(theta) * durdtheta
    #dudz = np.cos(theta) * durdtheta
    return xr.DataArray(
        dudz, dims=('z', 'lat'), coords=dict(z=expt.zc, lat=expt.yc[:, 0]))


def get_db_rdtheta_old(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    T = mu.rdmds(expt.path + 'dynDiag', it, rec=5)
    if np.isnan(it):
        T = np.mean(T, axis=0)
    b = expt.g * expt.alpha * T
    b = np.concatenate((b[:, [-1]], b), axis=1)
    dtheta = np.radians(np.diff(expt.yc[:, 0])[0])
    dy = (expt.rsphere + expt.zc) * dtheta
    dbdy = np.diff(b, axis=1) / dy[:, np.newaxis, np.newaxis]
    dbrdtheta = dbdy / 2 / expt.omega
    dbrdtheta = np.mean(dbrdtheta, axis=2)
    return xr.DataArray(
        dbrdtheta, dims=('z', 'lat'), coords=dict(z=expt.zc, lat=expt.yc[:, 0]))



def get_db_rdtheta(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    T = mu.rdmds(expt.path + 'dynDiag', it, rec=5)
    if np.isnan(it):
        T = np.mean(T, axis=0)
    b = expt.g * expt.alpha * T
    bdx = b * expt.dxc * expt.deepfacc[:, np.newaxis, np.newaxis]
    bdx = np.concatenate((bdx[:, [-1]], bdx), axis=1)
    dbrdtheta = np.diff(
        bdx, axis=1) / (expt.ras[np.newaxis, :, :] * expt.
                        deepfacc[:, np.newaxis, np.newaxis]**2)
    dbrdtheta /= (2 * expt.omega)
    dbrdtheta = np.mean(dbrdtheta, axis=2)
    return xr.DataArray(
        dbrdtheta, dims=('z', 'lat'), coords=dict(z=expt.zc, lat=expt.yc[:, 0]))


def get_thermal_wind(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    dudz = get_dudz(expt, it)
    dbrdtheta = get_db_rdtheta_old(expt, it)
    return dudz, dbrdtheta


def integrate_dudz_in_z(expt):
    dudz, rhs = get_thermal_wind(expt)

    dr = np.diff(expt.zf)[:, np.newaxis]
    phi = np.radians(expt.yc[np.newaxis, :, 0])
    sinphirhsdr = rhs * np.sin(phi) * dr
    intsinphirhsdr = np.cumsum(sinphirhsdr, axis=0)

    Rdphi = np.mean(
        expt.dyc[np.newaxis, :, :] * expt.deepfacc[:, np.newaxis, np.newaxis],
        axis=2)
    Rcosphidphi = Rdphi * np.cos(phi)
    rhsRcosphidphi = rhs * Rcosphidphi
    intrhsRcosphidphi = np.cumsum(rhsRcosphidphi, axis=1)
    return intsinphirhsdr + intrhsRcosphidphi
    return Rdphi


def remap_from_rphi_toxz(var, expt, npoints=1000, method='linear'):
    phi = np.radians(expt.yc[:, 0])[np.newaxis, :]
    r = expt.rsphere + expt.zc[:, np.newaxis]
    #phi = np.radians(expt.yc[:, 0])
    #r = expt.rsphere + expt.zc
    xp, zp = r * np.cos(phi), r * np.sin(phi)
    xi, zi = np.linspace(0, expt.rsphere, npoints), np.linspace(
        -expt.rsphere, expt.rsphere, 2 * npoints)
    x, z = np.meshgrid(xi, zi)
    var_new = griddata(
        (zp.ravel(), xp.ravel()), var.ravel(), (z, x), method=method)
    var_new[x**2 + z**2 < r[-1]**2] = np.Inf
    return xr.DataArray(var_new, dims=('z', 'x'), coords=dict(z=zi, x=xi))


def integrate_interpolated_dudz_in_z(expt):
    it,runpath=expt.get_runpath(np.NaN)
    u = np.mean(
        mu.rdmds(expt.path +runpath+ "dynDiag", np.NaN, rec=0).squeeze(), axis=(0, -1))
    uinterp = remap_from_rphi_toxz(u, expt)

    rhs = get_db_rdtheta(expt)
    rhs_interpolated = remap_from_rphi_toxz(rhs.values, expt, npoints=1000)

    ro = expt.rsphere
    ri = expt.zc[-1] + ro

    x = rhs_interpolated['x'].values
    z = rhs_interpolated['z'].values
    dz = np.diff(z)[0]
    xx, zz = np.meshgrid(x, z)

    # phi = np.arctan(zz/xx)

    # region1 = (phi < - np.acos(r[-1]/expt.rsphere)) & (zz<0)
    # region3 = (phi >   np.acos(r[-1]/expt.rsphere)) & (zz>0)
    # region2 = (phi <= np.acos(r[-1]/expt.rsphere)) & (phi >= -np.acos(r[-1]/expt.rsphere))

    ufromth = np.nancumsum(rhs_interpolated, axis=0)
    # ufromth[xx**2 + zz**2 < ri**2] = np.Inf
    # ufromth[xx**2 + zz**2 > ro**2] = np.Inf
    u_fromth = xr.DataArray(
        ufromth * dz, dims=('z', 'x'), coords=dict(z=z, x=x))
    u_interp = xr.DataArray(uinterp, dims=('z', 'x'), coords=dict(z=z, x=x))
    cI = (u_fromth - u_interp).sel(
        z=0, x=(ro + ri) / 2, method='nearest').values
    #u_fromth += cI
    return u_fromth, u_interp


def integrate_dbdr(expt, iteration=np.NaN):
    rhs = get_dudz(expt, iteration)
    rhs *= (2 * expt.omega)
    r = (expt.rsphere + expt.zc)
    theta = expt.yc[:, 0]
    dtheta = np.radians(np.diff(theta)[0])
    rhs *= r[:, np.newaxis]
    b = np.zeros_like(rhs)
    b[:, theta > 0] = np.cumsum(rhs[:, theta > 0], axis=1) * dtheta
    b[:, theta < 0] = -np.cumsum(
        rhs[:, theta < 0][:, ::-1], axis=1)[:, ::-1] * dtheta
    #b = np.cumsum(rhs, axis=1) * dtheta
    return b


def compare_integratedb_vs_origb(expt, iteration=np.NaN):
    it,runpath=expt.get_runpath(iteration)
    Torig = mu.rdmds(expt.path + 'dynDiag', it, rec=5)
    if np.isnan(it):
        Torig = Torig.mean(0)
    Torig = xr.DataArray(
        Torig.mean(2),
        dims=('z', 'lat'),
        coords=dict(z=expt.zc, lat=expt.yc[:, 0]))
    b = -integrate_dbdr(expt, it)
    T = b / (expt.g * expt.alpha)
    eqT = Torig.sel(lat=slice(-0.5, 0.5)).mean('lat').values
    T = T + eqT[:, np.newaxis]
    T = xr.DataArray(
        T, dims=('z', 'lat'), coords=dict(z=expt.zc, lat=expt.yc[:, 0]))
    return T, Torig



# #suyash_enc1=Experiment(base_dir="/net/fs08/d0/bire/mitgcm/enceladus_sector/run_1/")
# #suyash_enc2=Experiment(base_dir="/net/fs08/d0/bire/mitgcm/enceladus_sector/run_2/")
# v12_lowgamma0=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0")
# v12_lowgamma0_weakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_weakic")
# v12_lowgamma0_lowdiff=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowdiff")
# #v12_lowgamma0_allshell=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_allshell")
# v12_lowgamma0_flux10=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_flux10")
# v12_lowgamma0_flux100=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_flux100")
# v12_lowgamma0_flux100_weakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_flux100_weakic")
# #v12_lowgamma0_lowvert=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert")
# v12_lowgamma0_lowvert_weakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic")
# v12_lowgamma0_lowvert_weakic_adv33=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_adv33")
# #v12_lowgamma0_lowlowvert_weakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowlowvert_weakic")
# v12_lowgamma0_lowlowvert_weakweakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowlowvert_weakweakic")
# v12_lowgamma0_lowvert_weakic_allshell=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_allshell")
# v12_lowgamma0_lowvert_weakic_halfhalf=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_halfhalf")
# #v12_topo_lowgamma0_lowvert_weakic=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_topo_lowgamma0_lowvert_weakic")
# #v12_topo_lowgamma0_lowvert_weakic_allshell=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_topo_lowgamma0_lowvert_weakic_allshell")
# #v12_topo_lowgamma0_lowvert_weakic_halfhalf=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_topo_lowgamma0_lowvert_weakic_halfhalf")
# v12_lowgamma0_lowvert_weakic_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_4psu")
# v12_lowgamma0_lowvert_weakic_allshell_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_allshell_4psu")
# v12_lowgamma0_lowvert_weakic_halfhalf_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_lowgamma0_lowvert_weakic_halfhalf_4psu")
# #v12_topo_lowgamma0_lowvert_weakic_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="enceladus_z40x90y324_v12_topo_lowgamma0_lowvert_weakic_4psu")
# #v12_topo_lowgamma0_lowvert_weakic_allshell_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="topo_lowgamma0_lowvert_weakic_allshell_4psu")
# #v12_topo_lowgamma0_lowvert_weakic_halfhalf_4psu=Experiment(base_dir="/net/fs09/d0/wanying/data_",name="topo_lowgamma0_lowvert_weakic_halfhalf_4psu")
#  
#  
# expts_lowres = [v12_lowgamma0,v12_lowgamma0_weakic,v12_lowgamma0_lowdiff,v12_lowgamma0_flux10,v12_lowgamma0_flux100,v12_lowgamma0_flux100_weakic,v12_lowgamma0_lowvert_weakic,v12_lowgamma0_lowvert_weakic_adv33,v12_lowgamma0_lowlowvert_weakweakic,v12_lowgamma0_lowvert_weakic_allshell,v12_lowgamma0_lowvert_weakic_halfhalf]
# 
gv13_40psu_allcore_strongic=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic")
gv13_40psu_halfhalf=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_halfhalf")
gv13_40psu_allshell=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allshell")
gv13_40psu_allcore_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid")
gv13_40psu_allcore_strongic_merid_asym_1e_3vr=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-3vr")
gv13_40psu_allcore_strongic_merid_asym_1e_5vr=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-5vr")
gv13_40psu_allcore_strongic_merid_asym_1e_5vr_1e_2vh=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-5vr_1e-2vh")
gv13_40psu_halfhalf_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_halfhalf_strongic_merid")
gv13_40psu_allshell_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allshell_strongic_merid")
gexpts_highres=[v13_40psu_allcore_strongic,v13_40psu_halfhalf,v13_40psu_allshell, v13_40psu_allcore_strongic_merid, v13_40psu_allcore_strongic_merid_asym_1e_3vr,v13_40psu_allcore_strongic_merid_asym_1e_5vr,v13_40psu_allcore_strongic_merid_asym_1e_5vr_1e_2vh,v13_40psu_halfhalf_strongic_merid,v13_40psu_allshell_strongic_merid]

def expt_grid_meanvelocity(
        expts,
        iteration=np.Inf,
        runnum=np.Inf,
        runpath=None,
        sample=3,
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
    #styledict = {29000: '--', 100000: '-.', 500000: '-'}
    #coldict = {0.1: 'r', 10: 'b', 1000: 'k', 10000: 'g'}
    for expt in expts:
        u_prime = expt.get_uprime(lats, latn, iteration=iteration,runnum=runnum,runpath=runpath,sample=sample)

        z = u_prime.z / expt.Htot
        ls = expt.lstyle  #styledict[expt.htot]
        cs = expt.c  #coldict[expt.q]
        ax[0].plot(
            u_prime.values, u_prime.z / expt.Htot, ls=ls, c=cs, label=expt.name)
        ax[1].plot(
            u_prime.values / expt.ucone,
            u_prime.z / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[2].plot(
            u_prime.values / expt.unonrot,
            u_prime.z / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[3].plot(
            u_prime.values / expt.urot,
            u_prime.z / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        meanvels.append(func(u_prime))
        urots.append(expt.urot)
        ucones.append(expt.ucone)
        #cols.append(coldict[expt.Htot])
        cols.append(expt.c)
    return fig, meanvels, urots, ucones, cols


def expt_grid_rmst(
        expts,
        iteration=np.Inf,
        runnum=np.Inf,
        runpath=None,
        sample=3,
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
    #styledict = {29000: '--', 100000: '-.', 500000: '-'}
    #coldict = {0.1: 'r', 10: 'b', 1000: 'k', 10000: 'g'}
    for expt in expts:
        Trms = expt.get_tprime(lats, latn, iteration=iteration,runnum=runnum,runpath=runpath,sample=sample)
        ls = expt.lstyle  #styledict[expt.Htot]
        cs = expt.c  #coldict[expt.Q]
        z = Trms.z / expt.Htot
        zlims = (z > -0.9) & (z < -0.1)
        ax[0].plot(
            Trms.values[zlims],
            Trms.z[zlims] / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        #ax[1].plot(urms.values / expt.urot, urms.z / expt.Htot,ls=ls,c=cs, label=expt.name)
        ax[1].plot(
            Trms.values[zlims] / expt.gprime,
            Trms.z[zlims] / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        ax[2].plot(
            Trms.values[zlims] / expt.gprimenonrot,
            Trms.z[zlims] / expt.Htot,
            ls=ls,
            c=cs,
            label=expt.name)
        meanTs.append(func(Trms))
        gprimes.append(expt.gprime)
        #cols.append(coldict[expt.Htot])
        cols.append(expt.c)
    return fig, meanTs, gprimes, cols


def expt_grid_powerspectrums(expts,
                             iteration=np.Inf,
                             lats=35,
                             latn=45,
                             ncols=4,
                             z=10,
                             figsize=(12, 9)):
    labs = iter(string.ascii_lowercase)
    nax = len(expts)
    fig, ax = plt.subplots(int(np.ceil(nax / ncols)), ncols, figsize=figsize)
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

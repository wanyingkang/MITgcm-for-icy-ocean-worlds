import facets as fa
import sparsemap as sm
import exchange
import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mu
import cartopy.crs as ccrs
import string
import xarray as xr
from scipy.interpolate import griddata
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.io import loadmat
from os import path
import struct
import re

class Experiment():
    def __init__(self, **kwargs):
        self.base_dir = kwargs.get('base_dir',
                                   "./data_")
        base_dir = self.base_dir
        self.name = kwargs.get("name",None)
        self.csresol=kwargs.get("csresol",32)
        self.path = base_dir + self.name + "/"
        runpath=''

        if self.csresol==32:
            self.xq=np.arange(-180., 180.,2.8125)
            self.yq=np.arange(-90.,90.,2.8125)
            self.gridfile='grid_cs32.face{0:03d}.bin'
            # note that the gridfile assumes earth radius and therefore dxg, dyg, ac, aw, as etc is all wrong!
            # this can be fixed by rescaling everything with a_planet/a_earth
            #self.bklfile='/home/wanying/MITgcm/utils/matlab/cs_grid/bk_line/isoLat_cs32_59.mat'
            self.bklfile='/home/wanying/analysis/isoLat_cs32_59.mat'
        if self.csresol==96:
            self.xq=np.arange(-180.,180.,1.)
            self.yq=np.arange(-90.,90.,1.)
            self.gridfile='cs96_dxC3_dXYa.face{0:03d}.bin'
            self.bklfile=None
        if self.csresol==510:
            self.xq=np.arange(-180,180.,1/6)
            self.yq=np.arange(-90,90.,1/6)
            self.gridfile='cs510.face{0:03d}.bin'
            self.bklfile=None

        self.facedims = kwargs.get("facedims", 6 * [self.csresol, self.csresol])
        self.yqi=(self.yq[:-1]+self.yq[1:])/2.0
        self.yqi=np.append([self.yq[0]],self.yqi,axis=0)
        self.yqi=np.append(self.yqi,[self.yq[-1]],axis=0)
        self.yq=np.reshape(self.yq,[len(self.yq),1])
        self.xq=np.reshape(self.xq,[1,len(self.xq)])
        self.yqi=np.reshape(self.yqi,[len(self.yqi),1])

        if not path.exists(self.path+"XC.data"):
            runpath='run0/'
            if not path.exists(self.path+runpath+"XC.data"):
                runpath='coords/'

        self.grid = fa.MITGrid(self.path+self.gridfile, dims=self.facedims)
        self.a_earth=6371000
        self.xc = mu.rdmds(self.path+runpath + "XC")
        self.yc = mu.rdmds(self.path+runpath + "YC")
        self.xg = mu.rdmds(self.path+runpath + "XG")
        self.yg = mu.rdmds(self.path+runpath + "YG")
        self.zc = mu.rdmds(self.path+runpath + "RC").squeeze()
        self.zf = mu.rdmds(self.path+runpath + "RF").squeeze()
        self.dxc = mu.rdmds(self.path+runpath + "DXC")
        self.dyc = mu.rdmds(self.path+runpath + "DYC")
        self.dxg = mu.rdmds(self.path+runpath + "DXG")
        self.dyg = mu.rdmds(self.path+runpath + "DYG")
        self.dzg = -np.diff(self.zf)
        self.rac = mu.rdmds(self.path+runpath + "RAC")
        self.ras = mu.rdmds(self.path+runpath + "RAS")
        self.raw = mu.rdmds(self.path+runpath + "RAW")
        self.raz = mu.rdmds(self.path+runpath + "RAZ")
        hfacc = mu.rdmds(self.path+runpath + "hFacC")
        self.hfacc = xr.DataArray( hfacc, dims=('z', 'y', 'x'),
            coords={"z":(('z'),self.zc),"lat":(('y','x'),self.yc),"lon":(('y','x'),self.xc)})
        drf = mu.rdmds(self.path+runpath + "DRF").squeeze()
        self.drf=xr.DataArray(drf,dims=('z'),coords=dict(z=self.zc))
        self.mask=self.hfacc*0+1
        self.mask=self.mask.where(self.hfacc>0.0)
        self.wgt=self.rac/np.mean(self.rac)
        M = self.get_expt_resolution()
        self.hfaccgrid=M(self.hfacc)
        self.hfaccgrid=xr.DataArray(self.hfaccgrid,dims=('z','lat','lon'),
                coords=dict(z=self.zc,lat=self.yq[:,0],lon=self.xq[0,:]))
        self.maskgrid=self.hfaccgrid*0+1
        self.maskgrid=self.maskgrid.where(self.hfaccgrid>0.0)
        self.wgtgrid=len(self.yq[:,0])*(np.sin(np.radians(self.yqi[1:,0]))-np.sin(np.radians(self.yqi[:-1,0])))/(np.sin(np.radians(self.yqi[-1,0]))-np.sin(np.radians(self.yqi[0,0])))
        
        self.cp = 4000
        self.lf=334000
        self.beta=0.0
        self.topo=False
        self.Dfig = kwargs.get('Dfig', 500000)  # Used only for polar plots
        self.deepatm=False
        with open(self.path + "data") as namelist:
            for line in namelist:
                if line[0] is '#':
                    continue
                if 'gravity=' in line:
                    self.g = float(re.search('=(.*),',line).group(1))
                if 'rSphere=' in line:
                    self.a = float(re.search('=(.*),',line).group(1))
                    self.grid.dxg=self.grid.dxg/self.a_earth*self.a
                    self.grid.dyg=self.grid.dyg/self.a_earth*self.a
                    self.grid.dxc=self.grid.dxc/self.a_earth*self.a
                    self.grid.dyc=self.grid.dyc/self.a_earth*self.a
                    self.grid.dyu=self.grid.dyu/self.a_earth*self.a
                    self.grid.dxv=self.grid.dxv/self.a_earth*self.a

                    self.grid.ac=self.grid.ac*(self.a/self.a_earth)**2
                    #self.grid.as=self.grid.as*(self.a/self.a_earth)**2
                    self.grid.aw=self.grid.aw*(self.a/self.a_earth)**2
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
                if 'deepAtmosphere=' in line:
                    tmp=re.search('=.(.*).,',line).group(1)
                    if tmp=='False' or tmp=='FALSE':
                        self.deepatm=False
                    else:
                        self.deepatm=True
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
                    self.Q0base=float(re.search('bottom_(.*)mW',line).group(1))/1e3
        self.runmax=int(self.itermax/self.eachiter)-1

        self.meridionalTs=0
        self.tide2d=False
        self.pcond=-1.0
        self.realtopo=0
        self.Hice_P1=0.0
        self.Hice_P2=0.0
        self.Hice_P3=0.0
        self.addmixbend=False
        if path.exists(self.path + "data.shelfice"): 
            with open(self.path + "data.shelfice") as shelficedata:
                for line in shelficedata:
                    if 'rhoShelfIce=' in line:
                        self.rhoice=float(re.search('=(.*),',line).group(1))
                    if 'SHELFICEthetaSurface=' in line:
                        self.Ts0=float(re.search('=(.*),',line).group(1))+273.15
                    if 'SHELFICEheatTransCoeff' in line:
                        self.gammaT=float(re.search('=(.*),',line).group(1))
                    if 'ptide' in line:
                        self.ptide=float(re.search('=(.*),',line).group(1))
                    if 'pcond' in line:
                        self.pcond=float(re.search('=(.*),',line).group(1))
                    if 'obliquity' in line:
                        self.obliquity=float(re.search('=(.*),',line).group(1))
                    if 'tide2d' in line:
                        tmp=re.search('=.(.*).,',line).group(1)
                        if tmp == 'False' or tmp == 'FALSE':
                            self.tide2d=False
                        else:
                            self.tide2d=True

        if path.exists(self.path + "gendata.m"):
            with open(self.path + "gendata.m") as gendata:
                for line in gendata:
                    if '%%%%%%' in line:
                        break
                    if 'Htot=' in line:
                        self.Htot=float(re.search('=(.*);',line).group(1))
                    if 'kappa0=' in line:
                        self.kappa0=float(re.search('=(.*);',line).group(1))
                        self.fluxfac=self.kappa0/651
                    if 'Hice0=' in line:
                        self.Hice0=float(line[line.find('Hice0=')+len('Hice0='):line.find(';')])
                    if 'Htide0_portion=' in line:
                        self.tideportion=float(re.search('=(.*);',line).group(1))
                    if 'Hice_P1=' in line and '==' not in line and line[0] != '%':
                        icm=line.find('%')
                        icl=line.find(';')
                        if not (icm>=0 and icm<icl):
                            tmp=line[line.find('Hice_P1=')+len('Hice_P1='):]
                            self.Hice_P1=float(tmp[:tmp.find(';')])
                    if 'Hice_P2=' in line and '==' not in line and line[0] != '%':
                        icm=line.find('%')
                        icl=line.find(';')
                        if not (icm>=0 and icm<icl):
                            tmp=line[line.find('Hice_P2=')+len('Hice_P2='):]
                            self.Hice_P2=float(tmp[:tmp.find(';')])
                    if 'Hice_P3=' in line and '==' not in line and line[0] != '%':
                        icm=line.find('%')
                        icl=line.find(';')
                        if not (icm>=0 and icm<icl):
                            tmp=line[line.find('Hice_P3=')+len('Hice_P3='):]
                            self.Hice_P3=float(tmp[:tmp.find(';')])
                    if 'qbotvary=' in line:
                        tmp=float(re.search('=(.*);',line).group(1))
                        if tmp==1:
                            self.qbotvary=True
                        else:
                            self.qbotvary=False
                    if 'mixbend=' in line:
                        tmp=float(re.search('=(.*);',line).group(1))
                        if tmp==1:
                            self.mixbend=True
                    if 'addmixbend=' in line:
                        tmp=float(re.search('=(.*);',line).group(1))
                        if tmp==1:
                            self.addmixbend=True
                        else:
                            self.addmixbend=False
                    if 'Htidemode=' in line:
                        tmp=re.search('=\[(.*)\];*',line).group(1)
                        self.Htidemode = [float(item) for item in tmp.split(',')]
                    if 'Hmixbendmode=' in line:
                        tmp=re.search('=\[(.*)\];*',line).group(1)
                        self.Hmixbendmode = [float(item) for item in tmp.split(',')]
                    if 'realtopo=' in line and 'realtopo==' not in line:
                        self.realtopo=float(re.search('=(.*);',line).group(1))
                    if 'meridionalTs=' in line and 'meridionalTs==' not in line and 'replace_string' not in line:
                        self.meridionalTs=float(re.search('=(.*);',line).group(1))
                    if 'realtoposmoothx=' in line and 'realtoposmoothx==' not in line:
                        tmp=re.search('=\[(.*)\];*',line).group(1)
                        self.realtoposmoothx = [float(item) for item in tmp.split(',')]
                    if 'realtoposmoothy=' in line and 'realtoposmoothy==' not in line:
                        tmp=re.search('=\[(.*)\];*',line).group(1)
                        self.realtoposmoothy = [float(item) for item in tmp.split(',')]
                    if 'realtopopath=' in line:
                            self.realtopopath=re.search("='(.*)';",line).group(1)

            self.addmixbend = kwargs.get('addmixbend', self.addmixbend)
            if self.Hice_P1!=0 or self.Hice_P2!=0 or self.Hice_P3!=0 or self.realtopo!=0:
                self.topo=True

            # just for scale estimation, except Q0 is taken as the global mean heat flux released that is normalized by the surface area
            self.ainter=self.a-(self.rhoice/self.rhoref)*self.Hice0
            self.Q0=self.Q0base*((self.a-self.Htot)/self.ainter)**(2*self.deepatm)

            slat1=np.sin(np.radians(self.yq))
            P1=slat1
            P2=1.5*slat1**2-0.5
            P3=2.5*slat1**3-1.5*slat1
            P4=(35*slat1**4-30.0*slat1**2 + 3)/8.0
            P6=(231.0*slat1**6.0-315.0*slat1**4.0+105.0*slat1**2.0-5.0)/16.0
            self.Ts=self.Ts0*np.ones_like(self.yq)
#            if self.realtopo!=0:
#                f=open(self.realtopopath,'rb')
#                fomat='>'+720*1440*'d'
#                Hice_=np.array(struct.unpack(fomat,f.read())).reshape((1440,720)).transpose()
#                Hice_=Hice_*1e3
#                if self.realtopo==1:
#                    y_=np.arange(-90+0.125,90,0.25)
#                    y_=np.hstack((-90,y_,90))
#                    Hice_=np.mean(Hice_,1)
#                    Hice_=np.hstack((Hice_[0],Hice_,Hice_[-1]))
#                    fHice=interp1d(y_,Hice_)
#                    self.Hice=fHice(self.yc)
#                    self.Hiceq=fHice(self.yq)
#                else:
#                    y_=np.arange(-90+0.125,90,0.25)
#                    y_=np.hstack((-90,y_,90))
#                    x_=np.arange(0,360.1,0.25)
#                    Hice_=np.hstack((Hice_,Hice_[:,0,np.newaxis]))
#                    Hice_sp=np.mean(Hice_[0,:])*np.ones_like(x_)
#                    Hice_np=np.mean(Hice_[-1,:])*np.ones_like(x_)
#                    Hice_=np.vstack((Hice_sp[np.newaxis,:],Hice_,Hice_np[np.newaxis,:]))
#                    if hasattr(self,'realtoposmoothx') and self.realtoposmoothx[0]!=0:
#                        nx_=len(x_)
#                        Hice_zm=np.mean(Hice_,axis=1,keepdims=True)
#                        Hice_anom=Hice_-Hice_zm
#                        Hice_anomsmt=Hice_anom*0
#                        ks=np.reshape(np.array(self.realtoposmoothx)[:,np.newaxis],[len(self.realtoposmoothx),1])
#                        if np.any(ks>0):
#                            basecos=np.cos(np.radians(x_[np.newaxis,:]*ks[ks>0]))
#                            Hice_anomsmt=Hice_anomsmt+(2/nx_)*np.matmul(np.matmul(Hice_anom,basecos.transpose()),basecos)
#                        if np.any(ks<0):
#                            basesin=np.sin(np.radians(x_[np.newaxis,:]*abs(ks[ks<0])))
#                            Hice_anomsmt=Hice_anomsmt+(2/nx_)*np.matmul(np.matmul(Hice_anom,basesin.transpose()),basesin)
#                        Hice_=Hice_zm+Hice_anomsmt
#                    if hasattr(self,'realtoposmoothy') and self.realtoposmoothy[0]!=0:
#                        ny_=len(y_)
#                        Hice_zm=np.mean(Hice_,axis=1,keepdims=True)
#                        Hice_anom=Hice_-Hice_zm
#                        ...
#
#
#                    #y2_=y2_[::3,::3]
#                    #x2_=x2_[::3,::3]
#                    #Hice_=Hice_[::3,::3]
#                    #fHice=interp2d(y2_,x2_,Hice_)
#                    #self.Hice=fHice(self.yc,self.xc)
#                    fHice=RectBivariateSpline(y_,x_,Hice_)
#                    self.Hiceq=fHice(self.yq,self.xq+180)
#                self.Hice0=np.mean(self.Hiceq*self.wgtgrid[:,np.newaxis])
#
#            else:
#                self.Hiceq=self.Hice0+self.Hice_P1*P1+self.Hice_P2*P2+self.Hice_P3*P3

            self.Hunder=np.sum((1-self.hfacc)*self.drf,axis=0)
            self.Hice=self.Hunder/self.rhoice*self.rhoref
            M = self.get_expt_resolution()
            self.Hiceq=M(self.Hice)
            
            if self.meridionalTs==1:
                yqr=np.radians(self.yq)
                self.Ts[abs(self.yq[:,0])<=90-self.obliquity]=self.Ts0*np.cos(yqr[abs(self.yq[:,0])<=90-self.obliquity])**0.25
                self.Ts[abs(self.yq[:,0])>90-self.obliquity]=self.Ts0*(((np.pi/2-abs(yqr[abs(self.yq[:,0])>90-self.obliquity]))**2+(self.obliquity*np.pi/180)**2)/2)**0.125
            if self.meridionalTs==3:
                cobl=np.cos(np.radians(self.obliquity))
                p2obl=1.5*cobl**2-0.5
                p4obl=(35*cobl**4-30.0*cobl**2 + 3)/8.0
                p6obl=(231.0*cobl**6.0-315.0*cobl**4.0+105.0*cobl**2.0-5.0)/16.0
                self.Ts=self.Ts0*(1.0-(5/8)*p2obl*P2-(9/64)*p4obl*P4-(65/1024)*p6obl*P6)**0.25
                
            self.Hcond=self.kappa0*np.log((self.Tref+273.15)/(self.Ts))/self.Hice0
            self.Hcond=self.Hcond*(self.Hiceq/self.Hice0)**self.pcond
            self.Hcond0=np.mean(self.Hcond*self.wgtgrid[:,np.newaxis])
            self.Htide0=self.Hcond0*self.tideportion
            self.showname='core:{}W, shell:{}W, visr:{}, vish:{}'.format(self.Q0,self.Htide0,self.av,self.ah)
        else:
            self.Htot=100e3
            self.Hice0=0
            self.Hice=0

        self.GM=False
        if path.exists(self.path + "data.pkg"): 
            with open(self.path + "data.pkg") as pkgdata:
                for line in pkgdata:
                    if 'useGMRedi' in line and '#' not in line:
                        if 'TRUE' in line or 'True' in line:
                            self.GM=True
                        else:
                            self.GM=False

        self.D=self.Htot-self.Hice0
        self.ro=self.rsphere-self.Hice0
        self.ri=self.rsphere-self.Htot
        self.geomratio = self.ri / self.ro
        self.thetatc = np.degrees(np.arccos(self.geomratio))

        clat1=np.cos(np.radians(self.yq))
        slat1=np.sin(np.radians(self.yq))
        self.thetaeq3=np.degrees(np.arcsin(1/3))
        self.thetaeq2=np.degrees(np.arcsin(1/2))
        Y00=np.ones_like(clat1)/np.sqrt(4*np.pi)
        Y20=(1.5*slat1**2-0.5)/np.sqrt(4*np.pi/5)
        Y40=(35/8*slat1**4-30/8*slat1**2+3/8)/np.sqrt(4*np.pi/9)
        if self.tide2d:
            c2lon=np.cos(np.radians(self.xq)*2)
            c4lon=np.cos(np.radians(self.xq)*4)
            Y22=(3*clat1**2)*(2*c2lon)/np.sqrt(96*np.pi/5)
            Y42=(7.5*(7*slat1**2-1)*clat1**2)*(2*c2lon)/np.sqrt(1440*np.pi/9)
            Y44=(105*clat1**4)*(2*c4lon)/np.sqrt(40320*4*np.pi/9)
        else:
            Y22=0
            Y42=0
            Y44=0

        self.Q=self.Q0*np.ones_like(self.yq)
        if self.qbotvary:
            qprofile=1-25/2/(60-25/2)*(2*clat1**2-1)
            qprofile=qprofile/np.mean(qprofile*self.wgtgrid)
            self.Q=self.Q*qprofile

        Htideprof=np.sqrt(4*np.pi)*(Y00+self.Htidemode[0]*Y20+self.Htidemode[1]*Y40+self.Htidemode[2]*Y22+self.Htidemode[3]*Y42+self.Htidemode[4]*Y44)
        Htideprof3=Htideprof*(self.Hiceq/self.Hice0)**(-3.0)
        Htideprof2=Htideprof*(self.Hiceq/self.Hice0)**(-2.0)
        Htideprof15=Htideprof*(self.Hiceq/self.Hice0)**(-1.5)
        Htideprof1=Htideprof*(self.Hiceq/self.Hice0)**(-1.)
        Htideprof=Htideprof*(self.Hiceq/self.Hice0)**self.ptide
        if self.addmixbend:
            Hmixbendprof=np.sqrt(4*np.pi)*(self.Hmixbendmode[0]*Y00+self.Hmixbendmode[1]*Y20+self.Hmixbendmode[2]*Y40+self.Hmixbendmode[3]*Y22+self.Hmixbendmode[4]*Y42+self.Hmixbendmode[5]*Y44)
            Htideprof3=Htideprof3+Hmixbendprof
            Htideprof2=Htideprof2+Hmixbendprof
            Htideprof15=Htideprof15+Hmixbendprof
            Htideprof1=Htideprof1+Hmixbendprof
            Htideprof=Htideprof+Hmixbendprof
        Htideprof3=Htideprof3/np.mean(Htideprof3*self.wgtgrid[:,np.newaxis])
        Htideprof2=Htideprof2/np.mean(Htideprof2*self.wgtgrid[:,np.newaxis])
        Htideprof15=Htideprof15/np.mean(Htideprof15*self.wgtgrid[:,np.newaxis])
        Htideprof1=Htideprof1/np.mean(Htideprof1*self.wgtgrid[:,np.newaxis])
        Htideprof=Htideprof/np.mean(Htideprof*self.wgtgrid[:,np.newaxis])
        self.Htide3=self.Htide0*Htideprof3
        self.Htide2=self.Htide0*Htideprof2
        self.Htide15=self.Htide0*Htideprof15
        self.Htide1=self.Htide0*Htideprof1
        self.Htide=self.Htide0*Htideprof
        

        # nondimensional numbers
        self.B = self.g * self.alpha * (self.Q0) / self.rhoref / self.cp
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
        self.rastarq = self.alpha * self.g * self.Q0 / self.rhoref / self.cp / (
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
            self.D / 1000, self.Q0,
            np.mean(self.ra().values[-final_tsteps:]),
            np.mean(self.rastar().values[-final_tsteps:]), self.rastarq,
            self.E, self.Ev,
            np.mean(self.nustar().values[-final_tsteps:]),
            np.mean(self.nu().values[-final_tsteps:]),
            np.mean(self.roloc().values[-final_tsteps:])
        ]

    def get_deltaT(self):
        T = self.T(iteration=np.Inf)
        vmeanT = (T * self.racgrid*self.maskgrid).sum(('lat', 'lon')) / (self.racgrid*self.maskgrid).sum(('lat', 'lon'))
        vmeanT=vmeanT.dropna(dim='z')
        deltaT = vmeanT.isel(z=-1) - vmeanT.isel(z=0)
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

    def get_expt_resolution(self):
        if self.csresol==510:
            M = sm.frommap(
                '/home/wanying/analysis/cs510_to_llsixthdeg_conservative_stackx.i4i4f8_2160x1080x1560600.map'
            )
        if self.csresol==96:
            M = sm.frommap(
                '/home/wanying/analysis/cs96_to_ll1deg_conservative_oldmap.i4i4f8_360x180x55296.map'
            )
        if self.csresol==32:
            M = sm.frommap(
                '/home/wanying/analysis/cs32_to_ll128x64_conservative_oldmap.i4i4f8_128x64x6144.map'
            )
        return M

    def get_var(self,var,iteration=np.NaN,sample=1,dimnum=3,rec=None):
        # tuple iteration will cause the script to read all available records and select those within the range
        #it,runpath=self.get_runpath(iteration)
        if type(iteration)==tuple:
            it=np.NaN
        else:
            it=iteration
        Vardata,its,meta = mu.rdmds(self.path + var, it,returnmeta=True,rec=rec)
        its=np.array(its)
        if ((sample>1 and np.isnan(it)) or  ((type(iteration)==tuple) and len(iteration)>1)):
            itsid=np.arange(len(its))
            if (type(iteration)==tuple) and len(iteration)>1:
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
                dims=('Time','z', 'y', 'x'),
                coords={"Time":Time,
                        "z":(('z'),self.zc),
                        "lat":(('y','x'),self.yc),
                        "lon":(('y','x'),self.xc)})
            Varxdata=Varxdata.where(self.hfacc>0.0)

        if dimnum==2:
            Varxdata= xr.DataArray(
                Vardata,
                dims=('Time','y', 'x'),
                coords={"Time":Time,
                        "lat":(('y','x'),self.yc),
                        "lon":(('y','x'),self.xc)})

        return Varxdata
    
    def UV(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,iz=None):
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
        Vdata=self.get_var(runpath+'V',iteration=iteration,sample=sample)
        Time=Udata.Time
        z=Udata.z
        if tmean:
            Udata=Udata.mean(dim='Time',keep_attrs=True,keepdims=True)
            Vdata=Vdata.mean(dim='Time',keep_attrs=True,keepdims=True)
            Time=np.mean(Time,keepdims=True)
        if iz!=None:
            Udata=Udata.isel(z=iz)
            Vdata=Vdata.isel(z=iz)
                
        exch = exchange.cs()
        Udata = fa.fromglobal(Udata.values, dims=self.facedims, extrau=1)
        Vdata = fa.fromglobal(Vdata.values, dims=self.facedims, extrav=1)
        exch.uv(Udata,Vdata)
        Vdata = 0.5 * (Vdata[:, :, :-1, :] + Vdata[:, :, 1:, :])
        Udata = 0.5 * (Udata[:, :, :, :-1] + Udata[:, :, :, 1:])
        anglecs = self.grid.anglecs
        anglesn = self.grid.anglesn
        Urot=anglecs * Udata - anglesn * Vdata
        Vrot=anglesn * Udata + anglecs * Vdata
        M = self.get_expt_resolution()
        Ugrid = M(Urot.toglobal())
        Vgrid = M(Vrot.toglobal())

        Ugrid=xr.DataArray(Ugrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        Ugrid.name='U'
        Ugrid.attrs['units']='m/s'
        Ugrid.attrs['showname']='U (m/s)'
        Ugrid.attrs['longshowname']='Zonal flow speed (m/s)'

        Vgrid=xr.DataArray(Vgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        Vgrid.name='V'
        Vgrid.attrs['units']='m/s'
        Vgrid.attrs['showname']='V (m/s)'
        Vgrid.attrs['longshowname']='Meridional flow speed (m/s)'
        return Ugrid,Vgrid

    def T(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None, iz=None, tmean=True):
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
        Time=Tdata.Time
        if tmean:
            Tdata=Tdata.mean(dim='Time',keep_attrs=True,keepdims=True)
            Time=np.mean(Time,keepdims=True)
        if iz!=None:
            Tdata=Tdata.isel(z=iz)
        z=Tdata.z

        M = self.get_expt_resolution()
        Tgrid=M(Tdata.values)

        Tgrid=xr.DataArray(Tgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        Tgrid.name='T'
        Tgrid.attrs['units']='degC'
        Tgrid.attrs['showname']='T (degC)'
        Tgrid.attrs['longshowname']='Temperature (degC)'
        return Tgrid

    def PH(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,iz=None,tmean=True):
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
        PHdata=self.get_var(runpath+'PH',iteration=iteration,sample=sample)
        if tmean:
            PHdata=PHdata.mean(dim='Time',keep_attrs=True,keepdims=True)
        if iz!=None:
            PHdata=PHdata.isel(z=iz)
        Time=PHdata.Time
        z=PHdata.z

        M = self.get_expt_resolution()
        PHgrid=M(PHdata.values)

        PHgrid=xr.DataArray(PHgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        PHgrid.name='PH'
        PHgrid.attrs['units']='m^2/s'
        PHgrid.attrs['showname']='PH (m^2/s)'
        PHgrid.attrs['longshowname']='Nonhydro Pressure (m^2/s)'
        return PHgrid

    def PHL(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,iz=None,tmean=True):
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
        PHLdata=self.get_var(runpath+'PHL',iteration=iteration,sample=sample)
        if tmean:
            PHLdata=PHLdata.mean(dim='Time',keep_attrs=True,keepdims=True)
        if iz!=None:
            PHLdata=PHLdata.isel(z=iz)
        Time=PHLdata.Time
        z=PHLdata.z

        M = self.get_expt_resolution()
        PHLgrid=M(PHLdata.values)

        PHLgrid=xr.DataArray(PHLgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        PHLgrid.name='PHL'
        PHLgrid.attrs['units']='m^2/s'
        PHLgrid.attrs['showname']='PHL (m^2/s)'
        PHLgrid.attrs['longshowname']='Hydro Pressure (m^2/s)'
        return PHLgrid

    def PNH(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,iz=None,tmean=True):
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
        PNHdata=self.get_var(runpath+'PNH',iteration=iteration,sample=sample)
        if tmean:
            PNHdata=PNHdata.mean(dim='Time',keep_attrs=True,keepdims=True)
        if iz!=None:
            PNHdata=PNHdata.isel(z=iz)
        Time=PNHdata.Time
        z=PNHdata.z

        M = self.get_expt_resolution()
        PNHgrid=M(PNHdata.values)

        PNHgrid=xr.DataArray(PNHgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        PNHgrid.name='PNH'
        PNHgrid.attrs['units']='m^2/s'
        PNHgrid.attrs['showname']='PNH (m^2/s)'
        PNHgrid.attrs['longshowname']='Nonhydro Pressure (m^2/s)'
        return PNHgrid

    def W(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,iz=None,tmean=True):
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
        if tmean:
            Wdata=Wdata.mean(dim='Time',keep_attrs=True,keepdims=True)
        if iz!=None:
            Wdata=Wdata.isel(z=iz)
        Time=Wdata.Time
        z=Wdata.z

        M = self.get_expt_resolution()
        Wgrid=M(Wdata.values)

        Wgrid=xr.DataArray(Wgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        Wgrid.name='W'
        Wgrid.attrs['units']='m/s'
        Wgrid.attrs['showname']='W (m/s)'
        Wgrid.attrs['longshowname']='Vertical flow speed (m/s)'
        return Wgrid

    def heatflux(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,iz=None,component='advdif',zsum=True):
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
        if 'adv' in component:
            ADVy_TH = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=0)
            ADVx_TH = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=10)
        else:
            ADVy_TH=0
            ADVx_TH=0
        if 'dif' in component:
            DFyE_TH = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=2)
            DFxE_TH = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=12)
        else:
            DFyE_TH=0
            DFxE_TH=0

        Fy=(ADVy_TH+DFyE_TH)/self.dxg*self.rhoref*self.cp/1e6
        Fx=(ADVx_TH+DFxE_TH)/self.dyg*self.rhoref*self.cp/1e6

        Time=Fy.Time
        z=Fy.z
        if tmean:
            Fy=Fy.mean(dim='Time',keep_attrs=True,keepdims=True)
            Fx=Fx.mean(dim='Time',keep_attrs=True,keepdims=True)
            Time=np.mean(Time,keepdims=True)
        if iz!=None:
            Fy=Fy.isel(z=iz)
            Fx=Fx.isel(z=iz)
        if zsum:
            Fy=Fy.sum('z')
            Fx=Fx.sum('z')
                
        exch = exchange.cs()
        Fx = fa.fromglobal(Fx.values, dims=self.facedims, extrau=1)
        Fy = fa.fromglobal(Fy.values, dims=self.facedims, extrav=1)
        exch.uv(Fx,Fy)
        if Fy.ndim==4:
            Fy = 0.5 * (Fy[:, :, :-1, :] + Fy[:, :, 1:, :])
            Fx = 0.5 * (Fx[:, :, :, :-1] + Fx[:, :, :, 1:])
        if Fy.ndim==3:
            Fy = 0.5 * (Fy[:, :-1, :] + Fy[:, 1:, :])
            Fx = 0.5 * (Fx[:, :, :-1] + Fx[:, :, 1:])
        if Fy.ndim==2:
            Fy = 0.5 * (Fy[:-1, :] + Fy[1:, :])
            Fx = 0.5 * (Fx[:, :-1] + Fx[:, 1:])
        
        anglecs = self.grid.anglecs
        anglesn = self.grid.anglesn
        Fxrot=anglecs * Fx - anglesn * Fy
        Fyrot=anglesn * Fx + anglecs * Fy
        M = self.get_expt_resolution()
        Fxgrid = M(Fxrot.toglobal())
        Fygrid = M(Fyrot.toglobal())

        if (not tmean) and (not zsum):
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('Time','z', 'lat', 'lon'),
                    coords=dict(Time=Time,z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('Time','z', 'lat', 'lon'),
                    coords=dict(Time=Time,z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if tmean and (not zsum):
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('z', 'lat', 'lon'),
                    coords=dict(z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('z', 'lat', 'lon'),
                    coords=dict(z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if (not tmean) and zsum:
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('Time', 'lat', 'lon'),
                    coords=dict(Time=Time,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('Time', 'lat', 'lon'),
                    coords=dict(Time=Time,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if tmean and zsum:
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('lat', 'lon'),
                    coords=dict(lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('lat', 'lon'),
                    coords=dict(lat=self.yq[:,0],lon=self.xq[0,:]))

        Fxgrid.name='F_Heat_x'
        Fxgrid.attrs['units']='MW/m'
        Fxgrid.attrs['showname']='F_Heat_x (MW/m)'
        Fxgrid.attrs['longshowname']='Zonal heat flux (MW/m)'

        Fygrid.name='F_Heat_y'
        Fygrid.attrs['units']='MW/m'
        Fygrid.attrs['showname']='F_Heat_y (MW/m)'
        Fygrid.attrs['longshowname']='Meridional heat flux (MW/m)'
        return Fxgrid,Fygrid

    def saltflux(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,iz=None,component='advdif'):
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
        if 'adv' in component:
            ADVy_SLT = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=5)
            ADVx_SLT = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=11)
        else:
            ADVy_SLT=0
            ADVx_SLT=0
        if 'dif' in component:
            DFyE_SLT = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=7)
            DFxE_SLT = self.get_var(runpath + 'flxDiag', iteration,sample=sample,rec=13)
        else:
            DFyE_SLT=0
            DFxE_SLT=0

        Fy=(ADVy_SLT+DFyE_SLT)/self.dxg*self.rhoref/1e3
        Fx=(ADVx_SLT+DFxE_SLT)/self.dyg*self.rhoref/1e3

        Time=Fy.Time
        z=Fy.z
        if tmean:
            Fy=Fy.mean(dim='Time',keep_attrs=True,keepdims=True)
            Fx=Fx.mean(dim='Time',keep_attrs=True,keepdims=True)
            Time=np.mean(Time,keepdims=True)
        if iz!=None:
            Fy=Fy.isel(z=iz)
            Fx=Fx.isel(z=iz)
                
        exch = exchange.cs()
        Fx = fa.fromglobal(Fx.values, dims=self.facedims, extrau=1)
        Fy = fa.fromglobal(Fy.values, dims=self.facedims, extrav=1)
        exch.uv(Fx,Fy)
        if Fy.ndim==4:
            Fy = 0.5 * (Fy[:, :, :-1, :] + Fy[:, :, 1:, :])
            Fx = 0.5 * (Fx[:, :, :, :-1] + Fx[:, :, :, 1:])
        if Fy.ndim==3:
            Fy = 0.5 * (Fy[:, :-1, :] + Fy[:, 1:, :])
            Fx = 0.5 * (Fx[:, :, :-1] + Fx[:, :, 1:])
        if Fy.ndim==2:
            Fy = 0.5 * (Fy[:-1, :] + Fy[1:, :])
            Fx = 0.5 * (Fx[:, :-1] + Fx[:, 1:])
        anglecs = self.grid.anglecs
        anglesn = self.grid.anglesn
        Fxrot=anglecs * Fx - anglesn * Fy
        Fyrot=anglesn * Fx + anglecs * Fy
        M = self.get_expt_resolution()
        Fxgrid = M(Fxrot.toglobal())
        Fygrid = M(Fyrot.toglobal())

        if (not tmean) and (not zsum):
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('Time','z', 'lat', 'lon'),
                    coords=dict(Time=Time,z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('Time','z', 'lat', 'lon'),
                    coords=dict(Time=Time,z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if tmean and (not zsum):
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('z', 'lat', 'lon'),
                    coords=dict(z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('z', 'lat', 'lon'),
                    coords=dict(z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if (not tmean) and zsum:
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('Time', 'lat', 'lon'),
                    coords=dict(Time=Time,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('Time', 'lat', 'lon'),
                    coords=dict(Time=Time,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
        if tmean and zsum:
            Fxgrid=xr.DataArray(Fxgrid,
                    dims=('lat', 'lon'),
                    coords=dict(lat=self.yq[:,0],lon=self.xq[0,:]))
            Fygrid=xr.DataArray(Fygrid,
                    dims=('lat', 'lon'),
                    coords=dict(lat=self.yq[:,0],lon=self.xq[0,:]))

        Fxgrid.name='F_Salt_x'
        Fxgrid.attrs['units']='kg/m/s'
        Fxgrid.attrs['showname']='F_Salt_x (kg/m/s)'
        Fxgrid.attrs['longshowname']='Zonal salt flux (kg/m/s)'

        Fygrid.name='F_Salt_y'
        Fygrid.attrs['units']='kg/m/s'
        Fygrid.attrs['showname']='F_Salt_y (kg/m/s)'
        Fygrid.attrs['longshowname']='Meridional salt flux (kg/m/s)'
        return Fxgrid,Fygrid


    def Hlatent(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=False,**kwargs):
        q=self.q(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath,tmean=tmean,inmassunit=True)
        Hlatnt=self.lf*q
        Hlatnt.name='Hlatent'
        Hlatnt.attrs['units']='W/m^2'
        Hlatnt.attrs['showname']='Hlatent (W/m^2)'
        Hlatnt.attrs['longshowname']='Latent heating (W/m^2)'
        return Hlatnt

    def q(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,inmassunit=False,tmean=False,**kwargs):
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
        if tmean:
            qdata=qdata.mean(dim='Time',keep_attrs=True,keepdims=True)
        Time=qdata.Time

        M = self.get_expt_resolution()
        qgrid=M(qdata.values)
        if np.ndim(qgrid)==2:
            qgrid=qgrid[np.newaxis,:,:]
        qgrid=xr.DataArray(qgrid,
                dims=('Time', 'lat', 'lon'),
                coords=dict(Time=Time,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        if inmassunit:
            qgrid.name='q'
            qgrid.attrs['units']='kg/s/m^2'
            qgrid.attrs['showname']='q (kg/s/m^2)'
            qgrid.attrs['longshowname']='freezing rate (kg/s/m^2)'
        else:
            qgrid=qgrid/1000*(86400*360*1e6)/1000
            qgrid.name='q'
            qgrid.attrs['units']='km/Myr'
            qgrid.attrs['showname']='q (km/Myr)'
            qgrid.attrs['longshowname']='freezing rate (km/Myr)'
        return qgrid


    def S(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,iz=None, tmean=True):
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
        Time=Sdata.Time
        if tmean:
            Sdata=Sdata.mean(dim='Time',keep_attrs=True,keepdims=True)
            Time=np.mean(Time,keepdims=True)
        if iz!=None:
            Sdata=Sdata.isel(z=iz)
        z=Sdata.z

        M = self.get_expt_resolution()
        Sgrid=M(Sdata.values)

        Sgrid=xr.DataArray(Sgrid,
                dims=('Time','z', 'lat', 'lon'),
                coords=dict(Time=Time,z=z,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        Sgrid.name='S'
        Sgrid.attrs['units']='psu'
        Sgrid.attrs['showname']='S (psu)'
        Sgrid.attrs['longshowname']='Salinity (psu)'
        return Sgrid

    def Rho(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,Sgrid=None,Tgrid=None,tmean=True,iz=None):
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
        if not 'rray' in type(Sgrid).__name__:
            Sgrid=self.S(runpath=runpath,iteration=iteration,sample=sample,tmean=tmean,iz=iz)
        if not 'rray' in type(Tgrid).__name__:
            Tgrid=self.T(runpath=runpath,iteration=iteration,sample=sample,tmean=tmean,iz=iz)
        Tgrid['Time']=Sgrid.Time

        density=self.rhoref*(self.beta*(Sgrid-self.Sref)-self.alpha*(Tgrid-self.Tref))
        density.name='Rho'
        density.attrs['units']='kg/m3'
        density.attrs['showname']='Rho (kg/m3)'
        density.attrs['longshowname']='density (kg/m3)'
        return density
    
    def Psi(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,vv=None,uu=None,areamul=False,domask=True,iz=None,unitkg=True,**kwargs):
        bklF=loadmat(self.bklfile)
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
        if not areamul:
            if self.deepatm:
                dxg=self.dxg[np.newaxis,:,:]*np.array(self.deepfacc)[:,np.newaxis,np.newaxis]
                dyg=self.dyg[np.newaxis,:,:]*np.array(self.deepfacc)[:,np.newaxis,np.newaxis]
            else:
                dxg=self.dxg[np.newaxis,:,:]*np.array(self.deepfacc)[:,np.newaxis,np.newaxis]/np.array(self.deepfacc)[:,np.newaxis,np.newaxis]
                dyg=self.dyg[np.newaxis,:,:]*np.array(self.deepfacc)[:,np.newaxis,np.newaxis]/np.array(self.deepfacc)[:,np.newaxis,np.newaxis]

            hFacS = mu.rdmds(self.path+runpath + "hFacS")
            hFacW = mu.rdmds(self.path+runpath + "hFacW")
            delM=self.drf
        else:
            dxg=np.ones([len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])])
            dyg=dxg
            hFacS=dxg
            hFacW=dxg
            delM=np.ones_like(self.drf)

        nr=len(delM)
        ncx=np.size(dxg,2)
        nc=np.size(dxg,1)
        ylat=bklF['bkl_Ylat']
        ydim=np.size(ylat)

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
        if uu is None:
            uu,Time,meta=mu.rdmds(self.path+runpath+'U',iteration,returnmeta=True)
        else:
            Time=uu.Time
            uu=uu.values
        if vv is None:
            vv=mu.rdmds(self.path+runpath+'V',iteration)
        else:
            vv=vv.values
        #uu,Time,meta=mu.rdmds(self.path+runpath+'dynDiag',iteration,returnmeta=True,rec=0)
        #vv=mu.rdmds(self.path+runpath+'dynDiag',iteration,rec=1)
        Nit=len(Time)
        if uu.ndim==3:
            uu=uu[np.newaxis,:,:,:]
            vv=vv[np.newaxis,:,:,:]
        if tmean and Nit>1:
            uu=np.mean(uu,axis=0,keepdims=True)
            vv=np.mean(vv,axis=0,keepdims=True)
            Time=np.mean(Time,keepdims=True)
            Nit=1

        uu=np.reshape(np.transpose(uu,[0,1,2,3]),[Nit,nr,ncx*nc],order='C')
        vv=np.reshape(np.transpose(vv,[0,1,2,3]),[Nit,nr,ncx*nc],order='C')
        hFacW=np.reshape(np.transpose(hFacW,[0,1,2]),[nr,ncx*nc],order='C')
        hFacS=np.reshape(np.transpose(hFacS,[0,1,2]),[nr,ncx*nc],order='C')
        dxg=np.reshape(np.transpose(dxg,[0,1,2]),[nr,ncx*nc],order='C')
        dyg=np.reshape(np.transpose(dyg,[0,1,2]),[nr,ncx*nc],order='C')

        psi=np.zeros([Nit,1,nr+1,ydim+2])
        mskZ=np.zeros([1,nr+1,ydim+2])
        mskV=np.zeros([1,nr,ydim+2])
        mskG=np.zeros([1,nr,ydim+1])
        ufac=np.zeros((1,)+np.shape(bklF['bkl_Flg']))
        vfac=np.zeros((1,)+np.shape(bklF['bkl_Flg']))
        ufac[0,:,:]=np.fmod(bklF['bkl_Flg'],2)
        vfac[0,:,:]=np.fix(bklF['bkl_Flg']/2)

        for nt in range(Nit):
            for k in range(nr-1,-1,-1):
                ut=dyg[k,:]*uu[nt,k,:]*hFacW[k,:]
                vt=dxg[k,:]*vv[nt,k,:]*hFacS[k,:]
                
                for jl in range(ydim):
                    ie=bklF['bkl_Npts'][jl].squeeze()
                    IJuv=bklF['bkl_IJuv'][:ie,jl]-1
                    vz=np.sum(ufac[0,:ie,jl]*ut[IJuv] + vfac[0,:ie,jl]*vt[IJuv])
                    psi[nt,0,k,jl+1]=psi[nt,0,k+1,jl+1]-vz*delM[k]

        if domask:
            ufac=abs(ufac) 
            vfac=abs(vfac)
            for jl in range(ydim):
                ie=bklF['bkl_Npts'][jl].squeeze()
                IJuv=bklF['bkl_IJuv'][:ie,jl]-1
                hw=np.zeros([nr,ie])
                hs=np.zeros([nr,ie])
                hw=hFacW[:,IJuv]
                hs=hFacS[:,IJuv]
                for k in range(nr):
                    tmpv=ufac[0,:ie,jl]*hw[k,:]+vfac[0,:ie,jl]*hs[k,:]
                    mskV[0,k,jl+1]=mskV[0,k,jl+1]+max(tmpv)
            mskV=np.ceil(mskV)
            mskV=np.minimum(1,mskV)
            mskG=mskV[:,:,:ydim+1]+mskV[:,:,1:ydim+2]
            mskG=np.minimum(1,mskG)
            mskZ[:,1:nr+1,:]=mskV
            mskZ[:,:nr,:]=mskZ[:,:nr,:]+mskV
            mskZ[:,:,:ydim]=mskZ[:,:,:ydim]+mskZ[:,:,1:ydim+1]
            mskZ[:,:,1:ydim+1]=mskZ[:,:,1:ydim+1]+mskZ[:,:,2:ydim+2]
            mskZ=np.ceil(mskZ)
            mskZ=np.minimum(1,mskZ)

            psi=np.where(mskZ!=0,psi,np.NaN)

        if not iz is None:
            psi=psi[:,:,iz,:]
            zz=self.zf[iz]
#            if len(iz)==1:
#                psi=psi[:,:,np.newaxis,:]
#                zz=[zz]
        else:
            zz=self.zc

        yi=np.hstack((-90,ylat.squeeze(),90))
        intp=interp1d(yi,psi)
        Psigrid=intp(self.yq)
        Psigrid=Psigrid.squeeze()
        if unitkg:
            Psigrid=Psigrid*self.rhoref
        if len(zz)==1 and Nit==1:
            Psigrid=Psigrid[np.newaxis,np.newaxis,:]
        elif Nit==1:
            Psigrid=Psigrid[np.newaxis,:,:]
        elif len(zz)==1:
            Psigrid=Psigrid[:,np.newaxis,:]

        if len(zz)>1:
            Psigrid=(Psigrid[:,:-1,:]+Psigrid[:,1:,:])/2.
        Psigrid=xr.DataArray(Psigrid,
                dims=('Time','z', 'lat'),
                coords=dict(Time=Time,z=self.zc,
                        lat=self.yq[:,0]))
        Psigrid.name='Psi'
        Psigrid.attrs['units']='kg/s'
        Psigrid.attrs['showname']='Psi (kg/s)'
        Psigrid.attrs['longshowname']='Meridional Streamfunction (kg/s)'
        return Psigrid

    def Psi_tot(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,iz=None):
        PsiGM=self.Psi_GM(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath)
        PsiEu=self.Psi(iteration=iteration,runnum=runnum,sample=sample,runpath=runpath)
        PsiGM['Time']=PsiEu.Time
        PsiGM['lat']=PsiEu.lat
        PsiGM['z']=PsiEu.z
        Psi=PsiGM+PsiEu
        Psi.attrs['units']=PsiEu.attrs['units']
        Psi.attrs['longshowname']='Total meridional streamfunction ('+Psi.attrs['units']+')'
        Psi.attrs['showname']='Psi_tot ('+Psi.attrs['units']+')'
        return Psi

    def Psi_GM(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,tmean=True,iz=None):
        if self.GM:
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
            PsiGM=self.get_var(runpath+'dynDiag',iteration=iteration,sample=sample,rec=13)
            if tmean:
                PsiGM=PsiGM.mean(dim='Time',keep_attrs=True,keepdims=True)
                Time=np.mean(PsiGM.Time,keepdims=True)
            if iz!=None:
                PsiGM=PsiGM.isel(z=iz)
            z=PsiGM.z

            M = self.get_expt_resolution()
            PsiGMgrid=M(PsiGM.values)

            PsiGMgrid=xr.DataArray(PsiGMgrid,
                    dims=('Time','z', 'lat', 'lon'),
                    coords=dict(Time=Time,z=z,
                            lat=self.yq[:,0],lon=self.xq[0,:]))
            #PsiGMgrid.values=filters.gaussian_filter(tmp,[0.8,0.8],mode='constant',cval=0)[np.newaxis,:,:,np.newaxis]
            PsiGMgrid=PsiGMgrid.mean('lon')*self.rhoref*(2*np.pi*self.a*np.cos(np.radians(self.yq[np.newaxis,np.newaxis,:,0])))
            PsiGMgrid=PsiGMgrid.fillna(0.0)
            PsiGMgrid.name='Psi_GM'
            PsiGMgrid.attrs['units']='kg/s'
            PsiGMgrid.attrs['showname']='Psi_GM (kg/s)'
            PsiGMgrid.attrs['longshowname']='GM-induced meridional stream function (kg/s)'
        else:
            PsiGMgrid=np.zeros(np.shape(self.hfaccgrid.values[np.newaxis,:,:,:]))
            PsiGMgrid.name='Psi_GM'
            PsiGMgrid.attrs['units']='kg/s'
            PsiGMgrid.attrs['showname']='Psi_GM (kg/s)'
            PsiGMgrid.attrs['longshowname']='GM-induced meridional stream function (kg/s)'

        return PsiGMgrid

    def _Psi(self,iteration=np.NaN,runnum=np.Inf,sample=1,runpath=None,Vgrid=None,tmean=True,iz=None):
        if not 'rray' in type(Vgrid).__name__:
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
            _,Vgrid=self.UV(runpath=runpath,iteration=iteration,sample=sample,tmean=tmean,iz=iz)
        print('finish Vgrid')
        if not 'lon' in Vgrid.dims:
            vdr=Vgrid*self.hfaccgrid.mean(dim='lon')*self.drf*2*np.pi*self.a*np.cos(np.radians(Vgrid.lat))*self.rhoref
            if self.deepatm:
                vdr=vdr*self.deepfacc[:, np.newaxis]
        else:
            vdr=Vgrid*self.hfaccgrid*self.drf*self.a*np.cos(np.radians(Vgrid.lat))*np.radians(self.xq[0,1]-self.xq[0,0])*self.rhoref
            if self.deepatm:
                vdr=vdr*self.deepfacc[:, np.newaxis,np.newaxis]
        print('integrating Psi')
        Psigrid=-(vdr.reindex(z=vdr.z[::-1])).cumsum(dim='z').reindex(z=vdr.z).sum('lon')
        Psigrid=Psigrid.where(self.hfaccgrid>0.0)
        Psigrid.name='Psi'
        Psigrid.attrs['units']='kg/s'
        Psigrid.attrs['showname']='Psi (kg/s)'
        Psigrid.attrs['longshowname']='Meridional stream function (kg/s)'
        return Psigrid

    def get(self,varlist,iteration=np.NaN,runnum=np.Inf,runpathstr=None,sample=1,tmean=True,iz=None,dimmeths={}):
        dic={}
        codetemplate1="""dic['varname']=self.varname(iteration=it_,runnum=rn_,runpath=rp_,sample=sp_,tmean=tm_,iz=iz_)"""
        for var in varlist:
            print('reading {}'.format(var))
            if not var in dic:
                code=codetemplate1.replace('varname',var).replace('it_',str(iteration)).replace('rn_',str(runnum)).replace('nan','np.NaN').replace('inf','np.Inf').replace('sp_',str(sample)).replace('rp_',str(runpathstr)).replace('tm_',str(tmean)).replace('iz_',str(iz))
                if var=='U':
                    if 'V' in dic:
                        code=code.replace("dic['U']=self.U","dic['U'],dic['V']=self.UV")
                    else:
                        code=code.replace("dic['U']=self.U","dic['U'],_=self.UV")

                if var=='V':
                    if 'U' in dic:
                        code=code.replace("dic['V']=self.V","dic['U'],dic['V']=self.UV")
                    else:
                        code=code.replace("dic['V']=self.V","_,dic['V']=self.UV")

                if var=='Rho' and 'T' in dic:
                    code=code.replace(")",",Tgrid=dic['T'])")
                if var=='Rho' and 'S' in dic:
                    code=code.replace(")",",Sgrid=dic['S'])")
                if var=='Psi' and 'V' in dic:
                    code=code.replace(")",",Vgrid=dic['V'])")

                exec(code)
            dic[var]=dimcomp(dic[var],key=var,dimmeths=dimmeths)
        return dic

    def monitor(self,iteration=np.Inf,runnum=-1,runpathstr=None,pltarray=(3,2),wdirs=[0,0,0,1,1,1],pltvar=['T','S','Rho','U','W','Psi'],dimmeths={'Time':'mean','lon':'mean','lat':None,'z':None},figsize=(12,12),projection=None,pltcontour={},flip=True,pltdir='F',labelorientation=None,xlabelpad=0,ylabelpad=20,labelpad=0.04,labelshrink=1,labelaspect=20,linetype=['k-']*20,sharex=True,sharey=True,xlims=None,ylims=None,vminmax={},cms={},savefig=0,sample=3,tmean=True,iz=None,alwaysrefmean=False):
        d=self.get(pltvar,runpathstr=runpathstr,iteration=iteration,sample=sample,tmean=tmean,iz=iz,runnum=runnum)
        its=d[pltvar[0]].Time/self.dt*86400.0
        its=np.round(its.values)
        print('its={}-{}'.format(its[0],its[-1]))
        fig,axf,da,figname=my2dplt(self,d,pltarray,wdirs=wdirs,pltvar=pltvar,dimmeths=dimmeths,figsize=figsize,projection=projection,pltcontour=pltcontour,flip=flip,pltdir=pltdir,labelorientation=labelorientation,xlabelpad=xlabelpad,ylabelpad=ylabelpad,labelpad=labelpad,labelshrink=labelshrink,labelaspect=labelaspect,linetype=linetype,sharex=sharex,sharey=sharey,xlims=xlims,ylims=ylims,vminmax=vminmax,cms=cms,savefig=savefig,alwaysrefmean=alwaysrefmean)
        return fig,axf,da

    def diffusivity_eq(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=5,doplt=False):
        slat1=np.sin(np.radians(self.yc))
        P1=slat1
        P2=1.5*slat1**2-0.5
        P3=2.5*slat1**3-1.5*slat1
        self.Hice=self.Hice0+self.Hice_P1*P1+self.Hice_P2*P2+self.Hice_P3*P3
        T_bar=self.T(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample).mean(dim=('lon','Time')).values
        dTv=T_bar[-5,:]-T_bar[4,:]
        H_surf=self.heatflux_surf(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample).mean(dim=('Time','lon')).values
        kappav_eq=H_surf/dTv/(self.cp*self.rhoref)*self.D
        
        if doplt:
            with plt.style.context(('labelsize15')):
                fig, ax = plt.subplots(1,1)
                im=ax.plot(self.yc[:,0],kappav_eq,color='black')
                ax.set_title('vertical diffusivity',loc='right')
                plt.xlabel('lat')

        return(kappav_eq)



    def heatbudget(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=5):
        H_surf=self.heatflux_surf(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample).mean(dim=('Time','lon')).values
        H_ocny=self.bulk_heatflux_y(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample).values
        
        # devide the whole globe into 3 equal pieces
        #boollow=(abs(self.yq)<=self.thetaeq3)
        #boolhigh=((self.yq)>self.thetaeq3)
        #indbndN=np.argmin(abs(self.yq-self.thetaeq3))
        #indbndS=np.argmin(abs(-self.yq-self.thetaeq3))

        # devide the NH into 2 equal pieces
        boollow=(self.yq<=self.thetaeq2) & (self.yq>=0)
        boolhigh=(self.yq>self.thetaeq2)
        indbndN=np.argmin(abs(self.yq-self.thetaeq2))
        indbndS=np.argmin(abs(-self.yq-self.thetaeq2))

        zonalbandarea=np.sum(self.rac*((self.a-(self.rhoice/self.rhoref)*self.Hice)/self.a)**(2*self.deepatm),axis=1)
        arealow=np.sum(zonalbandarea[boollow])
        areahigh=np.sum(zonalbandarea[boolhigh])
        zonalbandarea_inter=np.sum(self.rac*(self.ainter/self.a)**(2*self.deepatm),axis=1)
        arealow_inter=np.sum(zonalbandarea_inter[boollow])
        areahigh_inter=np.sum(zonalbandarea_inter[boolhigh])

        scale=1e3 # W to mW
        dheat=dict(H_ocny=H_ocny[[indbndS,indbndN]]*1e6/np.array([arealow_inter,areahigh_inter])*scale,
                H_surf=[np.sum(H_surf[boollow]*zonalbandarea[boollow])/arealow_inter*scale,np.sum(H_surf[boolhigh]*zonalbandarea[boolhigh])/areahigh_inter*scale],
                Qbot=[np.sum(self.Q[boollow]*zonalbandarea_inter[boollow])/arealow_inter*scale,np.sum(self.Q[boolhigh]*zonalbandarea_inter[boolhigh])/areahigh_inter*scale],
                Htide=[np.sum(self.Htide[boollow]*zonalbandarea[boollow])/arealow_inter*scale,np.sum(self.Htide[boolhigh]*zonalbandarea[boolhigh])/areahigh_inter*scale],
                Hcond=[np.sum(self.Hcond[boollow]*zonalbandarea[boollow])/arealow_inter*scale,np.sum(self.Hcond[boolhigh]*zonalbandarea[boolhigh])/areahigh_inter*scale]
                )
        return dheat


    def bulk_heatflux_y_adv(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None,tmean=False):
        advflux_y=self.bulk_heatflux_y(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample,doplt=doplt,component='adv',layers=layers,tmean=tmean)
        return advflux_y

    def bulk_heatflux_y_dif(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None,tmean=False):
        difflux_y=self.bulk_heatflux_y(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample,doplt=doplt,component='dif',layers=layers,tmean=tmean)
        return difflux_y

    def bulk_heatflux_y(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,component='advdif',doplt=False,layers=None,tmean=False,**kwargs):
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

        if 'adv' in component:
            print('reading ADV_TH...')
            ADVy_TH,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=0)
            ADVx_TH,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=10)
            print('its={}-{}'.format(its[0],its[-1]))
            if len(its)==1:
                ADVy_TH=ADVy_TH[np.newaxis,:,:,:]
                ADVx_TH=ADVx_TH[np.newaxis,:,:,:]
            Time = np.array(its)*self.dt/86400
            if tmean and len(its)>1:
                ADVy_TH=np.mean(ADVy_TH,axis=0,keepdims=True)
                ADVx_TH=np.mean(ADVx_TH,axis=0,keepdims=True)
                Time=np.mean(Time,keepdims=True)
        else:
            ADVy_TH=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))
            ADVx_TH=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))

        if 'dif' in component:
            print('reading ADV_TH...')
            DFyE_TH,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=2)
            DFxE_TH,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=12)
            print('its={}-{}'.format(its[0],its[-1]))
            if len(its)==1:
                DFyE_TH=DFyE_TH[np.newaxis,:,:,:]
                DFxE_TH=DFxE_TH[np.newaxis,:,:,:]
            Time = np.array(its)*self.dt/86400
            if tmean and len(its)>1:
                DFyE_TH=np.mean(DFyE_TH,axis=0,keepdims=True)
                DFxE_TH=np.mean(DFxE_TH,axis=0,keepdims=True)
                Time=np.mean(Time,keepdims=True)
        else:
            DFyE_TH=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))
            DFxE_TH=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))

        Fy_TH=DFyE_TH+ADVy_TH
        Fx_TH=DFxE_TH+ADVx_TH
        Fx_TH=xr.DataArray(Fx_TH,
                dims=('Time', 'z', 'y', 'x'),
                coords={"Time":Time,"z":self.zc,
                    "lat":(['y','x'],self.yc),
                    "lon":(['y','x'],self.xc)})
        Fy_TH=xr.DataArray(Fy_TH,
                dims=('Time', 'z', 'y', 'x'),
                coords={"Time":Time,"z":self.zc,
                    "lat":(['y','x'],self.yc),
                    "lon":(['y','x'],self.xc)})
        Heatflux_y=-self.Psi(uu=Fx_TH,vv=Fy_TH,areamul=True,unitkg=False,domask=False,iz=[0],tmean=tmean).squeeze()
        Heatflux_y=Heatflux_y*self.rhoref*self.cp/1e9
        Heatflux_y.name='Heatflux_y'
        Heatflux_y.attrs['units']='GW'
        Heatflux_y.attrs['showname']='F_Heat_y (GW)'
        Heatflux_y.attrs['longshowname']='Meridional Heat Flux (GW)'

        if doplt:
            with plt.style.context(('labelsize15')):
                fig, ax = plt.subplots(1,1)
                im=ax.plot(Heatflux_y.lat,Heatflux_y,color='black')
                ax.plot(Heatflux_y.lat,Heatflux_y*0.,'k--')
                ax.set_title(Heatflux_y.longshowname,loc='right')
                plt.xlabel('lat')
        
        return Heatflux_y
        

    def bulk_saltflux_y_adv(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None,tmean=False):
        advflux_y=self.bulk_saltflux_y(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample,doplt=doplt,component='adv',layers=layers,tmean=tmean)
        return advflux_y

    def bulk_saltflux_y_dif(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,doplt=False,layers=None,tmean=False):
        difflux_y=self.bulk_saltflux_y(iteration=iteration,runnum=runnum,runpath=runpath,sample=sample,doplt=doplt,component='dif',layers=layers,tmean=tmean)
        return difflux_y

    def bulk_saltflux_y(self,iteration=np.NaN,runnum=np.Inf,runpath=None,sample=1,component='advdif',doplt=False,layers=None,tmean=False,**kwargs):
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

        if 'adv' in component:
            print('reading ADV_SLT...')
            ADVy_SLT,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=5)
            ADVx_SLT,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=11)
            print('its={}-{}'.format(its[0],its[-1]))
            if len(its)==1:
                ADVy_SLT=ADVy_SLT[np.newaxis,:,:,:]
                ADVx_SLT=ADVx_SLT[np.newaxis,:,:,:]
            Time = np.array(its)*self.dt/86400
            if tmean and len(its)>1:
                ADVy_SLT=np.mean(ADVy_SLT,axis=0,keepdims=True)
                ADVx_SLT=np.mean(ADVx_SLT,axis=0,keepdims=True)
                Time=np.mean(Time,keepdims=True)
        else:
            ADVy_SLT=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))
            ADVx_SLT=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))

        if 'dif' in component:
            print('reading ADV_SLT...')
            DFyE_SLT,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=7)
            DFxE_SLT,its,meta = mu.rdmds(self.path + runpath + 'flxDiag', iteration,returnmeta=True,rec=13)
            print('its={}-{}'.format(its[0],its[-1]))
            if len(its)==1:
                DFyE_SLT=DFyE_SLT[np.newaxis,:,:,:]
                DFxE_SLT=DFxE_SLT[np.newaxis,:,:,:]
            Time = np.array(its)*self.dt/86400
            if tmean and len(its)>1:
                DFyE_SLT=np.mean(DFyE_SLT,axis=0,keepdims=True)
                DFxE_SLT=np.mean(DFxE_SLT,axis=0,keepdims=True)
                Time=np.mean(Time,keepdims=True)
        else:
            DFyE_SLT=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))
            DFxE_SLT=np.zeros((1,len(self.zc),len(self.yc[:,0]),len(self.xc[0,:])))

        Fy_SLT=DFyE_SLT+ADVy_SLT
        Fx_SLT=DFxE_SLT+ADVx_SLT
        Fx_SLT=xr.DataArray(Fx_SLT,
                dims=('Time', 'z', 'y', 'x'),
                coords={"Time":Time,"z":self.zc,
                    "lat":(['y','x'],self.yc),
                    "lon":(['y','x'],self.xc)})
        Fy_SLT=xr.DataArray(Fy_SLT,
                dims=('Time', 'z', 'y', 'x'),
                coords={"Time":Time,"z":self.zc,
                    "lat":(['y','x'],self.yc),
                    "lon":(['y','x'],self.xc)})
        Saltflux_y=-self.Psi(uu=Fx_SLT,vv=Fy_SLT,areamul=True,unitkg=False,domask=False,iz=[0],tmean=tmean).squeeze()
        Saltflux_y=Saltflux_y/1e3
        Saltflux_y.name='Saltflux_y'
        Saltflux_y.attrs['units']='kg/s'
        Saltflux_y.attrs['showname']='F_Salt_y (kg/s)'
        Saltflux_y.attrs['longshowname']='Meridional Salt Flux (kg/s)'

        if doplt:
            with plt.style.context(('labelsize15')):
                fig, ax = plt.subplots(1,1)
                im=ax.plot(Saltflux_y.lat,Saltflux_y,color='black')
                ax.plot(Saltflux_y.lat,Saltflux_y*0.,'k--')
                ax.set_title(Saltflux_y.longshowname,loc='right')
                plt.xlabel('lat')
        
        return Saltflux_y

    def Htide_model(self,**kwargs):
        Htide=self.Htide
        Htide=xr.DataArray(Htide,
                dims=('lat', 'lon'),
                coords=dict(lat=self.yq[:,0],lon=self.xq[0,:]))
        Htide.name='Modeled ice dissipation'
        Htide.attrs['units']='W/m^2'
        Htide.attrs['showname']='Htide_mod (W/m^2)'
        Htide.attrs['longshowname']='Modeled ice dissipation (W/m^2)'
        return Htide

    def Htide_imply(self,runpath=None,runnum=np.Inf,iteration=np.NaN,sample=3,removeq=True,q=None,tmean=True,**kwargs):
        H_ocn=self.heatflux_surf(runpath=runpath,runnum=runnum,iteration=iteration,sample=sample,removeq=removeq,q=q,tmean=tmean)
        Hlatent=self.Hlatent(runpath=runpath,runnum=runnum,iteration=iteration,sample=sample,tmean=tmean)
        Htide_imply=H_ocn
        Htide_imply.values=self.Hcond-H_ocn-Hlatent
        Htide_imply.name='Implied ice dissipation'
        Htide_imply.attrs['units']='W/m^2'
        Htide_imply.attrs['showname']='Htide_imp (W/m^2)'
        Htide_imply.attrs['longshowname']='Implied ice dissipation (W/m^2)'
        return Htide_imply

    def heatflux_surf(self,runpath=None,runnum=np.Inf,iteration=np.NaN,sample=3,removeq=True,q=None,tmean=True,**kwargs):
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
        ForcTdata=self.get_var(runpath+'shiDiag',iteration=iteration,sample=sample,dimnum=2,rec=1)
        if tmean and len(ForcTdata.Time)>1:
            ForcTdata=ForcTdata.mean(dim='Time',keepdims=True)
        M = self.get_expt_resolution()
        ForcTgrid=M(ForcTdata.values)
        ForcTgrid=-ForcTgrid
        Time=ForcTdata.Time

        if removeq:
            if q is None:
                q=self.q(runpath=runpath,runnum=runnum,iteration=iteration,sample=sample,inmassunit=True)
            else:
                if q.units=='km/Myr':
                    q=q*1e3/1e6/360/86400*self.rhoref
            if tmean and len(q.Time)>1:
                q=q.mean(dim='Time',keepdims=True)
            ForcTgrid=ForcTgrid*(self.rhoref*self.gammaT/(self.rhoref*self.gammaT-q))

        ForcTgrid=xr.DataArray(ForcTgrid,
                dims=('Time','lat', 'lon'),
                coords=dict(Time=Time,
                        lat=self.yq[:,0],lon=self.xq[0,:]))
        ForcTgrid.name='Ice-ocean heat exchange'
        ForcTgrid.attrs['units']='W/m^2'
        ForcTgrid.attrs['showname']='Hocn (W/m^2)'
        ForcTgrid.attrs['longshowname']='Ice-ocean heat exchange (W/m^2)'
        return ForcTgrid
        

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
        dims = ('Time', 'z', 'y', 'x')
        coords = {"Time":Time,
                  "z":self.zc,
                  "lat":(('y','x'),self.yc),
                  "loni":(('y','x'),self.xg)}
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
            dims=('z', 'y', 'x'),
            coords={"z":(('z'),self.zc),
                    "lat":(('y','x'),self.yc),
                    "lon":(('y','x'),self.xc)})
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


def dimcomp(var,key=None,dimmeths={}):
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
    return var

def my2dplt(expt,d,pltarray,wdirs=np.zeros(30),pltvar=None,dimmeths={'Time':'mean','lon':'mean','lat':None,'z':None},figsize=(12,12),projection=None,pltcontour={},flip=True,pltdir='F',labelorientation=None,xlabelpad=0,ylabelpad=20,labelaspect=20,labelshrink=1,labelpad=0.04,linetype=['k-']*20,sharex=True,sharey=True,xlims=None,ylims=None,vminmax={},cms={},savefig=0,alwaysrefmean=False):
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
            var=dimcomp(d[key],key,dimmeths)

#            for dim in allvardim:
#                if not dim in dimmeths:
#                    continue
#                dimmeth=dimmeths[dim]
#                if dimmeth==None:
#                    continue
#                elif dimmeth=='mean':
#                    var=var.mean(dim=dim,keep_attrs=True)
#                elif dimmeth=='anom':
#                    if key=='T':
#                        var=var-expt.Tref
#                    if key=='S':
#                        var=var-expt.Sref
#                    if key=='rho':
#                        var=var-expt.rhoref
#                    var=var-var.mean(dim=dim,keep_attrs=True)
#                elif type(dimmeth) is not string:
#                    if type(dimmeth) is float:
#                        if dimmeth is np.Inf:
#                            var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)==len(var[dim])-1),drop=True).mean(dim=dim,keep_attrs=True)
#                        else:
#                            var=var.where(abs(var[dim]-dimmeth)==min(abs(var[dim]-dimmeth)),drop=True).mean(dim=dim,keep_attrs=True)
#                    elif type(dimmeth) is int:
#                        var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)==dimmeth),drop=True).mean(dim=dim,keep_attrs=True)
#                    elif type(dimmeth) is tuple or type(dimmeth) is list:
#                        if type(dimmeth[0]) is int:
#                            var=var.where((xr.DataArray(range(len(var[dim])),dims=dim)>=dimmeth[0])&(xr.DataArray(range(len(var[dim])),dims=dim)<=dimmeth[1]),drop=True)
#                        elif type(dimmeth[0]) is float:
#                            var=var.where((var[dim]>=dimmeth[0])&(var[dim]<=dimmeth[1]),drop=True)
#                        else:
#                            raise NameError('dimmeths format wrong.')
#                        if type(dimmeth) is list:
#                            var=var.mean(dim=dim,keep_attrs=True)
#                    else:
#                        raise NameError('dimmeths format wrong.')
#                else:
#                    raise NameError('dimmeths format wrong.')

            newvardim=var.squeeze(drop=True).dims  
            if len(newvardim)!=2 and len(newvardim)!=1:
                raise NameError('{} dimension doesn''t equal to 1 or 2'.format(key))

            if len(newvardim)==1:
                xcoord=var[newvardim[0]]
                if newvardim[0]=='lat':
                    var[0]=np.NaN
                    var[-1]=np.NaN
                if newvardim[0]=='z':
                    xcoord=xcoord/1e3
                    xcoord=xcoord+expt.Hice0/1e3*(not expt.topo)
                    if xlims==None and (not expt.topo):
                        xlims=[(-expt.Htot+expt.Hice0*(not expt.topo))/1e3,(expt.Hice0*(not expt.topo)-expt.Hice0*(not expt.topo))/1e3]
                if newvardim[0]=='Time':
                    xcoord=xcoord/360
                    if xlims==None:
                        xlims=[var.Time.min()/360,var.Time.max()/360]

                im=axf[iplt].plot(xcoord,var,linetype[iplt])
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
                    xcoord=xcoord+expt.Hice0/1e3*(not expt.topo)
                    if xlims==None and (not expt.topo):
                        xlims=[(-expt.Htot+expt.Hice0*(not expt.topo))/1e3,(expt.Hice0*(not expt.topo)-expt.Hice0*(not expt.topo))/1e3]
                if ydim=='z':
                    ycoord=ycoord/1e3
                    ycoord=ycoord+expt.Hice0/1e3*(not expt.topo)
                    if ylims==None and (not expt.topo):
                        ylims=[(-expt.Htot+expt.Hice0*(not expt.topo))/1e3,(expt.Hice0*(not expt.topo)-expt.Hice0*(not expt.topo))/1e3]
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
                if wdirs[iplt]==0:
                    if key=='S' and not alwaysrefmean: 
                        offset=expt.Sref
                    elif key=='T' and not alwaysrefmean:
                        offset=expt.Tref
                    else:
                        varmean=var.mean(dim=[xdim,ydim]).values
                        print('{} mean:{}'.format(key,varmean))
                        if np.isnan(varmean):
                            print(var)
                        if (not np.isnan(varmean)) and varmean!=0:
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
                    im=axf[iplt].pcolormesh(xcoord-xcoord.mean(),ycoord, var-offset,transform=ccrs.PlateCarree(),cmap=cm,vmin=v_min,vmax=v_max)
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

                if offset!=0:
                    if hasattr(var,'showname'):
                        title=var.showname.replace(')',',ref:{:g})'.format(offset))
                    else:
                        title='{} (ref:{:g})'.format(key,offset)

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
# v13_40psu_allcore_strongic=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic")
# v13_40psu_halfhalf=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_halfhalf")
# v13_40psu_allshell=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allshell")
# v13_40psu_allcore_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid")
# v13_40psu_allcore_strongic_merid_asym_1e_3vr=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-3vr")
# v13_40psu_allcore_strongic_merid_asym_1e_5vr=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-5vr")
# v13_40psu_allcore_strongic_merid_asym_1e_5vr_1e_2vh=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allcore_strongic_merid_asym_1e-5vr_1e-2vh")
# v13_40psu_halfhalf_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_halfhalf_strongic_merid")
# v13_40psu_allshell_strongic_merid=Experiment(base_dir="/net/fs08/d0/wanying/data_",name="enceladus_z59x90y648_v13_40psu_allshell_strongic_merid")
# 
# expts_highres=[v13_40psu_allcore_strongic,v13_40psu_halfhalf,v13_40psu_allshell, v13_40psu_allcore_strongic_merid, v13_40psu_allcore_strongic_merid_asym_1e_3vr,v13_40psu_allcore_strongic_merid_asym_1e_5vr,v13_40psu_allcore_strongic_merid_asym_1e_5vr_1e_2vh,v13_40psu_halfhalf_strongic_merid,v13_40psu_allshell_strongic_merid]

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
        cs = expt.c  #coldict[expt.Q0]
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

def save_netcdf(expt,dic,fname):
    ds=xr.Dataset(dic)
    ds.to_netcdf(expt.base_dir + fname)

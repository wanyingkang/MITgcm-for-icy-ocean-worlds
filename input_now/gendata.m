%% load packages
MITROOT='./'; % MITROOT should point to the directory where MITgcm is installed XX.
addpath([MITROOT,'/MITgcm/utils/matlab/'])
addpath(genpath([MITROOT,'/MITgcm/my_exp/gsw/'])); % gsw needs to be put under this path XX

%% initial condition
restartTS=0; % 0: don't use restart T/S, 1: use horizontal averaged profile, 2: use zonal mean, 3: use 3d, negative: don't write new IC files, 4: interpolate pickup files as IC, 5: interpolate pickup files as new restart file, 6: take diff between two states and use it to step current state forward to achieve faster convergence.
restartTSpath=''; % used when restartTS>0
restartname=''; % used when restartTS=0
gridpath='./'; % used when restartTS>3
restartTSiter=0; % NaN means time mean (for restartTS==1,2,3 only), otherwise need to specify an iteration number (for restartTS>0)

%% ice surface temperature
Ts0=62; % K, ice surface temperature, used to compute conductive heat loss.
meridionalTs=3; % 0: constant Ts, 1: Ojakangas and Stevenson 1989 approximation, 3: Nadeau and McGehee 2018
obl=27; % depending on the scheme, the obliquity of the moon may be used to calculate Ts profile

%% diffusion profile and boundary mixing rate
meridionalkappav=0; % 0: uniform kappav, 1: equator-pole grardient, 2: apply different boundary kappa wrt interior kappa, see code
kappav_interior=0.001;
kappav_bnd=1;
kappav_eq=0.01;
kappav_pole=0.01;
kappav_width=30/2; % deg
iceshelf_heatcoef=1e-5; % m/s, heat exchange rate between ice and ocean
iceshelf_saltcoef=1e-5; % m/s, salt exchange rate between ice and ocean
bottom_dragcoef=1e-4; % m/s, momentum exchange rate between core and ocean
iceshelf_dragcoef=1e-4; % m/s, momentum exchange rate between ice and ocean

%% domain
Htot=76e3;
nx=80 ; ny=672 ;
nr=70;
dx=0.25 ; dy=0.25 ; yyM=84 ; % yyM set latitudinal range yyM degS to yyM degN

%% ocean salinity, equation of state
Sref=60;
eos='LINEAR';
usePT=1; % consistent with linear EOS, density doesn't vary with pressure, also need to turn off the T->PT conversion in shelfice_thermodynamics.F

%% ice shell geometry
Hice0=20e3; Hice_P1=0e3; Hice_P2=-3e3; Hice_P3=0; % 0.025H0, 0.3H0
realtopo=0; % 0: dont use realtopo, 1: use zonal mean, 2: use the whole 2D field
realtopopath='/home/wanying/Hemingway_Mittal_2019_Enceladus_nominal_shell_thickness_Fig11d/Enceladus_nominal_shell_thickness_Fig11d.tab';
Mice_randic=0*10.0; % random ice shelf geometry amplitude (m)

%% heating terms
Htide0_portion=1; % portion of heat production in the ice shell, 1: 100% shell heating, 0: 100% core heating
twodtide=0; % account for zonal ice thickness variation
qbotvary=1; % 1: use poleward amplified bottom heating profile given by Beuthe 2019, 0: use uniform bottom heating
qbot0=-1; % when set to positive value, bottom heating rate will be set to qbot0; when set to negative value, bottom heating is set to make global heat budget match.
Htidemode=[0.250, 0.0825, -0.0834, -0.0546, -0.0562]; % ice dissipation profile (membrane mode), the 5 elements give the Y20, Y40, Y22, Y42, Y44 amp relative to Y00, calculated using Beuthe 2019 formula with Enceladus parameters
Hmixbendmode=[0.124,0.196,-0.0199,-0.0656,0.0132,0.0136]; % same as Htidemode except this is for mix+bend mode
addmixbend=1; % whether to include mix+bend mode
ptide=-2.0; % ice dissipation's power dependence on ice thickness
pcond=-1.0; % conductive heat loss rate's power dependence on ice thickness
HtidePs_ice=[0,0,0,0,0,0]; % virtual topography that only affect ice tidal dissipation and/or conductive heat loss
useHtidePsinHcond=0;
ptide_ext=-2.0; % same as ptide, but for HtidePs_ice
pcond_ext=-1.0; % same as pcond, but for HtidePs_ice if useHtidePsinHcond=1
tiltheat=0.0; % hemispherically asymmetric heating

%% ice freezing/melting mode
PrescribeFreezing=1; % 1: prescribe freezing/melting rate using thin shell ice flow model (see Kang and Flierl 2020), 0: allow ice to respond to ocean heat transport
EvolveIce=0; % 0: ice geometry doesn't evolve, 1: ice geometry evolve based on freezing/melting rate and ice flow amplified by SHIdtFactor
SHIdtFactor=1; % boost the total ice evolution tendency (including tendency induced by ice flow) by this factor to accelerate convergence.

%% physical constants
rhoice=917.; % kg/m3, ice density
kappa0=651; % W/m, ice heat conductivity constant
a0=250e3; % m, radius
G=6.67e-11; % gravitational const
rhocore=2500; % kg/m3, core density
rhoout=1000; % kg/m3, outer layer density
M=4*pi/3*((a0-Htot)^3*rhocore+(a0^3-(a0-Htot)^3)*rhoout);
g0=G*M/a0^2; % m/s2, gravity
eta_melt=3e14; % Pa s, viscosity at melting
eta_max_Q=Inf;
Ea=59.4e3; % J/mol, activation energy
Rg=8.31; % ideal gas constant

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SET UP END HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% -- save a copy
system('cp -f data data_back');
system('cp -f data.shelfice data.shelfice_back');

if (Hice_P1==0 && Hice_P2==0 && Hice_P3==0 && realtopo==0)
    appendix='flat'; % choose a name for the topography, IC profiles
else
    appendix='';
end

%% -- set up domain and legendre polynominals
nxy = nx*ny;
d2r=pi/180;
yc=-yyM+dy/2:dy:yyM; xc=[dx/2:dx:nx*dx]'-nx*dx/2;
yi=-yyM:dy:yyM; xi=[0:dx:nx*dx]'-nx*dx/2;
clat=cosd(yc);slat=sind(yc);
%dh= [18e3,2e3*ones([1,29])]; % https://kiss.caltech.edu/workshops/oceanworlds/presentations/Prockter.pdf, page 53
dh= [19e3,0.5e3*ones([1,52]),0.7e3,1.e3,1.3e3,1.7e3,2e3*ones([1,12]),2.3e3]; % total 76km
hf=[0,cumsum(dh)];
P1=slat;
P2=(3/2).*slat.^2-(1/2);
P3=(5/2).*slat.^3-(3/2).*slat;
P4=(35*slat.^4-30.*slat.^2+3)/8.0;
P5=(63.*slat.^5-70.*slat.^3+15.*slat)./8;
P6=(231.*slat.^6-315.*slat.^4+105.*slat.^2-5)/16.0;
Ps=[P1;P2;P3;P4;P5;P6];
wgt1=ny*(sind(yi(2:end))-sind(yi(1:end-1)))./(sind(yi(end))-sind(yi(1)));
wgt=repmat(wgt1,[nx,1]);
prec='real*4';

%% -- equation of state
P_inter0=rhoice*g0*Hice0;
Tfreeze0=gsw_t_freezing(Sref,P_inter0/1e4);
rhoNil=gsw_rho(Sref,Tfreeze0,P_inter0/1e4);
sBeta=gsw_beta(Sref,Tfreeze0,P_inter0/1e4); % P in unit of dbar
tAlpha=gsw_alpha(Sref,Tfreeze0,P_inter0/1e4);
pKappa=gsw_kappa_CT_exact(Sref,0,P_inter0/1e4);
fprintf('\nrhoNil=%.5g\n',rhoNil)
fprintf('\nsBeta=%.4g, tAlpha=%.4g, pKappa=%.4g\n',sBeta,tAlpha,pKappa)
func_replace_string('data','rhoNil',sprintf('rhoNil=%.3f,',rhoNil))
func_replace_string('data','tAlpha',sprintf('tAlpha=%.3g,',tAlpha))
func_replace_string('data','sBeta',sprintf('sBeta=%.3g,',sBeta))
func_replace_string('data','eosType',sprintf('eosType=%s,',eos))

%% -- gravity profile
rhobulk=g0/(G*4*pi/3*a0);
h=(hf(2:end)+hf(1:end-1))./2;
g=g0*a0^2*(1-rhoout/rhobulk*(a0^3-(a0-h).^3)./(a0^3))./(a0-h).^2;
func_replace_string('data','gravity=',sprintf('gravity=%f,',g0))
fprintf('gravity=%f',g0)
func_replace_string('data','rSphere=',sprintf('rSphere=%f,',a0))
fprintf('rSphere=%f\n',a0)

fname=['gravity_r_enceladus.bin'];
fid=fopen(fname,'w','b');
fwrite(fid,g,prec);
fclose(fid);
fprintf(['\nwrite file: ',fname]);
func_replace_string('data','gravityFile',['gravityFile=''',fname,''','])
fprintf(['gravityFile=''',fname,''','])

%% -- Bathy :
hh=ones(nx,ny);
hh(:,1)=zeros(nx,1);
hh(:,ny)=zeros(nx,1);
hh=-Htot*hh;

file_name=['flat_',sprintf('%d',Htot/1e3),'km.bin'] ;
fid=fopen(file_name,'w','b');
fwrite(fid,hh,prec); fclose(fid);
fprintf(['\nwrite file: ', file_name])
func_replace_string('data','bathyFile',['bathyFile=''',file_name,''','])
fprintf(['bathyFile=''',file_name,''','])

%% -- ice topography
if (realtopo==0)
    Hice=Hice0+Hice_P1.*P1+Hice_P2.*P2+Hice_P3.*P3;
    Hice=repmat(reshape(Hice,[1,ny]),[nx,1]);
else
    fid=fopen(realtopopath);
    Hice_=fread(fid,'double','ieee-be');
    Hice_=reshape(Hice_,[720,1440])*1e3;
    if realtopo==1
        y_=[-90:0.25:90-0.25]+0.25/2; y_=[-90,y_,90];
        Hice_=mean(Hice_,2);
        Hice_=[Hice_(1);Hice_;Hice_(end)];
        Hice=interp1(y_,Hice_,yc,'liear');
    else
        y_=[-90:0.25:90-0.25]+0.25/2; y_=[-90,y_,90];
        x_=[0:0.25:360];
        Hice_=[Hice_,Hice_(:,1)];
        Hice_=[Hice_(1,:);Hice_;Hice_(end,:)];
        [x2_,y2_]=meshgrid(x_,y_);
        [xc2,yc2]=meshgrid(xc,yc);
        Hice=interp2(x2_,y2_,Hice_,xc2,yc2);
    end
    Hice0=mean(mean(Hice.*wgt));
end

%% -- ice shelf topography and thickness
Hunder=-(rhoice/rhoNil).*Hice;
if (length(appendix)==0)
fname='icetopo.bin';
else
fname=['icetopo_',appendix,'.bin'];
end
fid=fopen(fname,'w','b');
fwrite(fid,Hunder,prec);
fclose(fid);
fprintf(['\nwrite file: ',fname]);
func_replace_string('data.shelfice','SHELFICEtopoFile',['SHELFICEtopoFile=''',fname,''','])
fprintf(['SHELFICEtopoFile=''',fname,''','])

Miceshelf=Hice.*rhoice;
if Mice_randic
    Miceshelf_=Miceshelf+rhoice.*smooth(rand(nx,ny)).*Mice_randic;
else
    Miceshelf_=Miceshelf;
end

if (length(appendix)==0)
fname='iceShelf_Mass.bin';
else
fname=['iceShelf_Mass_',appendix,'.bin'];
end
fid=fopen(fname,'w','b');
fwrite(fid,Miceshelf_,prec);
fclose(fid);
fprintf(['\nwrite file: ',fname]);
func_replace_string('data.shelfice','SHELFICEmassFile',['SHELFICEmassFile=''',fname,''','])
fprintf(['SHELFICEmassFile=''',fname,''','])

%% -- freezing temperature under varying ice topography
fa0= -0.0575;
fb= -7.61e-4;
fc0=0.0901;
P_thin=rhoice*g0*min(Hice(:));
P_thick=rhoice*g0*max(Hice(:));
P_bot=rhoout*g0*Htot;
Tfreeze_thin=gsw_t_freezing(Sref,P_thin/1e4);
Tfreeze_thick=gsw_t_freezing(Sref,P_thick/1e4);
Tfreeze0_modelcode=fa0*(Sref)+fc0+fb*P_inter0/1e4;
Tfreeze_thin_modelcode=fa0*(Sref)+fc0+fb*P_thin/1e4;
Tfreeze_thick_modelcode=fa0*(Sref)+fc0+fb*P_thick/1e4;
fprintf('\nTfreeze0=%.4g, Tfreeze_thin=%.4g,  Tfreeze_thick=%.4g',Tfreeze0,Tfreeze_thin,Tfreeze_thick)
fprintf('\nTfreeze0_mod=%.4g, Tfreeze_thin_mod=%.4g,  Tfreeze_thick_mod=%.4g',Tfreeze0_modelcode,Tfreeze_thin_modelcode,Tfreeze_thick_modelcode)
[Tfreeze0_SA, Tfreeze0_P] = gsw_t_freezing_first_derivatives(Sref,P_inter0/1e4);
func_replace_string('data','tRef',sprintf('tRef=%d*%.3f,',nr,Tfreeze0_modelcode))
func_replace_string('data','sRef',sprintf('sRef=%d*%.3f,',nr,Sref))

%p_correction=((1.-rhoout/rhobulk)./(1.-Hice/a0)+(rhoout/rhobulk).*(1.-Hice./a0/2.));
p_correction=1;
P_inter=Miceshelf.*(g0).*p_correction;
Tfreeze=fa0*(Sref)+fc0+fb*P_inter/1e4;

if usePT
    Thfreeze=fPT_T(Tfreeze,P_inter/1e4,Sref);
    % if turn on this, one also need to turn on the T->PT conversion in shelfice_thermodynamics.F
    % 40psu: approximately equivalent to minus 5mK; 4psu: to minus <3.5mK
    func_replace_string('data.shelfice','usePT','usePT=.TRUE.,')
    fprintf('usePT=.TRUE.')
else
    Thfreeze=Tfreeze;
    func_replace_string('data.shelfice','usePT','usePT=.FALSE.,')
    fprintf('usePT=.FALSE.')
end

%% -- heat conduction:
Ts=Ts0.*ones(1,ny);
if meridionalTs==1
ycr=yc.*d2r;
ipolar=abs(yc)>90-obl;
Ts(~ipolar)=Ts0.*(clat(~ipolar)).^(1/4);
Ts(ipolar)=Ts0.*(((pi/2-abs(ycr(ipolar))).^2+(obl*d2r)^2)./2).^(1/8);
elseif meridionalTs==3
    flux_prof=1-(5/8).*legendreP(2,cosd(obl)).*legendreP(2,slat)...
            -(9/64).*legendreP(4,cosd(obl)).*legendreP(4,slat)...
            -(65/1024).*legendreP(6,cosd(obl)).*legendreP(6,slat);
    Ts_prof=flux_prof.^(1/4);
    Ts=Ts_prof*Ts0;
end
func_replace_string('data.shelfice','meridionalTs',sprintf('meridionalTs=%d,',meridionalTs))
Tssmt=reshape(smooth(Ts),[1,ny]);
figure(1)
hold on
yyaxis right
plot(yc,Tssmt,'k-','LineWidth',2)
set(gca,'YColor','k','FontSize',16)
ylabel('Surface Temperature (K)')
xlabel('lat (deg)')

Hcond=kappa0.*log((Tfreeze0+273.15)./Ts)./Hice0;
Hcond=Hcond.*(Hice/Hice0).^pcond;
func_replace_string('data.shelfice','pcond=',['pcond=',sprintf('%f',pcond),','])
fprintf('pcond=%f,',pcond)
func_replace_string('data.shelfice','pcond_ext=',['pcond_ext=',sprintf('%f',pcond_ext),','])
fprintf('pcond_ext=%f,',pcond_ext)
if useHtidePsinHcond
    func_replace_string('data.shelfice','useHtidePsinHcond','useHtidePsinHcond=.TRUE.,')
    condtopo=1;
    for i=1:length(HtidePs_ice)
        condtopo=condtopo+HtidePs_ice(i)*Ps(i,:);
        func_replace_string('data.shelfice',sprintf('HP%d_ice',i),sprintf('HP%d_ice=%f,',i,HtidePs_ice(i)))
        fprintf('HP%d_ice=%f,',i,HtidePs_ice(i))
    end
    Hcond=Hcond.*(condtopo).^pcond_ext;
else
    func_replace_string('data.shelfice','useHtidePsinHcond','useHtidePsinHcond=.FALSE.,')
end

figure(1)
hold on
yyaxis left
plot(yc,1e3.*Hcond,'g-.','LineWidth',2)
Hcond0=mean(mean(Hcond.*wgt),2);
plot(yc,yc.*0+1e3.*Hcond0,'g-.')
fprintf('\nHcond=%.3g W/m^2', Hcond0)
func_replace_string('data.shelfice','SHELFICEkappa',sprintf('SHELFICEkappa=%.1f,',-kappa0))
fprintf('SHELFICEkappa=%.1f,',-kappa0)
func_replace_string('data.shelfice','SHELFICEthetaSurface',sprintf('SHELFICEthetaSurface=%.2f,',Ts0-273.15))
fprintf('SHELFICEthetaSurface=%.2f,',Ts0-273.15)
func_replace_string('data.shelfice','obliquity',sprintf('obliquity=%.2f,',obl))
fprintf('obliquity=%.2f,',obl)

%% -- tidal heating
Htidemean=Htide0_portion*Hcond0;
one=ones(nx,ny);
Y00=sqrt(1/4/pi).*one;
Y20=sqrt(5/4/pi)*(1.5*slat.^2-0.5).*one;
Y40=sqrt(9/4/pi)*(35/8*slat.^4-30/8*slat.^2+3/8).*one;
if nx>1 && twodtide
    c2lon=cosd(2.*xc);
    c4lon=cosd(4.*xc);
    Y22=sqrt(5/4/pi/24).*(3-3*slat.^2.).*(2*c2lon);
    Y42=sqrt(9/4/pi/360).*(7.5.*(7*slat.^2.-1).*(1-slat.^2)).*(2.*c2lon);
    Y44=sqrt(9/4/pi/40320).*(105.*clat.^4).*(2*c4lon);
else
    Y22=Y00.*0;
    Y42=Y00.*0;
    Y44=Y00.*0;
end
Htideprof=sqrt(4*pi).*(Y00+Htidemode(1).*Y20+Htidemode(2).*Y40+Htidemode(3).*Y22+Htidemode(4).*Y42+Htidemode(5).*Y44);
tidetopo=1;
for i=1:length(HtidePs_ice)
    tidetopo=tidetopo+HtidePs_ice(i)*Ps(i,:);
    func_replace_string('data.shelfice',sprintf('HP%d_ice',i),sprintf('HP%d_ice=%f,',i,HtidePs_ice(i)))
    fprintf('HP%d_ice=%f,',i,HtidePs_ice(i))
end
Htideprof=Htideprof.*tidetopo.^ptide_ext;
if addmixbend
    func_replace_string('data.shelfice','addmixbend','addmixbend=.TRUE.,')
    fprintf('addmixbend=.TRUE.')
    Hmixbendprof=sqrt(4*pi).*(Hmixbendmode(1)*Y00+Hmixbendmode(2).*Y20+Hmixbendmode(3).*Y40+Hmixbendmode(4).*Y22+Hmixbendmode(5).*Y42+Hmixbendmode(6).*Y44);
else
    func_replace_string('data.shelfice','addmixbend','addmixbend=.FALSE.,')
    fprintf('addmixbend=.FALSE.')
    Hmixbendmode=0;
end

Htidenorm=mean(mean(((Hice./Hice0).^ptide.*Htideprof + Hmixbendprof ).*wgt),2);
Htide0=Htidemean./Htidenorm;
fprintf('\nHtidemean=%.3g, Htide0=%.3g',Htidemean,Htide0)
func_replace_string('data.shelfice','Htide0',['Htide0=',sprintf('%f',Htide0),','])
fprintf('Htide0=%f,',Htide0)
if EvolveIce
    func_replace_string('data.shelfice','Htide0_portion',['Htide0_portion=',sprintf('%f',Htide0_portion),','])
    fprintf('Htide0_portion=%f,',Htide0_portion)
else
    func_replace_string('data.shelfice','Htide0_portion',['Htide0_portion=-1,']) % if ice is not evolving, there is no need to do online calculation of ice dissipation
    fprintf('Htide0_portion=%f,',Htide0_portion)
end
func_replace_string('data.shelfice','ptide=',['ptide=',sprintf('%f',ptide),','])
fprintf('ptide=%f,',ptide)
func_replace_string('data.shelfice','ptide_ext=',['ptide_ext=',sprintf('%f',ptide_ext),','])
fprintf('ptide_ext=%f,',ptide_ext)
if twodtide
    func_replace_string('data.shelfice','tide2d','tide2d=.TRUE.,')
    fprintf('tide2d=.TRUE.')
else
    func_replace_string('data.shelfice','tide2d','tide2d=.FALSE.,')
    fprintf('tide2d=.FALSE.')
end

func_replace_string('data.shelfice','tiltheat',['tiltheat=',sprintf('%f',tiltheat),',']) % hemispheric asymmetric forcing, see code

figure(1)
hold on
yyaxis left
plot(yc,Htideprof.*1e3.*Hcond0./Htidenorm,'r-','LineWidth',2)
set(gca,'YColor','k','FontSize',16)
box on
yl=ylim;
ylim([0 yl(2)])
ylabel('Heat flux (mW/m^2)')

%% -- Bottom Heat :
if qbot0<0
    qbot_inter=(Hcond0-Htidemean);
    qbot0=qbot_inter*(a0-Hice0*(rhoice/rhoNil))^2/(a0-Htot)^2;
end
qbot=qbot0*ones(nx,ny);
% meridional variation
if qbotvary
% qprofile= 1-25/2/(60-25/2).*(2.*clat.^2-1);
qprofile= 1.08449 + 0.252257*cosd(2*(90-yc)) + 0.00599489*cosd(4*(90-yc));
qprofile=qprofile/mean(qprofile.*wgt1);
qbot=qbot.*qprofile;
else
qprofile=1;
end
qbot=qbot(:);


file_name=sprintf('Q_bottom_%dmW.bin', round(qbot0*1e3)) ;
fid=fopen(file_name,'w','b');
fwrite(fid,qbot,prec); fclose(fid);
fprintf(['\nwrite file: ', file_name])
func_replace_string('data','geothermalFile',['geothermalFile=''',file_name,''','])
fprintf(['geothermalFile=''',file_name,''','])

figure(1)
hold on
yyaxis left
plot(yc,1e3.*qprofile.*Hcond0,'Color',[148,55,255]./256,'LineStyle','--','LineWidth',2)
xlim([yc(1), yc(end)])
saveas(gcf,'heat_profile.png')


%% -- Surface and Bottom masks :
%
%var=zeros(nxy,nr);
%var(:,nr)=1;
%fname='rbcs_bottom.bin';
%fid=fopen(fname,'w','b'); fwrite(fid,var,prec); fclose(fid);
%fprintf(['\nwrite file: ',fname]);
%
%var=zeros(nxy,nr);
%var(:,1)=1;
%fname='rbcs_surf.bin';
%fid=fopen(fname,'w','b'); fwrite(fid,var,prec); fclose(fid);
%fprintf(['\nwrite file: ',fname]);

%% -- interface exchange rate
 Cp=4000;
 Tfreeze=Tfreeze+qbot_inter/(iceshelf_heatcoef*rhoNil*Cp);
 func_replace_string('data','HeatCapacity_Cp',sprintf('HeatCapacity_Cp=%.3g,',Cp))
 fprintf('\nHeatCapacity_Cp=%.3g,',Cp)
 func_replace_string('data.shelfice','SHELFICEheatTransCoeff',sprintf('SHELFICEheatTransCoeff=%.3g,',iceshelf_heatcoef))
 fprintf('SHELFICEheatTransCoeff=%.3g,',iceshelf_heatcoef)
 func_replace_string('data.shelfice','SHELFICEsaltTransCoeff',sprintf('SHELFICEsaltTransCoeff=%.3g,',iceshelf_saltcoef))
 fprintf('SHELFICEsaltTransCoeff=%.3g,',iceshelf_saltcoef)
 func_replace_string('data','bottomDragLinear',sprintf('bottomDragLinear=%.3g,',bottom_dragcoef))
 fprintf('bottomDragLinear=%.3g,',bottom_dragcoef)
 func_replace_string('data.shelfice','SHELFICEDragLinear',sprintf('SHELFICEDragLinear=%.3g,',iceshelf_dragcoef))
 fprintf('SHELFICEDragLinear=%.3g,',iceshelf_dragcoef)
 SHELFICElatentHeat=334000;
 func_replace_string('data.shelfice','SHELFICElatentHeat',['SHELFICElatentHeat=',sprintf('%f',SHELFICElatentHeat),','])

%% - meridional variation of kappav
if meridionalkappav==1
    kappav1=kappav_pole+(kappav_eq-kappav_pole).*exp(-abs(yc)/kappav_width);
    kappav2=repmat(kappav1,[nx,1]);
    kappav3=repmat(kappav2(:),[1,nr]);
    fname=sprintf('kappav3d_eq%.1gpole%.1g.bin',kappav_eq,kappav_pole);
    fprintf('kappa_v varying with latitude. kappav_pole=%f, kappav_eq=%f. writing file %s.\n',kappav_pole,kappav_eq,fname)
elseif meridionalkappav==2
    Hunder1=reshape(Hunder,[nxy,1]);
    relh=max((Hunder1+h)./(Hunder1+Htot),0);
    kappav3=kappav_interior+(kappav_bnd-kappav_interior).*exp(-relh.*nr./1);
    fname=sprintf('kappav3d_bnd%.1ginterior%.1g.bin',kappav_bnd,kappav_interior);
    fprintf('kappa_v varying with latitude. kappav_interior=%f, kappav_bnd=%f. writing file %s.\n',kappav_interior,kappav_bnd,fname)
end
if meridionalkappav
    fid=fopen(fname,'w','b'); fwrite(fid,kappav3,prec); fclose(fid);
    func_replace_string('data','diffKrFile',['diffKrFile=''',fname,''','])
    fprintf(['diffKrFile=''',fname,''','])
end

%% -- initial temperaturee
 % freezing theta should be lower than the in-situ Tfreeze
 % but the top layer temperature should also be warmer than freezing theta to transport heat into the ice
if restartTS>=0
if restartTS==0 % prescribe analytical temp, salt IC profiles
     %  --- temperature
     Tini1=repmat(reshape(Thfreeze,[nxy,1]),[1,nr]);
     Hunder1=reshape(Hunder,[nxy,1]);
     Hunder0=-(rhoice/rhoNil).*Hice0;
     ibnd=find(hf>-Hunder0,1,'first')+1; % when turned on SHELFICEboundaryLayer, the layer below the top layer is also counted as boundary
     h_bnd=hf(ibnd)+Hunder0-1.5*dh(ibnd-1)/2; % boundary layer thickness
     relh=max((Hunder1+h-h_bnd)./(Hunder1+Htot),0);
     drelh=relh(:,2:end)-relh(:,1:end-1);
     drelh=drelh.*((a0-mean(hf(2:end-1)))./(a0-hf(2:end-1))).^2.2;
     relh1=[zeros(nxy,1),cumsum(drelh,2)];
     relh0=max((h+Hunder0)./(Htot+Hunder0),0);
     relh=0*relh0+1*relh1;
     %Tini=mean(mean(Thfreeze.*wgt,1),2).*(relh1)+Tini1.*(1-relh1);
     Tini=Tini1;
     %Tini=Tini.*(1-relh.^0.6)+mean(Tini.*reshape(wgt,[nxy,1]),1).*relh.^0.6;
     % set vertical temperature gradient
     if Sref>15
         delTemp0=min(Thfreeze(:))-max(Thfreeze(:));
         delTempvar=delTemp0;
     else
         delTemp0=max(Thfreeze(:))-min(Thfreeze(:));
         delTempvar=-delTemp0;
     end

     delTemp_pattern=delTemp0-delTempvar*cosd(2.*yc);
     delTemp_pattern=repmat(delTemp_pattern,[nx,1]);
     delTemp_pattern=delTemp_pattern(:);
     delTemp=delTemp_pattern; % this 10mK comes from eyeballing the result from 40mW experiments with horizontal viscosity/diffusivity equal 1 and vertical viscosity/diffusivity equal 0.01.
     Tini=Tini+delTemp.*ones(nxy,nr).*relh;
     dTemp=round(delTemp0*1000); % im mili-Kelvin
     sfx=sprintf('%i%s',dTemp,'mK.bin');

     dTnoise=delTemp0*0;
     dTn=rand([nx,ny]);
     dTn(:,ny/2+1:ny)=dTn(:,ny/2:-1:1);
     dTn=dTn-mean(mean(dTn,1).*wgt1);
     dTn=dTn(:);
     Tini(:,nr)=Tini(:,nr)+dTnoise*dTn;
     if (length(appendix)==0)
         fTname=['theta_ini.',sfx];
     else
         fTname=['theta_ini_',appendix,'.',sfx];
     end

    % --- salinity
     Sini=Sref+zeros(nxy,nr);
     delSalt0=0;
     Sini=Sini+min(h-abs(Hunder0),0)/(abs(Hunder0)-min(abs(Hunder(:))))*delSalt0;
     sfx=sprintf('%i%s',delSalt0*1e3,'mpsu.bin');
     if (length(appendix)==0)
         fSname=['salt_ini.',sfx];
     else
         fSname=['salt_ini_',appendix,'.',sfx];
     end

elseif restartTS<4 % restartTS ~= 0
    [Tini,its,~]=rdmds([restartTSpath,'/T'],restartTSiter);
    [Sini,its,~]=rdmds([restartTSpath,'/S'],restartTSiter);
    xcrest=squeeze(mean(rdmds([restartTSpath,'/XC'],NaN),1));
    ycrest=squeeze(mean(rdmds([restartTSpath,'/YC'],NaN),1));
    rcrest=squeeze(mean(rdmds([restartTSpath,'/RC'],NaN),1));
    if length(its)>1
        Tini=mean(Tini,4);
        Tini=Tini(:,:,:,1);
        Sini=mean(Sini,4);
        Sini=Sini(:,:,:,1);
    end
    if restartTS==1
        Tini=mean(Tini,[1,2]);
        Tini=repmat(Tini,[nx,ny,1]);
        Sini=mean(Sini,[1,2]);
        Sini=repmat(Sini,[nx,ny,1]);
    end
    if restartTS==2
        disp('entering 2')
        Tini=squeeze(mean(Tini,1));
        Sini=squeeze(mean(Sini,1));
        if size(Tini,1)~=ny || size(Tini,2)~=nr % interpolation
            % fill in missing values
            disp('fill in')
            Tini_=Tini;
            Tini_(Tini_==0)=NaN;
            Tbase=Tini_(:,end);
            Tdiff=-diff(Tini_,1,2);
            Tdiff(isnan(Tdiff))=0;
            Tini(:,1:end-1)=Tbase+cumsum(Tdiff,2,'reverse');
            Tini(1,:)=Tini(2,:);
            Tini(end,:)=Tini(end-1,:);

            Sini_=Sini;
            Sini_(Sini_==0)=NaN;
            Sbase=Sini_(:,end);
            Sdiff=-diff(Sini_,1,2);
            Sdiff(isnan(Sdiff))=0;
            Sini(:,1:end-1)=Sbase+cumsum(Sdiff,2,'reverse');
            Sini(1,:)=Sini(2,:);
            Sini(end,:)=Sini(end-1,:);

            % 2d interpolate
            disp('interpolate')
            [rrrest,yyrest]=meshgrid(rcrest,ycrest);
            [rr,yy]=meshgrid(-h,yc);
            Tini_=Tini;
            Tini=interp2(rrrest,yyrest,Tini_,rr,yy,'spline');
            Sini_=Sini;
            Sini=interp2(rrrest,yyrest,Sini_,rr,yy,'spline');

        end
        Tini=repmat(reshape(Tini,[1,ny,nr]),[nx,1,1]);
        Sini=repmat(reshape(Sini,[1,ny,nr]),[nx,1,1]);
    end

elseif restartTS==4 % use netcdf pickup files from other experiment to initialize the model
    if ~exist('./pickup.0000000000.t001.nc','file')
        system('rm -rf initial_condition')
        system('mkdir initial_condition')
        system(['cp ',gridpath,'*.nc ./initial_condition/'])
        system(['rename 0000000001 ',sprintf('%010d',0),' ./initial_condition/*.nc'])
        expandpickups(restartTSpath,restartTSiter,'./initial_condition/',0)
        system('cp -f initial_condition/pickup*.nc .')
    end
    func_replace_string('data','pickupSuff','pickupSuff=''0000000000'',')

elseif restartTS==5 % use netcdf pickup files from other experiment to restart the model
    if ~exist(['./pickup.',sprintf('%010d',restartTSiter),'.t001.nc'],'file')
        system('rm -rf restart_condition')
        system('mkdir restart_condition')
        system(['cp ',gridpath,'/*.nc ./restart_condition/'])
        system(['rename 0000000000 ',sprintf('%010d',restartTSiter),' ./restart_condition/*.nc'])
        expandpickups(restartTSpath,restartTSiter,'./restart_condition/',restartTSiter)
        system('cp -f restart_condition/pickup*.nc .')
    end
    func_replace_string('run.sub','export currentiter',['export currentiter=',sprintf('%d',restartTSiter)])

elseif restartTS==6 % use the long-term tendency to adjust current T/S state, so that the experiment converges faster.
    disp('start interpolate nc file')
    restartTSiterout=restartTSiter-1;
    if ~exist(sprintf('./pickup.%010d.t001.nc',restartTSiterout),'file')
        system('rm -rf restart_condition')
        system('mkdir restart_condition')
        system(['cp ',gridpath,'*.nc ./restart_condition/'])
        system(['rename ',sprintf('%010d',restartTSiter),' ',sprintf('%010d',restartTSiterout),' ./restart_condition/*.nc'])
        T_0=mean(mean(rdmds(['run',sprintf('%d',diffiter0),'/T'],NaN),4),1);
        S_0=mean(mean(rdmds(['run',sprintf('%d',diffiter0),'/S'],NaN),4),1);
        T_1=mean(mean(rdmds(['run',sprintf('%d',diffiter1),'/T'],NaN),4),1);
        S_1=mean(mean(rdmds(['run',sprintf('%d',diffiter1),'/S'],NaN),4),1);

        dTIC=repmat(reshape((T_1-T_0),[1,ny,nr]),[nx,1,1]);
        dSIC=repmat(reshape((S_1-S_0),[1,ny,nr]),[nx,1,1]);
        expandpickups(restartTSpath,restartTSiter,'./restart_condition/',restartTSiterout,1,dTIC,dSIC)
        system('cp -f restart_condition/pickup*.nc .')
    end
end % restartTS>0

if restartTS<4
    %- and add small noise
    Tini=reshape(Tini,[nxy,nr]);
    Sini=reshape(Sini,[nxy,nr]);
    dTnoise=0.001e-6;
    dTn=rand([nxy,nr]);
    %dTn(:,ny/2+1:ny)=dTn(:,ny/2:-1:1);
    Tini(:,nr)=Tini(:,nr)+dTnoise.*dTn(:,nr);

    fTname=['Tini_restart_',restartname,'.bin'];
    fSname=['Sini_restart_',restartname,'.bin'];

     fid=fopen(fTname,'w','b'); fwrite(fid,Tini,prec); fclose(fid);
     fprintf(['\nwrite file: ',fTname]);
    func_replace_string('data','hydrogThetaFile',['hydrogThetaFile=''',fTname,''','])
    fprintf(['hydrogThetaFile=''',fTname,''','])

     fid=fopen(fSname,'w','b'); fwrite(fid,Sini,prec); fclose(fid);
     fprintf(['\nwrite file: ',fSname]);
    func_replace_string('data','hydrogSaltFile',['hydrogSaltFile=''',fSname,''','])
    fprintf(['hydrogSaltFile=''',fSname,''','])

end
end % restartTS>=0

%% - ice flow
Tsmean=mean(Ts.*wgt1);
Tm=Tfreeze0+273.15;
ice_flow0=2*((rhoNil-rhoice)*rhoice/rhoNil*g0)*Hice0^3/((log(Tm/Tsmean))^3)*integral2(@(Tp,T) 1./min(eta_melt.*exp(-Ea/Rg/Tm+(Ea/Rg)./Tp),eta_max_Q).*((log(Tm./Tp)))./Tp./T.*(Tp<=T),Tsmean,Tm,Tsmean,Tm);
SHI_iceflow=ice_flow0/Hice0^3/rhoice^3;
func_replace_string('data.shelfice','SHI_iceflow',['SHI_iceflow=',sprintf('%g',SHI_iceflow),','])
fprintf('SHI_iceflow=%g,',SHI_iceflow)
func_replace_string('data.shelfice','SHELFICElatentHeat',['SHELFICElatentHeat=',sprintf('%g',SHELFICElatentHeat),','])


%% ice evolution
func_replace_string('data.shelfice','PrescribeFreezing',sprintf('PrescribeFreezing=%d,',PrescribeFreezing))
fprintf('PrescribeFreezing=%d\n',PrescribeFreezing)
if EvolveIce
    func_replace_string('data','quasiHydrostatic','quasiHydrostatic=.TRUE.,')
    func_replace_string('data','nonlinFreeSurf','nonlinFreeSurf=4,')
    func_replace_string('data','nonHydrostatic','nonHydrostatic=.FALSE.,')
    func_replace_string('data.shelfice','SHIdtFactor',['SHIdtFactor=',sprintf('%f',SHIdtFactor),','])
    func_replace_string('data.shelfice','SHELFICEMassStepping','SHELFICEMassStepping=.TRUE.,')
    func_replace_string('data.shelfice','SHELFICERemeshFrequency','SHELFICERemeshFrequency=2592000.,')
    func_replace_string('data.shelfice','SHELFICEwriteState', 'SHELFICEwriteState=.TRUE.,')
    fprintf('SHIdtFactor=%f\n',SHIdtFactor)
else
    func_replace_string('data','quasiHydrostatic','quasiHydrostatic=.FALSE.,')
    func_replace_string('data','nonlinFreeSurf','nonlinFreeSurf=0,')
    func_replace_string('data','nonHydrostatic','nonHydrostatic=.TRUE.,')
    func_replace_string('data.shelfice','SHELFICEMassStepping','SHELFICEMassStepping=.FALSE.,')
    func_replace_string('data.shelfice','SHELFICERemeshFrequency','SHELFICERemeshFrequency=0,')
    func_replace_string('data.shelfice','SHELFICEwriteState', 'SHELFICEwriteState=.TRUE.,')
end

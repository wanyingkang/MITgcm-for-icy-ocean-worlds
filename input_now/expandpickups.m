%interpickups('../data_enceladus_southpole_D2dz300dx100_c90s10ptide2_30psu_eject/',20160000,'./interp2d/',1)

function expandpickups(dirin,iterin,dirout,iterout,rpt,dTIC,dSIC,varargin)
addpath('~/MITgcm/utils/matlab/')
% function expandpickups(dirin,iterin,dirout,iterout,snap)
%
% wanying: based on interpickups.m, I modified the code to expand a 
% 2D experiment pickup to a 3D pickup
% add input iterin, iterout
% This function interpolates the data in 
% a set of mnc pickup files and grid files from the MITgcm 
% given in dirin/pickup.*.nc and in dirin/grid.*.nc
% (this can be a global file or a collection of per-tile pickup files)
% to the pickup files dirout/pickup.*.nc and the grid files dirout/grid.*.nc
% (this, too, can be a global file or a collection of per-tile pickup files)
%
% Extrapolation takes place near the domain edges if domain size
% of pickout is larger than that of pickin.
%
% The number of vertical levels must be the same in the two sets of files.
%
% Snap is an optional argument if there is more than one timestep
% in the file.  The default is 1.
%
% May be fishy near boundaries if grid is not uniform...

if nargin==7
  snap=1
else
  snap=varargin{1}
end

disp(dirin)
disp(dirout)
if (strcmp(dirin,dirout))
  error('dir','You cant use the same input and output directories!')
end

pickin=dir([dirin '/pickup.' sprintf('%010d',iterin) '*.nc'])
gridin=dir([dirin '/grid.*.nc'])
if length(pickin)~=length(gridin)
  error('in','Incompatible number of input pickups and gridfiles')
end

pickout=dir([dirout '/pickup.' sprintf('%010d',iterout) '*.nc'])
gridout=dir([dirout '/grid.*.nc'])
if length(pickout)~=length(gridout)
  error('out','Incompatible number of output pickups and gridfiles')
end 

%%%%%%%%%%%%INPUT SANITY
disp('input grid')
fin=[dirin '/' pickin(1).name];
gin=[dirin '/' gridin(1).name];
Zcomp=ncread(fin,'Z');
gZcomp=ncread(gin,'Z');
if (sum(Zcomp~=gZcomp)>0)
  error('in','Incompatible Z-axis input pickup and gridfile: 1')
end

Xin=ncread(fin,'X');
gXin=ncread(gin,'X');

if (sum(gXin~=gXin)>0)
  error('in','Incompatible x-axis input pickups and gridfile: 1')
end

Yin=ncread(fin,'Y');
gYin=ncread(gin,'Y');
if (sum(gYin~=gYin)>0)
  error('in','Incompatible y-axis input pickups and gridfile: 1')
end

Xp1in=ncread(fin,'Xp1');
gXp1in=ncread(gin,'Xp1');
if (sum(gXp1in~=gXp1in)>0)
  error('in','Incompatible x-axis input pickups and gridfile: 1')
end

Yp1in=ncread(fin,'Yp1');
gYp1in=ncread(gin,'Yp1');
if (sum(gYp1in~=gYp1in)>0)
  error('in','Incompatible y-axis input pickups and gridfile: 1')
end


for i=2:length(pickin)
  fin=[dirin '/' pickin(i).name];
  Z=ncread(fin,'Z');
  if (sum(Zcomp~=Z)>0)
    error('Z','Incompatible vertical axes in input pickups:',num2str(i))
  end

  gin=[dirin '/' pickin(i).name];
  Z=ncread(gin,'Z');
  if (sum(Zcomp~=Z)>0)
    error('Z','Incompatible vertical axes in input gridfiles:',num2str(i))
  end

  Xin=sort([Xin;ncread(fin,'X')]);
  Xp1in=sort([Xp1in;ncread(fin,'Xp1')]);

  gXin=sort([gXin;ncread(gin,'X')]);
  gXp1in=sort([gXp1in;ncread(gin,'Xp1')]);

  if (sum(gXin~=Xin)>0)
    error('X','Incompatible x-axes in input files:',num2str(i))
  end

  Yin=sort([Yin;ncread(fin,'Y')]);
  Yp1in=sort([Yp1in;ncread(fin,'Yp1')]);

  gYin=sort([gYin;ncread(fin,'Y')]);
  gYp1in=sort([gYp1in;ncread(fin,'Yp1')]);

  if (sum(gYin~=Yin)>0)
    error('Y','Incompatible y-axes in input files:',num2str(i))
  end



end

store=[Xin(1)];
for i=2:length(Xin)
  if Xin(i-1)~=Xin(i)
    store(end+1)=Xin(i);
  end
end
Xin=store';
clear gXin

store=[Xp1in(1)];
for i=2:length(Xp1in)
  if Xp1in(i-1)~=Xp1in(i)
    store(end+1)=Xp1in(i);
  end
end
Xp1in=store';
clear gXp1in

store=[Yin(1)];
for i=2:length(Yin)
  if Yin(i-1)~=Yin(i)
    store(end+1)=Yin(i);
  end
end
Yin=store';
clear gYin

store=[Yp1in(1)];
for i=2:length(Yp1in)
  if Yp1in(i-1)~=Yp1in(i)
    store(end+1)=Yp1in(i);
  end
end
Yp1in=store';
clear gYp1in

%%%%%%%%%%%%%%% OUTPUT SANITY
disp('output grid')
fout=[dirout '/' pickout(1).name];
gout=[dirout '/' gridout(1).name];
Zcomp=ncread(fout,'Z');
gZcomp=ncread(gout,'Z');
if (sum(Zcomp~=gZcomp)>0)
  error('out','Incompatible Z-axis output pickup and gridfile: 1')
end

Xout=ncread(fout,'X');
gXout=ncread(gout,'X');

if (sum(gXout~=gXout)>0)
  error('out','Incompatible x-axis output pickups and gridfile: 1')
end

Yout=ncread(fout,'Y');
gYout=ncread(gout,'Y');
if (sum(gYout~=gYout)>0)
  error('out','Incompatible y-axis output pickups and gridfile: 1')
end

Xp1out=ncread(fout,'Xp1');
gXp1out=ncread(gout,'Xp1');
if (sum(gXp1out~=gXp1out)>0)
  error('out','Incompatible x-axis output pickups and gridfile: 1')
end

Yp1out=ncread(fout,'Yp1');
gYp1out=ncread(gout,'Yp1');
if (sum(gYp1out~=gYp1out)>0)
  error('out','Incompatible y-axis output pickups and gridfile: 1')
end




for i=2:length(pickout)
  fout=[dirout '/' pickout(i).name];
  Z=ncread(fout,'Z');
  if (sum(Zcomp~=Z)>0)
    error('Z','Incompatible vertical axes in output pickups:',num2str(i))
  end

  gout=[dirout '/' pickout(i).name];
  Z=ncread(gout,'Z');
  if (sum(Zcomp~=Z)>0)
    error('Z','Incompatible vertical axes in output gridfiles:',num2str(i))
  end

  Xout=sort([Xout;ncread(fout,'X')]);
  Xp1out=sort([Xp1out;ncread(fout,'Xp1')]);

  gXout=sort([gXout;ncread(gout,'X')]);
  gXp1out=sort([gXp1out;ncread(gout,'Xp1')]);

  if (sum(gXout~=Xout)>0)
    error('X','Incompatible x-axes in output files:',num2str(i))
  end

  Yout=sort([Yout;ncread(fout,'Y')]);
  Yp1out=sort([Yp1out;ncread(fout,'Yp1')]);

  gYout=sort([gYout;ncread(fout,'Y')]);
  gYp1out=sort([gYp1out;ncread(fout,'Yp1')]);

  if (sum(gYout~=Yout)>0)
    error('Y','Incompatible y-axes in output files:',num2str(i))
  end



end

store=[Xout(1)];
for i=2:length(Xout)
  if Xout(i-1)~=Xout(i)
    store(end+1)=Xout(i);
  end
end
Xout=store';
clear gXout

store=[Xp1out(1)];
for i=2:length(Xp1out)
  if Xp1out(i-1)~=Xp1out(i)
    store(end+1)=Xp1out(i);
  end
end
Xp1out=store';
clear gXp1out

store=[Yout(1)];
for i=2:length(Yout)
  if Yout(i-1)~=Yout(i)
    store(end+1)=Yout(i);
  end
end
Yout=store';
clear gYout

store=[Yp1out(1)];
for i=2:length(Yp1out)
  if Yp1out(i-1)~=Yp1out(i)
    store(end+1)=Yp1out(i);
  end
end
Yp1out=store';
clear gYp1out


[ycin,xcin]=meshgrid(Yin,Xin);
[ycout,xcout]=meshgrid(Yout,Xout);

%%%%%%%%%%%%% HFacCoutk

HFacoutk=ones([size(xcout) length(Zcomp)]);

disp(['Calculating HFacC...'])
  for j=1:length(gridout)
    gout=[dirout '/' gridout(j).name];
    Xhere=ncread(gout,'X');
    Yhere=ncread(gout,'Y');
    HFacouthere=ncread(gout,'HFacC');
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcout).*(Yhere(jj)==ycout));
        HFacoutk(iii,jjj,:)=HFacouthere(ii,jj,:);
      end
    end

  end
clear HFacouthere

%%%%%%%%%%%%% 2D centered
%vars={'Eta','EtaH','dEtaHdt'};
vars={};

for iv=1:length(vars)
  var=vars(iv);

  Fieldin=NaN*ones(size(xcin));
  Fieldout=NaN*ones(size(xcout));
% #1 read data from input
  for j=1:length(pickin)
    fin=[dirin '/' pickin(j).name];
    gin=[dirin '/' gridin(j).name];
    Xhere=ncread(fin,'X');
    Yhere=ncread(fin,'Y');
    Fieldinhere=ncread(fin,char(var));
    HFacinhere=squeeze(ncread(gin,'HFacC',[1,1,1],[Inf,Inf,1]));
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcin).*(Yhere(jj)==ycin));
        Fieldin(iii,jjj)=Fieldinhere(ii,jj);
      end
    end
  end
% #2 expand x and add small noises
  Fieldout=inpaint_nans(repmat(Fieldin,[rpt,1,1]),0);
  Amp=0;
  if Amp~=0
      Fieldout=Fieldout+rand(size(Fieldout)).*Amp;
  end
  % return;
% #3 write file
  for j=1:length(pickout)
    fout=[dirout '/' pickout(j).name];
    Xhere=ncread(fout,'X');
    Yhere=ncread(fout,'Y');
    [ii,jj]=find((Xhere(1)<=xcout).*(xcout<=Xhere(end)).*(Yhere(1)<=ycout).*(ycout<=Yhere(end)));
    ncwrite(fout,char(var),Fieldout(min(ii):max(ii),min(jj):max(jj)));
  end
end
%S,gSnm1,Temp,gTnm1,phi_nh are on Xc,Rc

%%%%%%%%%%%%%% centered 3D
%vars={'S','gSnm1','Temp','gTnm1','phi_nh','gWnm1'};
vars={'Temp'};
for iv=1:length(vars)
 var=vars(iv);
 Fieldoutk=zeros([size(xcout) length(Zcomp)]);
 for k=1:length(Zcomp)
  clear Fieldinhere
  Fieldin=NaN*ones(size(xcin));
  Fieldout=NaN*ones(size(xcout));
% #1 read input variable
  for j=1:length(pickin)
    fin=[dirin '/' pickin(j).name];
    gin=[dirin '/' gridin(j).name];
    Xhere=ncread(fin,'X');
    Yhere=ncread(fin,'Y');
    Fieldinhere=squeeze(ncread(fin,char(var),[1,1,k,snap],[Inf,Inf,1,1]));
    HFacinhere=squeeze(ncread(gin,'HFacC',[1,1,k],[Inf,Inf,1]));
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcin).*(Yhere(jj)==ycin));
        Fieldin(iii,jjj)=Fieldinhere(ii,jj);
      end
    end
  end
% #2 expand x and add small noises
  Fieldout=inpaint_nans(repmat(Fieldin,[rpt,ones(1,2-1)]),0);
  if k==length(Zcomp) && strcmp(char(var),'Temp')
      Amp=1e-7;
      Fieldout=Fieldout+rand(size(Fieldout)).*Amp;
  end
  % return;
  disp([char(var),':',num2str(k),' within topography'])
  Fieldoutk(:,:,k)=Fieldout;
 end
 if strcmp(char(var),'Temp')
     Fieldoutk=Fieldoutk+dTIC
 end
 if strcmp(char(var),'S')
     Fieldoutk=Fieldoutk+dSIC
 end

for j=1:length(pickout)
  fout=[dirout '/' pickout(j).name];
  Xhere=ncread(fout,'X');
  Yhere=ncread(fout,'Y');
  [ii,jj]=find((Xhere(1)<=xcout).*(xcout<=Xhere(end)).*(Yhere(1)<=ycout).*(ycout<=Yhere(end)));
  ncwrite(fout,char(var),Fieldoutk(min(ii):max(ii),min(jj):max(jj),:));
end
end

%% U,gUnm1 is on XU,Rc
[ycin,xcin]=meshgrid(Yin,Xp1in);
[ycout,xcout]=meshgrid(Yout,Xp1out);

% HFacWoutk

HFacout=ones(size(xcout));
HFacoutk=ones([size(xcout) length(Zcomp)]);

disp(['Calculating HFacW...'])
  for j=1:length(gridout)
    gout=[dirout '/' gridout(j).name];
    Xhere=ncread(gout,'Xp1');
    Yhere=ncread(gout,'Y');
    HFacouthere=ncread(gout,'HFacW');
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcout).*(Yhere(jj)==ycout));
        HFacoutk(iii,jjj,:)=HFacouthere(ii,jj,:);
      end
    end

  end
clear HFacouthere

% variables
%vars={'U','gUnm1'};
vars={};
for iv=1:length(vars)
var=vars(iv);
 Fieldoutk=zeros([size(xcout) length(Zcomp)]);
 for k=1:length(Zcomp)
  Fieldin=NaN*ones(size(xcin));
  Fieldout=NaN*ones(size(xcout));

  for j=1:length(pickin)
    fin=[dirin '/' pickin(j).name];
    gin=[dirin '/' gridin(j).name];
    Xhere=ncread(fin,'Xp1');
    Yhere=ncread(fin,'Y');
    Fieldinhere=squeeze(ncread(fin,char(var),[1,1,k,snap],[Inf,Inf,1,1]));
    HFacinhere=squeeze(ncread(gin,'HFacW',[1,1,k],[Inf,Inf,1]));
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcin).*(Yhere(jj)==ycin));
        Fieldin(iii,jjj)=Fieldinhere(ii,jj);
      end
    end
  end
% #2 expand x and add small noises
if rpt>1
Fieldout=inpaint_nans([repmat(Fieldin(1:end-1,:),[rpt-1,1]);Fieldin],0);
else
Fieldout=inpaint_nans(Fieldin,0);
end

  Amp=0;
  if Amp~=0
      Fieldout=Fieldout+rand(size(Fieldout)).*Amp;
  end
  % return;
  disp([char(var),':',num2str(k),' within topography'])
  Fieldoutk(:,:,k)=Fieldout;
% wanying: end

end

for j=1:length(pickout)
  fout=[dirout '/' pickout(j).name];
  Xhere=ncread(fout,'Xp1');
  Yhere=ncread(fout,'Y');
  [ii,jj]=find((Xhere(1)<=xcout).*(xcout<=Xhere(end)).*(Yhere(1)<=ycout).*(ycout<=Yhere(end)));
  ncwrite(fout,char(var),Fieldoutk(min(ii):max(ii),min(jj):max(jj),:));
end
end

%% V,gVnm1 is on XV,Rc

[ycin,xcin]=meshgrid(Yp1in,Xin);
[ycout,xcout]=meshgrid(Yp1out,Xout);

% HFacSoutk

HFacout=ones(size(xcout));
HFacoutk=ones([size(xcout) length(Zcomp)]);

disp(['Calculating HFacS...'])
  for j=1:length(gridout)
    gout=[dirout '/' gridout(j).name];
    Xhere=ncread(gout,'X');
    Yhere=ncread(gout,'Yp1');
    HFacouthere=ncread(gout,'HFacS');
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcout).*(Yhere(jj)==ycout));
        HFacoutk(iii,jjj,:)=HFacouthere(ii,jj,:);
      end
    end

  end
clear HFacouthere

% variables
%vars={'V','gVnm1'};
vars={};
for iv=1:length(vars)
 var=vars(iv);
 Fieldoutk=zeros([size(xcout) length(Zcomp)]);
 for k=1:length(Zcomp)
  Fieldin=NaN*ones(size(xcin));
  Fieldout=NaN*ones(size(xcout));

  for j=1:length(pickin)
    fin=[dirin '/' pickin(j).name];
    gin=[dirin '/' gridin(j).name];
    Xhere=ncread(fin,'X');
    Yhere=ncread(fin,'Yp1');
    Fieldinhere=squeeze(ncread(fin,char(var),[1,1,k,snap],[Inf,Inf,1,1]));
    HFacinhere=squeeze(ncread(gin,'HFacS',[1,1,k],[Inf,Inf,1]));
    for ii=1:length(Xhere)
      for jj=1:length(Yhere)
        [iii,jjj]=find((Xhere(ii)==xcin).*(Yhere(jj)==ycin));
        Fieldin(iii,jjj)=Fieldinhere(ii,jj);
      end
    end
  end
% #2 expand x and add small noises
  Fieldout=inpaint_nans(repmat(Fieldin,[rpt,ones(1,2-1)]),0);
  Amp=0;
  if Amp~=0
      Fieldout=Fieldout+rand(size(Fieldout)).*Amp;
  end
  % return;
  disp([char(var),':',num2str(k),' within topography'])
  Fieldoutk(:,:,k)=Fieldout;
% wanying: end

end

for j=1:length(pickout)
  fout=[dirout '/' pickout(j).name];
  Xhere=ncread(fout,'X');
  Yhere=ncread(fout,'Yp1');
  [ii,jj]=find((Xhere(1)<=xcout).*(xcout<=Xhere(end)).*(Yhere(1)<=ycout).*(ycout<=Yhere(end)));
  ncwrite(fout,char(var),Fieldoutk(min(ii):max(ii),min(jj):max(jj),:));

end
end

%% --- pickup_ptracers files
%#%#pickin=dir([dirin '/pickup_ptracers.' sprintf('%010d',iterin) '*.nc'])
%#%#pickout=dir([dirout '/pickup_ptracers.' sprintf('%010d',iterout) '*.nc'])
%#%#
%#%#[ycin,xcin]=meshgrid(Yin,Xin);
%#%#[ycout,xcout]=meshgrid(Yout,Xout);
%#%#
%#%#% HFacoutk
%#%#HFacoutk=ones([size(xcout) length(Zcomp)]);
%#%#
%#%#disp(['Calculating HFacC...'])
%#%#  for j=1:length(gridout)
%#%#    gout=[dirout '/' gridout(j).name];
%#%#    Xhere=ncread(gout,'X');
%#%#    Yhere=ncread(gout,'Y');
%#%#    HFacouthere=ncread(gout,'HFacC');
%#%#    for ii=1:length(Xhere)
%#%#      for jj=1:length(Yhere)
%#%#        [iii,jjj]=find((Xhere(ii)==xcout).*(Yhere(jj)==ycout));
%#%#        HFacoutk(iii,jjj,:)=HFacouthere(ii,jj,:);
%#%#      end
%#%#    end
%#%#
%#%#  end
%#%#clear HFacouthere
%#%#
%#%#% variables
%#%#vars={'centerbot','sidebot','wholebot'};
%#%#for iv=1:length(vars)
%#%# var=vars(iv);
%#%# Fieldoutk=zeros([size(xcout) length(Zcomp) 2]);
%#%# for it=1:2
%#%# for k=1:length(Zcomp)
%#%#  clear Fieldinhere
%#%#  Fieldin=NaN*ones(size(xcin));
%#%#  Fieldout=NaN*ones(size(xcout));
%#%#% #1 read input variable
%#%#  for j=1:length(pickin)
%#%#    fin=[dirin '/' pickin(j).name];
%#%#    gin=[dirin '/' gridin(j).name];
%#%#    Xhere=ncread(fin,'X');
%#%#    Yhere=ncread(fin,'Y');
%#%#    Fieldinhere=squeeze(ncread(fin,char(var),[1,1,k,it],[Inf,Inf,1,1]));
%#%#    HFacinhere=squeeze(ncread(gin,'HFacC',[1,1,k],[Inf,Inf,1]));
%#%#    for ii=1:length(Xhere)
%#%#      for jj=1:length(Yhere)
%#%#        [iii,jjj]=find((Xhere(ii)==xcin).*(Yhere(jj)==ycin));
%#%#        Fieldin(iii,jjj)=Fieldinhere(ii,jj);
%#%#      end
%#%#    end
%#%#  end
%#%#% #2 expand x and add small noises
%#%#  Fieldout=inpaint_nans(repmat(Fieldin,[rpt,ones(1,2-1)]),0);
%#%#  % return;
%#%#  disp([char(var),':',num2str(k),' within topography'])
%#%#  Fieldoutk(:,:,k,it)=Fieldout;
%#%#  end
%#%#end
%#%#
%#%#for j=1:length(pickout)
%#%#  fout=[dirout '/' pickout(j).name];
%#%#  Xhere=ncread(fout,'X');
%#%#  Yhere=ncread(fout,'Y');
%#%#  [ii,jj]=find((Xhere(1)<=xcout).*(xcout<=Xhere(end)).*(Yhere(1)<=ycout).*(ycout<=Yhere(end)));
%#%#  ncwrite(fout,char(var),Fieldoutk(min(ii):max(ii),min(jj):max(jj),:,:));
%#%#end
%#%#end
%

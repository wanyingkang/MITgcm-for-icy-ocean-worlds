function matrixOut = smooth2a(matrixIn,Nr,Nc,Pr,Pc,Method)
% Smooths 2D array data.  Ignores NaN's.
%
%function matrixOut = smooth2a(matrixIn,Nr,Nc,Pr,Pc)
% 
% This function smooths the data in matrixIn using a mean filter over a
% rectangle of size (2*Nr+1)-by-(2*Nc+1).  Basically, you end up replacing
% element "i" by the mean of the rectange centered on "i".  Any NaN
% elements are ignored in the averaging.  If element "i" is a NaN, then it
% will be preserved as NaN in the output.  At the edges of the matrix,
% where you cannot build a full rectangle, as much of the rectangle that
% fits on your matrix is used (similar to the default on Matlab's builtin
% function "smooth").
% 
% "matrixIn": original matrix
% "Nr": number of points used to smooth rows
% "Nc": number of points to smooth columns.  If not specified, Nc = Nr.
% "Pr": bool, whether row is periodic
% "Pc": bool, whether column is periodic
% "Method": string, "uniform" or "gaussian"
% 
% "matrixOut": smoothed version of original matrix
% 
% 
% 	Written by Greg Reeves, March 2009.
% 	Division of Biology
% 	Caltech
% 
% 	Inspired by "smooth2", written by Kelly Hilands, October 2004
% 	Applied Research Laboratory
% 	Penn State University
% 
% 	Developed from code written by Olof Liungman, 1997
% 	Dept. of Oceanography, Earth Sciences Centre
% 	G�teborg University, Sweden
% 	E-mail: olof.liungman@oce.gu.se
%
%   Wanying Kang add periodic function, Jun 2019

%
% Initial error statements and definitions
%
if nargin < 2, error('Not enough input arguments!'), end

N(1) = Nr; 
if nargin < 3, N(2) = N(1); else N(2) = Nc; end
if nargin < 4, P(1) = false; else P(1) = Pr; end
if nargin < 5, P(2) = false; else P(2) = Pc; end
if nargin < 6, Med = 'uniform'; else Med = Method; end

if length(N(1)) ~= 1, error('Nr must be a scalar!'), end
if length(N(2)) ~= 1, error('Nc must be a scalar!'), end

%
% Building matrices that will compute running sums.  The left-matrix, eL,
% smooths along the rows.  The right-matrix, eR, smooths along the
% columns.  You end up replacing element "i" by the mean of a (2*Nr+1)-by- 
% (2*Nc+1) rectangle centered on element "i".
%
[row,col] = size(matrixIn);
l1=-N(1):N(1);
switch Med
    case 'uniform'
        wgt1=1;
    case 'gaussian'
        wgt1=exp(-l1.^2./N(1)^2);
end

if P(1) % periodic in row
    eL = spdiags(ones(row,4*N(1)+1).*[wgt1(end-N(1)+1:end),wgt1,wgt1(1:N(1))],[-(row-1):-(row-N(1)),-N(1):N(1),row-N(1):row-1],row,row);
else
    eL = spdiags(ones(row,2*N(1)+1).*wgt1,(-N(1):N(1)),row,row);
end
%figure
%image(eL.*50)

l2=-N(2):N(2);
switch Med
    case 'uniform'
        wgt2=1;
    case 'gaussian'
        wgt2=exp(-l2.^2./N(2)^2);
end

if P(2) % periodic in col
    eR = spdiags(ones(col,4*N(2)+1).*[wgt2(end-N(2)+1:end),wgt2,wgt2(1:N(2))],[-(col-1):-(col-N(2)),-N(2):N(2),col-N(2):col-1],col,col);
else
    eR = spdiags(ones(col,2*N(2)+1).*wgt2,(-N(2):N(2)),col,col);
end

%
% Setting all "NaN" elements of "matrixIn" to zero so that these will not
% affect the summation.  (If this isn't done, any sum that includes a NaN
% will also become NaN.)
%
A = isnan(matrixIn);
matrixIn(A) = 0;

%
% For each element, we have to count how many non-NaN elements went into
% the sums.  This is so we can divide by that number to get a mean.  We use
% the same matrices to do this (ie, "eL" and "eR").
%
nrmlize = eL*(~A)*eR;
nrmlize(A) = NaN;

%
% Actually taking the mean.
%
matrixOut = full(eL)*matrixIn*full(eR);
matrixOut = matrixOut./nrmlize;













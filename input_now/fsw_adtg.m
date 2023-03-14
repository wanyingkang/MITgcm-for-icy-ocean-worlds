function sw_adtg=fsw_adtg(S,T,P)
  sref = 35.e0;
  a0 =  3.5803e-5;
  a1 = +8.5258e-6;
  a2 = -6.836e-8;
  a3 =  6.6228e-10;

  b0 = +1.8932e-6;
  b1 = -4.2393e-8;

  c0 = +1.8741e-8;
  c1 = -6.7795e-10;
  c2 = +8.733e-12;
  c3 = -5.4481e-14;

  d0 = -1.1351e-10;
  d1 =  2.7759e-12;

  e0 = -4.6206e-13;
  e1 = +1.8676e-14;
  e2 = -2.1687e-16;

  sw_adtg =  a0 + (a1 + (a2 + a3.*T).*T).*T + (b0 + b1.*T).*(S-sref) ...
      + ( (c0 + (c1 + (c2 + c3.*T).*T).*T) + (d0 + d1.*T).*(S-sref) ).*P ...
      + (  e0 + (e1 + e2.*T).*T ).*P.*P;
 end
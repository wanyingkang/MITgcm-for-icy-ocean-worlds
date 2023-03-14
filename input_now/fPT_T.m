function PT=fPT_T(T,P,S)
  PR=0;
% theta1
  del_P  = PR - P;
  del_th = del_P.*fsw_adtg(S,T,P);
  th     = T + 0.5.*del_th;
  q      = del_th;
% theta2
  del_th = del_P.*fsw_adtg(S,th,P+0.5.*del_P);
;
  th     = th + (1 - 1/sqrt(2)).*(del_th - q);
  q      = (2-sqrt(2)).*del_th + (-2+3/sqrt(2)).*q;
;
% theta3
  del_th = del_P.*fsw_adtg(S,th,P+0.5.*del_P);
  th     = th + (1 + 1/sqrt(2)).*(del_th - q);
  q      = (2 + sqrt(2)).*del_th + (-2-3/sqrt(2)).*q;
;
% theta4
  del_th = del_P.*fsw_adtg(S,th,P+del_P);
  PT= th + (del_th - 2.*q)/(2.*3);
end

function ima=iocwt2dliu1(S,HW,WH,WW,etl,lr,hr)

J=length(WW);

for j=J:-1:1
   hf=padf0(lr,j);
   gf=padf0(hr,j);
   lf=length(hf)-1;
   
   GS=(conv2(WH{j},hf')+conv2(WW{j},gf'))/2;
   [nr,nc]=size(GS);
   GS=GS(lf+1:nr-lf,:);
   
   HS=(conv2(S,hf')+conv2(HW{j},gf'))/2;
   HS=HS(lf+1:nr-lf,:);
   
   S=(conv2(HS,hf)+conv2(GS,gf))/2;
   [nr,nc]=size(S);
   S=S(:,lf+1:nc-lf);
   clear HS GS;
end
[nr,nc]=size(S);
ima=S(etl+1:nr-etl,etl+1:nc-etl);
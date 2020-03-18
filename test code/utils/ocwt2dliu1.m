function [S,HW,WH,WW,etl] = ocwt2dliu1(ima,ld,hd,J)


n=length(ima);
if n<128 
   etl=n;
else
   etl=128;
end
etl = 0;  % ÎÒ¼ÓµÄ
%%%%%%%%%%%boundary extend%%%%%%
m=etl;
ima=[ima(:,m:-1:1) ima ima(:,n:-1:n-m+1)];
ima=[ima(m:-1:1,:);ima;ima(n:-1:n-m+1,:)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S=ima;
for j=1:J
   hf=padf0(ld,j);
   gf=padf0(hd,j);
   
   GS=conv2(S,gf);
   HS=conv2(S,hf);

   WH{j}=conv2(GS,hf');
   WW{j}=conv2(GS,gf');
   HW{j}=conv2(HS,gf');
   S=conv2(HS,hf');
   
   clear GS HS;
end

   
function fi=padf0(f,j)

l=length(f);
l0=2^(j-1)-1;

n=l+l0*(l-1);

fi=zeros(1,n);

ind=1;
for i=1:l
   fi(ind)=f(i);
   ind=ind+l0+1;
end


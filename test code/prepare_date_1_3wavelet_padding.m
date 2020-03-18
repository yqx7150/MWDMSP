function y = prepare_date_1_3wavelet_padding(output_im)
            x_LL_1 = output_im(:,:,1) ;
            x_LL_2 = output_im(:,:,2) ;
            x_LL_3 = output_im(:,:,3) ;
            x_HL_1 = output_im(:,:,4);
            x_HL_2 = output_im(:,:,5);
            x_HL_3 = output_im(:,:,6);
            x_LH_1 = output_im(:,:,7) ;
            x_LH_2 = output_im(:,:,8) ;
            x_LH_3 = output_im(:,:,9) ;
            x_HH_1 = output_im(:,:,10) ;
            x_HH_2 = output_im(:,:,11);
            x_HH_3 = output_im(:,:,12);
            
            a = size(x_LL_1,1);
            x_LL_1 = x_LL_1( 3 : a - 2 , 3 : a - 2);
            x_HL_1 = x_HL_1( 3 : a - 2 , 3 : a - 2);
            x_LH_1 = x_LH_1( 3 : a - 2 , 3 : a - 2);
            x_HH_1 = x_HH_1( 3 : a - 2 , 3 : a - 2);
            output(:,:,1)= idwt2(x_LL_1,x_HL_1,x_LH_1,x_HH_1,'haar');
            
            aa = size(x_LL_2,1);
            x_LL_2 = x_LL_2( 2 : aa - 2 , 2 : aa - 2);
            x_HL_2 = x_HL_2( 2 : aa - 2 , 2 : aa - 2);
            x_LH_2 = x_LH_2( 2 : aa - 2 , 2 : aa - 2);
            x_HH_2 = x_HH_2( 2 : aa - 2 , 2 : aa - 2);
            output(:,:,2)= idwt2(x_LL_2,x_HL_2,x_LH_2,x_HH_2,'db2');
            
            aaa = size(x_LL_3,1);
            x_LL_3 = x_LL_3( 1 : aaa - 1 , 1 : aaa - 1);
            x_HL_3 = x_HL_3( 1 : aaa - 1 , 1 : aaa - 1);
            x_LH_3 = x_LH_3( 1 : aaa - 1 , 1 : aaa - 1);
            x_HH_3 = x_HH_3( 1 : aaa - 1 , 1 : aaa - 1);
            output(:,:,3)= idwt2(x_LL_3,x_HL_3,x_LH_3,x_HH_3,'db4');

            y = output;




end

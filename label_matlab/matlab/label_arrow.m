clear all;
close all;
arrow_area = [213 1 14 12];

flist  = {
    [0:33,1449:1478,1999:3313],...
    [34:1448,3314:3783,5062:5333],...
    [1479:1998,3784:5061,5334:5545]};

fid_train_data = fopen('arrow_train_data.bin','w');
fid_train_label = fopen('arrow_train_label.bin','w');
fid_test_data = fopen('arrow_test_data.bin','w');
fid_test_label = fopen('arrow_test_label.bin','w');

rng(1234);
testdata = 0;
traindata = 0;
for ii = 1 : size(flist,2)  
    for jj = 1 : size(flist{ii},2) 
        fn = strcat('D:\workspace\vision\elevator_monitor\jpg\',int2str(flist{ii}(jj)),'.jpg');
        I = im2single(imread(fn));
        I2 = imcrop(I,arrow_area);
        I2 = imresize(I2, [28 28]);
        I2_gray = rgb2gray(I2);
        for kk = 0 : 9
            fn = strcat('D:\workspace\vision\elevator_monitor\jpg\arrow\gray\',int2str(ii-1),'_',int2str((jj-1)*10+kk),'.jpg');
            if(kk == 9)
                imwrite(im2uint8(I2_gray),fn);  
                fwrite(fid_test_data,im2uint8(I2_gray)','uint8');
                fwrite(fid_test_label,ii-1,'uint8');
                testdata = testdata + 1;
            else 
                sat_scale = abs(0.5 + 0.1*randn(1));
                sat_center = 0.15*randn(1);
                I2_gray_sat = (tanh(-1+(I2_gray+sat_center)/sat_scale) + 1)/2;
                I2_gray_sat = im2uint8(I2_gray_sat);
                imwrite(I2_gray_sat,fn);      
                fwrite(fid_train_data,I2_gray_sat','uint8');
                fwrite(fid_train_label,ii-1,'uint8');     
                traindata = traindata + 1;
            end
        
        end
    end
end
testdata
traindata
fclose(fid_train_data);
fclose(fid_train_label);
fclose(fid_test_data);
fclose(fid_test_label);
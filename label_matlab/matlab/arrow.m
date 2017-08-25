clear all;
close all;
arrow_area = [213 1 14 12];
floor_area = [213 12 14 13];
fn_noarrow = 'C:\workspace\ADAS\elevator_monitor\jpg\0.jpg';
fn_arrowup = 'C:\workspace\ADAS\elevator_monitor\jpg\34.jpg';
fn_arrowdown = 'C:\workspace\ADAS\elevator_monitor\jpg\5517.jpg';

I = imread(fn_noarrow);
I2 = imcrop(I,arrow_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\noarrow.jpg');

I = imread(fn_arrowup);
I2 = imcrop(I,arrow_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\arrowup.jpg');

I = imread(fn_arrowdown);
I2 = imcrop(I,arrow_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\arrowdown.jpg');

% floor number
fn = 'C:\workspace\ADAS\elevator_monitor\jpg\0.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\0.jpg');


fn = 'C:\workspace\ADAS\elevator_monitor\jpg\46.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\1.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\445.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\2.jpg');


fn = 'C:\workspace\ADAS\elevator_monitor\jpg\463.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\3.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\485.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\4.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\533.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\5.jpg');


fn = 'C:\workspace\ADAS\elevator_monitor\jpg\544.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\6.jpg');


fn = 'C:\workspace\ADAS\elevator_monitor\jpg\587.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\7.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\606.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\8.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\634.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\9.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\674.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\10.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\694.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\11.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\744.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\12.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\754.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\13.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\794.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\14.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\814.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\15.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\854.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\16.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\884.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\17.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\904.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\18.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\934.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\19.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\984.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\20.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\1004.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\21.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\1034.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\22.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\1054.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\23.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\1084.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\24.jpg');

fn = 'C:\workspace\ADAS\elevator_monitor\jpg\1144.jpg';
I = imread(fn);
I2 = imcrop(I,floor_area);
imshow(I2);
imwrite(I2,'C:\workspace\ADAS\elevator_monitor\25.jpg');


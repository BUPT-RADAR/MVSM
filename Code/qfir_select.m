function [b] = qfir_select( Fs ,rx_num,pg)
%FIR Returns a discrete-time filter object.

%
% MATLAB Code
% Generated by MATLAB(R) 7.14 and the Signal Processing Toolbox 6.17.
%
% Generated on: 27-Nov-2015 09:48:30
%

% FIR Window Bandpass filter designed using the FIR1 function.

% All frequency values are in GHz.
%Fs = 44;  % Sampling Frequency
Fs=Fs/10^9;
N    = 50;       % Order
% Fc1  = 5.65;     % First Cutoff Frequency
% Fc2  = 7.95;     % Second Cutoff Frequency
flag = 'scale';  % Sampling Flag
% Create the window vector for the design algorithm.
win = hamming(N+1);
fc=[5.3,5.4,5.7,6.1,6.4,6.8,7.3,7.7,7.8,8.2,8.8];       %center frequency
fb=[1.75,1.8,1.85,2.05,2.15,2.3,2.35,2.5,2.5,2.65,3.1]; %Bandwidh
fu=[6,8.5;7.2,10.2];
% Calculate the coefficients using the FIR1 function.
if rx_num==1
b  = fir1(N, [fc(pg)-fb(pg)/2 fc(pg)+fb(pg)/2]/(Fs/2), 'bandpass', win, flag);
else
b  = fir1(N, [fu(pg,:)]/(Fs/2), 'bandpass', win, flag);
end
Hd = dfilt.dffir(b);

% [EOF]

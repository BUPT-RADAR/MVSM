clc
close all
clear

Title='huxixkvln6';
Mat='.mat';
radar = load([Title,Mat]);
PureData0 =radar.data;

Sign=PureData0(1:200,150);
Sign=Sign';
 
figure(1)
set(gcf,'Position',[20 450 400 300]);
mesh(PureData0)
Fs=20;
[imf,residual,info] = emd(Sign,'Interpolation','pchip');
Im=imf(:,1);
Specimf = 2*abs(fft(Im,1024))/length(Im);
Specimf = Specimf(1:round(end/2));
Freqbinimf = linspace(0,Fs/2,length(Specimf));
% y2=fft(Im,1024);
% mag2=abs(y2);
% mag3=mag2(1:512);
[x, y]=max(Specimf);
y=(y-1)/512*600;
figure(200)
set(gcf,'Position',[20 100 640 500]);	    
set(gcf,'Color','w'); 
plot(Freqbinimf,Specimf,'linewidth',2);

figure(2)

SampFreq = 20;
t = 1/SampFreq:1/SampFreq:length(Sign)/SampFreq;
set(gcf,'Position',[20 100 320 250]);	    
set(gcf,'Color','w'); 
plot(t,Sign,'linewidth',1);
xlabel('Time / Sec','FontSize',12,'FontName','Times New Roman');
ylabel('Amplitude','FontSize',12,'FontName','Times New Roman');
set(gca,'FontSize',12)
set(gca,'linewidth',1);
% 
%% Fourier spectrum
Spec = 2*abs(fft(Sign))/length(Sign);
Spec = Spec(1:round(end/2));
Freqbin = linspace(0,SampFreq/2,length(Spec));

figure(3)
set(gcf,'Position',[720 100 640 500]);	    
set(gcf,'Color','w'); 
plot(Freqbin,Spec,'linewidth',2);
xlabel('Frequency / Hz','FontSize',24,'FontName','Times New Roman');
ylabel('Amplitude / AU','FontSize',24,'FontName','Times New Roman');
set(gca,'FontSize',24)
set(gca,'linewidth',2);
axis([0 10 0 1.1*max(Spec)])
figure(4)
window = 128;
Nfrebin = 1024;
[tfSpec1,f] = STFT(Sign',SampFreq,Nfrebin,window);%128
imagesc(t,f,abs(tfSpec1)); 
set(gcf,'Position',[720 700 320 250]);	 
xlabel('Time / Sec','FontSize',12,'FontName','Times New Roman');
ylabel('Frequency / Hz','FontSize',12,'FontName','Times New Roman');
set(gca,'YDir','normal')
set(gca,'FontSize',12);
set(gcf,'Color','w');	


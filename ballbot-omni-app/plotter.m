% ROB 311, Fall 2022
% Data importing / plotting from ball-bot
%
% University of Michigan
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
alpha = pi/4;
beta = pi/2;                                                                
Rk = 0.121;
Rw = 0.04778;

MAX_LEAN = 20; % degrees
MAX_PHI = 360; % degrees
MAX_DPHI = 270; % degrees/sec
MAX_DUTY = 1.0; % meters

% Input file name -  this will have to updated with the name of the trial
% you are analyzing
filename = 'ROB311_Filtering_Demo.txt';
% Use file name to load / create data matrix
eval(['load ' filename ';']);
eval(['data = ' erase(filename,'.txt') ';']);

% Define variables from data:
% data = 
% 1: [i] + 
% 2: [t] + 
% 3: [states['theta_roll']] + 
% 4: [states['theta_pitch']] + 
% 5: [phi[0]] + 
% 6: [phi[1]] + 
% 7: [phi[2]] + 
% 8: [dphi[0][0]] + 
% 9: [dphi[1][0]] + 
% 10: [dphi[2][0]] +
% 11: [theta_x_filtered] +
% 12: [theta_y_filtered] +
% 13: [dphi_x_filtered] +
% 14: [dphi_y_filtered]

index = data(:, 1);
time = data(:, 2);

theta_x = rad2deg(data(:, 3));
theta_y = rad2deg(data(:, 4));

phi_x = rad2deg(data(:, 5));
phi_y = rad2deg(data(:, 6));
phi_z = rad2deg(data(:, 7));

dphi_x = rad2deg(data(:, 8));
dphi_y = rad2deg(data(:, 9));
dphi_z = rad2deg(data(:, 10));

theta_x_filtered = rad2deg(data(:, 11));
theta_y_filtered = rad2deg(data(:, 12));

dphi_x_filtered = rad2deg(data(:, 13));
dphi_y_filtered = rad2deg(data(:, 14));

dt = [0.005; time(2:end) - time(1:end-1)];
terminal_dt_error = dt - 0.005;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%% FILTERING AND FFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %Filtering
% cutoff = 2;
% data_to_be_filtered = dphi_x;
% data_filtered = low_filt(1/mean(dt),2,cutoff,data_to_be_filtered);
% 
% %Frequency content
% [amplitude, frequency] = FFT(1/mean(dt), data_to_be_filtered);
% [amplitude_filtered, frequency_filtered] = FFT(1/mean(dt), low_filt(1/mean(dt),2,3,data_to_be_filtered));
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% figure
% subplot(121)
% plot(time, data_to_be_filtered,'linewidth',2)
% hold on
% plot(time, data_filtered,'linewidth',2)
% xlabel('Time (s)')
% ylabel('deg/sec') % Units will need to change if the signal being investigated gets changed
% ylim([-MAX_DPHI MAX_DPHI])
% subplot(122)
% plot(frequency, amplitude,'linewidth',2)
% hold on
% plot(frequency_filtered, amplitude_filtered,'linewidth',2)
% xlabel('Frequency')
% ylabel('Db/Hz')
% xlim([0 30])
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plotting Rotation of the Body (Theta)
figure
hold on
title('Rotation of the Body (\theta_F)')
plot(time, theta_x, 'linewidth',2)
plot(time, theta_y, 'linewidth',2)
legend('\theta_x_F', '\theta_y_F')
ylim([-MAX_LEAN MAX_LEAN])
xlabel('Time (s)')
ylabel('Angle (deg)')

% Plotting Filtering Theta X
figure
hold on
title('Filtering \theta_x)')
plot(time, theta_x, 'linewidth',2)
plot(time, theta_x_filtered, 'linewidth',2)
legend('\theta_x', '\theta_x_filtered')
ylim([-MAX_LEAN MAX_LEAN])
xlabel('Time (s)')
ylabel('deg/sec')

% Plotting Filtering Theta Y
figure
hold on
title('Filtering \theta_y)')
plot(time, theta_y, 'linewidth',2)
plot(time, theta_y_filtered, 'linewidth',2)
legend('\theta_y', '\theta_y_filtered')
ylim([-MAX_LEAN MAX_LEAN])
xlabel('Time (s)')
ylabel('deg/sec')

% Plotting Rotation of the Ball (Phi)
figure
hold on
title('Rotation of the Ball (\phi)')
plot(time, phi_x, 'linewidth',2)
plot(time, phi_y, 'linewidth',2)
legend('\phi_x', '\phi_y')
ylim([-MAX_PHI MAX_PHI])
xlabel('Time (s)')
ylabel('Angle (deg)')

% Plotting Velocity of the Ball (dPhi)
figure
hold on
title('Velocity of the Ball (d\phi)')
plot(time, dphi_x, 'linewidth',2)
plot(time, dphi_y, 'linewidth',2)
legend('\phi_x', '\phi_y')
ylim([-MAX_DPHI MAX_DPHI])
xlabel('Time (s)')
ylabel('deg/sec')

% Plotting Filtering dPhi
figure
hold on
title('Filtering d\phi)')
plot(time, dphi_x, 'linewidth',2)
plot(time, dphi_x_filtered, 'linewidth',2)
legend('d\phi_x', 'd\phi_x_filtered')
ylim([-MAX_DPHI MAX_DPHI])
xlabel('Time (s)')
ylabel('deg/sec')

% Plotting Filtering dPhi
figure
hold on
title('Filtering d\phi)')
plot(time, dphi_y, 'linewidth',2)
plot(time, dphi_y_filtered, 'linewidth',2)
legend('d\phi_y', 'd\phi_y_filtered')
ylim([-MAX_DPHI MAX_DPHI])
xlabel('Time (s)')
ylabel('deg/sec')

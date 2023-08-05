%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author: Ravi Ashok Pashchapur & Yuxi Chen                                    %
% @Date: 19/04/2022                                                             %
% @Project:Low-cost Multi-object Positioning System with Optical Sensor Fusion  %
% @Licence: MIT                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all

data = readtable('data.xlsx');

track_t = data.Var2(91:175) - 15;
track_x = data.Var3(91:175);
track_y = data.Var4(91:175);
track_z = data.Var5(91:175);

n = numel(track_t);

vicon_t = data.Var8 * 1.02 - 15;
vicon_x = data.Var9;
vicon_y = data.Var10 - 0.2;
vicon_z = data.Var11;

vicon_t_slow = track_t;
vicon_x_slow = zeros(n,1);
vicon_y_slow = zeros(n,1);
vicon_z_slow = zeros(n,1);

for i = 1:n
    [~, idx] = min(abs(vicon_t - track_t(i)));
    vicon_x_slow(i) = vicon_x(idx);
    vicon_y_slow(i) = vicon_y(idx);
    vicon_z_slow(i) = vicon_z(idx);
end

error_x = track_x - vicon_x_slow;
error_y = track_y - vicon_y_slow;
error_z = track_z - vicon_z_slow;

rmse_x = sqrt(sum(error_x.^2 ./ n));
rmse_y = sqrt(sum(error_y.^2 ./ n));
rmse_z = sqrt(sum(error_z.^2 ./ n));

error_x_mean = mean(abs(error_x));
error_y_mean = mean(abs(error_y));
error_z_mean = mean(abs(error_z));

figure('WindowState', 'maximized')
hold on
plot(track_t, track_x)
plot(vicon_t_slow, vicon_x_slow)
title('X-axis Plot')
xlabel('Time (s)')
ylabel('X-axis (m)')
legend('Estimated data', 'Ground Truth (Vicon System)')

figure('WindowState', 'maximized')
hold on
plot(track_t, track_y)
plot(vicon_t_slow, vicon_y_slow)
title('Y-axis Plot')
xlabel('Time (s)')
ylabel('Y-axis (m)')
legend('Estimated data', 'Ground Truth (Vicon System)')

figure('WindowState', 'maximized')
hold on
plot(track_t, track_z)
plot(vicon_t_slow, vicon_z_slow)
title('Z-axis Plot')
xlabel('Time (s)')
ylabel('Z-axis (m)')
legend('Estimated data', 'Ground Truth (Vicon System)')

figure('WindowState', 'maximized')
subplot(3, 1, 1)
plot(track_t, abs(error_x))
title('Error Plot')
ylabel('X-axis Error (m)')
subplot(3, 1, 2)
plot(track_t, abs(error_y))
ylabel('Y-axis Error (m)')
subplot(3, 1, 3)
plot(track_t, abs(error_z))
xlabel('Time (s)')
ylabel('Z-axis Error (m)')

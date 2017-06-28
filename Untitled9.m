
plotit = false;
data_gd = readtable('GD_convergence.txt','HeaderLines',1);
[n_gd,~] = size(data_gd.time);
%time_gd = zeros(n_gd,1);
time_gd = data_gd.time(1:n_gd);
if plotit
    figure;
    plot(time_gd(1:2000),data_gd.totalError(1:2000));
    hold on;
end



data_trm = readtable('TRM_convergence.txt','HeaderLines',1);
[n_trm,~] = size(data_trm.time);
%time_trm = zeros(n_trm+1,1);
time_trm = data_trm.time(1:n_trm);
if plotit
    plot(time_trm(1:n_trm),data_trm.totalError(1:n_trm));
end
title('error');

disp(data_trm.totalGmag);

if plotit
    figure
    %hold on;
    plot(time_gd(1:n_gd-1),data_gd.totalGmag(1:n_gd-1));
    hold on;
    plot(time_trm(1:n_trm-1),data_trm.totalGmag(1:n_trm-1));
end

% gotta fill in table


table_results = zeros(2,4);

indices = zeros(7,1);
c = 1;
for i = 1:7
    tval = data_gd.time(c);
    while tval < data_trm.time(i)
        c = c + 1;
        tval = data_gd.time(c);
    end
    indices(i) = c;
    disp(strcat('time: ', num2str(tval), ' iter: ', int2str(c), ' obj: ', num2str(data_gd.totalError(c)), ' gmag ', num2str(data_gd.totalGmag(c))));
end



disp(time_trm(n_trm));
disp(data_trm.totalError(n_trm));
disp(data_trm.totalGmag(n_trm));

disp(time_gd(n_gd-1));
disp(data_gd.totalError(n_gd));
disp(data_gd.totalGmag(n_gd));

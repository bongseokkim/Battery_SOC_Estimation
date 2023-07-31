
function [Y_est] = Model(X)

%Input X: Measured current, voltage, and temperature values
%X: 3 columns, T rows, where T is length of input data in seconds
Current = X(:,1); %Amps, column 1
Voltage = X(:,2); %Volts, column 2
Temperature = X(:,3); %degrees Celsius, column 3
Time = (0:1:(length(Current)-1))'; %seconds

%------------------ Model 1: %Coulomb Counting SOC Estimation  -------------------------

%Coulomb Counting SOC Estimator
SOC_Init = 1; %Assume battery always starts fully charged
Capacity = 4.4; %Ah, Nominal capacity of new Tesla 21700 NMC/NCA cell

%Coulomb counting: SOC = integral of current
for i=1:length(Time)
    if i==1
        %At time step 0, SOC is equal to initial setpoint
        SOC(i)=SOC_Init;
    else
        %Greater than time step 0
        %SOC = SOC of last time step + delta SOC
        SOC(i)=SOC(i-1)+Current(i)*((Time(i)-Time(i-1))/3600)/Capacity;
    end
end

%Output Y: Estimated SOC
%Y: 1 columns, T rows, where T is length of input data in seconds
Y_est_coul=SOC'; %Transpose SOC from columns to rows  


%------------------- Model 2: Pre-trained LSTM SOC Estimation  -------------------------

% Normalize 1 %Current, Voltage, Temp
MAX = [44.32819091,4.41959706 , 50.08918657];
MIN = [-1.87576232e+01, 0.00000000e+00,  -2.49560000e+01];
Y_min = -1.40268912e-03;
Y_max = 1.03237779;

X_2 = X;
X_2(:,1) = ((X_2(:,1) - MIN(1))./(MAX(1)-MIN(1)));
X_2(:,2) = ((X_2(:,2) - MIN(2))./(MAX(2)-MIN(2)));
X_2(:,3) = ((X_2(:,3) - MIN(3))./(MAX(3)-MIN(3)));
repeatedRows = repmat(X_2(1,:), 20, 1);

% Concatenate the original dataset with the repeated rows
X_2 = [repeatedRows; X_2];

% Reorder and transpose X data for LSTM
expanded_dataset = [X_2(:, 2), X_2(:, 1), X_2(:, 3)]';

% sequence length
windowSize = 20;  
dataSize = size(expanded_dataset, 2);
numWindows = dataSize - windowSize;

sequence = cell(1,numWindows);
for i = 1:numWindows
    sequence{i} = expanded_dataset(:, i:i+windowSize-1);
end

%load pre-trained lstm parameters
net = load("trained_lstm.mat").net;

Y_predict_2 = predict(net, sequence);
Y_predict_2 = Y_predict_2';
Y_predict_2(1,:) = Y_predict_2(1,:) * (Y_max-Y_min) + Y_min;

% ------------------------Ensemble prediction-----------------------------------

% Threshold for switching between LSTM prediction and average
threshold = 0.15; % Adjust the threshold as per your requirement

% Ensemble estimation with switching logic
for j = 1:length(Y_predict_2)
    diff = abs(Y_est_coul(j)' - Y_predict_2(j));
    if diff > threshold
        Y_predict_2(j) = Y_predict_2(j); % Use LSTM prediction
    else
        Y_predict_2(j) = 0.312689*Y_predict_2(j) + 0.687311*Y_est_coul(j); % Take average
    end
end

% Finally, do filtering with a moving average
window_size =300;
window = ones(window_size, 1) / window_size;
filtered_values = conv(Y_predict_2, window, 'same');

filtered_values(1:window_size) = Y_predict_2(1:window_size); 
filtered_values(end-window_size+1:end) = Y_predict_2(end-window_size+1:end);
Y_est = filtered_values';

% Adjusting Coulomb counting and ensemble
difference = filtered_values' - Y_est_coul;
len = length(difference);
start_idx = round(len/3) ;
diff_mean = mean(difference(start_idx:end));

Y_est_coul(:,1) = Y_est_coul(:,1) + diff_mean;
Y_est = 0.6*Y_est+ 0.4*Y_est_coul; 
end


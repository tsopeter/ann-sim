%% Define Red Pitaya as TCP client object
clc
clear all
close all
IP = '129.237.123.147';           % Input IP of your Red Pitaya...
port = 5000;
RP = tcpclient(IP, port);

flush(RP);                              % Flush input and output
% flush(RP, 'input')
% flush(RP, 'output')

%% Open connection with your Red Pitaya and close previous one
x = instrfind;
clear RP;


RP = tcpclient(IP, port);
RP.ByteOrder = "big-endian";
configureTerminator(RP,"CR/LF");

writeline(RP,'GEN:RST');                    % Reset Generator
writeline(RP,'SOUR2:FUNC SINE');            % Set function of output signal
                                            % {sine, square, triangle,
                                            % sawu,sawd, pwm}
writeline(RP,'SOUR2:FREQ:FIX 10000');        % Set frequency of output signal
writeline(RP,'SOUR2:VOLT 1');               % Set amplitude of output signal
writeline(RP,'OUTPUT2:STATE ON');           % Set output to ON

writeline(RP,'SOUR2:TRIG:INT');             % Generate trigger


% %% Calcualte arbitrary waveform with 16384 samples
% % Values of arbitrary waveform must be in range from -1 to 1.
% N = 16383;
% t = 0:(2*pi)/N:2*pi;
% x = sin(t) + 1/3*sin(3*t);
% y = 1/2*sin(t) + 1/4*sin(4*t);
% 
% figure;
% plot(t,x,t,y)
% grid on;
% 
% %% Convert waveforms to string with 5 decimal places accuracy
% waveform_ch_1_0 = num2str(x,'%1.5f,');
% waveform_ch_2_0 = num2str(y,'%1.5f,');
% 
% % the last two elements are empty spaces “,”.
% waveform_ch_1 =waveform_ch_1_0(1,1:length(waveform_ch_1_0)-3);
% waveform_ch_2 =waveform_ch_2_0(1,1:length(waveform_ch_2_0)-3);
% 
% %% Generation
% 
% writeline(RP,'GEN:RST')                     % Reset to default settings
% 
% writeline(RP,'SOUR1:FUNC ARBITRARY');       % Set function of output signal
% writeline(RP,'SOUR2:FUNC ARBITRARY');       % {sine, square, triangle, sawu, sawd}
% 
% writeline(RP,['SOUR1:TRAC:DATA:DATA ' waveform_ch_1])  % Send waveforms to Red Pitya
% writeline(RP,['SOUR2:TRAC:DATA:DATA ' waveform_ch_2])
% 
% writeline(RP,'SOUR1:VOLT 0.7');             % Set amplitude of output signal
% writeline(RP,'SOUR2:VOLT 1');
% 
% writeline(RP,'SOUR1:FREQ:FIX 10000');        % Set frequency of output signal
% writeline(RP,'SOUR2:FREQ:FIX 10000');
% 
% 
% writeline(RP,'OUTPUT:STATE ON');            % Start both channels simultaneously
% writeline(RP,'SOUR:TRIG:INT');              % Generate triggers

%% acq


writeline(RP,'ACQ:RST');
writeline(RP,'ACQ:DEC 4');


% Set trigger delay to 0 samples
% 0 samples delay sets trigger to center of the buffer
% Signal on your graph will have trigger in the center (symmetrical)
% Samples from left to the center are samples before the trigger
% Samples from center to the right are samples after the trigger

writeline(RP,'ACQ:TRIG:DLY 0');

% for SIGNALlab device there is a possiblity to set the trigger threshold
writeline(RP,'ACQ:TRIG:LEV 1')


%% Start & Trigg
% Trigger source setting must be after ACQ:START
% Set trigger to source 1 positive edge

writeline(RP,'ACQ:START');
% After acquisition is started some time delay is needed in order to acquire fresh samples in to buffer
pause(1);
% Here we have used time delay of one second but you can calculate the exact value taking in to account buffer
% length and sampling rate

writeline(RP,'ACQ:TRIG NOW');
% Wait for trigger
% Until trigger is true wait with acquiring
% Be aware of while loop if trigger is not achieved
% Ctrl+C will stop code execution in MATLAB

while 1
    trig_rsp = writeread(RP,'ACQ:TRIG:STAT?')

    if strcmp('TD',trig_rsp(1:2))  % Read only TD

        break;

    end
end

% % UNIFIED OS
% % wait for fill adc buffer
% while 1
%     fill_state = writeread(RP,'ACQ:TRIG:FILL?')
%
%     if strcmp('1', fill_state(1:1))
%
%         break;
%
%     end
% end

% Read data from buffer
signal_str   = writeread(RP,'ACQ:SOUR1:DATA?');
signal_str_2 = writeread(RP,'ACQ:SOUR2:DATA?');

% Convert values to numbers.
% The first character in string is “{“
% and the last 3 are 2 spaces and “}”.

signal_num   = str2num(signal_str  (1, 2:length(signal_str)  - 3));
signal_num_2 = str2num(signal_str_2(1,2:length(signal_str_2) - 3));

figure;
plot(signal_num)
hold on
plot(signal_num_2,'r')
grid on
ylabel('Voltage / V')
xlabel('samples')

clear RP;
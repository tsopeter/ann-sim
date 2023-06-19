%% Define Red Pitaya as TCP/IP object
IP = '129.237.123.147';                % Input IP of your Red Pitaya...
port = 5000;
RP = rp_awg.rp;

%% Open connection with your Red Pitaya

RP.ByteOrder = 'big-endian';
configureTerminator(RP, 'CR/LF');

%flush(RP);

% Set decimation value (sampling rate) in respect to your
% acquired signal frequency

writeline(RP,'ACQ:RST');
writeline(RP,'ACQ:DEC 1');
writeline(RP,'ACQ:TRIG:LEV 0.8');       % trigger level

% there is an option to select coupling when using SIGNALlab 250-12
% writeline(RP,'ACQ:SOUR1:COUP AC');    % enables AC coupling on channel 1

% by default LOW level gain is selected
writeline(RP,'ACQ:SOUR1:GAIN LV');    % sets gain to LV/HV (should the same as jumpers)


% Set trigger delay to 0 samples
% 0 samples delay sets trigger to the center of the buffer
% Signal on your graph will have the trigger in the center (symmetrical)
% Samples from left to the center are samples before trigger
% Samples from center to the right are samples after trigger

writeline(RP,'ACQ:TRIG:DLY 0');

%% Start & Trigg
% Trigger source setting must be after ACQ:START
% Set trigger to source 1 positive edge

writeline(RP,'ACQ:START');

% After acquisition is started some time delay is needed in order to acquire fresh samples in the buffer
pause(1);
% Here we have used time delay of one second, but you can calculate the exact value by taking into account buffer
% length and sampling rate

writeline(RP,'ACQ:TRIG CH1_PE');

% Wait for trigger
% Until trigger is true wait with acquiring
% Be aware of the while loop if trigger is not achieved
% Ctrl+C will stop code execution in MATLAB

while 1
    trig_rsp = writeread(RP,'ACQ:TRIG:STAT?')

    if strcmp('TD', trig_rsp(1:2))      % Read only TD

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
signal_str = writeread(RP,'ACQ:SOUR1:DATA?');

% Convert values to numbers.
% The first character in the received string is “{“
% and the last 3 are 2 empty spaces and a “}”.

signal_num = str2num(signal_str(1, 2:length(signal_str)-3));

figure;
plot(signal_num)
grid on;
ylabel('Voltage / V')
xlabel('Samples')

clear RP;
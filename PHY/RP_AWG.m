% arbitrary waveform generator
%
%
classdef RP_AWG
    properties
        rp      % red_pitaya class
        dac_freq
        dac_m_volt
        xf
        src
        source
        dst
        destination

        adc_sample_rate = 125e6;

        en_trig  = false
        trig_lvl = 0
        trig_del = 0

        en_trig_str = false
        trig_lvl_str = '0';
        trig_del_str = '0';
    end

    methods
        function awg = RP_AWG(ip_addr, port_n, dac_freq, dac_m_volt, src, dst)
            awg.rp = tcpclient(ip_addr, port_n);   % communicate with
            flush(awg.rp);
            clear awg.rp;
            awg.rp = tcpclient(ip_addr, port_n);
            awg.rp.ByteOrder = "big-endian";
            configureTerminator(awg.rp, "CR/LF");

            awg.dac_freq = dac_freq;        % operating frequency
            awg.dac_m_volt = dac_m_volt;    % max voltage

            awg.src = src;
            awg.dst = dst;

            % configure source and destination strings
            awg.source      = awg.get_source(src);
            awg.destination = awg.get_source(dst);
        end

        function reset(awg)
            writeline(awg.rp, 'GEN:RST');
        end

        function dst = get_source(awg, src)
            if src==1
                dst = 'SOUR1';
            else
                dst = 'SOUR2';
            end
        end

        % internal function
        function cmd = implt_writl_str(awg, tgt, instr)
            stgt = awg.get_source(tgt);
            cmd = [stgt, ':', instr];
        end

        function sine(awg)
            awg.reset();
            cmd = [awg.implt_writl_str(awg.source, 'FUNC ') 'SINE'];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.source, 'FREQ:FIX ') num2str(awg.dac_freq, '%1.0f')];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.source, 'VOLT '), num2str(awg.dac_m_volt, '%1.1f')];
            writeline(awg.rp, cmd);

            cmd = ['OUTPUT' num2str(awg.src) ':STATE ON'];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.source, 'TRIG:INT')];
            writeline(awg.rp, cmd);
        end

        function transmit(awg, dat)
            awg.reset()
            if max(dat)>=0  % if positive only, append spaces
                ch_1 = num2str(dat, '%1.5f, ');
            else
                ch_1 = num2str(dat, '%1.5f,');
            end
            ch_1 = ch_1(1,1:length(ch_1)-3);

            cmd = [awg.implt_writl_str(awg.src, 'FUNC '), 'ARBITRARY'];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.src, 'TRAC:DATA:DATA ') ch_1];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.src, 'VOLT'), num2str(awg.dac_m_volt, '%1.5f')];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.src, 'FREQ:FIX '), num2str(awg.dac_freq)];
            writeline(awg.rp, cmd);

            cmd = ['OUTPUT' num2str(awg.src) ':STATE ON'];
            writeline(awg.rp, cmd);

            cmd = [awg.implt_writl_str(awg.source, 'TRIG:INT')];
            writeline(awg.rp, cmd);
        end

        function rep = receive(awg)
            writeline(awg.rp, 'ACQ:RST');
            writeline(awg.rp, 'ACQ:DEC 1');

            % if trigger is enabled, set trigger threshold
            if (awg.en_trig)
                cmd = ['ACQ:TRIG:LEV ' num2str(awg.trig_lvl)];
                writeline(awg.rp, cmd);
            end

            cmd = ['ACQ:' awg.destination, ':GAIN LV'];
            writeline(awg.rp, cmd);

            cmd = ['ACQ:TRIG:DLY ' num2str(awg.trig_del)];
            writeline(awg.rp, cmd);

            writeline(awg.rp, 'ACQ:START');
            %writeline(awg.rp, 'ACQ:TRIG NOW');

            cmd = ['ACQ:TRIG ', 'CH', num2str(awg.dst), '_PE'];
            writeline(awg.rp, cmd);

            while 1
                trig_rsp = writeread(awg.rp, 'ACQ:TRIG:STAT?');
                if strcmp('TD',trig_rsp(1:2))
                    break;
                end
            end
            cmd = ['ACQ:',awg.destination,':DATA?'];
            signal_str = writeread(awg.rp, cmd);
            rep = sscanf(signal_str(1,2:length(signal_str)-3), '%f,');
        end

        function wait(awg, n_samples)
            % calculate with 16384 sames at 125 MSps
            % then wait
            wt = n_samples / (awg.adc_sample_rate);
            pause(wt);
        end

        function setup_adc(awg)
            writeline(awg.rp, 'ACQ:RST');
            writeline(awg.rp, 'ACQ:DEC 1');

            % if trigger is enabled, set trigger threshold
            if (awg.en_trig)
                cmd = ['ACQ:TRIG:LEV ' num2str(awg.trig_lvl)];
                writeline(awg.rp, cmd);
            end

            cmd = ['ACQ:' awg.destination, ':GAIN LV'];
            writeline(awg.rp, cmd);

            cmd = ['ACQ:TRIG:DLY ' num2str(awg.trig_del)];
            writeline(awg.rp, cmd);
            
        end

        function rep = adc_read(awg)
            writeline(awg.rp, 'ACQ:START');
            %writeline(awg.rp, 'ACQ:TRIG NOW');

            cmd = ['ACQ:TRIG ', 'CH', awg.dst+'0', '_PE'];
            writeline(awg.rp, cmd);

            while 1
                trig_rsp = writeread(awg.rp, 'ACQ:TRIG:STAT?');
                if strcmp('TD',trig_rsp(1:2))
                    break;
                end
            end
            cmd = ['ACQ:',awg.destination,':DATA?'];
            signal_str = writeread(awg.rp, cmd);
            rep = sscanf(signal_str(1,2:length(signal_str)-3), '%f,');
        end
    end
end
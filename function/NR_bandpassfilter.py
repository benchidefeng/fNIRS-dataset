from scipy.signal import butter, filtfilt

def IIR_filter(data, hpf, lpf):
    # This function performs a IIR bandpass filter on time course data.
    #
    # USAGE:
    # Filt_data = IIR_filter(data, hpf, lpf)
    #
    # INPUT:
    # data - SNIRF data type containing data time course to filter, time vector, and channels.
    # hpf - high pass filter frequency (Hz)
    #       Typical value is 0 to 0.02.
    # lpf - low pass filter frequency (Hz)
    #       Typical value is 0.5 to 3.
    #
    # OUTPUT:
    # Filt_data - SNIRF data type containing the filtered data time course data
    T = data['T']
    fs = 1 / T
    Filt_data = data['Data']

    # First Low-pass filter
    lpf_norm = lpf / (fs / 2)
    Ch = Filt_data.shape[1]
    if lpf_norm > 0:
        filter_order = 3
        b, a = butter(filter_order, lpf_norm, btype='low')
        # pdb.set_trace()
        for i in range(Ch):
            Filt_data[:, i] = filtfilt(b, a, Filt_data[:, i])

    # Then High-pass filter
    hpf_norm = hpf / (fs / 2)
    if hpf_norm > 0:
        filter_order = 3
        b, a = butter(filter_order, hpf_norm, 'high')
        for i in range(Ch):
            Filt_data[:, i] = filtfilt(b, a, Filt_data[:, i])

    return Filt_data

def NR_bandpassfilter(nirsdata):
    T = nirsdata['T']
    HighPass = 0.015
    LowPass = 0.085
    oxyData = nirsdata['oxyData']
    oxyData = {'Data': oxyData, 'T': T}
    dxyData = nirsdata['dxyData']
    dxyData = {'Data': dxyData, 'T': T}
    totalData = nirsdata['totalData']
    totalData = {'Data': totalData, 'T': T}

    nirsdata['oxyData'] = IIR_filter(oxyData, HighPass, LowPass)
    nirsdata['dxyData'] = IIR_filter(dxyData, HighPass, LowPass)
    nirsdata['totalData'] = IIR_filter(totalData, HighPass, LowPass)

    return nirsdata
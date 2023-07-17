import numpy as np


def NR_detrend(nirsdata, order):
    # This function is using polynomial regression models to estimate linear or nonlinear trends.
    #   This trend is then subtracted from the original hemoglobin concentration signal.
    #
    # Usage:
    #   Detr_nirsdata = NR_detrend(nirsdata, order)
    #
    # Input:
    #   nirsdata: the original hemoglobin concentration signal containing data time course to filter, time vector, and channels.
    #   order: typical value is 1 or 2.
    #
    # Output:
    #   Detr_nirsdata: SNIRF data type containing the filtered data time course data
    tp = nirsdata['oxyData'].shape[0]
    oxyData = nirsdata['oxyData']
    dxyData = nirsdata['dxyData']
    totalData = nirsdata['totalData']

    for ch in range(nirsdata['oxyData'].shape[1]):
        # oxyData
        p_oxy = np.polyfit(np.arange(1, tp + 1).T, oxyData[:, ch], order)
        base_oxy = np.polyval(p_oxy, np.arange(1, tp + 1))
        oxyData[:, ch] = oxyData[:, ch] - base_oxy.T

        # dxyData
        p_dxy = np.polyfit(np.arange(1, tp + 1).T, dxyData[:, ch], order)
        base_dxy = np.polyval(p_dxy, np.arange(1, tp + 1))
        dxyData[:, ch] = dxyData[:, ch] - base_dxy.T

        # totalData
        p_total = np.polyfit(np.arange(1, tp + 1).T, totalData[:, ch], order)
        base_total = np.polyval(p_total, np.arange(1, tp + 1))
        totalData[:, ch] = totalData[:, ch] - base_total.T

    Detr_nirsdata = {'oxyData':oxyData,'dxyData':dxyData,'totalData':totalData,'T':0.020,'nch':8}

    return Detr_nirsdata

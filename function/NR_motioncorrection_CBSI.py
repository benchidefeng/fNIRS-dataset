import numpy as np


def NR_motioncorrection_CBSI(nirsdata):
    # This function performs a correlation-based signal improvement of the concentration
    # changes in order to correct for motion artifacts.
    # The algorithm follows the procedure described by
    # Cui et al.,NeuroImage, 49(4), 3039-46 (2010).
    #
    # USAGE:
    # Cbsi_nirsdata = NR_motioncorrection_CBSI(nirsdata)
    #
    # INPUT:
    # nirsdata: Concentration changes (it works with HbO and HbR)
    #
    # OUTPUT:
    # Cbsi_nirsdata: nirsdata after correlation-based signal improvement correction, same
    #           size as nirsdata (Channels that are not in the active ml remain unchanged)
    nch = nirsdata['nch']
    for ii in range(nch):
        oxyData = nirsdata['oxyData'][:, ii]
        dxyData = nirsdata['dxyData'][:, ii]
        sd_oxy = np.std(oxyData, axis=0)
        sd_dxy = np.std(dxyData, axis=0)
        alfa = sd_oxy / sd_dxy
        nirsdata['oxyData'][:, ii] = 0.5 * (oxyData - alfa * dxyData)
        nirsdata['dxyData'][:, ii] = -(1 / alfa) * nirsdata['oxyData'][:, ii]
        nirsdata['totalData'][:, ii] = nirsdata['oxyData'][:, ii] + nirsdata['dxyData'][:, ii]

    return nirsdata
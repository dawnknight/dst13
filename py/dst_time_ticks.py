
# -------- 
#  Generate some utilities for putting hourly time labels on plots
#
#  2013/12/16 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def time_ticks(units=10.):

    """ Ticks for times.  Input unit is seconds per time step. """

    # -- set the number of time steps per hour and range
    hour = 3600./units


    # -- return the ticks
    htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]
    ticks  = [hour*j for j in range(10+1)]

    return ticks, htimes

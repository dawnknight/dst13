# -------- 
#  Set the start and end times for nighttime slices
#
#  2013/11/30 - Written by Greg Dobler (CUSP/NYU)
# -------- 
def night_times():

    """ Set the start and end times for the nighttime slices """

    # -- set the parameters
    days  = ["10/26", "10/27", "10/28", "10/29", "10/30", "10/31", "11/01",
             "11/02", "11/03", "11/04", "11/05", "11/06", "11/07", "11/08",
             "11/09", "11/10", "11/11", "11/12", "11/13", "11/14", "11/15",
             "11/16", "11/17"]
    hours = ["19:00:00", "05:00:00"]


    # -- initialize start and end times and loop through days
    start = []
    end   = []

    for i in range(len(days)-1):
        start.append(days[i] + "/13 " + hours[0])
        end.append(days[i+1] + "/13 " + hours[1])

    return start, end

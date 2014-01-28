import numpy as np
import copy as cp


# -------- 
#  Sort an input matrix to minimize the cost using Hungarian/Munkres algorithm
#
#  2014/01/27 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def match(cost_in, sorts=[]):

    # -- make a copy since cost will be modified
    cost = cp.deepcopy(cost_in)
    sost = np.zeros(cost.shape) # sorted cost matrix to be returned

    # -- utilities
    nn      = cost.shape[0]
    lines   = np.zeros(cost.shape, dtype=np.int)
    uncov0  = np.zeros(cost.shape, dtype=bool)
    rows    = np.arange(nn**2).reshape(nn,nn) / nn
    cols    = np.arange(nn**2).reshape(nn,nn) % nn
    indices = np.arange(nn)
    nzr     = np.zeros(nn,dtype=np.int)
    nzc     = np.zeros(nn,dtype=np.int)


    # -- initialize the matrix
    cost[:,:] = (cost.T-cost.min(1)).T # sub min val from each row
    cost     -= cost.min(0) # sub min val from each col


    # -- reduce matrix by interatively identifying the minimum number
    #    of lines, nline, needed to cover the zeros and updating.
    #    when nline=nn (the dimensionality of the matrix), stop.
    nline = 0

    while nline<nn:
        lines[:,:]  = 0
        uncov0[:,:] = (cost<1e-6) & (lines==0) # uncovered zeros
        nzr[:]      = uncov0.sum(1) # number of uncovered zeros in a given row
        nzc[:]      = uncov0.sum(0) # number of uncovered zeros in a given col
        nline       = 0

        while nzr.max()>0: # stop when there are no more uncovered zeros
            index = 0
            mval  = min(nzr[nzr>0].min(),nzc[nzc>0].min())

            if mval in nzr:
                rind = indices[nzr==mval][0] # identify an uncovered 0
                cind = cols[rind,uncov0[rind]][index] # corresponding col
            elif mval in nzc:
                cind = indices[nzc==mval][0] # identify an uncovered 0
                rind = rows[uncov0[:,cind],cind][index] # corresponding row
            else:
                print("DST_MATCH: logical error!!! bailing out...")
                return


            if nzr[rind]==1: # 1 zero, cover corresponding col
                lines[:,cind] +=1
            elif nzc[cind]==1: # 1 zero, cover corresponding row
                lines[rind,:] += 1
            else: # multiple zeros, cover maximum possible (or row)
                if nzr[rind]<nzc[cind]:
                    lines[:,cind] += 1
                else:
                    lines[rind,:] += 1
            nline += 1

            # generate new list of uncovered zeros
            uncov0[:,:] = (cost<1e-6) & (lines==0) 
            nzr[:]      = uncov0.sum(1)
            nzc[:]      = uncov0.sum(0)

        if see:
            print nline

        # update the cost matrix
        if nline<nn:
            delta           = cost[lines==0].min()
            cost[lines==0] -= delta
            cost[lines==2] += delta


    # -- some consistency checks
    if nline>nn:
        print("DST_MATCH: number of lines is greater than dimensionality!!!")
        print("DST_MATCH:   bailing out...")
        return

    if (0 in (cost<1e-6).sum(1)) or (0 in (cost<1e-6).sum(0)):
        print("DST_MATCH: there is not a zero in each row and column!!!")
        print("DST_MATCH:   bailing out...")
        return



    # -- matrix is reduced, assign indices by recursively going
    #    through the cost matrix, assigning the rows and cols with
    #    only one zero when possible, and choosing when impossible.
    #    stop when all rows have been assigned exactly once.
    lines[:,:] = 0
    nassign    = 0

    while nassign<nn:

        # -- calculate the number of zeros in the rows and columns
        uncov0[:,:] = (cost<1e-6) & (lines==0)
        nzr[:]      = uncov0.sum(1)
        nzc[:]      = uncov0.sum(0)

        index = 0
        mval  = min(nzr[nzr>0].min(),nzc[nzc>0].min())

        if mval in nzr:
            rind = indices[nzr==mval][0]
            cind = cols[rind,uncov0[rind]][index]
        elif mval in nzc:
            cind = indices[nzc==mval][0]
            rind = rows[uncov0[:,cind],cind][index]
        else:
            print("DST_MATCH: logical error in assignment!!! bailing out...")
            return

        sorts.append([rind,cind])
        lines[rind,:] += 1
        lines[:,cind] += 1
        nassign       += 1

        if see:
            print nassign


    # -- reorder input cost matrix to minimize the trace and return
    for old_row, new_row in sorts:
        sost[new_row] = cost[old_row]

    return sost

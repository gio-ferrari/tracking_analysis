import numpy as np
import sys

def load_tcspc_data(tcspc_data_path):
    
    """   
    Open exp TCSPC data
    Input
    ----------   
        supfolder
        folder   
        subfolder 
        name : file name of experiment
    
    Returns
    -------    
        absTime : array, absTime tags of collected photons
        relTime : array, relTime tags of collected photon 
        channel : array, channel tags of collected photons (=originating APD)
    """
    
    inputfile = open(tcspc_data_path, "rb")
    data = []
    for line in inputfile:
    #    print(line)
        data.append(int(line))
        
        
    print("length", len(data))
#    print(data)
    relTime, absTime = convertHT3(data)
    
    # channel = coord[2, :]
    # absTime = coord[1, :]
    # relTime = coord[0, :]
    globRes = 1/ 19489040  #syncrate
    timeRes = 2 * 1e-12
    # globRes = 1e-3 # absTime resolution 
    # timeRes = 1 # relTime resolution (it's already in ns)
    
    absTime = absTime * globRes #s
    relTime = relTime * timeRes * 1e9 #ns
    
    return absTime, relTime

def convertHT3(countlist):
    oflcorrection = 0
    dlen = 0
    T3WRAPAROUND = 1024
    ntries = 0
    dtime_array = np.zeros(len(countlist))
    truensync_array = np.zeros(len(countlist))
    channel_array = np.zeros(len(countlist))
    for recNum in range(0, len(countlist)):
        if True:
            tempvalue = bin(countlist[recNum])[2:]
            if len(tempvalue) < 32:
                recordData = (32 - len(tempvalue))*"0"+ tempvalue #mit 0 auffuellen
            else:
                recordData = tempvalue
            ntries += 1

            #print("\n")
            #print("ntries = ", ntries)
        else:
            print("The file ended earlier than expected, at record %d/%d %d."\
                % (recNum, len(countlist), dlen))
            #return dtime_array, truensync_array #TODO:NAJA

            #exit(0)
        #print(recordData)
        #print(len(recordData))
        #print(type(recordData))
        special = int(recordData[0:1], base=2)
        channel = int(recordData[1:7], base=2)
        dtime = int(recordData[7:22], base=2)
        nsync = int(recordData[22:32], base=2)
        
        
        
        if special == 1:
            if channel == 0x3F: # Overflow
                # Number of overflows in nsync. If 0 or old version, it's an
                # old style single overflow
                if nsync == 0:
                    oflcorrection += T3WRAPAROUND
                    #print("%u OFL * %2x\n" % (recNum, 1))
                else:
                    oflcorrection += T3WRAPAROUND * nsync
                    #print("%u OFL * %2x\n" % (recNum, 1))
            if channel >= 1 and channel <= 15: # markers
                truensync = oflcorrection + nsync
                dtime_array[recNum] = dtime
                truensync_array[recNum] = truensync
                channel_array[recNum] = channel
        else: # regular input channel
            truensync = oflcorrection + nsync
            #print("%u CHN %1x %u %8.0lf %10u\n" % (recNum, channel,\
            #truensync, (truensync * 1 * 1e9), dtime)) #TODO changeglobRes
            channel_array[recNum] = channel
            dtime_array[recNum] = dtime
            truensync_array[recNum] = truensync
            
            dlen += 1
            #gotPhoton(truensync, channel, dtime)
        if recNum % 100000 == 0:
            sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(len(countlist))))
            
            sys.stdout.flush()  
    return dtime_array[truensync_array != 0], truensync_array[truensync_array != 0]
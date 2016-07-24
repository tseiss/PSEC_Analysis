import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import Pulse
import Event
import DataSet
import Group

#Computation of systematics, rise time, and amplitude information for pads

#outFile = "Results.txt"
#print "Writing all output to file", outFile
#sys.stdout = open(outFile, 'w')

estRiseTime = 30
risePointTrig = 2.5
baselineLength = 30
PulseWidth = 130 
minAmplitude = -50

filenameArray = ["../../DataStorage/SummerData/1Inch_0706.txt", "../../DataStorage/SummerData/1Inch_0713_Systematics.txt", "../../DataStorage/SummerData/1Inch_0713_Systematics2.txt"]
filenameArray_clean = ["Sys1_Clean.txt", "Sys2_Clean.txt", "Sys3_Clean.txt"]
group = Group.Group(filenameArray)
group.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
group.writeAll(filenameArray_clean)
group = Group.Group(filenameArray_clean)
sysAmpErrFrac = group.getSystematicAmplitudeError()
print "\n##############################\n"
sysRtErrFrac = group.getSystematicRiseTimeError()
print "\n##############################\n"

filenamed  = "../../DataStorage/SummerData/1Inch_Direct_0706.txt"
filename5  = "../../DataStorage/SummerData/PositionScan/5mm.txt"
filename10 = "../../DataStorage/SummerData/1Inch_0706.txt"
filename15 = "../../DataStorage/SummerData/15Inch_0707.txt"

group = Group.Group([filenamed, filename5, filename10, filename15])
group.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
cleanNames = ["d_clean.txt", "5_clean.txt", "10_clean.txt", "15_clean.txt"]
group.writeAll(cleanNames)
group = Group.Group(cleanNames)
#group.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude, minPeakLoc = 0, maxPeakLoc = 0)
print "\n##############################\n"

nameArray = ["Direct", "0.5 in", "1.0 in", "1.5 in"]
i = 0
for data in group:
	amps = data.getAvAmplitudes(ordered = True)
	rt = [entry*data.timestep for entry in data.getAvRiseTime()]
	print nameArray[i]
	statErr = amps[0][1]
	sysErr = amps[0][0]*sysAmpErrFrac
	print "Amplitude:", amps[0][0], "pm", statErr, "(stat) pm", sysErr, "(sys) mV"
	print "Total Error:", np.sqrt(statErr**2+sysErr**2), "mV"
	statErr = rt[1]
	sysErr = sysRtErrFrac*rt[0]
	print "Rise Time:", rt[0], "pm", statErr, "(stat) pm", sysErr, "(sys) ns"
	print "Total Rise Time Error:", np.sqrt(statErr**2+sysErr**2), "ns"
	print
	i += 1


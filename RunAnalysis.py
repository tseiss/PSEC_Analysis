import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import Pulse
import Event
import DataSet
import spheDataSet
import Group

spheFile = "../../DataStorage/SummerData/Strips/sphe_nicr.txt"
spheFile_clean = "sphe_nicr_clean.txt"
#spheFile = "../../DataStorage/SummerData/1Inch_0706.txt"
data = spheDataSet.spheDataSet(spheFile)
data.cleanEvents()
data.write(spheFile_clean)
data.plotRandomEvents(10)
print "Averaging..."
data.averagePulses().plotPulses()

"""
#Computation of systematics, rise time, and amplitude information for pads
outFile = "Results.txt"
print "Writing all output to file", outFile
sys.stdout = open(outFile, 'w')

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
sys.exit()
"""
#data = DataSet.DataSet("../../DataStorage/SummerData/Strips/direct_sphe_take2.txt")
#data.cleanEvents(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
#data.averagePulses().plotPulses()
#sys.exit()

#data = DataSet.DataSet("10_clean.txt")
#data.averagePulses().plotPulses()
#data.eventArray[0].getAmpsNearMax()
#sys.exit()

"""
for data in group:
	print max(data.getAvPower())
	print np.sum(data.getAvPower())
	print "\n###############################\n"
	#print data.getAvAmplitudes(ordered = True)
	#print [entry*data.timestep for entry in data.getAvRiseTime()]
	#print
"""
#group.getSystematicAmplitudeError()
#group.getSystematicRiseTimeError()


#filename = "../../DataStorage/SummerData/1Inch_Direct_0706.txt"
#data = DataSet.DataSet(filename)
#data.cleanEvents(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude) #minPeakLoc = 50,
#data.plotEvent(100, [0, 1, 2, 3, 4, 5])
#data.plotRandomEvents(30)#, RisePoint = True)

#filename = "../../DataStorage/SummerData/1Inch_0706.txt"
#filename = "../../DataStorage/SummerData/PositionScan_Cleaned/0mm_clean.txt"
#data = DataSet.DataSet(filename)
#avAmps = data.getAvAmplitudes()
#data.plotRandomEvents(10)
#numFiles = 30
#addFactor = 0
#filenameArray = ["../../DataStorage/SummerData/PositionScan/" + str(i+addFactor) + "mm.txt" for i in range(0, numFiles)] #5mm has problems
#group = Group.Group(filenameArray)
#group.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
#group.dataSetArray[0].plotAllEvents(1)
#cleanFilenames = ["../../DataStorage/SummerData/PositionScan_Cleaned/" + str(i+addFactor) + "mm_clean.txt" for i in range(0, numFiles)]
#group.writeAll(cleanFilenames)


#group2 = Group.Group(cleanFilenames)
#group2.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude, minPeakLoc = 0, maxPeakLoc = 0)
#group2.PositionScan(range(0, numFiles), [1, 2], filename = "PosScan.txt")
#group3 = Group.Group([])
#group3.plotPositionScan(filename = "PosScan.txt")
#group2.dataSetArray[0].plotAllEvents(1)
#data = DataSet.DataSet(filename)
#data.cleanEvents(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
#data.write("TestOut.txt")
#data2 = DataSet.DataSet("TestOut.txt")
#data2.plotRandomEvents(10)
#data.plotRandomEvents(5)
#data.plotAmplitudeDist()
#data.plotMeasRiseTimes()
#print data.getPrimaryChannelArray()
"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data.plotEvent(ax, 794, [4], show = False, fmtArray = ['r-'])
data2.plotEvent(ax, 1746, [4], show = False, fmtArray = ['b-'])
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(['Direct', 'Inside-Out'], loc = 2, fontsize = 18)
plt.show()
"""


"""
fig, ax = plt.subplots()
data.plotMeasRiseTimes(ax, show = False)
#data.averagefft()

filename = "../../DataStorage/1Inch_10e4_Direct.txt"
data = DataSet.DataSet(filename)
data.cleanEvents(baselineLength, PulseWidth, estRiseTime, risePointTrig) #minPeakLoc = 50,
data.plotMeasRiseTimes(ax, show = False)
plt.legend(["Inside-Out", "Direct"])
plt.show()

#data.eventArray[10].plotPulsesWithRisepoint(data.findPrimaryChannel(), estRiseTime, risePointTrig)
#data.plotMeasRiseTimes(show = True)
#data.plotEventsAboveThresholdRiseTime(1.5, 5)
"""
"""
#Plots amplitude distibution
amplitudeDist = data.getAmplitudeDist()
amplitudeDist = [-x for x in amplitudeDist]
print "Average Amplitude:", round(np.average(amplitudeDist),3), "mV"
print "Standard Dev:", round(np.std(amplitudeDist), 3), "mV"
fig, ax = plt.subplots()
plt.hist(amplitudeDist, bins = 50, alpha=0.75)
plt.xlabel("Peak Amplitude (mV)", fontsize = 14)
plt.ylabel("Counts", fontsize = 14)
majorLocator = MultipleLocator(10)
minorLocator = MultipleLocator(2)
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_minor_locator(minorLocator)
majorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(majorLocator)
minorLocator = MultipleLocator(10)
ax.yaxis.set_minor_locator(minorLocator)
plt.show()
"""
#data.plotRandomEvents(10, RisePoint = True)
#data.plotAllEvents(data.findPrimaryChannel())
#data.plotRiseTimes(estRiseTime = estRiseTime, risePointTrig = risePointTrig)

"""Plots average event
avEvent = data.averagePulses()
fig, ax = plt.subplots()
avEvent.plotPulses([4, 1, 2, 3, 0, 5], show = False, fmtArray = [ 'k-', 'k--', 'k--', 'k--', 'k--', 'k--'])
plt.legend(["Center Pad", "Adjacent Pad"], loc = 2)
plt.title("Average Pulse Shape", fontsize = 16)
majorLocator = MultipleLocator(1)
#majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.25)
ax.xaxis.set_major_locator(majorLocator)
#ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)

majorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(majorLocator)
minorLocator = MultipleLocator(10)
ax.yaxis.set_minor_locator(minorLocator)
plt.show()
avEvent.pulseArray[4].plotfft()
"""

#print data
#data.plotEvent(0, [0, 1]) #Plots channels 0, 1 of event 0 in the DataSet
#avEvent = data.averagePulses()
#avEvent.plotPulses([0, 1, 2, 3, 4, 5])
#data.plotIntegralDist([0, 1])

#data.findPrimaryChannel()
#minPeakLoc, baselineLength, PulseWidth

"""
for event in data: #Plot if > 130 in channel 1
	if -150 < event.getPulse(0).integral < -149.9 :
		event.getPulse(0).plot()
"""
"""
for i in range(0, 100):
	data.plotEvent(i, [0, 1, 2], show = False)
plt.show()

filename = "../../DataStorage/Sample00_noise.txt"
data = DataSet.DataSet(filename)
avNoise = data.averagePulses()

avNoiseNeg = avNoise*(-1)
Result = (avEvent + avNoiseNeg)*(0.5)
Result.plotPulses([0, 1, 2, 3, 4, 5])
"""

#data.plotEvent(1, [0, 1])

"""
eventArray[0].plotPulses([True, True, False, False, False, False])
eventArray[0].getffts()
eventArray[0].plotPulsesfft([True, True, False, False, False, False])
"""

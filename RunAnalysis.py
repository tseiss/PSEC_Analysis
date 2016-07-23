import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import Pulse
import Event
import DataSet
import Group


estRiseTime = 30
risePointTrig = 2.5
baselineLength = 30
PulseWidth = 130 
minAmplitude = -40


#filenameArray = ["../../NIM/nicr_12-17.txt", "../../NIM/nicr_12-17_systematic1.txt", "../../NIM/nicr_12-17_systematic2.txt", "../../NIM/nicr_12-17_systematic3.txt"]
#group = Group.Group(filenameArray)
#group.cleanAll(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude)
filenameArray_clean = ["../../NIM/nicr_12-17_clean.txt", "../../NIM/nicr_12-17_systematic1_clean.txt", "../../NIM/nicr_12-17_systematic2_clean.txt", "../../NIM/nicr_12-17_systematic3_clean.txt"]
#group.writeAll(filenameArray_clean)
group = Group.Group(filenameArray_clean)
#group.getSystematicAmplitudeError()
group.getSystematicRiseTimeError()

sys.exit()

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

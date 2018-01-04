import DataSet
import Event
import Pulse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

#A group of DataSets for the largest scale analysis
class Group:
	def __init__(self, filenameArray = None):
		self.dataSetArray = []
		if filenameArray:
			self.fill(filenameArray)
		else:
			print "Creating empty group."
		self.iterIndex = 0
		self.numDataSets = len(self.dataSetArray)

	def __iter__(self):
		return self	
	def next(self):
		if self.iterIndex >= self.numDataSets:
			self.iterIndex = 0
			raise StopIteration
		else:
			self.iterIndex += 1
			return self.dataSetArray[self.iterIndex-1]

	#Reads all the files in filename array
	def fill(self, filenameArray):
		numFiles = len(filenameArray)
		print "Reading", numFiles, "files... ", 
		oldstdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')
		i = 0
		for filename in filenameArray:
			oldstdout.write(str(i+1)+" "),
			i += 1
			self.dataSetArray.append(DataSet.DataSet(filename))
		sys.stdout = oldstdout
		print "Done."

	def getArray(self):
		return self.dataSetArray

	#Cleans all data sets in the group. Params are same as in DataSet.cleanEvents()
	def cleanAll(self, baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude, minPeakLoc = None, maxPeakLoc = None):
		i = 0
		print "Cleaning data."
		for dataSet in self:
			i += 1
			print str(i) + ":", 
			dataSet.cleanEvents(baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude, minPeakLoc, maxPeakLoc)

	#Writes all dataSets, in order, one to each file in filenameArray
	def writeAll(self, filenameArray):
		if len(filenameArray) != self.numDataSets:
			print "Number of files to write into needs to equal number of data sets in Group."
			return
		print "Writing all data sets...",
		for i in range(0, self.numDataSets):
			self.dataSetArray[i].write(filenameArray[i])
		print "Done."

	#Returns array of number of events in each dataset
	def getNumEvents(self):
		numEvents = []
		for data in self:
			numEvents.append(data.getNumEvents())
		return numEvents

	#Computes the average amplitudes and stat errors on the amps for all files in the group for the channels in channels (ie channels = [1, 2])
	#If filename given, writes all this info to file. Can be read with plotPositionScan()
	def PositionScan(self, xArray, channels, filename = None):
		if len(xArray) != self.numDataSets:
			print "xArray length needs to equal number of dataSets in group."
			return
		amps = [[] for _ in channels]
		stds = [[] for _ in channels]
		counter = 0
		for data in self:
			counter += 1
			print str(counter)+":",
			dataAmps = data.getAvAmplitudes()
			for i in range(0, len(channels)):
				amps[i].append(-dataAmps[channels[i]][0]/1000.0)
				stds[i].append(dataAmps[channels[i]][1]/1000.0)
		fmtArray = ['bo-', 'ro-']
		errfmtArray = ['b', 'r']
		errNegArray = [np.array([amps[i][j]-stds[i][j]/2.0 for j in range(0, len(amps[i]))]) for i in range (0, len(channels))]
		errPosArray = [np.array([amps[i][j]+stds[i][j]/2.0 for j in range(0, len(amps[i]))]) for i in range (0, len(channels))]
		if filename:
			print "Writing position scan data to file."
			f = open(filename, 'w')
			for i in range(0, len(channels)):
				f.write(str(channels[i]) + "\n")
				for j in range(0, len(amps[i])):
					f.write(str(xArray[j]) + " " + str(amps[i][j]) + " " + str(errNegArray[i][j]) + " " + str(errPosArray[i][j]) + "\n")
			f.close()
		return (amps, errNegArray, errPosArray)

	#Plots a position scan. If given filename, reads position data from there that has been written by PositionScan()
	def plotPositionScan(self, xArray = None, channels = None, filename = None, show = True):
		if filename:
			f = open(filename)
			amps = []
			errNeg = []
			errPos = []
			channels = []
			xArray = []
			xArrayFilled = False
			for line in f:
				words = line.split()
				if len(words) == 1:
					channels.append(int(words[0]))
					amps.append([])
					errNeg.append([])
					errPos.append([])
					if len(xArray) > 0:
						xArrayFilled = True
				else:
					if not xArrayFilled:
						xArray.append(float(words[0]))
					amps[-1].append(float(words[1]))
					errNeg[-1].append(float(words[2]))
					errPos[-1].append(float(words[3]))
			f.close()
		elif xArray and channels:
			amps, errNeg, errPos = self.PositionScan(xArray, channels, filename)
		else:
			print "Not enough information given to plotPositionScan"
			return None
		fmtArray = ['bo-', 'ro-']
		errfmtArray = ['b', 'r']
		fig, ax = plt.subplots()
		for i in range(0, len(channels)):
			plt.plot(xArray, amps[i], fmtArray[i])
			plt.fill_between(xArray, errNeg[i], errPos[i], facecolor = errfmtArray[i], alpha = 0.5)
		plt.xlabel("Laser Position (mm)", fontsize = 16)
		plt.ylabel("Average Pulse Amplitude (mV)", fontsize = 16)
		plt.legend(["Pad 1", "Pad 2"])
		majorLocator = MultipleLocator(5)
		minorLocator = MultipleLocator(1)
		ax.xaxis.set_major_locator(majorLocator)
		ax.xaxis.set_minor_locator(minorLocator)
		majorLocator = MultipleLocator(50)
		ax.yaxis.set_major_locator(majorLocator)
		minorLocator = MultipleLocator(10)
		ax.yaxis.set_minor_locator(minorLocator)
		if show:
			plt.show()

		#cutNum = -4
		#xArray = xArray[:cutNum]
		#amps = [amps[i][:cutNum] for i in range(0, len(amps))]
		#errPos = [errPos[i][:cutNum] for i in range(0, len(errPos))]
		#errNeg = [errNeg[i][:cutNum] for i in range(0, len(errNeg))]
		f2 = [amps[0][i]/(amps[0][i] + amps[1][i]) for i in range(0, len(amps[0]))] 
		sigmaA1 = [errPos[0][i]-errNeg[0][i] for i in range(0, len(errPos[0]))]
		sigmaA2 = [errPos[1][i]-errNeg[1][i] for i in range(0, len(errPos[1]))]
		sigmaf2 = [f2[i]/(amps[0][i]+amps[1][i])*np.sqrt((amps[1][i]/amps[0][i]*sigmaA1[i])**2 + sigmaA2[i]**2) for i in range(0, len(amps[0]))]
		deltax = xArray[1]-xArray[0]
		slopef2 = [np.abs((f2[i+1]-f2[i]))/deltax for i in range(0, len(f2)-1)]
		plt.plot(xArray, f2)
		plt.title("f2")
		plt.show()
		PosErrf2 = [sigmaf2[i]/slopef2[i] for i in range(0, len(slopef2))]
		print "Min Position Error with f2:", round(min(PosErrf2), 10), "mm"
		fig, ax = plt.subplots()
		plt.plot(xArray[:-1], PosErrf2, 'k', lw = 2)
		plt.xlabel("Laser Position (mm)", fontsize = 16)
		plt.ylabel("Position Resolution (mm)", fontsize = 16)
		ax.xaxis.set_major_locator(MultipleLocator(5))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.yaxis.set_major_locator(MultipleLocator(0.5))
		ax.yaxis.set_minor_locator(MultipleLocator(0.1))
		plt.show()
		return (amps, errNeg, errPos)

	#Assumes all data sets in self are to be compared for systematic errors
	#Prints a lot of systematic and statistical info in for the amplitudes
	def getSystematicAmplitudeError(self):
		ampArray = []
		statErrArray = []
		for data in self:
			ampArray.append([])
			statErrArray.append([])
			for entry in data.getAvAmplitudes(ordered = True):
				ampArray[-1].append(entry[0])
				statErrArray[-1].append(entry[1])
		primAmpArray = [entry[0] for entry in ampArray]	
		primErrArray = [entry[0] for entry in statErrArray]
		sigmaStat = 1/float(len(primErrArray)) * np.sqrt(np.sum([x**2 for x in primErrArray]))
		sigmaSys = np.std(primAmpArray)
		print "Pulse Amplitudes (mV):"
		for i in range(0, len(primAmpArray)):
			print primAmpArray[i], "pm", primErrArray[i]
		print "Average Amplitude:", np.average(primAmpArray), "mV"
		print "Statistical Error (mV):", sigmaStat
		print "Systematic Error:", sigmaSys, "mV"
		print "Total Error:", np.sqrt(sigmaSys**2+sigmaStat**2), "mV"
		fracErrSys = np.abs(sigmaSys/np.average(primAmpArray)) 
		print "Fractional Systematic Error:", fracErrSys
		return fracErrSys

	#Assumes all data sets in self are to be compared for systematic errors
	#Prints a lot of systematic and statistical info in for the rise times
	def getSystematicRiseTimeError(self):
		rtArray = []
		errArray = []
		for data in self:
			rt = data.getAvRiseTime()
			rtArray.append(rt[0]*data.timestep)
			errArray.append(rt[1]*data.timestep)
		print "Rise Times (ns):"
		for i in range(0, len(rtArray)):
			print round(rtArray[i], 3), "pm", round(errArray[i], 3)
		print "Average (ns):", np.average(rtArray)
		sigmaStat = 1/float(len(errArray))*np.sqrt(np.sum([x**2 for x in errArray]))
		print "Statistical Error (mV)", sigmaStat, "ns"
		sigmaSys = np.std(rtArray)
		print "Systematic Error (ns):", sigmaSys, "ns"
		fracSysErr = np.abs(sigmaSys)/np.average(rtArray)	
		print "Fractional Systematic Error:", fracSysErr
		return fracSysErr

	#goes in all datasets and counts 
	#pulses from all events and all channels
	#returns number of pulses and number of events total
	def countAllPulses(self, threshold, buttFreq):
		n_pulses = 0
		n_events = 0 
		for dset in self.dataSetArray:
			np, ne = dset.countAllPulses(threshold, buttFreq)
			n_pulses += np
			n_events += ne 

		return (n_pulses, n_events)




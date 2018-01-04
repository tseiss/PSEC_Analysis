import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import sys
import Pulse
import Event
import DataSet
import spheDataSet
import Group
import cPickle as pickle 

def plotPhdOneFile(filename):
	d = DataSet.DataSet(filename)
	fig, ax = plt.subplots()
	d.plotAllAmplitudeDist(ax)
	ax.set_title("Phd for pulses in \n" + filename)
	plt.show()


def plotRiseTimes(filename):
	d = DataSet.DataSet(filename)
	fig, ax = plt.subplots()
	d.plotMeasRiseTimes(ax)
	plt.show()

def plotPSD(filename):
	d = DataSet.DataSet(filename)
	fig, ax = plt.subplots()
	d.plotSummedPSD(ax)
	ax.set_title("PSD for " + filename)
	plt.show()

def showFilteredpulses(filename):
	d = DataSet.DataSet(filename)
	buttFreq = 2 #GHz
	channel = 2
	d.plotRandomFilterComparison(20, channel, buttFreq)

def triggeredDarkRate(filename):
	d = DataSet.DataSet(filename)
	threshold = -5 #mV threshold on calling things pulses
	fig, ax = plt.subplots()
	lamb = d.multiplePulseDarkRateMeasure(threshold, 2, ax)
	plt.show()


def noiseDarkRate(filename_list):
	group = Group.Group(filename_list)
	threshold = -5 #mV
	buttFreq = 2 #GHz
	np, ne = group.countAllPulses(threshold, buttFreq)
	def poisson(k, lamb):
			return (lamb**k/factorial(k))*np.exp(-lamb)

	#get one event's timestep assuming
	#all events in all files are the same
	timestep = group.getArray()[0].getTimestep() * 10**(-9) #in seconds
	samples = group.getArray()[0].getNumSamples()

	#rate constant given P = npulses/nevents = R*(dt*nsamples)
	rate_constant = np/(ne*timestep*samples) #in Hz
	print "Approximate dark rate is " + str(rate_constant) + " Hz"




if __name__ == "__main__":
	#filename = "../Pre-cesiation/Noise/2.7kV_auxtrig_darkbox_171202.mat"
	filename = "../Pre-cesiation/Pulses/2.7kV_2mvCh2_nosource_50mvDiv.mat"
	filename_list = ["../Pre-cesiation/Noise/2." + str(i)+"kV_auxtrig_darkbox_171202.mat" for i in [7,6,5,4,3,2]]
	plotRiseTimes(filename)




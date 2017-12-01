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


estRiseTime = 30
risePointTrig = 2.5
baselineLength = 30
PulseWidth = 130 
minAmplitude = [-50, 45, 20]
pol = [1, -1, -1]



direct_filenameArrayClean = ["../../NIM/direct_12-17_clean.txt", "../../NIM/direct_18-23_clean.txt", "../../NIM/direct_24-29_clean.txt"]
direct_filenameArrayCleanTake2 = ["../../NIM/strips_12-17_take2_clean.txt", "../../NIM/strips_18-23_take2_clean.txt", "../../NIM/strips_24-29_take2_clean.txt"]
nicr_filenameArrayClean = ["../../NIM/nicr_12-17_clean.txt", "../../NIM/nicr_18-23_clean.txt", "../../NIM/nicr_24-29_clean.txt"]
direct_filenameArraySystematics = ["../../NIM/strips_12-17_take2_clean.txt", "../../NIM/strips_systematics1_clean.txt", "../../NIM/strips_systematics2_clean.txt"]
nicr_filenameArraySystematics = ["../../NIM/nicr_12-17_clean.txt", "../../NIM/nicr_12-17_systematic1_clean.txt", "../../NIM/nicr_12-17_systematic2_clean.txt", "../../NIM/nicr_12-17_systematic3_clean.txt"]
direct_pad_1inch = ["../../NIM/pads/d_clean.txt"]
nicr_pad_1inch = ["../../NIM/pads/10_clean.txt"]



#Plot pulse inside-out and direct with nearest neighbors

nicr = DataSet.DataSet(nicr_pad_1inch[0])
direct = DataSet.DataSet(direct_pad_1inch[0])
fig, ax = plt.subplots(figsize=(15, 10))
nicr.plotEvent(15, [4,2], show=False, fmtArray=['b--', 'b--'])
direct.plotEvent(15, [4,2], show=False, fmtArray=['r', 'r'])
ax.lines[0].set_linewidth(4)
ax.lines[1].set_linewidth(2)
ax.lines[2].set_linewidth(4)
ax.lines[3].set_linewidth(2)

ax.get_xaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(length=10, width=2, which='minor')
ax.get_yaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_yaxis().set_tick_params(length=10, width=2, which='minor')
majorLocatorY = MultipleLocator(50)
minorLocatorY = MultipleLocator(10)
majorLocatorX = MultipleLocator(2)
minorLocatorX = MultipleLocator(0.5)
ax.get_xaxis().set_major_locator(majorLocatorX)
ax.get_xaxis().set_minor_locator(minorLocatorX)
ax.get_yaxis().set_major_locator(majorLocatorY)
ax.get_yaxis().set_minor_locator(minorLocatorY)
ax.set_ylabel("mV", fontsize=35)
ax.set_xlabel("time (ns)", fontsize=35)
plt.legend(("Capacitive pads", "Capacitive nearest neighbor", "Direct pads", "Direct nearest neighbor"), loc=(0.4,0.5), fontsize=20)
#plt.show()
plt.savefig("pulse_comparison_pads.png", bbox_inches='tight')
sys.exit()


#plot two SCALED sets of pulses on top of eachother
#one inside out, one direct

nicr = DataSet.DataSet(nicr_filenameArrayClean[0])
direct = DataSet.DataSet(direct_filenameArrayCleanTake2[0])
nicrT, nicrW = nicr.getEventWaveform(9, 2)
directT, directW = direct.getEventWaveform(9, 2)
nicrmax = min(nicrW)
directmax = min(directW)
factor = directmax/nicrmax
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot([(t + 0) for t in nicrT], [w*factor for w in nicrW], 'b', linestyle='--', linewidth=3, label="Capacitive pulse, scaled by x" + str(round(factor, 2)))
ax.plot(directT, directW, 'r', linestyle='-', linewidth=4, label="Direct coupled pulse")
ax.get_xaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(length=10, width=2, which='minor')
ax.get_yaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_yaxis().set_tick_params(length=10, width=2, which='minor')
majorLocatorY = MultipleLocator(50)
minorLocatorY = MultipleLocator(10)
majorLocatorX = MultipleLocator(2)
minorLocatorX = MultipleLocator(0.5)
ax.get_xaxis().set_major_locator(majorLocatorX)
ax.get_xaxis().set_minor_locator(minorLocatorX)
ax.get_yaxis().set_major_locator(majorLocatorY)
ax.get_yaxis().set_minor_locator(minorLocatorY)
ax.set_ylabel("millivolts", fontsize=35)
ax.set_xlabel("time (ns)", fontsize=35)
plt.legend(loc=(0.4, 0.5), fontsize=20)
#plt.show()
plt.savefig("shape_comparison_strips.png", bbox_inches='tight')
sys.exit()









#single channel analysis on amplitude
nicr_amp = []
direct_amp = []
nicr_err = []
direct_err = []
strip = [17, 16, 15, 14, 13, 12, 23, 22, 21, 20, 19, 18, 29, 28, 27, 26, 25, 24]
for i in range(len(nicr_filenameArrayClean)):
	data = DataSet.DataSet(nicr_filenameArrayClean[i])
	print "loaded filename: " + nicr_filenameArrayClean[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		nicr_amp.append(x[0])
		nicr_err.append(abs(x[1]))


for i in range(len(direct_filenameArrayCleanTake2)):
	data = DataSet.DataSet(direct_filenameArrayCleanTake2[i])
	print "loaded filename: " + direct_filenameArrayCleanTake2[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		direct_amp.append(x[0])
		direct_err.append(abs(x[1]))

#order the lists simultaneously to make the trace
#on the plot look nice
strip, nicr_amp, direct_amp, nicr_err, direct_err = (list(t) for t in zip(*sorted(zip(strip, nicr_amp, direct_amp, nicr_err, direct_err))))

'''
#symmetrize the array based on source being at strip 15
#complete it for striplines less than number 12
for n in range(1, 12):
	strip.append(n)
	nicr_amp.append(nicr_amp[17 - n])
	direct_amp.append(direct_amp[17 - n])
	nicr_err.append(nicr_err[17 - n])
	direct_err.append(nicr_err[17 - n])

#reorder again
strip, nicr_amp, direct_amp, nicr_err, direct_err = (list(t) for t in zip(*sorted(zip(strip, nicr_amp, direct_amp, nicr_err, direct_err))))
'''

#flip the polarity for looks
nicr_amp = [-1*x for x in nicr_amp]
direct_amp = [-1*x for x in direct_amp]

#convert strip array to distance from center
ref = 15 	#strip where pulse occurred
stripCTC = .18209 #center to center distance between strips (in)
dists = []
for d in strip:
	dists.append((stripCTC)*(d - ref))


def integrateRange(rang, x, y):
	integralArray = []
	newx = []
	for i in range(len(x)):
		if(abs(x[i]) <= rang):
			integralArray.append(y[i])
			newx.append(x[i])

	return np.trapz(integralArray, x=newx)

'''
print "Integral of both around 1'' from center"
nicr_int = integrateRange(1*stripCTC, dists, nicr_amp)
direct_int = integrateRange(1*stripCTC, dists, direct_amp)
print "NiCr integral : " + str(nicr_int)
print "Direct integral : " + str(direct_int)
print "Fraction: " + str(nicr_int/direct_int)
'''

#multiply by the ratio to put them on top of eachother
#using the center strip as a reference
#ratio = 0.3389
#direct_amp = [x*ratio for x in direct_amp]
#direct_err = [x*ratio for x in direct_err]


def Gaussian(x, mu, sigma, A):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))


stripNumber = range(-3, 15)
newdists = range(-2, 3)
(mu, sigma, A) = curve_fit(Gaussian, stripNumber, direct_amp, p0=[0, .5, 220])[0]
directsigma = str(round(sigma, 2))
fit_direct = Gaussian(newdists, mu, sigma, A)
(mu, sigma, A) = curve_fit(Gaussian, dists, nicr_amp, p0=[0, .2, 75])[0]
nicrsigma = str(round(sigma, 2))
fit_nicr = Gaussian(newdists, mu, sigma, A)
print nicrsigma
print directsigma




fig, ax = plt.subplots(figsize=(15, 10))	
ax.errorbar(stripNumber, direct_amp, yerr=direct_err, fmt='ro-', label=r"Direct coupling", linewidth=4, markersize=13)
ax.errorbar(stripNumber, nicr_amp, yerr=nicr_err, fmt='bs--', label=r"Capacitive coupling", linewidth=4, markersize=13)
#ax.plot(newdists, fit_direct, 'b', linewidth=3, alpha=0.5)
#ax.plot(newdists, fit_nicr, 'b', linewidth=3, alpha=0.5)
ax.set_xlabel("Strip number", fontsize=26)
ax.set_ylabel("Pulse Amplitude (mV)", fontsize=26)
#ax.set_title("Pulse amplitude distribution along a 30 strip-line pickup board", fontsize=26)
ax.get_yaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(length=10, width=2, which='minor')
major_ticks = stripNumber
ax.set_xticks(major_ticks)  
ax.set_xlim([min(major_ticks), max(major_ticks)])
#minorLocator = MultipleLocator(0.5)
#ax.xaxis.set_minor_locator(minorLocator)
ax.get_yaxis().set_ticks(np.arange(-50, 250, 25))
#ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.75)
ax.legend(fontsize=25, loc="center")
plt.savefig("strip_distribution_nim.png", bbox_inches='tight')
#plt.show()
sys.exit()





#calculate noise of electronics
data = DataSet.DataSet("../../NIM/noise.txt")
allSamples = data.getListOfSampleVoltages()
(mu, sigma) = norm.fit(allSamples)
fig, ax = plt.subplots(figsize=(15,10))
n, bins, patches = ax.hist(allSamples, bins = 50, alpha=0.75, normed=True)
y = mlab.normpdf(bins, mu, sigma)
l = ax.plot(bins,y,'r--', linewidth=2, label=r"$\mu = $" + str(mu) + r", $\sigma = $" + str(sigma))
ax.set_title("Histogram of all voltages on all samples of 10,000 events", fontsize=26)
ax.set_xlabel("mV",fontsize=24)
plt.legend(loc=4)
plt.savefig("noisehisto.png", bbox_inches='tight')
sys.exit()



#list the fraction of each channels C/D
nicr_amp = []
direct_amp = []
nicr_err = []
direct_err = []
strip = [17, 16, 15, 14, 13, 12, 23, 22, 21, 20, 19, 18, 29, 28, 27, 26, 25, 24]
for i in range(len(nicr_filenameArrayClean)):
	data = DataSet.DataSet(nicr_filenameArrayClean[i])
	print "loaded filename: " + nicr_filenameArrayClean[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		nicr_amp.append(x[0])
		nicr_err.append(abs(x[1]))


for i in range(len(direct_filenameArrayCleanTake2)):
	data = DataSet.DataSet(direct_filenameArrayCleanTake2[i])
	print "loaded filename: " + direct_filenameArrayCleanTake2[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		direct_amp.append(x[0])
		direct_err.append(abs(x[1]))

#order the lists simultaneously to make the trace
#on the plot look nice
strip, nicr_amp, direct_amp, nicr_err, direct_err = (list(t) for t in zip(*sorted(zip(strip, nicr_amp, direct_amp, nicr_err, direct_err))))

#flip the polarity for looks
nicr_amp = [-1*x for x in nicr_amp]
direct_amp = [-1*x for x in direct_amp]

#convert strip array to distance from center
ref = 15 	#strip where pulse occurred
stripCTC = 0.75 #center to center distance between strips (cm)
dists = []
for d in strip:
	dists.append((stripCTC)*(d - ref))

fracList = []
for i in range(len(dists)):
	fracAmp = nicr_amp[i]*nicr_amp[i]/(direct_amp[i]*direct_amp[i])
	fracList.append(fracAmp**(0.5))

print fracList
print dists

'''
fig, ax = plt.subplots(figsize=(15, 10))	
ax.plot(dists, fracList, 'go-', label=r"Fractional squared amplitude ($C^2/D^2)", linewidth=3, markersize=10)
ax.set_xlabel("Distance from laser position (cm)", fontsize=26)
ax.set_ylabel(r"Fractional squared amplitude ($C^2/D^2)", fontsize=26)
#ax.set_title("Pulse amplitude distribution along a 30 strip-line pickup board", fontsize=26)
ax.get_yaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(length=10, width=2, which='minor')
major_ticks = [dists[i] for i in range(len(dists)) if (i % 2) == 1]
ax.set_xticks(major_ticks)
ax.set_xlim([min(dists), max(dists)])
minorLocator = MultipleLocator(0.3)
ax.xaxis.set_minor_locator(minorLocator)
ax.get_yaxis().set_ticks(np.arange(-50, 250, 25))
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.75)
ax.legend(fontsize=25, loc="center right")
ax.set_xlim([-2, 6])
#plt.savefig("strip_distribution_nim.png", bbox_inches='tight')
plt.show()
'''
sys.exit()



#plot rise time on top of eachother

nicr = DataSet.DataSet(nicr_filenameArrayClean[0])
direct = DataSet.DataSet(direct_filenameArrayCleanTake2[0])
nicr_riseTimeArray = nicr.getMeasRiseTimes()
nicr_riseTimeArray = [i*nicr.timestep*1000 for i in nicr_riseTimeArray]
direct_riseTimeArray = direct.getMeasRiseTimes()
direct_riseTimeArray = [i*direct.timestep*1000 for i in direct_riseTimeArray]

print "Nicr Average Rise Time:", round(np.average(nicr_riseTimeArray),3), "ns"
print "Nicr Standard Dev:", round(np.std(nicr_riseTimeArray), 3), "ns"
print "Direct Average Rise Time:", round(np.average(direct_riseTimeArray),3), "ns"
print "Direct Standard Dev:", round(np.std(direct_riseTimeArray), 3), "ns"


fig, ax = plt.subplots(figsize=(15, 10))
#fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(nicr_riseTimeArray, bins = 26, color='b', alpha=0.7, label="NiCr capacitive coupling")
ax.hist(direct_riseTimeArray, bins = 8, color='r', alpha=0.9, label="Direct")
ax.set_xlabel("Rise Time (ps)", fontsize = 25)
#ax.set_title("Hisotgram of rise times, Direct vs. Capacitive couping", fontsize=26)
ax.set_ylabel("Counts", fontsize = 25)
ax.legend(loc='center right', fontsize=25)
majorLocator = MultipleLocator(50)
minorLocator = MultipleLocator(10)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.75)
ax.get_xaxis().set_major_locator(majorLocator)
ax.get_xaxis().set_minor_locator(minorLocator)
#ax.get_yaxis().set_minor_locator(minorLocator)
ax.get_xaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_xaxis().set_tick_params(length=10, width=2, which='minor')
ax.get_yaxis().set_tick_params(labelsize=26, length=20, width=2, which='major')
ax.get_yaxis().set_tick_params(length=10, width=2, which='minor')
ax.set_xlim([350, 750])
#plt.show()
plt.savefig("rise_time_histo_nim_bigbins.png", bbox_inches='tight')



sys.exit()



#average powers for each channel
strip_powtot = []
nicr_powtot = []
for i in range(len(nicr_filenameArrayClean)):
	data = DataSet.DataSet(nicr_filenameArrayClean[i])
	print "loaded filename: " + nicr_filenameArrayClean[i]
	avpows = data.getAvPower()
	for x in avpows:
		nicr_powtot.append(x)


for i in range(len(direct_filenameArrayCleanTake2)):
	data = DataSet.DataSet(direct_filenameArrayCleanTake2[i])
	print "loaded filename: " + direct_filenameArrayCleanTake2[i]
	avpows = data.getAvPower()
	print "##### Averaged Amplitudes #####"
	for x in avpows:
		strip_powtot.append(x)



print "Direct vector length of amplitude: " + str(sum(strip_powtot))
print "Nicr vector length of total amplitude: " + str(sum(nicr_powtot))
print "Fraction: " + str(sum(nicr_powtot)/sum(strip_powtot))

sys.exit()





#total amplitude summed
strip_amptot = 0
nicr_amptot = 0
for i in range(len(nicr_filenameArrayClean)):
	data = DataSet.DataSet(nicr_filenameArrayClean[i])
	print "loaded filename: " + nicr_filenameArrayClean[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		nicr_amptot += x[0]


for i in range(len(direct_filenameArrayCleanTake2)):
	data = DataSet.DataSet(direct_filenameArrayCleanTake2[i])
	print "loaded filename: " + direct_filenameArrayCleanTake2[i]
	avamps = data.getAvAmplitudes()
	print "##### Average Amplitudes #####"
	for x in avamps:
		strip_amptot += x[0]


print "Direct total amplitude: " + str(strip_amptot)
print "Nicr total amplitude: " + str(nicr_amptot)
print "Fraction: " + str(nicr_amptot/strip_amptot)





#do a systematics analysis
group = Group.Group(direct_filenameArraySystematics)
sysAmpErrFrac = group.getSystematicAmplitudeError()
print "\n##############################\n"
sysRtErrFrac = group.getSystematicRiseTimeError()
print "\n##############################\n"

group = Group.Group(nicr_filenameArraySystematics)
sysAmpErrFrac = group.getSystematicAmplitudeError()
print "\n##############################\n"
sysRtErrFrac = group.getSystematicRiseTimeError()
print "\n##############################\n"



sys.exit()





#!/usr/bin/env python
# coding: utf-8

# In[235]:


import matplotlib.pyplot as plt
import math

def con(momentum):
    # Constants
    c = 1 #as energy is in GeV
    beam_mass = 0.139570  # pion particle mass in GeV/c^2
    target_mass = 0.938270  # proton mass in GeV/c^2
    #Since the target particle is at rest, the momentum is taken to be zero
    target_energy =  (target_mass * c**2)

    # Total energy of the system, sum of both particles energies
    return math.sqrt((momentum**2)* (c ** 2) + (beam_mass*c**2)**2) + target_energy

"""
def convert_momentum_to_energy(momentum):
    # Constants
    c = 299792458  # Speed of light in meters per second
    beam_mass = 0.139570  # Beam particle mass in AMU
    target_mass = 0.938270  # Target particle mass in AMU
    # Convert masses to energy units (GeV)
    beam_energy = math.sqrt((beam_mass ** 2) * (c ** 4) + (beam_mass * c) ** 2)
    target_energy = math.sqrt((target_mass ** 2) * (c ** 4) + (target_mass * c) ** 2)

    # Calculate the total energy of the particle
    total_energy = math.sqrt((momentum ** 2) * (c ** 2) + (beam_mass * c) ** 2) + target_energy

    return total_energy
"""

print("Energy:", "GeV")


# In[262]:


import pandas as pd
import numpy as np


print("Information about the system")
print("----------------------------")
reaction = "π⁻ + p → p + π⁻"
print("Reaction:", reaction)
print("Total Elastic Cross-Section")
print("Beam Mass:", 0.139570)
print("Target Mass:", 0.938270 )
print("Threshold:", 0)
print("Final State Multiplicity:", 2)
print("Number of Data Points:", 277)

file_path = 'https://pdg.lbl.gov/2023/hadronic-xsections/rpp2022-pimp_elastic.dat'
data = pd.read_fwf(file_path, skiprows=10)
#colnames=['POINT_NUMBER',' PLAB(GEV/C)',' PLAB_MIN ','PLAB_MAX SIG(MB)',' STA_ERR+',' STA_ERR-',' SY_ER+(PCT)',' SY_ER-(PCT)',' REFERENCE','FLAGFORMAT(I5,1X,4F11.5,2F8.4,1X,2F6.1,A)']
#df = pd.read_fwf(file_path, names=colnames)

data.columns=['POINT_NUMBER',
                   'PLAB(GEV/C)',
                   'PLAB_MIN',
                   'PLAB_MAX',
                   'SIG(MB)',
                   'STA_ERR+',
                   'STA_ERR-',
                   'SY_ER+(PCT)',
                   'SY_ER-(PCT)',
                   'REFERENCE',
                   'FLAGFORMAT(I5,1X,4F11.5,2F8.4,1X,2F6.1,A)'];

df = pd.DataFrame(data)

momentum=df["PLAB(GEV/C)"]
cs=df["SIG(MB)"]
en=np.array( [ con(x) for x in momentum ] )
sta_err_plus = np.array(df["STA_ERR+"])
sta_err_minus = np.array(df["STA_ERR-"])

plt.rcParams['axes.facecolor'] = 'white'

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

font1 = {'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 11,
        }
plt.plot(en, df["SIG(MB)"],'--', markersize=0.4, label='Energy vs Cross-section', linewidth=0.7)
plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.6)
plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.4)
plt.xlabel("Energy (GeV)",fontdict=font1)
plt.ylabel("Cross-Section (MB)", fontdict=font1)
plt.legend()
plt.grid(True)
plt.title("Energy vs Cross-section", fontdict=font)
plt.show()

def plotoerr(lim):
 plt.figure()
 plt.plot(en, df["SIG(MB)"], '--', linewidth='0.7')
 plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
 plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.2, label="error bar")   
 plt.xlabel("Energy (GeV)",fontdict=font1)
 plt.ylabel("Cross-Section (MB)", fontdict=font1)
 plt.legend()
 plt.grid(True)
 plt.ylim([0,30])
 #plt.xlim([min(en),lim])
 plt.xlim([-1,lim])
 plt.show()


def ploto(lim):
 plt.figure()
 plt.plot(en, df["SIG(MB)"], '--', linewidth='0.7')
 plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
 #plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.2, label="error bar")   
 plt.xlabel("Energy (GeV)",fontdict=font1)
 plt.ylabel("Cross-Section (MB)", fontdict=font1)
 plt.legend()
 plt.grid(True)
 plt.ylim([0,30])
 plt.xlim([-1,lim])

 #plt.xlim([min(en),lim])
 plt.show()

def plotoerrbar(lim):
 plt.figure()
 #plt.plot(en, df["SIG(MB)"], 'r--', linewidth='0.4')
 plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
 #plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.2, label="error bar")
 plt.errorbar(en, cs,yerr=[sta_err_minus, sta_err_plus], fmt='o', label='Error-Bar', markersize='0.1', ecolor='red',capsize=0.4,linewidth=0.5)
 plt.xlabel("Energy (GeV)",fontdict=font1)
 plt.ylabel("Cross-Section (MB)", fontdict=font1)
 plt.legend()
 plt.grid(True)
 plt.ylim([0,30])
 plt.xlim([-1,lim])

    #plt.xlim([min(en),lim])
 plt.show()


ploto(50)
plotoerrbar(50)
plotoerr(50)

ploto(30)
plotoerrbar(30)
plotoerr(30)

ploto(20)
plotoerrbar(20)
plotoerr(20)

ploto(10)
plotoerrbar(10)
plotoerr(10)



#ploto(min(en)+9e8)
#plotoerrbar(min(en)+9e8)
#plotoerr(min(en)+9e8)




"""
plt.plot(en, df["SIG(MB)"], '--', linewidth='0.7')
plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.2, label="error bar")
plt.errorbar(en, cs,yerr=[sta_err_minus, sta_err_plus], fmt='o', label='Cross-section', markersize='0.1', ecolor='red',capsize=0.4,linewidth=0.5)
plt.legend()
plt.grid(True)
plt.ylim([0,30])
plt.xlim([min(en),min(en)+3e8])
plt.show()



plt.plot(en, df["SIG(MB)"], '--', markersize=1, label='Energy vs Cross-section', linewidth=0.7)
plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.4)
plt.legend()
plt.grid(True)
plt.ylim([0,30])
plt.xlim([min(en),min(en)+9e8])
plt.show()

plt.plot(en, df["SIG(MB)"], '--', markersize=1, label='Energy vs Cross-section',linewidth=0.7)
plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.4)
plt.legend()
plt.grid(True)
plt.ylim([0,30])
plt.xlim([min(en),min(en)+3e9])
plt.show()





#cross_section = data['PLAB(GEV/C)']
#energy = data['E']

#print(cross_section)
#print(energy)
"""


# In[276]:


import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

#a function to identify peaks

def peaks(lim):
 # Plotting the energy versus cross-section data
 plt.plot(en, df["SIG(MB)"], '--', linewidth='0.7')
 plt.plot(en, df["SIG(MB)"], 'ro', markersize=0.5, label='Energy vs Cross-section')
 plt.fill_between(en, cs - sta_err_minus, cs + sta_err_plus, color='gray', alpha=0.2, label="error bar")   

 # perform peak finding using scipy's signal
 
 peaks, _ = find_peaks(df["SIG(MB)"])
 print("Possible resonances at Energy (GeV) values:",en[peaks])
 print("Cross-section (MB) values:", cs[peaks])
 
 # Plot the identified peaks
 plt.plot(en[peaks], df["SIG(MB)"][peaks], 'x', color='green', markersize=3, label='Peaks')

 plt.legend()
 plt.grid(True)
 plt.ylim([0, 30])
 plt.xlim([0,lim])
 plt.show()



peaks(50)
peaks(30)
peaks(10)

peaks, properties = find_peaks(df["SIG(MB)"])



results_half = peak_widths(cs,peaks, rel_height=0.5)
results_full = peak_widths(cs, peaks, rel_height=1)

print(results_half[0])

plt.plot(cs,"--",linewidth=1)
plt.plot(peaks, cs[peaks], "rx",markersize=1)
plt.hlines(*results_half[1:], color="green",linewidth=0.5,label="Width of Peaks")
#plt.hlines(*results_full[1:], color="red")
plt.legend()
plt.grid(True)
plt.show()

#According to scipy's find_peaks, peaks are directly found from the single dataset of the signal, but
#we need peaks of Cross-section across our Energy values, so we perform interpolation using scipy's interpolate

from scipy.signal import find_peaks, peak_widths,peak_prominences
from scipy.interpolate import interp1d

def index_to_xdata(xdata, indices):
     #interpolate the values from signal.peak_widths to xdata
         ind = np.arange(len(xdata))
         f = interp1d(ind,xdata)
         return f(indices)

#redefined peak-finding, here we can parametrise our peaks so that the algorithm finds peaks only of above a 
#certain threshold of prominence and height 

def repeaks(minx,maxx,h,prom):
 peaks, _ = find_peaks(cs,height=h,prominence=prom)
 widths, width_heights, left_ips, right_ips = peak_widths(cs, peaks)
 widths = index_to_xdata(en, widths)
 left_ips = index_to_xdata(en, left_ips)
 right_ips = index_to_xdata(en, right_ips)
 print("---------------------------------")
 print("Energy band:(",minx,",",maxx,")")   
 print("Prominence:", prom)
 print("Height:",h)   
 print("Peaks detected at indices:",peaks)
 print("Energies (in GeV) corresponding to detected peaks:", en[peaks])
 print("Mass of detected resonances (in GeV/c^2):",en[peaks])   
 print("Mass of detected resonances (in MeV/c^2):",en[peaks]*1000)   
 print("Width of detected peaks (in GeV):",widths)
 plt.plot(en,cs,"--",linewidth=1)
 plt.plot(en[peaks], cs[peaks], "rx",markersize=5)
 plt.hlines(width_heights, left_ips, right_ips, color='green')
 plt.xlabel('Energy (GeV)')
 plt.ylabel('Cross-Section (MB)')
 plt.xlim(minx,maxx)
 plt.grid(True)
 plt.show()
  

repeaks(0,30,1,10)
repeaks(0,30,1,4)
repeaks(0,1.5,1,10)
repeaks(1,1.5,1,4)


# In[ ]:





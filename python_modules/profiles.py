# ----------------------------------------------------------------------------------------------------------------
# This program provides an interface to relaxation parameterization routines logmoments(), lmfit_kohlrausch(), and
# lmfit_exponential(). Input relaxation functions must be in a tab separated text file, in XY columns (same length, 
# no headers).
# ----------------------------------------------------------------------------------------------------------------
# Parameters: datadir: text
#                 Data directory, e.g. "C:/Documents and Settings/NMR/"
#             datafile: text
#                 Text (ASCII) file containing relaxation data, e.g. "errano_adipose.txt"
#             method: text 
#                 Parameterization method:
#                 -- 'stretched': the stretched exponential fit; 
#                 -- 'monoexp': mono-exponential fit;
#                 -- 'biexp': double-exponential fit;
#                 -- 'logmoments': logarithmic moment analysis.      
#             plotfit_toggle: Boolean
#                 Plot best-fit curves with original data (default: True)
#             reusable_toggle: Boolean
#                 Cash and retrieve best-fit parameters as initial for the next fit (default: False)
#             extrapol_toggle: Boolean
#                 Enable data extrapolation toward zero time; relevant to the 'logmoments' method (default: True)
#             verbose_toggle: Boolean
#                 Report best-fit parameters and several statistics from the minimizer being used.
#             rcParams: dictionary
#                 These control the appearance of two subplots: one for original relaxation functions and one for relaxation parameters. They are less frequently modified.
#
# Output: The profile.py outputs the sought-after relaxation parameters in three complementary ways. One way is a console output; 
#         the other is plotting the parameter(s) against experiment's index; and the third way is writing them to a file in the 
#         data directory (the filename is profile_output.txt).
#
# v1.0 August 2017
# Author: Oleg V. Petrov, TU Ilmenau
# E-mail: oleg.petrov@tu-ilmenau.de

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import logarithmic_moment_analysis as lma
import lmfit_exponential
import lmfit_kohlrausch
from math import gamma

def main():	
   # __________________ Adjustable parameters: __________________ 
    datadir =  "C:/Documents and Settings/Oleg Petrov/My Documents/Projects/Graviera/"	# data directory
    datafile = "MegaT1.txt"		# data file in ASCII format (.txt file)
				
   # parameterization options:
    method = 'logmoments'			# 'monoexp': mono-exponential fit; 'stretched': stretched-exponential fit; 'logmoments': logarithmic moment analysis
    plotfit_toggle  = True			# show best-fit curves along with relaxation functions
    reusable_toggle = False			# use best-fit parameters as initial values in the next fiting session
    extrapol_toggle = True			# use best-fit parameters to extrapolate relaxation functions toward zero time
    verbose_toggle  = True			# print whatever a fitting routine reports

   # plotting parameters (the following are less frequently modified):
    default_rcParams = {'axes.grid': True,
                        'axes.formatter.limits': [-2, 2],
                        'figure.subplot.hspace': 0.3, 
                        'figure.subplot.left': 0.13, 
                        'figure.subplot.right': 0.89,
                        'font.family': 'sans-serif', 
                        'font.size': 12, 
                        'font.weight':'normal',
                        'grid.alpha': 0.8,
                        'grid.color': 'k',
                        'legend.fontsize': 'large',
                        'legend.frameon': True,
                        'legend.loc': 'upper right',
                        'lines.color': 'b',
                        'lines.linestyle': 'None',
                        'lines.linewidth': 1.0,
                        'lines.marker': 'o',
                        'lines.markersize': 6,
                        'xtick.color': 'k',
                        'xtick.labelsize': 'medium',
                        'ytick.color': 'k',
                        'ytick.labelsize': 'medium'}

    ax1_rcParams = default_rcParams.copy()
    ax1_rcParams.update({})

    ax2_rcParams = default_rcParams.copy()
    ax2_rcParams.update({})

    ax3_rcParams = default_rcParams.copy()
    ax3_rcParams.update({'axes.grid': False, 
                        'axes.color_cycle': 'r',
                        'axes.labelcolor': 'r',
                        'lines.linestyle': '-',
                        'lines.marker': '^',
                        'ytick.color':'r'})
    mpl.rc('axes', grid = True)
   # ____________ end of adjustable parameters ____________

    with plt.rc_context(default_rcParams):
        fig = plt.figure()
        fig.canvas._master.geometry("512x704-0+0")

    with plt.rc_context(ax1_rcParams): 
        ax1 = fig.add_subplot(2, 1, 1) 	# ax1 will show relaxation functions and best-fit curves
        ax1.set_title('Relaxation functions')    
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('Magnetization, arb. u.')
        ax1.plot([], [], clip_on=False)

    with plt.rc_context(ax2_rcParams):
        ax2 = fig.add_subplot(2, 1, 2) 	# ax2 and ax3 will show relaxation parameters
        ax2.set_title('Relaxation parameter(s)') 
        ax2.set_xlabel('Experiment #')
        ax2.set_ylabel('T1, s')
        ax2.plot([], [], clip_on=False)
    
    if method is not 'monoexp':
        with plt.rc_context(ax3_rcParams):
            ax3 = ax2.twinx()
            if method == 'stretched':
                ax2.set_ylabel('Arithmetic mean T1, s')
                ax3.set_ylabel('Beta')
            elif method == 'logmoments':
                ax2.set_ylabel('Geometric mean T1, s')
                ax3.set_ylabel('Geometric SD')
            else:
                raise TypeError 
            ax3.plot([], [], clip_on=False)
            for yticklabels in ax3.get_yticklabels():
                yticklabels.set_rotation(45)
        ax2.spines['right'].set_edgecolor(ax3_rcParams['ytick.color'])

    plt.show(block=False)
    ax1.set_xscale("log")
    ax2.set_yscale("log")

    for ax in fig.axes:
        ax.xaxis._autolabelpos = False
        ax.yaxis._autolabelpos = False
        ax.margins(0.02, 0.02)

   # _______________ Start parameterization... _______________
    profiles = []
   # read specific columns from file to 2d arrays:
    with open(datadir + datafile) as f:
       # get number of columns:
        ncols = len(f.next().split())
        for j in range(ncols/2):
            f.seek(0)
            magnetization = np.loadtxt(f, usecols=(2*j,2*j+1))
            magnetization = magnetization[magnetization[:,0].argsort()]  

           # ________________ Parameterization: _________________
            magnetization_fit = magnetization.copy()
            accus = 5
              
            if method == 'monoexp':
                model = lmfit_exponential.minimize(magnetization[:,0], magnetization[:,1], options={'order':1, 'reusable':reusable_toggle, 'verbose':verbose_toggle})
                magnetization_fit[:,1] = magnetization[:,1] + model.residual
               # update profiles:
                profiles.append([j+1, 1./model.params['r1']])

            elif method == 'stretched':    
                meanT1 = 0.
                beta = 0.
                for i in range(accus):
                    model = lmfit_kohlrausch.minimize(magnetization[:,0], magnetization[:,1], options={'reusable':reusable_toggle})
                    beta += model.params['beta']
                    meanT1 += (gamma(1./model.params['beta']) / model.params['r']) / model.params['beta']
                magnetization_fit[:,1] = magnetization[:,1] + model.residual
                meanT1 /= accus
                beta   /= accus
               # update profiles:
                profiles.append([j+1, meanT1, beta])

            elif method == 'logmoments':
                logm1 = 0.
                logm2 = 0.
                logm3 = 0.
                for i in range(accus):
                    logms, y_fit = lma.logmoments(magnetization[:,0], magnetization[:,1], extrapol=extrapol_toggle, fit_options={'order':3, 'reusable':reusable_toggle, 'verbose':verbose_toggle})
                    logm1 += logms[0]
                    logm2 += logms[1]
                    logm3 += logms[2]
                logm1 /= accus
                logm2 /= accus
                logm3 /= accus
                magnetization_fit[:,1] = y_fit
               # update profiles:   
                profiles.append([j+1, np.exp(logm1), np.exp(logm2**0.5), logm3/logm2**1.5])

            else:
                raise TypeError      
 
           # update subplots:
            with plt.rc_context(ax1_rcParams): 
                ax1.plot(magnetization[:,0], magnetization[:,1])
                if plotfit_toggle:
                    the_color = ax1.lines[-1].get_color()
                    ax1.plot(magnetization_fit[:,0], magnetization_fit[:,1], '-', color=the_color)
            ax2.lines[0].set_xdata([item[0] for item in profiles])   
            ax2.lines[0].set_ydata([item[1] for item in profiles])
            if 'ax3' in locals():
                ax3.lines[0].set_xdata([item[0] for item in profiles])     
                ax3.lines[0].set_ydata([item[2] for item in profiles])
            for ax in fig.axes: 
                ax.relim()
            for ax in fig.axes: 
                ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

       # save the figure as png:
        for ax in fig.axes:
            ax.xaxis._autolabelpos = True
            ax.yaxis._autolabelpos = True
        if 'ax3' in locals():
            ax3.yaxis._autolabelpos = True
        # fig.savefig('test.png', format = 'png', dpi=175)

       # write profile(s) in a text file:
        if method == 'monoexp': the_header = ' Exp. #	T1(s)'
        if method == 'stretched': the_header = ' Exp. #	Mean T1(s)	Beta'
        if method == 'logmoments': the_header = ' Exp. #	Mean T1(s)	SD	Skewness'
        np.savetxt(datadir+'profiles_output.txt', profiles, fmt='%.4e', delimiter = '\t', header = the_header)	
		
if __name__ == "__main__":
    main()
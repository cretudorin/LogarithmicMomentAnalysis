# ----------------------------------------------------------------------------------------------------------------
# This program evaluates data stored a Stelar Data File (a sdf-file) into magnetization relaxation functions and 
# then calls one of the parameterization routines: logmoments(), lmfit_kohlrausch(), or lmfit_exponential().
# ----------------------------------------------------------------------------------------------------------------
# In particular, the following steps are executed:
# 1) open the sdf-file for reading;
# 2) read the file line by line looking for the parameters ZONE, T1MX, BRLX, BS, NBLK, BGRD, BINI, and BEND; 
# 3) once a DATA block is encountered, read the data to a 2-d array of Re& Im values;
# 4) quantify magnitization using time-domain data (FID's);
# 5) reconstruct a time scale from the read parameters and stacks it against the quantified magnetization.
# 6) proceed with parameterization of thus-built relaxation functions, exercising one of the three options:  
#   -- mono-exponential fit to the magnetization relaxation curve, which yields one output parameter (T1);
#   -- stretched-exponential fit, which yields two output parameters (arithmetic mean T1 and stretching exponent beta); 
#   -- logarithmic moment analysis, which gives up to three parameters (geometric mean T1, geometric standard deviation, 
#      and a skewness parameter).
# 7) plot the magnetization decays and the sough-after parameters against the relaxation field (a.k.a. relaxation profiles). 
# 8) write the results in a text file profile_output.txt.
#
# v1.0 August 2017
# Author: Oleg V. Petrov, TU Ilmenau
# E-mail: oleg.petrov@tu-ilmenau.de
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import logarithmic_moment_analysis as lma
import lmfit_exponential
import lmfit_kohlrausch
import re
from math import gamma
import warnings

warnings.filterwarnings("always")

datafile = "land_SM_dips.sdf"  # Stelar's data file (sdf)
datadir = "input"  # data directory DO NOT CHANGE


def export(save_path, profiles, the_header, fig):
    input_basename = datafile.split('.')[0]
    fig.savefig(os.path.join(save_path, '{}_out.png'.format(input_basename)), format='png', dpi=175)
    np.savetxt(
        os.path.join(save_path, '{}_out.txt'.format(input_basename)),
        profiles,
        fmt='%.4e',
        delimiter='\t',
        header=the_header
    )


def main():
    # __________________ Adjustable parameters: __________________

    [zone_from, zone_to] = [0, 999]  # zones of interest ([1, 999]: full scope)
    [fid_from, fid_to] = [0, 20]  # FID integration limits, in points

    # parameterization options:
    method = 'stretched'  # 'monoexp': mono-exponential fit; 'stretched': stretched-exponential fit; 'logmoments': logarithmic moment analysis
    plotfit_toggle = True  # show best-fit curves along with relaxation functions
    reusable_toggle = False  # use best-fit parameters as initial values in the next fiting session
    extrapol_toggle = True  # use best-fit parameters to extrapolate relaxation functions toward zero time
    verbose_toggle = True  # print whatever a fitting routine reports
    accus = 2

    # plotting parameters (the following are less frequently modified):
    default_rcParams = {'axes.grid': True,
                        'axes.formatter.limits': [-2, 2],
                        'figure.subplot.hspace': 0.3,
                        'figure.subplot.left': 0.13,
                        'figure.subplot.right': 0.89,
                        'font.family': 'sans-serif',
                        'font.size': 12,
                        'font.weight': 'normal',
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
                         'ytick.color': 'r'})
    mpl.rc('axes', grid=True)
    # ____________ end of adjustable parameters ____________

    with plt.rc_context(default_rcParams):
        fig = plt.figure()
        fig.canvas._master.geometry("512x704-0+0")

    with plt.rc_context(ax1_rcParams):
        ax1 = fig.add_subplot(2, 1, 1)  # ax1 will show relaxation functions and best-fit curves
        ax1.set_title('Relaxation functions')
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('Magnetization, arb. u.')
        ax1.plot([], [], clip_on=False)

    with plt.rc_context(ax2_rcParams):
        ax2 = fig.add_subplot(2, 1, 2)  # ax2 and ax3 will show relaxation parameters
        ax2.set_title('Relaxation parameter(s)')
        ax2.set_xlabel('Relaxation field, MHz')
        ax2.set_ylabel('T1, s')
        ax2.plot([], [], clip_on=False)

    if (method is not 'monoexp') and (method is not 'biexp'):

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
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    if 'ax3' in locals():
        ax3.set_xscale("log")

    for ax in fig.axes:
        ax.xaxis._autolabelpos = False
        ax.yaxis._autolabelpos = False
        ax.margins(0.02, 0.02)

    # _______________ Start magnetization evaluation... _______________
    profiles = []
    # read the sdf file in lines:
    with open(os.path.join(datadir, datafile)) as sdf:
        for line in sdf:
            if "ZONE" in line:
                zone = int(re.search(r'\d+', line).group())
                if zone > zone_to: break
            elif "T1MX=" in line:
                try:
                    t1mx = float(re.search(r'\d+\.\d+', line).group())
                except:
                    t1mx = float(re.search(r'\s\d+', line).group())
            elif "BRLX" in line:
                brlx = float(re.search(r'\d+\.\d+', line).group())
            elif "BS" in line:
                bs = int(re.search(r'\d+', line).group())
            elif "NBLK" in line:
                nblk = int(re.search(r'\d+', line).group())
            elif "BGRD" in line:
                bgrd = line.split('=\t')[1]
                bgrd = bgrd.strip()
            elif "BINI" in line:
                try:
                    bini = float(re.search(r'\d+\.\d+', line).group())
                except:
                    bini = float(re.search(r'\d+', line).group())
                if re.search('T1MX', line):
                    bini *= t1mx
            elif "BEND" in line:
                try:
                    bend = float(line.split('=\t')[1])
                except:
                    bend = float(re.search(r'\d+\.\d+', line).group())
                # bend = float(re.search(r'\d+', line).group())
                if re.search('T1MX', line):
                    bend *= t1mx
                if bgrd == 'LOG':
                    tau = np.logspace(np.log10(bini), np.log10(bend), nblk)
                    # print tau
                else:
                    tau = np.linspace(bini, bend, nblk)
            elif "DATA" in line:
                if zone < zone_from: continue
                data = np.zeros((nblk * bs, 2))
                for i in range(nblk * bs):
                    line = sdf.next()
                    data[i, 0] = line.split()[0]
                    data[i, 1] = line.split()[1]
                fids = data[:, 0] + 1j * data[:, 1]
                fids = fids.reshape((nblk, bs))

                # __________________ Quantitation: ___________________
                magnetization = np.zeros((nblk, 2))
                magnetization[:, 0] = tau
                magnetization[:, 1] = abs(fids[:, fid_from:fid_to]).sum(axis=1)
                magnetization = magnetization[magnetization[:, 0].argsort()]

                # ________________ Parameterization: _________________
                magnetization_fit = magnetization.copy()

                if method == 'monoexp':
                    model = lmfit_exponential.minimize(magnetization[:, 0], magnetization[:, 1],
                                                       options={'order': 1, 'reusable': reusable_toggle,
                                                                'verbose': verbose_toggle})
                    magnetization_fit[:, 1] = magnetization[:, 1] + model.residual

                    # update profiles:
                    profiles.append([brlx, 1. / model.params['r1']])

                elif method == 'biexp':
                    model = lmfit_exponential.minimize(magnetization[:, 0], magnetization[:, 1],
                                                       options={'order': 2, 'reusable': reusable_toggle,
                                                                'verbose': verbose_toggle})
                    magnetization_fit[:, 1] = magnetization[:, 1] + model.residual

                    # update profiles:
                    profiles.append([brlx, 1. / model.params['r1'], 1. / model.params['r2'], model.params['a1'],
                                     model.params['a2']])

                elif method == 'stretched':
                    meanT1 = 0.
                    beta = 0.
                    for i in range(accus):
                        model = lmfit_kohlrausch.minimize(magnetization[:, 0], magnetization[:, 1],
                                                          options={'reusable': reusable_toggle})
                        beta += model.params['beta']
                        meanT1 += (gamma(1. / model.params['beta']) / model.params['r']) / model.params['beta']
                    magnetization_fit[:, 1] = magnetization[:, 1] + model.residual
                    meanT1 /= accus
                    beta /= accus
                    # update profiles:
                    profiles.append([brlx, meanT1, beta])

                elif method == 'logmoments':
                    logm1 = 0.
                    logm2 = 0.
                    logm3 = 0.
                    for i in range(accus):
                        logms, y_fit = lma.logmoments(magnetization[:, 0], magnetization[:, 1],
                                                      extrapol=extrapol_toggle,
                                                      fit_options={'order': 3, 'reusable': reusable_toggle,
                                                                   'verbose': verbose_toggle})
                        logm1 += logms[0]
                        logm2 += logms[1]
                        logm3 += logms[2]
                    logm1 /= accus
                    logm2 /= accus
                    logm3 /= accus
                    magnetization_fit[:, 1] = y_fit
                    # update profiles:
                    profiles.append([brlx, np.exp(logm1), np.exp(logm2 ** 0.5), logm3 / logm2 ** 1.5])
                else:
                    raise TypeError

                    # update subplots:
                with plt.rc_context(ax1_rcParams):
                    ax1.plot(magnetization[:, 0], magnetization[:, 1])
                    if plotfit_toggle:
                        the_color = ax1.lines[-1].get_color()
                        ax1.plot(magnetization_fit[:, 0], magnetization_fit[:, 1], '-', color=the_color)
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

        # write profile(s) in a text file:
        if method == 'monoexp': the_header = ' Field(MHz)	T1(s)'
        if method == 'biexp': the_header = ' Field(MHz)	T11(s)	T11(s)	A1	A2'
        if method == 'stretched': the_header = ' Field(MHz)	Mean T1(s)	Beta'
        if method == 'logmoments': the_header = ' Field(MHz)	Mean T1(s)	SD	Skewness'

        try:
            save_dir = os.path.join('./out')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            export(save_dir, profiles, the_header, fig)
        except IOError:
            save_dir = os.path.join(datadir, 'out/')
            export(save_dir, profiles, the_header, fig)


if __name__ == "__main__":
    main()

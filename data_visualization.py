# This file contains data visualization functions for the karmouche24esd_AtlanticPacific_Pacemaker package.
# These are called in jupyter notebooks to perform various plotting routines. 
# The jupyter notebooks are used to generate the figures and tables in the paper: 
# Karmouche, S., Galytska, E., Meehl, G. A., Runge, J., Weigel, K., and Eyring, V.: 
# Changing effects of external forcing on Atlantic-Pacific interactions, EGUsphere [preprint], 
# https://doi.org/10.5194/egusphere-2023-1861, 2023.
# Author: Soufiane Karmouche

import os
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings while loading
import numpy as np
import xarray as xr
import seaborn as sns
from matplotlib import pyplot as plt

class DataVisualization:

    @staticmethod
    def plot_and_save_obs_dataframes(tig_data, tig_data_iso, N, var_names, yearsarrobs, FIGS_DIR, DF_DIR):
        # For plotting
        num_shades = 10
        light_green_palette = sns.light_palette("green", n_colors=num_shades)

        line1_color = "black"
        line2_color = light_green_palette[6]
        line3_color = "red"
        for i in range(N):
            fig, ax = plt.subplots(figsize=(18, 6))
            x1 = range(len(tig_data[:, i]))

            # Plot HadISST
            sns.lineplot(
                x=x1, y=tig_data[:, i], ax=ax, linewidth=3, alpha=0.9, color=line1_color
            )

            # Plot HadISST minus MEM
            sns.lineplot(
                x=x1, y=tig_data_iso[:, i], ax=ax, linewidth=6, alpha=.75, color=line2_color
            )

            # Plot difference represented by MEM
            diff = np.subtract(tig_data[:, i].data, tig_data_iso[:, i].data)
            sns.lineplot(x=x1, y=diff, ax=ax, linewidth=3, color=line3_color)
            squad = yearsarrobs
            ax.set_xticks(x1[1::40])
            ax.set_xticklabels(squad[1::40], size=18)
            ax.set_title(var_names[i], size=30)
            ax.tick_params(axis="both", which="major", labelsize=18)
            plt.savefig(FIGS_DIR+"plot_OBS_{}.png".format(var_names[i]))
            plt.show()
            plt.close()

        # Save the data
        np.save(file=DF_DIR+"obs_tig_data.npy", arr=tig_data)
        np.save(file=DF_DIR+"obs_tig_data_iso.npy", arr=tig_data_iso)

    @staticmethod
    def plot_obs_amv_pdv(amv_lp_short, pdv_lp_short, yearsarrobs, FIGS_DIR):

        # Set Seaborn style
        sns.set(style="whitegrid", font_scale=1.2)

        fig, ax = plt.subplots(figsize=(15, 5))
        sns.set_style("whitegrid")
        # Calculate low-pass filtered AMV and PDV
        plt.plot(pdv_lp_short, "k--", linewidth=3, label="PDV")
        plt.plot(amv_lp_short, "k", linewidth=3.5, label="AMV")

        plt.fill_between(
            range(len(pdv_lp_short)),
            pdv_lp_short,
            0.0,
            where=(pdv_lp_short > 0),
            alpha=0.8,
            color="pink",
            interpolate=True,
        )
        plt.fill_between(
            range(len(pdv_lp_short)),
            pdv_lp_short,
            0.0,
            where=(pdv_lp_short < 0),
            alpha=0.8,
            color="lightblue",
            interpolate=True,
        )
        plt.fill_between(
            range(len(amv_lp_short)),
            amv_lp_short,
            0.0,
            where=(amv_lp_short > 0),
            alpha=0.6,
            color="red",
            interpolate=True,
        )
        plt.fill_between(
            range(len(amv_lp_short)),
            amv_lp_short,
            0.0,
            where=(amv_lp_short < 0),
            alpha=0.6,
            color="blue",
            interpolate=True,
        )

        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlim(0, len(pdv_lp_short))
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Temperature Anomaly (degC)", fontsize=16)
        ax.set_title("10-year low-pass filtered PDV and AMV (HadISST)", fontsize=18)
        x1 = range(len(yearsarrobs))
        squad = yearsarrobs
        plt.xticks(x1[::120], squad[::120])
        plt.legend(fontsize=14)
        plt.show()
        plt.savefig(FIGS_DIR + "low_pass_filtered_PDV_and_AMV_HadISST.png")
        plt.close()

    @staticmethod
    def plot_amv_and_pdv_timeseries(amv_obs_dict_veryraw, amv_obs_dict_iso, amv_EM_ATL_veryraw, amv_EM_ATL_iso,
                                    pdv_obs_dict_veryraw, pdv_obs_dict_iso, pdv_EM_ATL_veryraw, pdv_EM_ATL_iso,
                                    yearsarrobs, utils):
        # construct 3 days and 10 days low pass filters
        window = 50

        hfw = utils.low_pass_weights(window, 1. / 85.)
        weight_high = xr.DataArray(hfw, dims=['window'])

        sns.set(style="darkgrid", font_scale=1.2)
        x1 = range(len(yearsarrobs))

        num_shades = 10

        light_green_palette = sns.light_palette("green", n_colors=num_shades)
        # Mean of all ensemble members for amv_EM_ATL
        mean_amv = np.nanmean([(amv_EM_ATL_veryraw[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                                                   center=True).construct('window').dot(
            weight_high))
                                for ensm in amv_EM_ATL_veryraw.keys()], axis=0)

        # Interquartile range (IQR) for amv_EM_ATL
        percentile_25_amv = np.nanpercentile(
            [amv_EM_ATL_veryraw[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                                 center=True).construct('window').dot(weight_high)
             for ensm in amv_EM_ATL_veryraw.keys()], 25, axis=0)
        percentile_75_amv = np.nanpercentile(
            [amv_EM_ATL_veryraw[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                                 center=True).construct('window').dot(weight_high)
             for ensm in amv_EM_ATL_veryraw.keys()], 75, axis=0)
        # Mean of all ensemble members for amv_EM_ATL_iso
        mean_amv_iso = np.nanmean([amv_EM_ATL_iso[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                                                  center=True).construct('window').dot(
            weight_high)
                                    for ensm in amv_EM_ATL_iso.keys()], axis=0)
        # Interquartile range (IQR) for amv_EM_ATL_iso
        percentile_25_amv_iso = np.nanpercentile(
            [amv_EM_ATL_iso[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                             center=True).construct('window').dot(weight_high)
             for ensm in amv_EM_ATL_iso.keys()], 25, axis=0)
        percentile_75_amv_iso = np.nanpercentile(
            [amv_EM_ATL_iso[ensm]['amv_timeseries'].rolling(time=len(hfw),
                                                             center=True).construct('window').dot(weight_high)
             for ensm in amv_EM_ATL_iso.keys()], 75, axis=0)

        # Plot mean lines with shading for the interquartile range (IQR) - AMV
        fig, ax = plt.subplots(figsize=(13, 5))

        plt.hlines(y=0, xmin=0, xmax=len(yearsarrobs), alpha=0.3, color='black')

        plt.plot(amv_obs_dict_veryraw['amv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
            weight_high),
                 color='black', linewidth=2, alpha=.9, linestyle='dashed', label='$OBS \ (HadISST)$')
        plt.plot(
            amv_obs_dict_iso['amv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(weight_high),
            color=light_green_palette[5], linewidth=2, alpha=.9, linestyle='dashed',
            label='$OBS \ (HadISST \ minus \ MEM)$')

        plt.plot(mean_amv, color='darkorange', linestyle='solid', linewidth=3, label='$Pacemaker \ (Ensemble \ Mean)$')
        plt.fill_between(range(len(mean_amv)), percentile_25_amv, percentile_75_amv, color='darkorange', alpha=0.2)

        plt.plot(mean_amv_iso, color='steelblue', linestyle='solid', linewidth=3,
                 label='$Pacemaker \ minus \ MEM \ (Ensemble \ Mean)$')
        plt.fill_between(range(len(mean_amv_iso)), percentile_25_amv_iso, percentile_75_amv_iso, color='steelblue',
                         alpha=0.2)

        plt.title('$AMV \ Timeseries \ (7$-$yr \ Low$-$Pass \ Filtered)$', fontsize=20)
        plt.xlabel('$Time$')
        plt.ylabel('$Anomaly$')
        plt.xticks(x1[::120], yearsarrobs[::120])
        ax.set_xlim(0, len(yearsarrobs))
        ax.legend(fontsize=12)  # ,bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='both', labelsize=15)
        plt.show()
        plt.close()

        # Similar modifications for PDV...

        # Mean of all ensemble members for pdv_EM_ATL
        mean_pdv = np.nanmean(
            [(pdv_EM_ATL_veryraw[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high))
             for ensm in pdv_EM_ATL_veryraw.keys()], axis=0)
        # Interquartile range (IQR) for pdv_EM_ATL
        percentile_25_pdv = np.nanpercentile(
            [pdv_EM_ATL_veryraw[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high)
             for ensm in pdv_EM_ATL_veryraw.keys()], 25, axis=0)
        percentile_75_pdv = np.nanpercentile(
            [pdv_EM_ATL_veryraw[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high)
             for ensm in pdv_EM_ATL_veryraw.keys()], 75, axis=0)
        # Mean of all ensemble members for pdv_EM_ATL_iso
        mean_pdv_iso = np.nanmean(
            [pdv_EM_ATL_iso[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high)
             for ensm in pdv_EM_ATL_iso.keys()], axis=0)

        # Interquartile range (IQR) for pdv_EM_ATL_iso
        percentile_25_pdv_iso = np.nanpercentile(
            [pdv_EM_ATL_iso[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high)
             for ensm in pdv_EM_ATL_iso.keys()], 25, axis=0)
        percentile_75_pdv_iso = np.nanpercentile(
            [pdv_EM_ATL_iso[ensm]['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
                weight_high)
             for ensm in pdv_EM_ATL_iso.keys()], 75, axis=0)

        # Plot mean lines with shading for the interquartile range (IQR) - PDV
        plt.figure(figsize=(13, 5))
        plt.hlines(y=0, xmin=0, xmax=len(yearsarrobs), alpha=0.3, color='black')

        plt.plot((pdv_obs_dict_veryraw['pdv_timeseries']).rolling(time=len(hfw), center=True).construct('window').dot(
            weight_high),
                 color='black', linewidth=2, alpha=.9, linestyle='dashed', label='$OBS \ (HadISST)$')
        plt.plot(pdv_obs_dict_iso['pdv_timeseries'].rolling(time=len(hfw), center=True).construct('window').dot(
            weight_high),
                 color=light_green_palette[5], linewidth=2, alpha=.9, linestyle='dashed',
                 label='$OBS \ (HadISST \ minus \ MEM)$')

        plt.plot(mean_pdv, color='darkorange', linestyle='solid', linewidth=3,
                 label='$Pacemaker \ (Ensemble \ Mean)$')
        plt.fill_between(range(len(mean_pdv)), percentile_25_pdv, percentile_75_pdv, color='darkorange', alpha=0.2)

        plt.plot(mean_pdv_iso, color='steelblue', linestyle='solid', linewidth=3,
                 label='$Pacemaker \ minus \ MEM \ (Ensemble \ Mean)$')
        plt.fill_between(range(len(mean_pdv_iso)), percentile_25_pdv_iso, percentile_75_pdv_iso, color='steelblue',
                         alpha=0.2)

        plt.title('$PDV \ Timeseries \ (7$-$yr \ Low$-$Pass \ Filtered)$', fontsize=20)
        plt.xlabel('$Time$')
        plt.ylabel('$Anomaly$')
        plt.xticks(x1[::120], yearsarrobs[::120])
        ax.set_xlim(0, len(yearsarrobs))
        # plt.legend(fontsize=10.5)#bbox_to_anchor=(.905, -0.1), loc='upper left')
        ax.tick_params(axis='both', labelsize=15)
        plt.show()


    @staticmethod
    def plot_average_time_series(data_EM_ATL, data_EM_ATL_iso, yearsarrobs, var_names):


        # Load data
        average_time_series_data = np.mean(data_EM_ATL[:, 1:, :], axis=0)
        average_time_series_data_iso = np.mean(data_EM_ATL_iso[:, 1:, :], axis=0)
        T, N = data_EM_ATL[0, :, :].shape
        # Define colors for better readability
        color_data = 'darkorange'
        color_data_iso = 'steelblue'
        alpha = 0.2

        # Create a 3x2 subplot grid
        fig, axs = plt.subplots(3, 2, figsize=(18, 15))

        # Loop through each variable
        for i, ax in enumerate(axs.flatten()):
            x1 = range(len(average_time_series_data[:, 0]))
            # Extract the time series for the current variable for both datasets
            variable_data = average_time_series_data[:, i]
            variable_data_iso = average_time_series_data_iso[:, i]
            ax.hlines(y=0, xmin=0, xmax=T, color='black', alpha=alpha)
            if i == 0:
                # Plot the mean time series for both datasets with a bolder line
                sns.lineplot(x=np.arange(len(variable_data)), y=variable_data, label='Pacemaker Ensemble Mean', color=color_data, linewidth=3, ax=ax)
                sns.lineplot(x=np.arange(len(variable_data_iso)), y=variable_data_iso, label='(Pacemaker minus MEM) Ensemble Mean', color=color_data_iso, alpha=0.8, linewidth=3, ax=ax)
            else:
                sns.lineplot(x=np.arange(len(variable_data)), y=variable_data, color=color_data, linewidth=3, ax=ax)
                sns.lineplot(x=np.arange(len(variable_data_iso)), y=variable_data_iso, color=color_data_iso, alpha=0.8, linewidth=3, ax=ax)
            # Shade the 5th-95th percentile range
            percentile_data = np.percentile(data_EM_ATL[:, 1:, i], [5, 95], axis=0)
            ax.fill_between(x=np.arange(len(variable_data)), y1=percentile_data[0, :], y2=percentile_data[1, :], color=color_data, alpha=0.3)

            percentile_data_iso = np.percentile(data_EM_ATL_iso[:, 1:, i], [5, 95], axis=0)
            ax.fill_between(x=np.arange(len(variable_data_iso)), y1=percentile_data_iso[0, :], y2=percentile_data_iso[1, :], color=color_data_iso, alpha=0.2)

            # Customize the plot
            ax.set_title(f'{chr(97 + i)}) {var_names[i]}', fontsize=16)
            ax.set_xlabel('$Time \ (years)$')
            ax.set_ylabel('$Anomaly$')
            ax.set_xticks(x1[::20])
            ax.set_xticklabels(yearsarrobs[::20], fontsize=10)

        # Add a legend to the first subplot (a)
        axs[0, 0].legend(loc='upper left')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def plot_picontrol_amv_pdv(amv_lp,pdv_lp,save_dir):
        fig, ax = plt.subplots(figsize=(28,6))
        sns.set_style("whitegrid")
        ax.set_title('Pre-industrial Control: AMV and PDV (13-yr lowpass filtered)',fontsize=18)
        plt.plot(pdv_lp,'k--',linewidth=1.2, label='PDV')
        plt.plot(amv_lp,'k',linewidth=3.5, label='AMV')

        plt.fill_between(range(len(pdv_lp)), pdv_lp, 0.0,
                        where=(pdv_lp > 0),
                        alpha=0.8, color='pink', interpolate=True)
        plt.fill_between(range(len(pdv_lp)), pdv_lp, 0.0,
                        where=(pdv_lp < 0),
                        alpha=0.8, color='lightblue', interpolate=True)
        plt.fill_between(range(len(amv_lp)), amv_lp, 0.0,
                        where=(amv_lp > 0),
                        alpha=0.6, color='red', interpolate=True)
        plt.fill_between(range(len(amv_lp)), amv_lp, 0.0,
                        where=(amv_lp < 0),
                        alpha=0.6, color='blue', interpolate=True)

        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xlabel('seasons (3-monthly avg)',fontsize=16)
        ax.set_xlim(0, len(pdv_lp))

        plt.savefig(save_dir+'picontrol_timeseries_AMV_PDV.png')
        plt.legend(fontsize=20)
        plt.show()
        plt.close()
        
    @staticmethod
    def create_directories(figs_dir, df_dir):
        # Check and create FIGS_DIR if it doesn't exist
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
            print("Directory '{}' created successfully!".format(figs_dir))
        else:
            print("Directory '{}' already exists!".format(figs_dir))

        # Check and create DF_DIR if it doesn't exist
        if not os.path.exists(df_dir):
            os.mkdir(df_dir)
            print("Directory '{}' created successfully!".format(df_dir))
        else:
            print("Directory '{}' already exists!".format(df_dir))

# Example usage:
if __name__ == "__main__":
    FIGS_DIR = "path_to_your_figs_directory"
    DF_DIR = "path_to_your_df_directory"
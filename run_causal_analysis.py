# This file contains functions for the causal analyses using PCMCI+ in karmouche24esd_AtlanticPacific_Pacemaker package.
# These are called in jupyter notebooks to perform various analysis and plotting routines.
# The jupyter notebooks are used to generate the figures and tables in the paper:
# Karmouche, S., Galytska, E., Meehl, G. A., Runge, J., Weigel, K., and Eyring, V.:
# Changing effects of external forcing on Atlantic-Pacific interactions, EGUsphere [preprint],
# https://doi.org/10.5194/egusphere-2023-1861, 2023.
# Author: Soufiane Karmouche

import os
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings while loading
import numpy as np
from matplotlib import pyplot as plt
from utils import Utils as utils
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.data_processing import DataFrame
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class run_obs_analysis:
    """
    A class that contains methods for calculating indices and patterns for observational datasets,
    running analysis on the observational data, and building link assumptions for the analysis.
    """

    @staticmethod
    def calculate_obs_indices(
        ds_ts_obs, ds_psl_obs, ds_ua_obs, ds_ts_em, ds_psl_em, ds_ua_em, utils
    ):
        """
        Calculate various indices and patterns for the observational dataset.

        Parameters:
        - ds_ts_obs (xarray.Dataset): Dataset containing the observational time series data.
        - ds_psl_obs (xarray.Dataset): Dataset containing the observational sea level pressure data.
        - ds_ua_obs (xarray.Dataset): Dataset containing the observational zonal wind data.
        - ds_ts_em (xarray.Dataset): Dataset containing the ensemble mean time series data.
        - ds_psl_em (xarray.Dataset): Dataset containing the ensemble mean sea level pressure data.
        - ds_ua_em (xarray.Dataset): Dataset containing the ensemble mean zonal wind data.
        - utils (object): Object containing utility functions for calculating indices and patterns.

        Returns:
        - tig_data (numpy.ndarray): Array containing the calculated indices and patterns for the observational dataset.
        - tig_data_iso (numpy.ndarray): Array containing the calculated indices and patterns for the observational dataset as isolated internal variability.
        - amv_obs_dict (dict): Dictionary containing the calculated AMV (Atlantic Multidecadal Variability) for the observational dataset.
        - amv_obs_dict_iso (dict): Dictionary containing the calculated AMV for the observational dataset as isolated internal variability.
        - amv_obs_dict_veryraw (dict): Dictionary containing the calculated AMV for the observational dataset without removing the global mean.
        - pdv_obs_dict (dict): Dictionary containing the calculated PDV (Pacific Decadal Variability) for the observational dataset.
        - pdv_obs_dict_iso (dict): Dictionary containing the calculated PDV for the observational dataset as isolated internal variability.
        - pdv_obs_dict_veryraw (dict): Dictionary containing the calculated PDV for the observational dataset without removing the global mean.
        - yearsarrobs (numpy.ndarray): Array containing the years corresponding to the observational dataset.
        - lenP1 (int): Length of the first period for calculating indices and patterns.
        - lenP2 (int): Length of the second period for calculating indices and patterns.
        """

        # Calculate indices and patterns for the observational dataset
        # without removing the global mean between 60°S and 70°N

        tna_obs_dict, nino34_obs_dict = (
            utils.calculate_tna(ds_ts_obs["ts"], None, remove_gm=False),
            utils.calculate_nino34(ds_ts_obs["ts"], None, remove_gm=False),
        )  # Calculate TNA and Nino3.4 indices

        alt3 = utils.calculate_ATL3_index(
            ds_ts_obs["ts"], None, remove_gm=False, period=["1949-01-01", "2015-12-01"]
        )  # Calculate ATL3 index

        naoo_obs_seas, _ = utils.calculate_nao(
            ds_psl_obs["psl"], None, seasonal=True
        )  # Calculate NAO index

        pnaa_obs_seas, _ = utils.calculate_pna(
            ds_psl_obs["psl"], None, seasonal=True
        )  # Calculate PNA index

        _, raw_zonal_wind_pacific_obs = utils.extract_reg_timeseries1(
            ds_ua_obs["ua"],
            ds_em=None,
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )  # Extract zonal wind in the Pacific region

        # Calculate indices and patterns for the observational dataset as isolated internal variability (MEM is removed)
        tna_obs_dict_iso, nino34_obs_dict_iso = (
            utils.calculate_tna(
                ds_ts_obs["ts"], ds_ts_em["ts"][12 * 50 :], remove_gm=False
            ),
            utils.calculate_nino34(
                ds_ts_obs["ts"], ds_ts_em["ts"][12 * 50 :], remove_gm=False
            ),
        )  # Calculate TNA and Nino3.4 indices with MEM removed

        alt3_iso = utils.calculate_ATL3_index(
            ds_ts_obs["ts"],
            ds_ts_em["ts"][12 * 50 :],
            remove_gm=False,
            period=["1949-01-01", "2015-12-01"],
        )  # Calculate ATL3 index with MEM removed

        naoo_obs_seas_iso, _ = utils.calculate_nao(
            ds_psl_obs["psl"], ds_psl_em["psl"][12 * 50 :], seasonal=True
        )  # Calculate NAO index with MEM removed

        pnaa_obs_seas_iso, _ = utils.calculate_pna(
            ds_psl_obs["psl"], ds_psl_em["psl"][12 * 50 :], seasonal=True
        )  # Calculate PNA index with MEM removed

        _, raw_zonal_wind_pacific_obs_iso = utils.extract_reg_timeseries1(
            ds_ua_obs["ua"],
            ds_em=ds_ua_em["ua"][12 * 50 :],
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )  # Extract zonal wind in the Pacific region with MEM removed

        # Calculate AMV and PDV (mixed and isolated) for the observational dataset
        # with and without removing the global mean between 60°S and 70°N
        amv_obs_dict = utils.calculate_amv(
            ds_ts_obs["ts"], period=["1950-01-16", "2014-12-16"]
        )  # Calculate AMV

        amv_obs_dict_iso = utils.calculate_amv(
            ds_ts_obs["ts"],
            ds_ts_em["ts"][12 * 50 :],
            remove_gm=False,
            period=["1950-01-16", "2014-12-16"],
        )  # Calculate AMV with MEM removed

        amv_obs_dict_veryraw = utils.calculate_amv(
            ds_ts_obs["ts"], remove_gm=False, period=["1950-01-16", "2014-12-16"]
        )  # Calculate AMV without removing the global mean

        pdv_obs_dict = utils.calculate_pdv(
            ds_ts_obs["ts"], period=["1950-01-16", "2014-12-16"]
        )  # Calculate PDV

        pdv_obs_dict_iso = utils.calculate_pdv(
            ds_ts_obs["ts"],
            ds_ts_em["ts"][12 * 50 :],
            remove_gm=False,
            period=["1950-01-16", "2014-12-16"],
        )  # Calculate PDV with MEM removed

        pdv_obs_dict_veryraw = utils.calculate_pdv(
            ds_ts_obs["ts"], remove_gm=False, period=["1950-01-16", "2014-12-16"]
        )  # Calculate PDV without removing the global mean

        # Flip the sign of PDV if correlation with Nino34 is negative
        if (
            np.corrcoef(
                pdv_obs_dict["pdv_timeseries"], nino34_obs_dict["nino34_timeseries"]
            )[0, 1]
            < 0
        ):
            pdv_obs_dict["pdv_timeseries"] *= -1
        if (
            np.corrcoef(
                pdv_obs_dict_iso["pdv_timeseries"], nino34_obs_dict["nino34_timeseries"]
            )[0, 1]
            < 0
        ):
            pdv_obs_dict_iso["pdv_timeseries"] *= -1
        if (
            np.corrcoef(
                pdv_obs_dict_veryraw["pdv_timeseries"],
                nino34_obs_dict["nino34_timeseries"],
            )[0, 1]
            < 0
        ):
            pdv_obs_dict_veryraw["pdv_timeseries"] *= -1

        # Prepare dataframe for Tigramite's PCMCIplus
        tig_data = np.zeros((len(naoo_obs_seas), 6))
        tig_data[:, 0] = (
            tna_obs_dict["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data[:, 1] = pnaa_obs_seas
        tig_data[:, 2] = (
            nino34_obs_dict["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data[:, 3] = raw_zonal_wind_pacific_obs
        tig_data[:, 4] = naoo_obs_seas
        tig_data[:, 5] = (
            alt3["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()
        )

        # Dataframe for inddices where MEM is removed
        tig_data_iso = np.zeros(tig_data.shape)
        tig_data_iso[:, 0] = (
            tna_obs_dict_iso["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data_iso[:, 1] = pnaa_obs_seas_iso
        tig_data_iso[:, 2] = (
            nino34_obs_dict_iso["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data_iso[:, 3] = raw_zonal_wind_pacific_obs_iso
        tig_data_iso[:, 4] = naoo_obs_seas_iso
        tig_data_iso[:, 5] = (
            alt3_iso["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()
        )
        # Years corresponding to the x-axis
        yearsarrobs = pnaa_obs_seas["time.year"].values[1:]
        # Flip the sign of NAO index in the dataframes
        tig_data[:, -2] *= -1  # NAO flip sign.
        tig_data_iso[:, -2] *= -1  # NAO iso flip sign

        # Length of the first and second periods for calculating indices and patterns
        lenP1, lenP2 = len(
            pnaa_obs_seas_iso.sel(time=slice("1949-12-01", "1983-03-01"))
        ), len(pnaa_obs_seas_iso.sel(time=slice("1983-01-01", "2014-12-01")))

        return (
            tig_data,
            tig_data_iso,
            amv_obs_dict,
            amv_obs_dict_iso,
            amv_obs_dict_veryraw,
            pdv_obs_dict,
            pdv_obs_dict_iso,
            pdv_obs_dict_veryraw,
            yearsarrobs,
            lenP1,
            lenP2,
        )

    @staticmethod
    def run_analysis_obs(
        pcmci,
        exp,
        list_titleperiods,
        FIGS_DIR,
        var_names=None,
        min_tau=0,
        max_tau=4,
        pc_alpha=None,
        node_pos=None,
        test_assumption=None,
    ):
        """
        Run causal analysis for observational data and plot results.

        Args:
            pcmci (PCMCI): The PCMCI object used for causal discovery.
            exp (int): The experiment number.
            list_titleperiods (list): A list of title periods.
            FIGS_DIR (str): The directory to save the figures.
            var_names (list, optional): A list of variable names. Defaults to None.
            min_tau (int, optional): The minimum lag value. Defaults to 0.
            max_tau (int, optional): The maximum lag value. Defaults to 4.
            pc_alpha (float, optional): The significance level for partial correlation test. Defaults to None.
            node_pos (dict, optional): The positions of the nodes in the graph. Defaults to None.
            test_assumption (str, optional): The assumptions for building links. Defaults to None.

        Returns:
            None
        """
        title = list_titleperiods[exp - 1]
        N = len(var_names)
        results_masking = pcmci.run_pcmciplus(
            tau_min=min_tau,
            tau_max=max_tau,
            pc_alpha=pc_alpha,
            link_assumptions=test_assumption,
        )  # , verbosity=0)

        lag_p_matrix = tp.setup_matrix(
            N=N,
            tau_max=max_tau,
            minimum=0.0,
            maximum=1,
            figsize=(10, 10),
            x_base=1,
            y_base=0.1,
            tick_label_size=8,
            lag_units="seasons",
            plot_gridlines=True,
            var_names=var_names,
            legend_fontsize=15,
            label_space_left=0.2,
            label_space_top=0.05,
        )
        lag_p_matrix.add_lagfuncs(val_matrix=results_masking["p_matrix"], color="black")
        lag_p_matrix.add_lagfuncs(
            val_matrix=(
                np.where(
                    results_masking["p_matrix"] <= pc_alpha,
                    results_masking["p_matrix"],
                    None,
                )
            ),
            color="red",
        )
        lag_p_matrix.savefig(
            FIGS_DIR
            + f"pcmciplus_with_assumptions_lagfuncs_OBS_{list_titleperiods[exp-1].replace(' ', '_')}.png"
        )

        fig = plt.figure(figsize=(12, 15))
        ax = plt.axes(
            projection=ccrs.PlateCarree(central_longitude=210.0)
        )  # again the projection is defined for the axes, so that node_pos combines with the map
        ax.set_extent(
            [-10, 150, -20, 67], crs=ccrs.PlateCarree(central_longitude=210.0)
        )  # section of the map is defined, with the corresponding coordinate reference system (crs)
        # Additional features from cartopy
        ax.add_feature(cfeature.LAND, alpha=0.3)
        # ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE, alpha=0.3)
        ax.set_title(title, size=20)
        # ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
        tp.plot_graph(
            fig_ax=(fig, ax),
            graph=results_masking["graph"],
            val_matrix=results_masking["val_matrix"],
            var_names=var_names,
            vmin_edges=-1,
            vmax_edges=1,
            vmin_nodes=-1,
            vmax_nodes=1,
            node_pos=node_pos,
            cmap_nodes="RdBu_r",
            node_size=11,
            arrowhead_size=12,
            curved_radius=0.23,
            node_label_size=19,
            link_label_fontsize=17,
            node_colorbar_label="auto-MCI",
            link_colorbar_label="cross-MCI",
            save_name=FIGS_DIR
            + f"pcmciplus_graph_with_assumptions_OBS_{list_titleperiods[exp-1].replace(' ', '_')}.png",
            figsize=(8, 4),
            show_colorbar=False,
        )
        plt.show()
        plt.close()

    @staticmethod
    def build_obs_assumption(max_tau, pcmci):
        """
        Builds the assumptions for causal analysis.

        Args:
            max_tau (int): The maximum time lag.
            pcmci: The PCMCI object.

        Returns:
            dict: The link assumptions for causal analysis.
        """
        # Here we implement our assumptions. For more about build_link_assumptions see the Tigramite code here tigramite/tigramite/pcmci_base.py
        # Plot the results
        link_assumptions_absent_link_means_no_knowledge = {
            3: {(2, 0): "<?-"},
            2: {(3, 0): "-?>"},
        }  #  PWC -> Nino34
        test_assumption = pcmci.build_link_assumptions(
            link_assumptions_absent_link_means_no_knowledge, 6, max_tau, 0
        )
        test_assumption[2][5, 0] = "<?-"  # Nino34 --> ATL3
        test_assumption[5][2, 0] = "-?>"  # Nino34 --> ATL3

        for tau in range(0, max_tau + 1):
            test_assumption[4].pop((5, -tau))  # No ATL3 --> NAO
            test_assumption[5].pop((4, -tau))  # No ATL3 --> NAO
        for tau in range(0, max_tau + 1):
            test_assumption[1].pop((5, -tau))  # No ATL3 --> PNA
            test_assumption[5].pop((1, -tau))  # No ATL3 --> PNA
        for tau in range(0, max_tau + 1):
            test_assumption[0].pop((5, -tau))  # No ATL3 --> TNA
            test_assumption[5].pop((0, -tau))  # No ATL3 --> TNA
        for tau in range(0, max_tau + 1):
            test_assumption[3].pop((1, -tau))  # No PNA --> PWC
            test_assumption[1].pop((3, -tau))  # No PNA --> PWC
        return test_assumption


class run_sliding_window:
    """
    A class that contains methods to load and calculate various indices for the Pacific and Atlantic pacemaker ensembles.
    """

    @staticmethod
    def load_and_calculate_indices_PACIFIC(
        ensm, ds_ts_em, ds_ua_em, ds_psl_em, ts_nc, ua_nc, psl_nc, T, N
    ):
        """
        Load and calculate various indices for the Pacific pacemaker ensemble.

        Parameters:
            ensm (int): The ensemble number.
            ds_ts_em (xarray.Dataset): The dataset containing TS values.
            ds_ua_em (xarray.Dataset): The dataset containing U wind values.
            ds_psl_em (xarray.Dataset): The dataset containing PSL values.
            T (int): The number of time steps.
            N (int): The number of indices.

        Returns:
            tuple: A tuple containing the following:
                - ensm-1 (int): The ensemble member (id).
                - tig_data_PAC (numpy.ndarray): The calculated indices for the Pacific region.
                - tig_data_PAC_iso (numpy.ndarray): The calculated indices for the Pacific region with isotope data.
                - amv_dict (dict): Dictionary containing AMV values.
                - amv_dict_iso (dict): Dictionary containing AMV values with isotope data.
                - amv_dict_veryraw (dict): Dictionary containing AMV values without removing the global mean.
                - pdv_dict (dict): Dictionary containing PDV values.
                - pdv_dict_iso (dict): Dictionary containing PDV values with isotope data.
                - pdv_dict_veryraw (dict): Dictionary containing PDV values without removing the global mean.
                - yearsarrobs (numpy.ndarray): Array of years for observations.
        """

        perioda = ["1950-01-01", "2014-12-01"]

        # Load datasets
        ds_psl_pac = xr.open_dataset(psl_nc).sel(time=slice("1950-01-01", "2014-12-01"))
        ds_ua_pac = (
            xr.open_dataset(ua_nc)
            .sel(lev=925, method="nearest")
            .sel(time=slice("1950-01-01", "2014-12-01"))
        )
        ds_ua_pac = ds_ua_pac.rename({"lev": "plev"})
        ds_ts_pac = xr.open_dataset(ts_nc).sel(time=slice("1950-01-01", "2014-12-01"))

        # Interpolate data to match MEM grid
        ds_ts_pac, ds_ua_pac, ds_psl_pac = (
            ds_ts_pac.interp(lon=ds_ts_em["lon"], lat=ds_ts_em["lat"]),
            ds_ua_pac.interp(lon=ds_psl_em["lon"], lat=ds_psl_em["lat"]),
            ds_psl_pac.interp(lon=ds_psl_em["lon"], lat=ds_psl_em["lat"]),
        )

        # Calculate indices
        tna_dict, nino34_dict = utils.calculate_tna(
            ds_ts_pac["TS"], None, remove_gm=False
        ), utils.calculate_nino34(ds_ts_pac["TS"], None, remove_gm=False)
        naoo_seas, _ = utils.calculate_nao(ds_psl_pac["PSL"], None, seasonal=True)
        pnaa_seas, _ = utils.calculate_pna(ds_psl_pac["PSL"], None, seasonal=True)
        _, raw_zonal_wind_pacific = utils.extract_reg_timeseries1(
            ds_ua_pac["U"],
            ds_em=None,
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )
        alt3 = utils.calculate_ATL3_index(
            ds_ts_pac["TS"], None, remove_gm=False, period=["1950-01-01", "2014-12-01"]
        )
        tna_dict_iso, nino34_dict_iso = utils.calculate_tna(
            ds_ts_pac["TS"], ds_ts_em["ts"], remove_gm=False
        ), utils.calculate_nino34(ds_ts_pac["TS"], ds_ts_em["ts"], remove_gm=False)
        naoo_seas_iso, _ = utils.calculate_nao(
            ds_psl_pac["PSL"], ds_psl_em["psl"], seasonal=True
        )
        pnaa_seas_iso, _ = utils.calculate_pna(
            ds_psl_pac["PSL"], ds_psl_em["psl"], seasonal=True
        )
        _, raw_zonal_wind_pacific_iso = utils.extract_reg_timeseries1(
            ds_ua_pac["U"],
            ds_em=ds_ua_em["ua"],
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )
        alt3_iso = utils.calculate_ATL3_index(
            ds_ts_pac["TS"],
            ds_ts_em["ts"],
            remove_gm=False,
            period=["1950-01-01", "2014-12-01"],
        )

        # Calculate tig_data_PAC and tig_data_PAC_iso
        tig_data_PAC = np.zeros((T, N))
        tig_data_PAC[:, 0] = (
            tna_dict["tna_timeseries"].resample(time="QS-DEC", keep_attrs=True).mean()
        )
        tig_data_PAC[:, 1] = pnaa_seas
        tig_data_PAC[:, 2] = (
            nino34_dict["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data_PAC[:, 3] = raw_zonal_wind_pacific
        tig_data_PAC[:, 4] = naoo_seas
        tig_data_PAC[:, 5] = (
            alt3["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()
        )

        tig_data_PAC_iso = np.zeros(tig_data_PAC.shape)
        tig_data_PAC_iso[:, 0] = (
            tna_dict_iso["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data_PAC_iso[:, 1] = pnaa_seas_iso
        tig_data_PAC_iso[:, 2] = (
            nino34_dict_iso["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()
        )
        tig_data_PAC_iso[:, 3] = raw_zonal_wind_pacific_iso
        tig_data_PAC_iso[:, 4] = naoo_seas_iso
        tig_data_PAC_iso[:, 5] = (
            alt3_iso["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()
        )

        # Flip indices if necessary
        if (np.corrcoef(tig_data_PAC[:, 1], tig_data_PAC[:, 2])[0, 1]) < 0:
            tig_data_PAC[:, 1] *= -1
        if (np.corrcoef(tig_data_PAC[:, 1], tig_data_PAC[:, 4])[0, 1]) > 0:
            tig_data_PAC[:, 4] *= -1
        if (np.corrcoef(tig_data_PAC_iso[:, 1], tig_data_PAC_iso[:, 2])[0, 1]) < 0:
            tig_data_PAC_iso[:, 1] *= -1
        if (np.corrcoef(tig_data_PAC_iso[:, 1], tig_data_PAC_iso[:, 4])[0, 1]) > 0:
            tig_data_PAC_iso[:, 4] *= -1

        # Calculate AMV and PDV values
        amv_dict = utils.calculate_amv(ds_ts_pac["TS"], None, period=perioda)
        amv_dict_veryraw = utils.calculate_amv(
            ds_ts_pac["TS"], remove_gm=False, period=perioda
        )
        amv_dict_iso = utils.calculate_amv(
            ds_ts_pac["TS"], ds_ts_em["ts"], remove_gm=False, period=perioda
        )
        pdv_dict = utils.calculate_pdv(ds_ts_pac["TS"], None, period=perioda)
        pdv_dict_veryraw = utils.calculate_pdv(
            ds_ts_pac["TS"], remove_gm=False, period=perioda
        )
        pdv_dict_iso = utils.calculate_pdv(
            ds_ts_pac["TS"], ds_ts_em["ts"], remove_gm=False, period=perioda
        )
        yearsarrobs = pnaa_seas_iso[1:]["time.year"].values

        return (
            ensm - 1,
            tig_data_PAC,
            tig_data_PAC_iso,
            amv_dict,
            amv_dict_iso,
            amv_dict_veryraw,
            pdv_dict,
            pdv_dict_iso,
            pdv_dict_veryraw,
            yearsarrobs,
        )

    @staticmethod
    def load_and_calculate_indices_ATLANTIC(
        ensm, ds_ts_em, ds_ua_em, ds_psl_em, ts_nc, ua_nc, psl_nc, T, N
    ):
        """
        Load and calculate various indices for the Atlantic pacemaker ensemble.

        Parameters:
            ensm (int): The ensemble number.
            ds_ts_em (xarray.Dataset): The dataset containing TS values.
            ds_ua_em (xarray.Dataset): The dataset containing U wind values.
            ds_psl_em (xarray.Dataset): The dataset containing PSL values.
            T (int): The number of time steps.
            N (int): The number of indices.

        Returns:
            tuple: A tuple containing the following:
                - ensm-1 (int): The ensemble member (id).
                - tig_data_PAC (numpy.ndarray): The calculated indices for the Pacific region.
                - tig_data_PAC_iso (numpy.ndarray): The calculated indices for the Pacific region with isotope data.
                - amv_dict (dict): Dictionary containing AMV values.
                - amv_dict_iso (dict): Dictionary containing AMV values with isotope data.
                - amv_dict_veryraw (dict): Dictionary containing AMV values without removing the global mean.
                - pdv_dict (dict): Dictionary containing PDV values.
                - pdv_dict_iso (dict): Dictionary containing PDV values with isotope data.
                - pdv_dict_veryraw (dict): Dictionary containing PDV values without removing the global mean.
                - yearsarrobs (numpy.ndarray): Array of years for observations.
        """
        # Format the ensemble number
        fensm = "{:02d}".format(ensm)

        # Open and select data from netCDF files for Atlantic region
        ds_psl_atl = xr.open_dataset(psl_nc).sel(time=slice("1950-01-01", "2013-12-01"))
        ds_ua_atl = (
            xr.open_dataset(ua_nc)
            .sel(lev=925, method="nearest")
            .sel(time=slice("1950-01-01", "2013-12-01"))
        )
        ds_ua_atl = ds_ua_atl.rename({"lev": "plev"})
        ds_ts_atl = xr.open_dataset(ts_nc).sel(time=slice("1950-01-01", "2013-12-01"))

        # Assign coordinates to match the ensemble data
        ds_psl_atl = ds_psl_atl.assign_coords({"time": ds_psl_em["time"]})
        ds_ts_atl = ds_ts_atl.assign_coords({"time": ds_psl_em["time"]})
        ds_ua_atl = ds_ua_atl.assign_coords({"time": ds_psl_em["time"]})

        # Interpolate data to match the ensemble grid
        ds_ts_atl, ds_ua_atl, ds_psl_atl = (
            ds_ts_atl.interp(lon=ds_ts_em["lon"], lat=ds_ts_em["lat"]),
            ds_ua_atl.interp(lon=ds_psl_em["lon"], lat=ds_psl_em["lat"]),
            ds_psl_atl.interp(lon=ds_psl_em["lon"], lat=ds_psl_em["lat"]),
        )

        # Calculate indices for the Atlantic region
        tna_dict, nino34_dict = utils.calculate_tna(
            ds_ts_atl["TS"], None, remove_gm=False
        ), utils.calculate_nino34(ds_ts_atl["TS"], None, remove_gm=False)
        naoo_seas, _ = utils.calculate_nao(ds_psl_atl["PSL"], None, seasonal=True)
        pnaa_seas, _ = utils.calculate_pna(ds_psl_atl["PSL"], None, seasonal=True)
        _, raw_zonal_wind_pacific = utils.extract_reg_timeseries1(
            ds_ua_atl["U"],
            ds_em=None,
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )
        alt3 = utils.calculate_ATL3_index(
            ds_ts_atl["TS"], None, remove_gm=False, period=["1950-01-01", "2014-12-01"]
        )

        # Calculate indices for the Atlantic pacemaker ensemble (with MEM removed)
        tna_dict_iso, nino34_dict_iso = utils.calculate_tna(
            ds_ts_atl["TS"], ds_ts_em["ts"], remove_gm=False
        ), utils.calculate_nino34(ds_ts_atl["TS"], ds_ts_em["ts"], remove_gm=False)
        naoo_seas_iso, _ = utils.calculate_nao(
            ds_psl_atl["PSL"], ds_psl_em["psl"], seasonal=True
        )
        pnaa_seas_iso, _ = utils.calculate_pna(
            ds_psl_atl["PSL"], ds_psl_em["psl"], seasonal=True
        )
        _, raw_zonal_wind_pacific_iso = utils.extract_reg_timeseries1(
            ds_ua_atl["U"],
            ds_em=ds_ua_em["ua"],
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )
        alt3_iso = utils.calculate_ATL3_index(
            ds_ts_atl["TS"],
            ds_ts_em["ts"],
            remove_gm=False,
            period=["1950-01-01", "2014-12-01"],
        )

        # Fill the tig_data_ATL array with calculated indices for the Atlantic pacemaker ensemble
        tig_data_ATL = np.zeros((T, N))
        tig_data_ATL[:, 0] = (
            tna_dict["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_ATL[:, 1] = pnaa_seas[1:]
        tig_data_ATL[:, 2] = (
            nino34_dict["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_ATL[:, 3] = raw_zonal_wind_pacific[1:]
        tig_data_ATL[:, 4] = naoo_seas[1:]
        tig_data_ATL[:, 5] = (
            alt3["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()[1:]
        )

        # Fill the tig_data_ATL_iso array with calculated indices for the Atlantic pacemaker ensemble (with MEM removed)
        tig_data_ATL_iso = np.zeros(tig_data_ATL.shape)
        tig_data_ATL_iso[:, 0] = (
            tna_dict_iso["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_ATL_iso[:, 1] = pnaa_seas_iso[1:]
        tig_data_ATL_iso[:, 2] = (
            nino34_dict_iso["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_ATL_iso[:, 3] = raw_zonal_wind_pacific_iso[1:]
        tig_data_ATL_iso[:, 4] = naoo_seas_iso[1:]
        tig_data_ATL_iso[:, 5] = (
            alt3_iso["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()[1:]
        )

        # Flip indices if necessary
        if (np.corrcoef(tig_data_ATL[:, 1], tig_data_ATL[:, 2])[0, 1]) < 0:
            tig_data_ATL[:, 1] *= -1
        if (np.corrcoef(tig_data_ATL[:, 1], tig_data_ATL[:, 4])[0, 1]) > 0:
            tig_data_ATL[:, 4] *= -1
        if (np.corrcoef(tig_data_ATL_iso[:, 1], tig_data_ATL_iso[:, 2])[0, 1]) < 0:
            tig_data_ATL_iso[:, 1] *= -1
        if (np.corrcoef(tig_data_ATL_iso[:, 1], tig_data_ATL_iso[:, 4])[0, 1]) > 0:
            tig_data_ATL_iso[:, 4] *= -1
        # Print information about the analysis

        # Calculate AMV and PDV values for the Atlantic pacemaker ensemble
        amv_dict = utils.calculate_amv(
            ds_ts_atl["TS"], None, period=["1950-01-01", "2014-12-01"]
        )
        amv_dict_veryraw = utils.calculate_amv(
            ds_ts_atl["TS"], remove_gm=False, period=["1950-01-01", "2014-12-01"]
        )
        amv_dict_iso = utils.calculate_amv(
            ds_ts_atl["TS"],
            ds_ts_em["ts"],
            remove_gm=False,
            period=["1950-01-01", "2014-12-01"],
        )
        pdv_dict = utils.calculate_pdv(
            ds_ts_atl["TS"], None, period=["1950-01-01", "2014-12-01"]
        )
        pdv_dict_veryraw = utils.calculate_pdv(
            ds_ts_atl["TS"], remove_gm=False, period=["1950-01-01", "2014-12-01"]
        )
        pdv_dict_iso = utils.calculate_pdv(
            ds_ts_atl["TS"],
            ds_ts_em["ts"],
            remove_gm=False,
            period=["1950-01-01", "2014-12-01"],
        )

        # Extract years for observations
        yearsarrobs = pnaa_seas_iso[1:]["time.year"].values
        return (
            ensm - 1,
            tig_data_ATL,
            tig_data_ATL_iso,
            amv_dict,
            amv_dict_iso,
            amv_dict_veryraw,
            pdv_dict,
            pdv_dict_iso,
            pdv_dict_veryraw,
            yearsarrobs,
        )

    @staticmethod
    def run_analysis_sliding(
        pcmci,
        var_names=None,
        min_tau=0,
        max_tau=4,
        pc_alpha=0.01,
        window_step=40,
        window_length=80,
        subtitles=None,
        node_posld=None,
        test_assumption=None,
        FIGS_DIR="figures/",
    ):
        """
        Perform sliding window analysis using PCMCI+ algorithm.

        Args:
            pcmci: The PCMCI object used for causal discovery.
            var_names (list): List of variable names.
            min_tau (int): Minimum time lag.
            max_tau (int): Maximum time lag.
            pc_alpha (float): Significance level for partial correlation test.
            window_step (int): Step size for sliding window.
            window_length (int): Length of each sliding window.
            subtitles (list): List of subtitles for each window.
            node_posld: Node positions for plotting the graph.
            test_assumption: Assumptions for the link test.
            FIGS_DIR (str): Directory to save the figures.

        Returns:
            None
        """
        conf_lev = 0.95
        method = "run_pcmciplus"
        method_args = {
            "tau_min": min_tau,
            "tau_max": max_tau,
            "pc_alpha": pc_alpha,
            "link_assumptions": test_assumption,
        }

        # Run PCMCI+ algorithm with sliding window
        results = pcmci.run_sliding_window_of(
            method=method,
            method_args=method_args,
            window_step=window_step,
            window_length=window_length,
            conf_lev=conf_lev,
        )

        graphs = results["window_results"]["graph"]
        val_matrices = results["window_results"]["val_matrix"]
        p_matrices = results["window_results"]["p_matrix"]
        n_windows = len(graphs)

        mosaic = [["plot{}".format(i) for i in range(n_windows)]]

        # Create a figure and axes for the mosaic layout
        fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(18, 6))

        # Loop through each window and plot the graph
        for w, subtitle in zip(range(n_windows), subtitles):

            # Plot the graph for the current window
            tp.plot_graph(
                graphs[w],
                val_matrix=val_matrices[w],
                var_names=var_names,
                node_size=0.9,
                link_label_fontsize=14,
                node_label_size=18,
                node_pos=node_posld,
                save_name=None,
                show_colorbar=False,
                vmin_edges=-1,
                vmax_edges=1,
                vmin_nodes=-1,
                vmax_nodes=1,
                fig_ax=(fig, axs["plot{}".format(w)]),
            )

            # Set the subtitle for the current plot below the plot
            axs["plot{}".format(w)].set_title("")
            axs["plot{}".format(w)].set_xlabel(subtitle, fontdict={"fontsize": 18})

        # Adjust layout for better appearance
        plt.tight_layout()

        # Save the figure
        if test_assumption is None:
            plt.savefig(
                FIGS_DIR
                + f"pcmciplus_sliding_window_analysis_pacem_with_NO_assumptions.png",
                format="png",
            )
        else:
            plt.savefig(
                FIGS_DIR
                + f"pcmciplus_sliding_window_analysis_pacem_with_assumptions.png",
                format="png",
            )
        plt.show()

        output_directory_sld = (
            FIGS_DIR + "pcmciplus_sliding_window_singlewindow_analysis"
        )
        os.makedirs(output_directory_sld, exist_ok=True)

        # Create and save individual figures
        for i, w in enumerate(range(n_windows)):
            lag_p_matrix = tp.setup_matrix(
                N=len(var_names),
                tau_max=max_tau,
                minimum=0.0,
                maximum=1,
                figsize=(5, 12),
                x_base=1,
                y_base=0.1,
                tick_label_size=8,
                lag_units="seasons",
                plot_gridlines=True,
                var_names=var_names,
                legend_fontsize=15,
                label_space_left=0.2,
            )
            lag_p_matrix.add_lagfuncs(val_matrix=p_matrices[w], color="black")
            lag_p_matrix.add_lagfuncs(
                val_matrix=(np.where(p_matrices[w] <= pc_alpha, p_matrices[w], None)),
                color="red",
            )

            filename = os.path.join(
                output_directory_sld,
                f"p_matrix_win{w}_{subtitles[i]}_sliding_window_with_NO_assumptions_analysis.png",
            )
            lag_p_matrix.savefig(filename)  # Adjust dpi as needed
            plt.close("all")

        # Create a single figure with subplots
        fig, axs = plt.subplots(1, n_windows, figsize=(25, 12))

        # Load and display individual images in the subplots
        for i, w in enumerate(range(n_windows)):
            filename = os.path.join(
                output_directory_sld,
                f"p_matrix_win{w}_{subtitles[i]}_sliding_window_with_NO_assumptions_analysis.png",
            )
            img = plt.imread(filename)

            axs[i].imshow(img)
            axs[i].set_title(subtitles[i])

            # Remove ticks, labels, and grid for each subplot
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            axs[i].grid(False)
            axs[i].set_facecolor("white")

        # Save the complete figure with high resolution
        if test_assumption is None:
            plt.savefig(
                FIGS_DIR + f"complete_lag_p_matrix_with_NO_assumptions.png",
                format="png",
            )
        else:
            plt.savefig(
                FIGS_DIR + f"complete_lag_p_matrix_with_assumptions.png", format="png"
            )
        plt.show()

        # Close the individual figures
        plt.close("all")

        # Delete individual image files
        for i, w in enumerate(range(n_windows)):

            filename = os.path.join(
                output_directory_sld,
                f"p_matrix_win{w}_{subtitles[i]}_sliding_window_with_NO_assumptions_analysis.png",
            )
            os.remove(filename)

        # Remove the empty directory
        os.rmdir(output_directory_sld)

    @staticmethod
    def build_atlantic_assumption(max_tau, pcmci):
        """
        Build the assumptions for the Atlantic Pacemaker.

        Args:
            max_tau (int): The maximum time lag.
            pcmci: The PCMCI object.

        Returns:
            dict: The assumptions for the Atlantic region.
        """
        # Here we implement our assumptions. For more about build_link_assumptions see the Tigramite code here tigramite/tigramite/pcmci_base.py
        link_assumptions_absent_link_means_no_knowledge = {
            5: {(2, 0): "<?-"},
            2: {(5, 0): "-?>"},
        }  # ATL3 --> Nino34
        test_assumption = pcmci.build_link_assumptions(
            link_assumptions_absent_link_means_no_knowledge, 6, max_tau, 0
        )

        test_assumption[4][1, 0] = "<?-"  # NAO --> PNA
        test_assumption[1][4, 0] = "-?>"  # NAO --> PNA
        test_assumption[3][2, 0] = "<?-"  #  PWCu -->  Nino34
        test_assumption[2][3, 0] = "-?>"  #  PWCu -->  Nino34
        # test_assumption[2][4,0]='<?-' # Nino34 --> NAO
        # test_assumption[4][2,0]='-?>' # Nino34 --> NAO
        test_assumption[0][2, 0] = "<?-"  # TNA --> Nino34
        test_assumption[2][0, 0] = "-?>"  # TNA --> Nino34
        test_assumption[0][1, 0] = "<?-"  # TNA --> PNA
        test_assumption[1][0, 0] = "-?>"  # TNA --> PNA
        test_assumption[0][4, 0] = "<?-"  # TNA --> NAO
        test_assumption[4][0, 0] = "-?>"  # TNA --> NAO
        for tau in range(max_tau + 1):
            test_assumption[0].pop((5, -tau))  # No ATL3 --> TNA
            test_assumption[5].pop((0, -tau))  # No ATL3 --> TNA
        for tau in range(0, max_tau + 1):
            test_assumption[3].pop((1, -tau))  # No PWCu --> PNA
            test_assumption[1].pop((3, -tau))  # No PWCu --> PNA
        for tau in range(0, max_tau + 1):
            test_assumption[4].pop((5, -tau))  # No ATL3 --> NAO
            test_assumption[5].pop((4, -tau))  # No ATL3 --> NAO
        for tau in range(0, max_tau + 1):
            test_assumption[1].pop((5, -tau))  # No ATL3 --> PNA
            test_assumption[5].pop((1, -tau))  # No ATL3 --> PNA
        for tau in range(max_tau + 1):
            test_assumption[0].pop((1, -tau))  # No lagged TNA --> PNA
            # test_assumption[1].pop((0, -tau))# No lagged TNA --> PNA
        return test_assumption

    @staticmethod
    def build_pacific_assumption(max_tau, pcmci):
        """
        Builds the link assumption for Pacific pacemaker.

        Args:
            max_tau (int): The maximum tau value.
            pcmci: The PCMCI object.

        Returns:
            dict: The test assumption dictionary.
        """
        # Here we implement our assumptions. For more about build_link_assumptions see the Tigramite code here tigramite/tigramite/pcmci_base.py

        link_assumptions_absent_link_means_no_knowledge = {
            2: {(5, 0): "<?-"},
            5: {(2, 0): "-?>"},
        }  # Nino34 --> ATL3
        test_assumption = pcmci.build_link_assumptions(
            link_assumptions_absent_link_means_no_knowledge, 6, max_tau, 0
        )

        test_assumption[2][0, 0] = "<?-"  # Nino34 --> TNA
        test_assumption[0][2, 0] = "-?>"  # Nino34 --> TNA
        # test_assumption[4].pop((5, 0)) # No ATL3 --> NAO
        # test_assumption[5].pop((4, 0)) # No ATL3 --> NAO
        test_assumption[2][3, 0] = "<?-"  # Nino34 --> PWCu
        test_assumption[3][2, 0] = "-?>"  # Nino34 --> PWCu
        for tau in range(0, max_tau + 1):
            test_assumption[2].pop((3, -tau))  # # Nino34 --> PWCu
            # test_assumption[3][2,-tau]='-?>' # Nino34 --> PWCu
        for tau in range(0, max_tau + 1):
            test_assumption[4].pop((5, -tau))
            test_assumption[5].pop((4, -tau))
        for tau in range(0, max_tau + 1):
            test_assumption[0].pop((5, -tau))  #
            test_assumption[5].pop((0, -tau))
        for tau in range(0, max_tau + 1):
            test_assumption[3].pop((1, -tau))  #
            test_assumption[1].pop((3, -tau))
        return test_assumption


class run_picontrol:
    """
    Class for running the piControl experiment.
    """

    @staticmethod
    def load_calc_piControl(ds_ts_pic, ds_psl_pic, ds_ua_pic, peeriood, peerioodpsl):
        """
        Load and calculate indices for the piControl run.

        Args:
            ds_ts_pic (xarray.Dataset): The dataset containing TS values.
            ds_psl_pic (xarray.Dataset): The dataset containing PSL values.
            ds_ua_pic (xarray.Dataset): The dataset containing U wind values.
            peeriood (list): The period for calculating indices.
            peerioodpsl (list): The period for calculating PSL indices.

        Returns:
            tuple: A tuple containing the following:
                - amv_pic_dict (dict): Dictionary containing AMV values.
                - pdv_pic_dict (dict): Dictionary containing PDV values.
                - tig_data_pic (numpy.ndarray): The calculated indices.
        """
        # Calculate AMV and PDV values
        amv_pic_dict, pdv_pic_dict = utils.calculate_amv(
            ds_ts_pic["ts"], remove_gm=False, period=peeriood
        ), utils.calculate_pdv(ds_ts_pic["ts"], remove_gm=False, period=peeriood)

        # Calculate TNA and Nino34 values
        tna_pic_dict, nino34_pic_dict = utils.calculate_tna(
            ds_ts_pic["ts"], None, remove_gm=False
        ), utils.calculate_nino34(ds_ts_pic["ts"], None, remove_gm=False)

        # Calculate ATL3 index
        alt3_pic = utils.calculate_ATL3_index(
            ds_ts_pic["ts"], None, remove_gm=False, period=peeriood
        )

        # Calculate NAO and PNA values
        naoo_pic_seas, _ = utils.calculate_nao(
            ds_psl_pic["psl"], None, period=peerioodpsl, seasonal=True
        )
        pnaa_pic_seas, _ = utils.calculate_pna(
            ds_psl_pic["psl"], None, period=peerioodpsl, seasonal=True
        )

        # Calculate raw zonal wind in the Pacific
        _, raw_zonal_wind_pacific_pic = utils.extract_reg_timeseries1(
            ds_ua_pic["ua"],
            ds_em=None,
            minlon=180,
            maxlon=210,
            minlat=-6,
            maxlat=6,
            seasonal=True,
        )

        # Check correlation between PDV and Nino34, and flip PDV if necessary
        if (
            np.corrcoef(
                pdv_pic_dict["pdv_timeseries"], nino34_pic_dict["nino34_timeseries"]
            )[0, 1]
        ) < 0:
            print(
                np.corrcoef(
                    pdv_pic_dict["pdv_timeseries"], nino34_pic_dict["nino34_timeseries"]
                )[0, 1]
            )
            pdv_pic_dict["pdv_timeseries"] *= -1

        # Create the TIG data array
        tig_data_pic = np.zeros((len(naoo_pic_seas), 6))
        tig_data_pic[:, 0] = (
            tna_pic_dict["tna_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_pic[:, 1] = pnaa_pic_seas
        tig_data_pic[:, 2] = (
            nino34_pic_dict["nino34_timeseries"]
            .resample(time="QS-DEC", keep_attrs=True)
            .mean()[1:]
        )
        tig_data_pic[:, 3] = raw_zonal_wind_pacific_pic[1:]
        tig_data_pic[:, 4] = naoo_pic_seas
        tig_data_pic[:, 5] = (
            alt3_pic["atl3_index"].resample(time="QS-DEC", keep_attrs=True).mean()[1:]
        )

        # EOF sign flip. Check correlation between PNA and Nino34, and flip PNA if necessary
        if (np.corrcoef(tig_data_pic[:, 1], tig_data_pic[:, 2])[0, 1]) < 0:
            tig_data_pic[:, 1] *= -1
        if (np.corrcoef(tig_data_pic[:, 1], tig_data_pic[:, 4])[0, 1]) > 0:
            tig_data_pic[:, 4] *= -1

        return amv_pic_dict, pdv_pic_dict, tig_data_pic

    @staticmethod
    def make_regime_dict(amv_lp, pdv_lp, T, N):
        """
        Create a dictionary of regime masks based on the given AMV and PDV data.

        Parameters:
        amv_lp (numpy.ndarray): AMV (Atlantic Meridional Overturning Circulation) data.
        pdv_lp (numpy.ndarray): PDV (Pacific Decadal Variability) data.
        T (int): Number of time steps.
        N (int): Number of grid points.

        Returns:
        dict: A dictionary containing the following regime masks:
            - '$PDV+ | AMV+$': Mask for positive PDV and positive AMV.
            - '$PDV+ | AMV-$': Mask for positive PDV and negative AMV.
            - '$PDV- | AMV+$': Mask for negative PDV and positive AMV.
            - '$PDV- | AMV-$': Mask for negative PDV and negative AMV.
        """
        # Create empty arrays for the masks
        pic_amvplus_mask = np.zeros((T, N))
        pic_amvminus_mask = np.zeros((T, N))
        pic_pdvplus_mask = np.zeros((T, N))
        pic_pdvminus_mask = np.zeros((T, N))

        pic_pdvplus_amvplus_mask = np.zeros((T, N))
        pic_pdvplus_amvminus_mask = np.zeros((T, N))
        pic_pdvminus_amvplus_mask = np.zeros((T, N))
        pic_pdvminus_amvminus_mask = np.zeros((T, N))
        #####################################################

        # Fix any NaN values in the AMV and PDV data
        maskingarr_picamv = utils.fix_nans(amv_lp)
        maskingarr_picpdv = utils.fix_nans(pdv_lp)

        # Create masks based on AMV values
        amvplus_pic = np.where(maskingarr_picamv < 0, 1, 0)
        amvminus_pic = np.where(maskingarr_picamv > 0, 1, 0)

        # Create masks based on PDV values
        pdvplus_pic = np.where(maskingarr_picpdv < 0, 1, 0)
        pdvminus_pic = np.where(maskingarr_picpdv > 0, 1, 0)

        # Create masks based on PDV and AMV combinations
        pdvplus_amvplus = np.where((pdvplus_pic + amvplus_pic == 0), 0, 1)  # PDV+/AMV+
        pdvplus_amvminus = np.where(
            (pdvplus_pic + amvminus_pic == 0), 0, 1
        )  # PDV+/AMV-
        pdvminus_amvplus = np.where(
            (pdvminus_pic + amvplus_pic == 0), 0, 1
        )  # PDV-/AMV+
        pdvminus_amvminus = np.where(
            (pdvminus_pic + amvminus_pic == 0), 0, 1
        )  # PDV-/AMV-

        # Assign the masks to the corresponding arrays
        for i in range(len(pic_amvplus_mask[0, :])):
            pic_amvplus_mask[:, i] = amvplus_pic
            pic_amvminus_mask[:, i] = amvminus_pic

            pic_pdvplus_mask[:, i] = pdvplus_pic
            pic_pdvminus_mask[:, i] = pdvminus_pic

            pic_pdvplus_amvplus_mask[:, i] = pdvplus_amvplus
            pic_pdvplus_amvminus_mask[:, i] = pdvplus_amvminus
            pic_pdvminus_amvplus_mask[:, i] = pdvminus_amvplus
            pic_pdvminus_amvminus_mask[:, i] = pdvminus_amvminus

        # Create a dictionary of regime masks
        regime_masks_dict = {
            "$PDV+ | AMV+$": pic_pdvplus_amvplus_mask,
            "$PDV+ | AMV-$": pic_pdvplus_amvminus_mask,
            "$PDV- | AMV+$": pic_pdvminus_amvplus_mask,
            "$PDV- | AMV-$": pic_pdvminus_amvminus_mask,
        }

        return regime_masks_dict

    @staticmethod
    def run_regime_masked_pcmciplus(
        regime,
        exp_data,
        var_names,
        regime_masks_dict,
        min_tau,
        max_tau,
        pc_alpha,
        figs_dir,
        scenario_id,
        link_assumption=True,
    ):
        """
        Run PCMCI+ algorithm with regime masked data and plot results.

        Args:
            regime (str): The regime name.
            exp_data (numpy.ndarray): The experimental data.
            var_names (list): List of variable names.
            regime_masks_dict (dict): Dictionary of regime masks.
            min_tau (int): Minimum time lag.
            max_tau (int): Maximum time lag.
            pc_alpha (float): Significance level for partial correlation test.
            figs_dir (str): Directory to save the figures.
            scenario_id (str): The scenario ID.
            link_assumption (bool): Whether to use link assumptions.

        Returns:
            None
        """
        x = np.array([110, 10, 5, 18, 117, 130])  # degree to the east
        y = np.array(
            [13.5, 52, 0, 14, 52, 0]
        )  # degree to the south (therefore negative)
        # now the dictionary node_pos is defined
        node_pos = {
            "x": x,
            "y": y,
            "transform": ccrs.PlateCarree(central_longitude=210),
        }

        yrs_per_regime = np.count_nonzero(regime_masks_dict[regime][1000:2000, 0] == 0)
        T, N = exp_data[:, :].shape
        # Create a dataframe
        dataframe = DataFrame(
            exp_data, var_names=var_names, mask=regime_masks_dict[regime][1000:2000]
        )
        title = regime + " (" + str(yrs_per_regime) + " seasons)"
        # Run PCMCI+ algorithm
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(mask_type="y"))
        # Build link assumptions
        if link_assumption == True:
            link_assumptions_absent_link_means_no_knowledge = {
                3: {(2, 0): "<?-"},
                2: {(3, 0): "-?>"},
            }  # Nino34 --> PWCu
            test_assumption = pcmci.build_link_assumptions(
                link_assumptions_absent_link_means_no_knowledge,
                len(var_names),
                max_tau,
                0,
            )

            test_assumption[2][0, 0] = "<?-"  # Nino34 --> TNA
            test_assumption[0][2, 0] = "-?>"  # Nino34 --> TNA
            for tau in range(0, max_tau + 1):
                test_assumption[4].pop((5, -tau))  # No NAO --> ATL3
                test_assumption[5].pop((4, -tau))  # No NAO --> ATL3
            for tau in range(0, max_tau + 1):
                test_assumption[0].pop((5, -tau))  # No ATL3 --> TNA
                test_assumption[5].pop((0, -tau))  # No ATL3 --> TNA
            for tau in range(0, max_tau + 1):
                test_assumption[1].pop((5, -tau))  # No ATL3 --> PNA
                test_assumption[5].pop((1, -tau))  # No ATL3 --> PNA
            for tau in range(0, max_tau + 1):
                test_assumption[3].pop((1, -tau))  # PNA  --> PWC
                test_assumption[1].pop((3, -tau))  # PNA  --> PWC
        else:
            test_assumption = None
        # Run PCMCI+ algorithm
        results = pcmci.run_pcmciplus(
            tau_min=min_tau,
            tau_max=max_tau,
            pc_alpha=pc_alpha,
            link_assumptions=test_assumption,
        )
        # Plot timeseries
        tp.plot_timeseries(dataframe=dataframe, grey_masked_samples="data")
        plt.show()
        # Plot lagged p-matrix
        lag_p_matrix = tp.setup_matrix(
            N=N,
            tau_max=max_tau,
            minimum=0.0,
            maximum=1,
            figsize=(10, 10),
            x_base=1,
            y_base=0.1,
            tick_label_size=8,
            lag_units="seasons",
            plot_gridlines=True,
            var_names=var_names,
            legend_fontsize=15,
            label_space_left=0.2,
            label_space_top=0.05,
        )
        lag_p_matrix.add_lagfuncs(val_matrix=results["p_matrix"], color="black")
        lag_p_matrix.add_lagfuncs(
            val_matrix=(
                np.where(results["p_matrix"] <= pc_alpha, results["p_matrix"], None)
            ),
            color="red",
        )
        # plot the PCMCI+ results (causal network)
        fig = plt.figure(figsize=(12, 15))
        ax = plt.axes(
            projection=ccrs.PlateCarree(central_longitude=210.0)
        )  # again the projection is defined for the axes, so that node_pos combines with the map
        ax.set_extent(
            [-10, 150, -20, 67], crs=ccrs.PlateCarree(central_longitude=210.0)
        )  #
        # Additional features from cartopy
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, alpha=0.3)
        ax.set_title(title, size=20)
        # Plot the graph using Tigramite's plot_graph function
        tp.plot_graph(
            fig_ax=(fig, ax),
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            var_names=var_names,
            vmin_edges=-1,
            vmax_edges=1,
            vmin_nodes=-1,
            vmax_nodes=1,
            node_pos=node_pos,
            cmap_nodes="RdBu_r",
            node_size=11,
            arrowhead_size=12,
            curved_radius=0.23,
            node_label_size=19,
            link_label_fontsize=17,
            node_colorbar_label="auto-MCI",
            link_colorbar_label="cross-MCI",
            save_name=figs_dir + f"picontrol_scenario_{scenario_id}_causal_graph.png",
            figsize=(8, 4),
            show_colorbar=False,
        )
        plt.show()

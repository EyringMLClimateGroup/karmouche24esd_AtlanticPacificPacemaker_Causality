# This file contains utility functions for the karmouche24esd_AtlanticPacific_Pacemaker package.
# These are called in jupyter notebooks to calculate indices and perform other operations. 
# The jupyter notebooks are used to generate the figures and tables in the paper: 
# Karmouche, S., Galytska, E., Meehl, G. A., Runge, J., Weigel, K., and Eyring, V.: 
# Changing effects of external forcing on Atlantic-Pacific interactions, EGUsphere [preprint], 
# https://doi.org/10.5194/egusphere-2023-1861, 2023.
# Author: Soufiane Karmouche
# ## Description
# 
# Indices
# We use the following indices from observational and reanalyses datasets:
# 
# 
# | **Index**     | **Definition**                                                        | **Region**                  | **Dataset**                                      |
# |---------------|-------------------------------------------------------|---------------------------------------------|------------------------------|
# | **TNA**       | Area-weighted monthly SSTAs over the North Tropical Atlantic region   | 5.5–23.5°N, 58°–15°W        | HadISST                      |
# | **Niño3.4**   | Area-weighted monthly SSTAs over the equatorial Pacific region        | 5°N–5°S, 170°–120°W         | HadISST                      |
# | **PNA**       | Leading EOF of (3-monthly averaged and area-weighted) SLP             | 20–85°N, 120°E–120°W        | NCEP-NCAR-R1                 |
# |                  anomalies over the Pacific North America region                      | 20–85°N, 120°E–120°W        | NCEP-NCAR-R1                 |
# | **NAO**       | Leading EOF of (3-monthly averaged and area-weighted) SLP anomalies   | 20–80°N, 90°W–40°E          | NCEP-NCAR-R1                 |
# |                  over the North Atlantic region                                       | 20–80°N, 90°W-40°E          | NCEP-NCAR-R1                 |
# | **PWCu**      | Monthly zonal wind anomaly at 925 hPa over the equatorial Pacific     | 6°N–6°S, 180°-150°E         | NCEP-NCAR-R1                 |
# | **ATL3**      | Area-weighted monthly SSTAs over the equatorial Atlantic region       | 3°N–3°S, 20°W–0°            | HadISST                      |
# | AMV           | Monthly SSTAs averaged over the North Atlantic region                 | 0–60°N, 80–0°W              | HadISST                      |
# | PDV           | PC associated with 1st EOF of area-weighted monthly SSTAs over the    | 20–70°N, 110°E–100°W        | HadISST                      |
# |                  North Pacific region                                                 | 20–70°N, 110°E–100°W        | HadISST                      |


# Import necessary libraries
import numpy as np  
import xarray as xr 
from scipy import stats
from eofs.xarray import Eof


class Utils:
    """
    A utility class containing various static methods for data processing and analysis.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def wgt_areaave(indat, latS, latN, lonW, lonE):
        """
        Calculate the area-weighted average of a given input dataset within a specified latitude and longitude range.

        Parameters:
        - indat (xarray.Dataset): The input dataset containing latitude and longitude coordinates.
        - latS (float): The southernmost latitude of the desired area.
        - latN (float): The northernmost latitude of the desired area.
        - lonW (float): The westernmost longitude of the desired area.
        - lonE (float): The easternmost longitude of the desired area.

        Returns:
        - odat (xarray.Dataset): The area-weighted average of the input dataset within the specified latitude and longitude range.
        """
        # Get latitude and longitude coordinates from the input dataset
        lat = indat.lat
        lon = indat.lon

        # Adjust longitude values if necessary
        if ((lonW < 0) or (lonE < 0)) and (lon.values.min() > -1):
            anm = indat.assign_coords(lon=((lon + 180) % 360 - 180))
            lon = (lon + 180) % 360 - 180
        else:
            anm = indat

        # Select latitude and longitude values within the specified range
        iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
        iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

        # Calculate the weights based on latitude values
        wgt = np.cos(np.deg2rad(lat))

        # Calculate the area-weighted average within the specified range
        odat = anm.sel(lat=iplat, lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
        return odat
    
    @staticmethod
    def interpolate_dataset(ds, target_ds):
        """
        Interpolate a dataset to the grid of another dataset."""
        return ds.interp(lon=target_ds["lon"], lat=target_ds["lat"], method="nearest")
    
    @staticmethod
    def extract_reg_timeseries1(
        global_ds, minlat, maxlat, minlon, maxlon, seasonal=False, ds_em=None
    ): 
        """
        Extract regional time series from a global dataset.
        
        Parameters:
        - global_ds (xarray.Dataset): The global dataset containing latitude and longitude coordinates.
        - minlat (float): The southernmost latitude of the desired area.
        - maxlat (float): The northernmost latitude of the desired area.
        - minlon (float): The westernmost longitude of the desired area.
        - maxlon (float): The easternmost longitude of the desired area.
        - seasonal (bool, optional): Flag to calculate seasonal averages. Default is False.
        - ds_em (xarray.Dataset, optional): Ensemble mean of the global dataset.

        Returns:
        - region_data (xarray.Dataset): The regional time series extracted from the global dataset.

        """
        multidarray=global_ds.copy()
        if ds_em is not None:
            ensmean_ua=ds_em.copy()
            multidarray.values=np.subtract(multidarray.values,ensmean_ua.values)
        climatologymultid = multidarray.groupby("time.month").mean("time")
        multidarray = multidarray.groupby("time.month") - climatologymultid 
        # Resample data to seasonal frequency
        if seasonal:
            multidarray = multidarray.resample(time="QS-DEC", keep_attrs=True).mean()

        onedarr = Utils.wgt_areaave(multidarray, minlat, maxlat, minlon, maxlon)
        return multidarray, onedarr

    @staticmethod
    def low_pass_weights(window, cutoff):
        """Calculate weights for a low pass Lanczos filter.

        Args:

        window: int
            The length of the filter window.

        cutoff: float
            The cutoff frequency in inverse time steps.

        """
        order = ((window - 1) // 2 ) + 1
        nwts = 2 * order + 1
        w = np.zeros([nwts])
        n = nwts // 2
        w[n] = 2 * cutoff
        k = np.arange(1., n)
        sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
        firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
        w[n-1:0:-1] = firstfactor * sigma
        w[n+1:-1] = firstfactor * sigma
        return w[1:-1]

    @staticmethod
    def calculate_low_pass(data, window):
        weights = Utils.low_pass_weights(window, 1.0 / 121.0)
        data_lp = (
            data.rolling(time=len(weights), center=True)
            .construct("window")
            .dot(xr.DataArray(weights, dims=["window"]))
        )
        return data_lp

    @staticmethod
    def calculate_nao(
        ds_psl, psl_em=None, period=["1949-01-01", "2014-12-01"], seasonal=True
    ): 
        """
        Calculate the North Atlantic Oscillation (NAO) index.

        Parameters:
        - ds_psl (xarray.Dataset): Dataset containing the sea level pressure data.
        - psl_em (xarray.Dataset, optional): Dataset containing the ensemble mean sea level pressure data. Default is None.
        - period (list, optional): List of two strings representing the start and end dates of the period to calculate NAO. Default is ["1949-01-01", "2014-12-01"].
        - seasonal (bool, optional): Flag indicating whether to calculate NAO on a seasonal basis. Default is True.

        Returns:
        - naooo (xarray.DataArray): NAO index.
        - eofs (xarray.DataArray): Empirical Orthogonal Function (EOF) of the NAO index.
        """
        global_psl = ds_psl.copy()
        if psl_em is not None:
            ensmean_psl = psl_em.copy()
            global_psl.values = np.subtract(global_psl.values, ensmean_psl.values)
        climatologymultid = global_psl.groupby("time.month").mean("time")
        global_psl = global_psl.groupby("time.month") - climatologymultid
        if seasonal:
            global_psl = global_psl.assign_coords(
                lon=(((global_psl.lon + 180) % 360) - 180)
            )
            global_psl = global_psl.roll(
                lon=int(len(global_psl["lon"]) / 2), roll_coords=True
            )
            naoarea = global_psl.sel(lat=slice(20, 80), lon=slice(-90, 40))
            naoarea = naoarea.resample(time="QS-DEC", keep_attrs=True).mean()
        coslat = np.cos(np.deg2rad(naoarea.lat.values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(naoarea.sel(time=slice(period[0], period[1])), weights=wgts)
        eofs = solver.eofsAsCorrelation(neofs=1)
        naooo = solver.pcs(npcs=1, pcscaling=1)
        if xr.corr(naooo[:, 0], naoarea.sel(time=slice(period[0], period[1]))) < -0.5:
            naooo = naooo[:, 0] * -1
            eofs = eofs[0] * -1
        else:
            naooo = naooo[:, 0]
            eofs = eofs[0]
        return naooo, eofs

    @staticmethod
    def calculate_pna(
        ds_psl, psl_em=None, period=["1949-01-01", "2014-12-01"], seasonal=True
    ):
        """
        Calculate the Pacific-North American (PNA) index.

        Parameters:
        - ds_psl (xarray.Dataset): Dataset containing the sea level pressure data.
        - psl_em (xarray.Dataset, optional): Dataset containing the ensemble mean sea level pressure data. Default is None.
        - period (list, optional): List of two strings representing the start and end dates of the period to calculate PNA. Default is ["1949-01-01", "2014-12-01"].
        - seasonal (bool, optional): Flag indicating whether to calculate PNA on a seasonal basis. Default is True.

        Returns:
        - pnaaa (xarray.DataArray): PNA index.
        - eofs (xarray.DataArray): Empirical Orthogonal Function (EOF) of the PNA index.
        """
        global_psl = ds_psl.copy()

        if psl_em is not None:
            ensmean_psl = psl_em.copy()
            global_psl.values = np.subtract(global_psl.values, ensmean_psl.values)

        climatologymultid = global_psl.groupby("time.month").mean("time")
        global_psl = global_psl.groupby("time.month") - climatologymultid

        if seasonal:
            pnaarea = global_psl.sel(lat=slice(20, 85), lon=slice(120, 240))
            pnaarea = pnaarea.resample(time="QS-DEC", keep_attrs=True).mean()

        coslat = np.cos(np.deg2rad(pnaarea.lat.values))
        wgts = (np.sqrt(coslat))[..., np.newaxis]
        solver = Eof(pnaarea.sel(time=slice(period[0], period[1])), weights=wgts)

        eofs = solver.eofsAsCorrelation(neofs=1)
        pnaaa = solver.pcs(npcs=1, pcscaling=1)

        if xr.corr(pnaaa[:, 0], pnaarea.sel(time=slice(period[0], period[1]))) < -0.5:
            pnaaa = pnaaa[:, 0] * -1
            eofs = eofs[0] * -1
        else:
            pnaaa = pnaaa[:, 0]
            eofs = eofs[0]

        return pnaaa, eofs

    @staticmethod
    def calculate_slp_grad(ds_psl, psl_em=None, seasonal=True):
        """
        Calculate the sea level pressure (SLP) gradient between the Indo-Pacific and Pacific regions.

        Parameters:
        - ds_psl (xarray.Dataset): Dataset containing the sea level pressure data.
        - psl_em (xarray.Dataset, optional): Dataset containing the ensemble mean sea level pressure data. Default is None.
        - seasonal (bool, optional): Flag indicating whether to calculate the gradient on a seasonal basis. Default is True.

        Returns:
        - psl_grad (xarray.DataArray): Sea level pressure gradient between the Indo-Pacific and Pacific regions.
        """
        global_psl = ds_psl.copy()

        if psl_em is not None:
            global_psl.values = np.subtract(global_psl.values, psl_em.values)

        climatologymultid = global_psl.groupby("time.month").mean("time")
        global_psl = global_psl.groupby("time.month") - climatologymultid

        if seasonal:
            global_psl = global_psl.resample(time="QS-DEC", keep_attrs=True).mean()

        raw_psl_indo = Utils.wgt_areaave(global_psl, -5, 5, 100, 160)

        raw_psl_pac = Utils.wgt_areaave(global_psl, -5, 5, 200, 260)

        psl_grad = xr.zeros_like(raw_psl_pac)
        psl_grad.values = raw_psl_pac.values - raw_psl_indo.values

        return psl_grad

    @staticmethod
    def calculate_amv(
        ds_sst, sst_em=None, remove_gm=True, period=["1950-01-01", "2021-12-01"]
    ):
        """
        Calculate the Atlantic Multidecadal Variability (AMV) using sea surface temperature (SST) data.

        Parameters:
        - ds_sst (xarray.Dataset): Dataset containing sea surface temperature data.
        - sst_em (xarray.DataArray, optional): Multi Ensemble mean (MEM) of sea surface temperature data. Default is None.
        - remove_gm (bool, optional): Flag to remove the global mean from the AMV calculation. Default is True.
        - period (list, optional): Time period for the AMV calculation. Default is ["1950-01-01", "2021-12-01"].

        Returns:
        - amv_dict (dict): Dictionary containing the AMV timeseries, low-pass filtered AMV timeseries,
          AMV pattern, and low-pass filtered AMV pattern.

        """
        amv_dict = {}
        global_sst = ds_sst.copy() - 273.15
        global_sst.values = np.where(global_sst.values < -1.8, -1.8, global_sst.values)
        if sst_em is not None:
            global_sst.values = np.subtract(global_sst.values, sst_em.values)

        climatologymultid = global_sst.groupby("time.month").mean("time")
        global_sst = global_sst.groupby("time.month") - climatologymultid

        noratlmean = Utils.wgt_areaave(global_sst, 0, 60, 280, 360)

        area6070 = global_sst.sel(lat=slice(-60, 70))
        area6070_mean = Utils.wgt_areaave(area6070, -60, 70, 0, 360)
        if remove_gm:
            amoo = noratlmean - area6070_mean
        else:
            amoo = noratlmean

        sstanom = global_sst - area6070_mean.broadcast_like(global_sst)
        amo_pattern = xr.corr(amoo.sel(time=slice(period[0], period[1])), sstanom, dim="time")
        amoo_lp = amoo.rolling(time=121, center=True).mean()
        amo_lp_pattern = xr.corr(amoo_lp.sel(time=slice(period[0], period[1])), sstanom.rolling(time=121, center=True).mean(), dim="time")
        amv_dict["amv_timeseries"] = amoo
        amv_dict["amv_timeseries_lp"] = amoo_lp
        amv_dict["amv_pattern"] = amo_pattern
        amv_dict["amv_pattern_lp"] = amo_lp_pattern
        return amv_dict

    @staticmethod
    def calculate_pdv(
        ds_sst, sst_em=None, remove_gm=True, period=["1950-01-01", "2021-12-01"]
    ): 
        """
        Calculate the Pacific Decadal Variability (PDV) using sea surface temperature (SST) data.

        Parameters:
        - ds_sst (xarray.Dataset): Sea surface temperature data.
        - sst_em (xarray.Dataset, optional): Ensemble mean of sea surface temperature data. Default is None.
        - remove_gm (bool, optional): Flag to remove the global mean 60°S and 70°N from the SST data. Default is True.
        - period (list, optional): Time period for analysis in the format ["start_date", "end_date"]. Default is ["1950-01-01", "2021-12-01"].

        Returns:
        - pdv_dict (dict): Dictionary containing the PDV timeseries and pattern.

        """
        pdv_dict = {}
        global_sst = ds_sst.copy() - 273.15
        global_sst.values = np.where(global_sst.values < -1.8, -1.8, global_sst.values)
        if sst_em is not None:
            ensmean_sst = sst_em.copy()
            global_sst.values = np.subtract(global_sst.values, ensmean_sst.values)
        climatologymultid = global_sst.groupby("time.month").mean("time")
        global_sst = global_sst.groupby("time.month") - climatologymultid

        norpac = global_sst.sel(lat=slice(20,70),lon=slice(110,260))

        if remove_gm:
            area6070 = global_sst.sel(lat=slice(-60, 70))
            weights = np.cos(np.deg2rad(area6070.lat))
            weighted_area6070 = area6070.weighted(weights)
            area6070_mean = weighted_area6070.mean(dim=["lat", "lon"])
            norpac = norpac - area6070_mean

        weights = np.cos(np.deg2rad(norpac.lat))
        weighted_norpac = norpac.weighted(weights)
        norpacmean = weighted_norpac.mean(dim=["lat", "lon"])

        coslat = np.cos(np.deg2rad(norpac.lat.values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(norpac.sel(time=slice(period[0], period[1])), weights=wgts)
        eof1 = solver.eofsAsCorrelation(neofs=1)
        pc1 = solver.pcs(npcs=1, pcscaling=1)
        if xr.corr(pc1[:, 0], norpacmean.sel(time=slice(period[0], period[1]))) < 0:
            pdoo = pc1[:, 0] * -1
            pdv_dict["pdv_timeseries"] = pdoo
            pdv_dict["pdv_pattern"] = eof1[0] * -1
        else:
            pdoo = pc1[:, 0]
            pdv_dict["pdv_timeseries"] = pdoo
            pdv_dict["pdv_pattern"] = eof1[0]

        return pdv_dict

    @staticmethod
    def calculate_tna(ds_sst, sst_em=None, remove_gm=True, period=["1950-01-01", "2021-12-01"]):
        """
        Calculate the Tropical North Atlantic (TNA) index.

        Parameters:
        - ds_sst (xarray.Dataset): Sea surface temperature dataset.
        - sst_em (xarray.Dataset, optional): Ensemble mean of sea surface temperature dataset.
        - remove_gm (bool, optional): Flag to remove the global mean. Default is True.
        - period (list, optional): Time period for analysis. Default is ["1950-01-01", "2021-12-01"].

        Returns:
        - tnaaa_dict (dict): Dictionary containing the TNA index and related variables.
        """

        tnaaa_dict = {}
        global_sst = ds_sst.copy()  # .sel(lat=slice(-60,70))#, lon=slice(110,260))
        global_sst = global_sst - 273.15  # .sel(lat=slice(-60,70))#, lon=slice(110,260))
        global_sst.values = np.where(
            global_sst.values < -1.8, -1.8, global_sst.values
        )  # below -1.8 to -1.8

        if sst_em is not None:
            ensmean_sst = sst_em.copy()
            global_sst.values = np.subtract(global_sst.values, ensmean_sst.values)

        climatologymultid = global_sst.groupby("time.month").mean("time")
        global_sst = global_sst.groupby("time.month") - climatologymultid

        noratlmean = Utils.wgt_areaave(global_sst, 5.5, 23.5, 302, 345)
        area6070_mean = Utils.wgt_areaave(global_sst, -60, 70, 0, 360)
        sstanom = global_sst

        if remove_gm == True:
            tnaaa = noratlmean - area6070_mean
        else:
            tnaaa = noratlmean

        sstanom = sstanom - area6070_mean.broadcast_like(sstanom)

        tnaasd = (tnaaa / np.std(tnaaa)).sel(time=slice(period[0], period[1]))

        tnaaa_pattern = xr.corr(tnaasd, sstanom, dim="time")
        tnaaa_pattern_reg = (
            xr.cov(tnaasd, sstanom, dim="time") / tnaasd.var(dim="time", skipna=True).values
        )

        tnaaa_lp = tnaaa.rolling(time=21, center=True).mean()
        tnaasd_lp = (tnaaa_lp / np.std(tnaaa_lp)).sel(time=slice(period[0], period[1]))

        tnaaa_lp_pattern = xr.corr(
            (tnaaa_lp / np.std(tnaaa_lp)).sel(time=slice(period[0], period[1])),
            sstanom.rolling(time=21, center=True).mean(),
            dim="time",
        )
        tnaaa_pattern_reg_lp = (
            xr.cov(tnaasd_lp, sstanom.rolling(time=21, center=True).mean(), dim="time")
            / tnaasd_lp.var(dim="time", skipna=True).values
        )

        tnaaa_dict["sst_anom"] = sstanom
        tnaaa_dict["glb_mean"] = area6070_mean
        tnaaa_dict["tna_timeseries"] = tnaaa
        tnaaa_dict["tna_timeseries_lp"] = tnaaa_lp
        tnaaa_dict["tna_pattern"] = tnaaa_pattern
        tnaaa_dict["tna_pattern_lp"] = tnaaa_lp_pattern
        tnaaa_dict["tnaaa_pattern_reg"] = tnaaa_pattern_reg
        tnaaa_dict["tnaaa_pattern_reg_lp"] = tnaaa_pattern_reg_lp

        return tnaaa_dict

    @staticmethod
    def calculate_nino34(ds_sst, sst_em=None, remove_gm=True):
        """
        Calculate the NINO3.4 index.

        Parameters:
        - ds_sst (xarray.Dataset): Sea surface temperature dataset.
        - sst_em (xarray.Dataset, optional): Ensemble mean of sea surface temperature dataset.
        - remove_gm (bool, optional): Flag to remove the global mean. Default is True.

        Returns:
        - nino34_dict (dict): Dictionary containing the NINO3.4 index and related variables.
        """

        # Copy the dataset and convert temperature units
        global_sst = ds_sst.copy()
        global_sst = global_sst - 273.15
        global_sst.values = np.where(global_sst.values < -1.8, -1.8, global_sst.values)

        # Subtract ensemble mean if provided
        if sst_em is not None:
            global_sst.values = np.subtract(global_sst.values, sst_em.values)

        # Calculate the climatology
        climatology = global_sst.groupby("time.month").mean("time")
        global_sst = global_sst.groupby("time.month") - climatology

        # Calculate the NINO3.4 index
        nino34_region_mean = Utils.wgt_areaave(global_sst, -5, 5, 190, 240)
        global_mean = Utils.wgt_areaave(global_sst, -60, 70, 0, 360)
        sstanom = global_sst

        if remove_gm:
            nino34 = nino34_region_mean - global_mean
        else:
            nino34 = nino34_region_mean

        nino34_lp = nino34.rolling(time=21, center=True).mean()

        nino34_dict = {
            "sst_anom": sstanom,
            "global_mean": global_mean,
            "nino34_timeseries": nino34,
            "nino34_timeseries_lp": nino34_lp
        }

        return nino34_dict

    @staticmethod
    def calculate_ATL3_index(ds_sst, sst_em=None, remove_gm=True, period=["1950-01-01", "2021-12-01"]):
        """
        Calculate the ATL3 index.

        Parameters:
        - ds_sst (xarray.Dataset): Input dataset containing sea surface temperature data.
        - sst_em (xarray.Dataset, optional): Ensemble mean dataset. If provided, the ensemble mean will be subtracted from the input dataset.
        - remove_gm (bool, optional): Flag indicating whether to remove the global mean from the input dataset. Default is True.
        - period (list, optional): List containing the start and end dates of the time period to consider. Default is ["1950-01-01", "2021-12-01"].

        Returns:
        - atl3_dict (dict): Dictionary containing the ATL3 index.

        """
        atl3_dict = {}
        global_sst = ds_sst.copy() - 273.15
        global_sst.values = np.where(global_sst.values < -1.8, -1.8, global_sst.values)

        if sst_em is not None:
            global_sst.values = np.subtract(global_sst.values, sst_em.values)

        climatologymultid = global_sst.groupby("time.month").mean("time")
        global_sst = global_sst.groupby("time.month") - climatologymultid

        atl3_index = Utils.wgt_areaave(global_sst, -3, 3, 340, 360).sel(time=slice(period[0], period[1]))

        atl3_dict["atl3_index"] = atl3_index

        return atl3_dict

    @staticmethod
    def detrend_kw(data): 
        """
        Detrends the given data using linear regression.

        Parameters:
        data (array-like): The input data to be detrended.

        Returns:
        array-like: The detrended data.

        """
        reg = stats.linregress(range(0, len(data)), data)
        detr_data = data - (reg[1] + (reg[0] * range(0, len(data))))
        return detr_data
    
    @staticmethod
    def fix_nans(maskaaaray):
        '''Analysis
        '''
        maskaaar=maskaaaray.copy()
        start_index = 0
        while start_index < len(maskaaar) and np.isnan(maskaaar[start_index]):
            start_index += 1

        # Find the index where consecutive NaNs start at the end
        end_index = len(maskaaar) - 1
        while end_index >= 0 and np.isnan(maskaaar[end_index]):
            end_index -= 1

        # Replace consecutive NaNs at the beginning with the first non-NaN value
        maskaaar[:start_index] = maskaaar[start_index]

        # Replace consecutive NaNs at the end with the last non-NaN value
        maskaaar[end_index + 1:] = maskaaar[end_index]
        return maskaaar

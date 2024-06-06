# **Changing effects of external forcing on Atlantic-Pacific interactions**

# Project Overview
 This repository contains the code used for the analyses and results associated with the paper:
> Karmouche, S., Galytska, E., Meehl, G. A., Runge, J., Weigel, K., and Eyring, V.: Changing effects of external forcing on Atlantic-Pacific interactions, Earth Syst. Dynam. [Accepted], 2024.

**Author:** Soufiane Karmouche, sou_kar@uni-bremen.de \
**Year:** 2024

# Requirements

The code in this repository requires specific Python packages. You can set up the required environment using Conda. Here are the steps to create the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/EyringMLClimateGroup/karmouche24esd_AtlanticPacificPacemaker_Causality
   cd karmouche24esd_AtlanticPacificPacemaker_Causality

2. Create Conda environment:
    ```bash
    conda env create -f environment.yml
    ```

3. Activate the environment:
    ```bash
    conda activate causal_esd24
    cd ..
    ```

4. **Install Tigramite Package**:  
  The Tigramite package for causal discovery is available on [GitHub](https://github.com/jakobrunge/tigramite) (last access: 29 April 2024). Please follow instructions [here](https://github.com/jakobrunge/tigramite). Version 5.2.0.0 was used in this study.
    ```bash
    git clone https://github.com/jakobrunge/tigramite.git
    cd tigramite
    python setup.py install

    ```
5. Add environment to Jupyter kernels: \
After installing Tigramite in the Conda environment, add the environment to Jupyter kernels using the following command:
    ```bash
    python -m ipykernel install --user --name=causal_esd24
    ```

# Data

### Multi-Ensemble Mean (MEM) Calculation

To isolate internal variability from the pacemaker simulations, a multi-ensemble mean (MEM) was calculated for each variable using three CMIP6 historical large ensemble means:
- CESM2 (11 members)
- MIROC6 (50 members)
- UKESM1-0-LL (16 members).

For a sample recipe to produce CMIP6 MEM using ESMValTool, please see [example_recipe_esmvaltool.yml](../main/example_recipe_esmvaltool.yml).

### Observational and Reanalyses Datasets

To calculate indices for the observed historical record, the following datasets were used at different sections of the study to extract sea surface temperature (SST), sea level pressure (PSL), eastward wind (U), wind<sub>Stress</sub>, and the depth of 20°C isotherm.
- Hadley Centre Sea Ice and Sea Surface Temperature [**(HadISST)**](http://dx.doi.org/10.1029/2002JD002670) (SST)
- National Center for Environmental Prediction-National Center for Atmospheric Research reanalysis 1  [**(NCEP-NCAR-R1)**](http://dx.doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2) (PSL, U)
- Ocean Reanalysis System version 5 [**(ORAS5)**](https://cds.climate.copernicus.eu/doi/10.24381/cds.67e8eeb7) (wind<sub>Stress</sub> , depth of 20°C isotherm)

### Pacemaker Simulations

- **CESM2 Pacific Pacemaker Ensemble Dataset:**  
  The CESM2 Pacific pacemaker ensemble dataset can be found [here](https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2.pacific.pacemaker.html) (last access: 29 April 2024). Variables: surface temperature **(TS), PSL and U**

- **CESM1 Atlantic Pacemaker Ensemble Dataset:**  
  The CESM1 Atlantic pacemaker ensemble dataset can be found [here](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.ATL-PACEMAKER.html) (last access: 29 April 2024). Variables: **TS, PSL and U**

- **Additional Information and Documentation:**  
  The complete description and documentation of the Pacific and Atlantic pacemaker datasets are available on the [Climate Variability and Change Working Group's (CVCWG) webpage](https://www.cesm.ucar.edu/working-groups/climate/simulations/) (last access: 29 April 2024).

### Pre-Industrial Control Run

A 250-year segment from the [CESM2 pre-industrial control run](http://dx.doi.org/10.1029/2019MS001916) was used. Variables: **TS, PSL and U**

For more detailed information and documentation on these datasets and simulations, please refer to the published version of the paper.

It is the user's responsibility to ensure access to the necessary datasets and simulations referenced in this study.

# Jupyter notebooks

**We recommend running the notebooks in the order below:**   

1. [section3_proof_of_concept.ipynb](../main/section3_proof_of_concept.ipynb)
    The code here is detailed. This notebook is used to reproduce the panels in **Fig 2** and respective supplementary material

In the following notebooks, the code is wrapped in seperate class functions to perform different preprocessing, analysis and plotting routines.
See [data_visualization.py](../main/data_visualization.py) and [run_causal_analysis.py](../main/run_causal_analysis.py) (PS: the classes are only used as function holders)

2. [section4_obs_reanalysis.ipynb](../main/section4_obs_reanalysis.ipynb)
    This notebook is used to reproduce the panels in **Figs 1, 3, and 4** and respective supplementary material
3. [section4_PAC_pacemaker_ensemble.ipynb](../main/section4_PAC_pacemaker_ensemble.ipynb)(to reproduce  **Fig 5**) and/or [section4_ATL_pacemaker_ensemble.ipynb](../main/section4_ATL_pacemaker_ensemble.ipynb) (to reproduce **Fig 6**)
4. [section4_PiControl.ipynb](../main/section4_PiControl.ipynb). This notebook is used to reproduce the panels in **Fig 7** and respective supplementary material.


# Results
Results from running the jupyter notebooks will be saved under the `FIGS/` directory. 

# Miscellaneous
- **Earth System Model Evaluation Tool (ESMValTool):**  
  The [ESMValTool](http://dx.doi.org/10.5194/gmd-13-1179-2020) was used for preprocessing and calculating the CMIP6 MEM and and for applying land mask on TS fields. For a sample recipe to produce the CMIP6 MEM using ESMValTool preprocessor, please see [example_recipe_esmvaltool.yml](../main/example_recipe_esmvaltool.yml).


- **Multidata-PCMCI Functionality:**  
  Details on the Multidata-PCMCI functionality can be found on Tigramite's [GitHub repository](https://github.com/jakobrunge/tigramite/blob/master/tutorials/dataset_challenges/tigramite_tutorial_multiple_datasets.ipynb) (last access: 29 April 2024).


---------------

# References

> Copernicus Climate Change Service. (2021). ORAS5 global ocean reanalysis monthly data from 1958 to present. [ECMWF](https://cds.climate.copernicus.eu/doi/10.24381/cds.67e8eeb7).

> Danabasoglu, G., et al. (2020). The Community Earth System Model Version 2 (CESM2). [Journal of Advances in Modeling Earth Systems](http://dx.doi.org/10.1029/2019MS001916).

> Kalnay, E., et al. (1996). The NCEP/NCAR 40-Year Reanalysis Project. [Bulletin of the American Meteorological Society](http://dx.doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2).

> PCMCI+: J. Runge (2020): Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. [Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence, UAI 2020,Toronto, Canada, 2019, AUAI Press, 2020.](http://auai.org/uai2020/proceedings/579_main_paper.pdf)

> Rayner, N. A., et al. (2003). Global analyses of sea surface temperature, sea ice, and night marine air temperature since the late nineteenth century. [Journal of Geophysical Research: Atmospheres](http://dx.doi.org/10.1029/2002JD002670).

> Righi, M., et al. (2020). Earth System Model Evaluation Tool (ESMValTool) v2.0 – technical overview. [Geoscientific Model Development](http://dx.doi.org/10.5194/gmd-13-1179-2020).

# License

[Apache License 2.0](../main/license.txt)

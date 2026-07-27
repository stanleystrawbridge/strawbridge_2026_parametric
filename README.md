# Quantification and statistical comparison of cell-state transition kinetics using a parametric failure-based model

Code, digitized data, and analysis outputs associated with the manuscript:

**Strawbridge SE and Fletcher AG**  
*Parametric quantification and comparison of cell-state transition kinetics using failure-based modeling.*

## Overview

This repository contains the code, digitized datasets, and analysis outputs used to fit delayed Weibull models to published cell-state transition time courses.

The repository includes:

- core fitting code for the delayed Weibull transition model
- dataset-specific analysis scripts for each application example
- digitized input data extracted from published figures
- generated fit summaries and pairwise statistical comparisons
- scripts used to generate parameter and topology figures
- supplemental tables accompanying the manuscript

## Repository structure

```text
.
├── src/
│   ├── data/
│   ├── hanna2009nature/
│   ├── leeb2014cellStemCell/
│   ├── mulas2017stemCellReports/
│   ├── parameterFigures/
│   ├── topologyModelFigures/
│   ├── analysis_hanna.py
│   ├── analysis_leeb.py
│   ├── analysis_mulas.py
│   ├── cstk_fit.py
│   ├── make_parameter_figures.py
│   └── simulate_topology_models.py
└── supplementary_tables/
```

## Contents

### `src/`

Main source directory containing code, input data, and generated outputs.

### `src/data/`

Digitized input datasets used for fitting the delayed Weibull model.

### `src/analysis_leeb.py`

Analysis of bulk RT-qPCR down-regulation kinetics during exit from naive pluripotency.  
Source study: Leeb et al. (2014).

### `src/analysis_mulas.py`

Analysis of directed differentiation kinetics from distinct embryonic stem cell starting states.  
Source study: Mulas et al. (2017).

### `src/analysis_hanna.py`

Analysis of reprogramming kinetics in monoclonal B-cell populations.  
Source study: Hanna et al. (2009).

### `src/cstk_fit.py`

Core functions for fitting the delayed Weibull transition model and extracting fitted parameters and derived timing metrics.

### `src/make_parameter_figures.py`

Script used to generate parameter-interpretation figures for the manuscript.

### `src/simulate_topology_models.py`

Script used to generate simulated transition curves and hazard profiles for illustrative network topologies.

### `src/leeb2014cellStemCell/`, `src/mulas2017stemCellReports/`, `src/hanna2009nature/`

Output folders containing fitted summaries, pairwise statistical comparisons, and generated figures for each application dataset.

### `src/parameterFigures/`

Generated output for parameter-interpretation figures.

### `src/topologyModelFigures/`

Generated output for topology simulation figures.

### `supplementary_tables/`

Supplemental tables associated with the manuscript, including digitized data, fitting summaries, and pairwise statistical comparisons.

## Reproducing the analyses

Run the dataset-specific scripts from within the `src/` directory:

```bash
python analysis_leeb.py
python analysis_mulas.py
python analysis_hanna.py
```

Additional manuscript figures can be generated with:

```bash
python make_parameter_figures.py
python simulate_topology_models.py
```

## Software requirements

The analysis code was written in Python.

Main Python dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Data provenance

The datasets in this repository were digitized from published figures, as described in the manuscript Methods.  
Each dataset remains associated with its original source publication:

- Hanna J, Saha K, Pando B, van Zon J, Lengner CJ, Creyghton MP, van Oudenaarden A, Jaenisch R. (2009). Direct cell reprogramming is a stochastic process amenable to acceleration. *Nature* 462(7273):595–601.
- Leeb M, Dietmann S, Paramor M, Niwa H, Smith A. (2014). Genetic exploration of the exit from self-renewal using haploid embryonic stem cells. *Cell Stem Cell* 14(3):385–393.
- Mulas C, Kalkan T, Smith A. (2017). NODAL secures pluripotency upon embryonic stem cell progression from the ground state. *Stem Cell Reports* 9(1):77–91.

## Supplemental tables

The `supplementary_tables/` directory contains the tables provided with the manuscript, including:

- digitized datasets
- fitting summaries
- pairwise statistical comparisons

These correspond to the supplemental tables cited in the manuscript.

## Citation

If you use this code or data, please cite the associated manuscript.

## License

This repository is released under the **GNU General Public License v3.0**.

## Contact

**Stanley E. Strawbridge**  
Centre for Stem Cell Biology, University of Sheffield  
s.strawbridge@sheffield.ac.uk

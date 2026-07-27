# Quantification and statistical comparison of cell-state transition kinetics using a parametric failure-based model

Code, digitized data, analysis outputs, and supplemental tables associated with:

**Stanley E. Strawbridge and Alexander G. Fletcher**

*Quantification and statistical comparison of cell-state transition kinetics using a parametric failure-based model.*

## Overview

This repository contains the code and digitized datasets used to quantify cell-state transition kinetics with a delayed Weibull model.

The framework describes the transitioned fraction as

\[
F(t)=
\begin{cases}
0, & t<t_0,\\
\pi\left[1-\exp\left(-\left(\frac{t-t_0}{\lambda}\right)^k\right)\right], & t\geq t_0,
\end{cases}
\]

where:

- \(t_0\) is the onset delay;
- \(\lambda\) is the characteristic transition timescale;
- \(k\) determines whether transition hazard decreases, remains constant, or increases with time;
- \(\pi\) is the fraction of the population competent to respond.

The model is intended as a compact, phenomenological description of transition timing rather than a detailed mechanistic reconstruction of the underlying regulatory network.

The repository includes:

- core delayed Weibull fitting functions;
- dataset-specific analyses for differentiation and reprogramming examples;
- digitized data extracted from published figures;
- fitted parameter summaries and pairwise statistical comparisons;
- simulations of minimal constant-, increasing-, and decreasing-hazard topologies;
- comparison of simulated first-passage distributions with delayed Weibull fits and analytic solutions;
- benchmarking against Gamma, log-normal, Gompertz, and log-logistic models;
- scripts used to generate manuscript figures;
- Supplemental Tables S1–S4.

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
│   ├── benchmark_parametric.py
│   ├── cstk_fit.py
│   ├── make_parameter_figures.py
│   └── simulate_topology_models.py
├── supplementary_tables/
│   ├── Supplemental_Table_S1.xlsx
│   ├── Supplemental_Table_S2.xlsx
│   ├── Supplemental_Table_S3.xlsx
│   └── Supplemental_Table_S4.xlsx
├── LICENSE
└── README.md
```

## Contents

### `src/cstk_fit.py`

Core implementation of the delayed Weibull fraction model.

The script contains functions for:

- evaluating the delayed Weibull cumulative distribution;
- scaling the distribution by the competent fraction \(\pi\);
- nonlinear least-squares fitting;
- fixing or estimating \(t_0\) and \(\pi\);
- estimating an approximate covariance matrix;
- calculating \(t_{10}\), \(t_{50}\), and \(t_{90}\);
- reporting \(R^2\), AIC, and BIC.

### `src/analysis_leeb.py`

Reanalysis of bulk RT-qPCR measurements during exit from naïve pluripotency.

Source:

Leeb et al. (2014), *Cell Stem Cell*.

The script analyses:

- down-regulation of naïve pluripotency-associated genes;
- the effects of Pum1 knockdown;
- fitted \(\lambda\), \(t_{50}\), \(k\), and \(\pi\);
- approximate pairwise Wald comparisons with multiple-testing correction.

For these analyses, \(t_0\) is fixed to zero and \(\pi\) is estimated.

### `src/analysis_mulas.py`

Reanalysis of directed differentiation from three mouse embryonic stem-cell starting populations:

- 2i;
- Rex1-Hi;
- Rex1-Lo.

Source:

Mulas et al. (2017), *Stem Cell Reports*.

The script analyses differentiation toward:

- primitive streak;
- neural lineage;
- lateral mesoderm;
- definitive endoderm.

For all fits, \(t_0\) is fixed to zero.

For primitive streak and neural differentiation, \(\pi\) is fixed to one.

For lateral mesoderm and definitive endoderm differentiation, \(\pi\) is estimated.

### `src/analysis_hanna.py`

Reanalysis of population-rescaled reprogramming kinetics in monoclonal pre-B-cell populations.

Source:

Hanna et al. (2009), *Nature*.

The analysed conditions include:

- parental NGFP1;
- Lin28 overexpression;
- Nanog overexpression;
- p21 knockdown;
- p53 knockdown.

For these fits, \(t_0\) is fixed to zero and \(\pi\) is fixed to one.

### `src/make_parameter_figures.py`

Generates the parameter-interpretation curves used to illustrate the effects of varying:

- \(t_0\);
- \(\lambda\);
- \(k\);
- \(\pi\).

PNG, SVG, and text summary files are written to:

```text
src/parameterFigures/
```

### `src/simulate_topology_models.py`

Simulates first-passage times for three minimal stochastic transition topologies:

1. Constant hazard:

```text
S → T
```

2. Increasing hazard:

```text
S0 → S1 → S2 → S3 → T
```

3. Decreasing hazard:

```text
S → T
S → R → T
```

The script:

- simulates stochastic first-passage times;
- calculates empirical cumulative distribution functions;
- estimates empirical hazard functions;
- fits Weibull distributions to the simulated transition times;
- compares simulations with analytic first-passage solutions;
- generates the topology panels used in Figure 2 and Supplemental Figure S1.

The topology examples are illustrative rather than uniquely mechanistic.

The delayed Weibull is exact for the constant-hazard model, provides a close approximation for the increasing-hazard model over the fitted interval, and captures the qualitative hazard regime but not the full long-time tail of the decreasing-hazard model.

Outputs are written to:

```text
src/topologyModelFigures/
```

### `src/benchmark_parametric.py`

Benchmarks the delayed Weibull model against four alternative parametric distributions:

- Gamma;
- log-normal;
- Gompertz;
- log-logistic.

All models are expressed as competent-fraction-scaled cumulative distributions with parameter counts matched within each dataset.

Models are compared using:

- small-sample-corrected Akaike information criterion, AICc;
- \(\Delta\)AICc;
- Akaike weights;
- \(R^2\).

For the Mulas datasets, the reported standard deviations are used as fitting weights.

For the Leeb and Hanna datasets, ordinary nonlinear least squares is used because the source figures provide digitized trajectories rather than underlying count-level observations.

The benchmarking outputs underpin Supplemental Table S4.

## Input data

Digitized input data are stored in:

```text
src/data/
```

The datasets were extracted from published figures using graph digitization, as described in the manuscript Methods.

The repository contains data derived from:

- Hanna et al. (2009);
- Leeb et al. (2014);
- Mulas et al. (2017).

The digitized values should be interpreted in the context of the original studies and their reported experimental designs.

## Reproducing the analyses

Clone the repository and move into the source directory:

```bash
git clone https://github.com/stanleystrawbridge/strawbridge_2026_parametric.git
cd strawbridge_2026_parametric/src
```

Run the three dataset-specific analyses:

```bash
python analysis_leeb.py
python analysis_mulas.py
python analysis_hanna.py
```

Generate the parameter-interpretation figures:

```bash
python make_parameter_figures.py
```

Generate the stochastic topology figures:

```bash
python simulate_topology_models.py
```

### Running the parametric benchmark

`benchmark_parametric.py` reads its input files from the directory specified by the `BENCH_DATA_DIR` environment variable.

From PowerShell, while in the `src` directory:

```powershell
$env:BENCH_DATA_DIR = "data"
python benchmark_parametric.py
```

From Windows Command Prompt:

```bat
set BENCH_DATA_DIR=data
python benchmark_parametric.py
```

From macOS or Linux:

```bash
BENCH_DATA_DIR=data python benchmark_parametric.py
```

Benchmark outputs are written to:

```text
benchmark_out/
```

The principal output files are:

```text
parametric_benchmark_long.csv
parametric_benchmark_wins.csv
parametric_benchmark_featured.csv
```

## Software requirements

The analyses require Python 3 and the following packages:

```text
numpy
pandas
matplotlib
scipy
```

A minimal environment can be created using:

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install numpy pandas matplotlib scipy
```

## Statistical interpretation

The available datasets consist largely of digitized summary trajectories rather than raw cell- or well-level counts.

The repository therefore uses nonlinear least squares for model fitting.

Where raw binary counts are available, a binomial or beta-binomial likelihood would generally provide a more appropriate observation model.

Approximate covariance matrices and Wald comparisons should be interpreted cautiously for sparsely sampled time courses.

In particular:

- \(k\) may be difficult to distinguish from an unresolved onset delay;
- \(\pi\) is defined relative to the observation window and measured marker;
- \(t_{50}\) refers to the competent subpopulation when \(\pi < 1\);
- a good Weibull fit does not uniquely identify the underlying molecular mechanism.

## Supplemental tables

The `supplementary_tables/` directory contains:

### Supplemental Table S1

Digitized Leeb et al. data, fitted parameters, and pairwise comparisons for pluripotency-factor down-regulation and Pum1 perturbation.

### Supplemental Table S2

Digitized Mulas et al. data, fitted parameters, and pairwise comparisons for directed differentiation from different starting states.

### Supplemental Table S3

Digitized Hanna et al. data, fitted parameters, and pairwise comparisons for reprogramming perturbations.

### Supplemental Table S4

Benchmarking of the delayed Weibull model against Gamma, log-normal, Gompertz, and log-logistic alternatives.

The table reports:

- \(\Delta\)AICc;
- \(R^2\);
- best-supported model;
- Weibull Akaike weight;
- whether AICc is defined for the available number of observations.

AICc is informative only where the number of observations is sufficient relative to the number of fitted parameters.

## Data provenance

The digitized datasets originate from the following publications:

Hanna J, Saha K, Pando B, van Zon J, Lengner CJ, Creyghton MP, van Oudenaarden A, Jaenisch R.  
2009.  
Direct cell reprogramming is a stochastic process amenable to acceleration.  
*Nature* 462:595–601.

Leeb M, Dietmann S, Paramor M, Niwa H, Smith A.  
2014.  
Genetic exploration of the exit from self-renewal using haploid embryonic stem cells.  
*Cell Stem Cell* 14:385–393.

Mulas C, Kalkan T, Smith A.  
2017.  
NODAL secures pluripotency upon embryonic stem cell progression from the ground state.  
*Stem Cell Reports* 9:77–91.

## Citation

Please cite the associated manuscript when using the code or digitized datasets:

Strawbridge SE and Fletcher AG.  
*Quantification and statistical comparison of cell-state transition kinetics using a parametric failure-based model.*

Publication details and a DOI will be added following publication.

## License

This repository is released under the GNU General Public License v3.0.

## Contact

Stanley E. Strawbridge  
Centre for Stem Cell Biology  
University of Sheffield  
s.strawbridge@sheffield.ac.uk

Alexander G. Fletcher  
School of Mathematical and Physical Sciences  
University of Sheffield  
a.g.fletcher@sheffield.ac.uk

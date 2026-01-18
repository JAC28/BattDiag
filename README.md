[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

# BattDiag - Battery Diagnostics Functions

A Python package for multi-cell battery analysis providing single-cell and cross-cell diagnostics (statistical measures, entropy metrics, anomaly detection).

**Supplementary Material**: This repository accompanies the academic paper draft *Comparative evaluation of data-based methods for detection of internal short circuits on cell-level*. The code is provided as supplementary material to ensure reproducibility and transparency. For the official manuscript, see the citation section below and the link to the paper after acceptance.

## Overview

In agreement to the original publication, the functions are classified and separated based on the underlying principles. Thus, the repository contains the following two modules:

- `src/battdiag/cellEval.py`: Single-cell metrics that return fault signal with shape `(T,N)` for voltage signals of shape `(T,N)`
  - **Inter-Cell**
    - $\Delta\mu$ -- Mean deviation from the module mean
    - z-Score -- Mean standardized difference
    - $\max\Delta U$d$t$ -- Differentiation between cell and maximum value
    - LOF -- Local outlier factor
    - LoOP -- Local outlier probability
    - ApEn -- Approximate Entropy
    - SampEn -- Sample Entropy
    - EnShanEn -- Ensemble Shannon Entropy
  - **Intra-Cell**
    - LoShanEn -- Local Shannon Entropy calculation without taking the other cells into account
- `src/battdiag/crossCellEval.py`: Cross-cell metrics that return fault signal with shape `(T,N,N)` for voltage signals of shape `(T,N)`
  - **Cross-Cell**
    - X-ApEn -- Cross-Approximate Entropy
    - X-SampEn -- Cross-Sample Entropy
    - PearCorr -- Pearson Correlation coefficient
    - ICC -- Intraclass correlation coefficient in multiple variants

All rolling functions support configurable pre-, mid-, and post-processing pipelines.

## Installation

```bash
# Clone
git clone https://github.com/JAC28/battdiag.git
cd battdiag

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

Requires Python 3.12.

## Quick Start

```python
import numpy as np
from battdiag import cellEval, crossCellEval

data = np.random.randn(1000, 5)  # 1000 time steps, 5 cells
window = 50

pearCorr = crossCellEval.numba_rolling_PearCorr(data, 100, 8,
    preProcessing="None", preParameters=List([10.0]),  
    midProcessing = "None", midParameters=List([0.1,2.0]), 
    postProcessing="zScore", postParameters=List([1.0]))
```

See `examples/` for a notebook using a sample dataset.

## Features

- Single-cell: z-Score, $\Delta\mu$, $\max\Delta U$d$t$, EnShanEn, ApEn, SampEn, LOF, LoOP
- Cross-cell: PearCorr, ICC, X-SampEn, X-ApEn
- Processing pipeline: pre (`None`, `Min-Downsample`, `Mean-Downsample`), mid (`None`, `Rectangle`), post (`None`, `zScore`)

## Testing

```bash
uv run python -m unittest discover tests -p "test_functional.py" -v
```

## Citation

If you use BattDiag in your research, please cite:

> Battery diagnostics implementations are available at: https://github.com/JAC28/battdiag (Version 0.1.0).

BibTeX:
```bibtex
@software{battdiag2026,
  author  = {Klink, Jacob},
  title   = {BattDiag: Battery Diagnostics Functions},
  year    = {2026},
  url     = {https://github.com/JAC28/battdiag},
  version = {0.1.0}
}
```

Link to the paper: *Updated after acceptance*

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Implementation Notes

This project used GitHub Copilot for preparing the developed functions for publication namely structuring the projects, generating tests to verify the transferred functions and documenting the code; all code has been reviewed and validated.

## Project Structure

- `src/battdiag/cellEval.py` - Single-cell battery metrics
- `src/battdiag/crossCellEval.py` - Cross-cell metrics and comparisons
- `tests/test_functional.py` - Functional tests (shapes, finite values)

## Dependencies

- numba: JIT compilation for performance
- numpy: Array operations
- scikit-learn: Statistical methods
- PyNomaly: Outlier detection

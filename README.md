# BattDiag - Battery Diagnosis Functions

Extracted and refactored battery diagnostics functions from EvalModupy for dissertation supplementary material.

## Project Structure

- `src/battdiag/cellEval.py` - Single-cell battery metrics
- `src/battdiag/crossCellEval.py` - Cross-cell metrics and comparisons
- `tests/test_functional.py` - Functional tests (shapes, finite values)
- `tests/test_refactor.py` - Comparison tests with original implementation (in .gitignore)

## Note on Implementation

**Setup and test generation were created with extensive use of GitHub Copilot, but carefully reviewed and validated for correctness.** All functionality has been verified against the original EvalModupy implementation.

## Testing

Run functional tests:
```bash
uv run python -m unittest discover tests -p "test_functional.py"
```

Run all tests (requires EvalModupy):
```bash
uv run python -m unittest discover tests
```

## Dependencies

- numba: JIT compilation for performance
- numpy: Array operations
- scikit-learn: Statistical methods
- antropy: Entropy calculations
- PyNomaly: Outlier detection
- rolling: Rolling window operations

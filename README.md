# Exact Discretisation and Boundary Observables in Lorentzian Causal Diamonds
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19337577.svg)](https://doi.org/10.5281/zenodo.19338306) 
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Doc License: CC BY 4.0](https://img.shields.io/badge/Doc_License-CC_BY_4.0-green.svg)](LICENSE-CC-BY.txt)

**Author:** Yannick Schmitt  
**Date:** March 2026  
**Status:** Preprint 1.0.3


## Overview

This paper studies the ternary Minkowski lattice `L = {-1, 0, +1}^4` under the metric `η = diag(-1, +1, +1, +1)` and establishes a sequence of exact geometric and algebraic results about its lightlike structure and the discrete causal diamond two-complex built from it.

Five main results are established:

1. **Lightlike enumeration** (Lemma 2.3): The lattice contains exactly 12 lightlike nearest-neighbour vectors, partitioning into two null sheets of 6 mutually spacelike channels each.
2. **D4 root identification** (Proposition 2.5): These 12 vectors are precisely the mixed roots of the D4 root system, giving the exact Minkowski partition 24 = 12 + 12.
3. **Causal diamond geometry** (Theorem 3.2): The boundary `∂D` satisfies five independent geometric conditions and spans `R^4`.
4. **Lorentzian boundary sum** (Theorem 4.2): The discrete boundary sum of a U(1) gauge field over `∂D` evaluates to `n^μ_eff = (12, 0, 0, 0)` — purely temporal.
5. **Plaquette Laplacian spectrum** (Propositions 5.4–5.5): The 21 order-4 plaquettes of `D` yield a Laplacian with exact integer spectrum `{0^(4), 6^(2), 8^(3), 10^(2), 28^(1)}` and a 4-dimensional flat-connection null space.

A leading-order U(1) lattice BF partition function is also constructed, exhibiting a finite-system crossover at `β_c ≈ 2.7364`.

## Repository Structure
* `/paper` - LaTeX source files and PDF pre-print of the manuscript.
* `/script` - Verification script

## Verification Script

All numerical claims in the paper are verified by exhaustive enumeration in `verification_ED_BO_CD_paper.py`. The script re-derives every structure from scratch — no matrix or enumeration is hardcoded — and prints a `[PASS]` or `[FAIL]` line for each check.

### What the script verifies

The script is organised into seven parts, each corresponding to a section of the paper:

| Part | Checks |
|---|---|
| 1 — Lattice classification | 81 lattice points; 12 lightlike, 66 spacelike, 2 timelike |
| 2 — D4 root system | `\|D4\| = 24`; partition 12 + 12; lightlike roots coincide with lattice |
| 3 — Causal diamond conditions | All five conditions (a)–(e) of Theorem 3.2 |
| 4 — Boundary sum | Lorentzian sum gives `(12,0,0,0)`; Riemannian sum vanishes |
| 5 — Plaquettes & Laplacian | 21 plaquettes, exact integer spectrum, flat-connection space |
| 5b–5d — CW-complex & extended complex | Obstruction in Remark 5.7; extended K_{6,6} Betti numbers |
| 6 — BF theory | `K_bdy` and `K_total` spectra; boundary coupling `2/13` |
| 7 — Character expansion | 60 compatible plaquette pairs; crossover `β_c ≈ 2.7364`, `C(β_c) ≈ 12.69` |

### Requirements

```
numpy
```

### Running the script

```bash
python verification_ED_BO_CD_paper.py
```


## Key Numerical Results

| Quantity | Value |
|---|---|
| Lightlike vectors | 12 |
| Order-4 plaquettes | 21 |
| Plaquette Laplacian spectrum | `{0^4, 6^2, 8^3, 10^2, 28^1}` |
| Flat-connection dimension | 4 |
| Effective boundary vector | `(12, 0, 0, 0)` |
| Boundary coupling | `β/13` |
| BF crossover | `β_c ≈ 2.7364` |

## Related Papers

- **Paper 2** (companion): *Algebraic Structure of the D4 Causal Diamond* — extends these results to symmetry group decomposition, CSS codes, mass renormalisation, and MOND phenomenology.
- **Paper 3**: *A Lorentzian CSS Duality in Causal Diamond Quantum Error-Correcting Codes* — derives a four-code CSS family from the incidence matrix `M`.


## Citation

If you use this work, please cite it as:

> Yannick Schmitt. (2026). Exact Discretisation and Boundary Observables in Lorentzian Causal Diamonds. Zenodo. https://doi.org/10.5281/zenodo.19338306


## License
 * The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
 * The documentation, LaTeX source files, and PDF papers are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC-BY.txt).

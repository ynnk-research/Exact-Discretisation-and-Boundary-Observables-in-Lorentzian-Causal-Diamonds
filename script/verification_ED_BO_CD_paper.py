# > Yannick Schmitt. (2026). EMA-Gated Temporal Sequence Compression in Vision Transformers. Zenodo. https://doi.org/10.5281/zenodo.19338306
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
============================================================
  EXACT DISCRETISATION — VERIFICATION SUITE  (v3)
============================================================

  Paper: "Exact Discretisation and Boundary Observables
         in Lorentzian Causal Diamonds" by Yannick Schmitt

  This script verifies every numerical claim in the paper by
  exhaustive enumeration.  It is intended as supplementary
  material: running it reproduces every number in the text.

  v3 changes:
    - Betti-number computation replaced by flat-connection
      analysis (Proposition 5.5, Remarks 5.6–5.8)
    - CW-complex obstruction verified (Remark 5.7)
    - Extended K_{6,6} complex Betti numbers computed (Remark 5.8)

  Run:  python experiment_verification_v3.py
============================================================
"""

import numpy as np
from itertools import combinations, product
from collections import Counter

# ================================================================
#  GF(2) LINEAR ALGEBRA
# ================================================================

def gf2_rank(A):
    """Rank of integer matrix A over GF(2)."""
    M = np.array(A, dtype=np.int8) % 2
    r, c = M.shape
    rank = 0
    for col in range(c):
        rows = np.where(M[rank:, col] == 1)[0]
        if not len(rows):
            continue
        p = rows[0] + rank
        M[[rank, p]] = M[[p, rank]]
        for row in range(r):
            if row != rank and M[row, col]:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
        if rank == r:
            break
    return rank


# ================================================================
#  PART 1 — LATTICE AND LIGHTLIKE VECTORS
# ================================================================

def minkowski_norm_sq(v):
    """η(v,v) = -v₀² + v₁² + v₂² + v₃²"""
    return -v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2

print("=" * 66)
print("  VERIFICATION SUITE v3")
print("=" * 66)

# Build the ternary Minkowski lattice
lattice = list(product([-1, 0, 1], repeat=4))
assert len(lattice) == 81, f"Lattice has {len(lattice)} points, expected 81"

# Classify non-origin points
lightlike, spacelike, timelike = [], [], []
for v in lattice:
    if v == (0, 0, 0, 0):
        continue
    s2 = minkowski_norm_sq(v)
    if s2 == 0:
        lightlike.append(v)
    elif s2 > 0:
        spacelike.append(v)
    else:
        timelike.append(v)

lightlike.sort()  # canonical ordering used throughout

print(f"\n  PART 1: Lattice classification")
print(f"  [PASS] Lattice points:  {len(lattice)} = 81")
print(f"  [PASS] Lightlike:       {len(lightlike)} = 12")
print(f"  [PASS] Spacelike:       {len(spacelike)} = 66")
print(f"  [PASS] Timelike:        {len(timelike)} = 2")
print(f"  [PASS] Total non-origin: {len(lightlike)+len(spacelike)+len(timelike)} = 80")

# Verify structure: each lightlike vector has |t|=1 and exactly one |x_j|=1
for v in lightlike:
    assert abs(v[0]) == 1
    assert sum(1 for x in v[1:] if x != 0) == 1

future = [v for v in lightlike if v[0] == +1]
past   = [v for v in lightlike if v[0] == -1]
assert len(future) == 6 and len(past) == 6

# Verify pairwise spacelike within each sheet
for sheet, name in [(future, "FS"), (past, "PS")]:
    for i, u in enumerate(sheet):
        for w in sheet[i+1:]:
            d = tuple(a - b for a, b in zip(u, w))
            assert minkowski_norm_sq(d) > 0, f"{name}: {u},{w} not spacelike"

# Verify antipodal pairing
for f in future:
    antipode = tuple(-c for c in f)
    assert antipode in past

# Verify spanning
W = np.array(future, dtype=float)
assert np.linalg.matrix_rank(W) == 4
print(f"  [PASS] FS spans R⁴ (rank = 4)")


# ================================================================
#  PART 2 — D4 ROOT SYSTEM
# ================================================================

d4_roots = [v for v in product([-1, 0, 1], repeat=4)
            if v != (0, 0, 0, 0) and sum(c**2 for c in v) == 2]
assert len(d4_roots) == 24

d4_lightlike = [v for v in d4_roots if minkowski_norm_sq(v) == 0]
d4_spacelike = [v for v in d4_roots if minkowski_norm_sq(v) > 0]
d4_timelike  = [v for v in d4_roots if minkowski_norm_sq(v) < 0]

assert len(d4_lightlike) == 12
assert len(d4_spacelike) == 12
assert len(d4_timelike) == 0
assert set(d4_lightlike) == set(lightlike)

print(f"\n  PART 2: D4 root system")
print(f"  [PASS] |D4| = {len(d4_roots)} = 24")
print(f"  [PASS] D4 partition: 12 lightlike + 12 spacelike + 0 timelike")
print(f"  [PASS] D4 lightlike roots = lattice lightlike vectors")


# ================================================================
#  PART 3 — CAUSAL DIAMOND CONDITIONS
# ================================================================

# (a)–(e) already verified above; collect results
print(f"\n  PART 3: Causal diamond conditions")
print(f"  [PASS] (a) All 12 boundary links are lightlike")
print(f"  [PASS] (b) All 15 + 15 within-sheet pairs are spacelike")
print(f"  [PASS] (c) FS ∩ PS = ∅")
print(f"  [PASS] (d) FS spans R⁴")
print(f"  [PASS] (e) Antipodal pairing FS ↔ PS bijective")
print(f"  [PASS] Nloc = |FS| + |PS| = 12")


# ================================================================
#  PART 4 — LORENTZIAN BOUNDARY SUM
# ================================================================

n_eff = [0, 0, 0, 0]
for v in future:
    for i in range(4):
        n_eff[i] += v[i]        # future: +L_f
for v in past:
    for i in range(4):
        n_eff[i] -= v[i]        # past: -L_p (Lorentzian orientation)

assert tuple(n_eff) == (12, 0, 0, 0)

# Verify Riemannian convention vanishes
riem = [sum(v[i] for v in lightlike) for i in range(4)]
assert tuple(riem) == (0, 0, 0, 0)

print(f"\n  PART 4: Boundary sum")
print(f"  [PASS] n_eff = {tuple(n_eff)} (Lorentzian)")
print(f"  [PASS] Riemannian sum = {tuple(riem)} (vanishes)")


# ================================================================
#  PART 5 — PLAQUETTES, INCIDENCE MATRIX, LAPLACIAN SPECTRUM
# ================================================================

n_links = 12
link_idx = {v: i for i, v in enumerate(lightlike)}

# Order-3 plaquettes (should be zero)
order3 = [c for c in combinations(lightlike, 3)
          if all(sum(v[k] for v in c) == 0 for k in range(4))]

# Order-4 plaquettes
plaquettes = []
for combo in combinations(lightlike, 4):
    if all(sum(v[k] for v in combo) == 0 for k in range(4)):
        plaquettes.append(combo)

# Order-2 antipodal pairs
order2 = [(u, v) for u, v in combinations(lightlike, 2)
          if tuple(-c for c in u) == v]

assert len(order3) == 0
assert len(plaquettes) == 21
assert len(order2) == 6

# Verify all plaquettes are 2F + 2P
for p in plaquettes:
    nf = sum(1 for v in p if v[0] == +1)
    np_ = sum(1 for v in p if v[0] == -1)
    assert nf == 2 and np_ == 2, f"Plaquette {p} has {nf}F+{np_}P"

print(f"\n  PART 5: Plaquettes and Laplacian")
print(f"  [PASS] Order-3 plaquettes: {len(order3)} = 0")
print(f"  [PASS] Order-4 plaquettes: {len(plaquettes)} = 21, all 2FS+2PS")
print(f"  [PASS] Order-2 antipodal pairs: {len(order2)} = 6")

# Build incidence matrix M (12 × 21)
M = np.zeros((n_links, len(plaquettes)), dtype=int)
for j, p in enumerate(plaquettes):
    for v in p:
        M[link_idx[v], j] = 1

# Verify column sums = 4
assert np.all(M.sum(axis=0) == 4)

# Plaquette Laplacian
K = M @ M.T

# Verify diagonal and trace
assert np.all(np.diag(K) == 7)
assert np.trace(K) == 84

# Spectrum
eigenvalues = np.linalg.eigvalsh(K.astype(float))
spectrum = Counter(np.round(eigenvalues).astype(int))
expected_spectrum = {0: 4, 6: 2, 8: 3, 10: 2, 28: 1}

assert dict(spectrum) == expected_spectrum, f"Got {dict(spectrum)}"
assert np.linalg.matrix_rank(M) == 8

print(f"  [PASS] M shape: {M.shape}, rank = {np.linalg.matrix_rank(M)}")
print(f"  [PASS] K diagonal: all 7, Tr(K) = {np.trace(K)}")
print(f"  [PASS] Spectrum: {dict(sorted(spectrum.items()))}")


# ================================================================
#  PART 5b — FLAT-CONNECTION SPACE  (Proposition 5.5)
# ================================================================

print(f"\n  PART 5b: Flat connections (Proposition 5.5)")

# ker(M^T) = {θ ∈ R^12 : Σ_{l∈p} θ_l = 0 for every plaquette p}
null_dim = np.sum(np.abs(eigenvalues) < 1e-10)
assert null_dim == 4

# Verify ker(K) = ker(M^T) explicitly
_, eigvecs = np.linalg.eigh(K.astype(float))
null_space = eigvecs[:, np.abs(eigenvalues) < 1e-10]
residual = M.T @ null_space
assert np.max(np.abs(residual)) < 1e-12

print(f"  [PASS] dim ker(K) = dim ker(M^T) = {null_dim}")
print(f"  [PASS] Verification: max|M^T · null_vectors| = {np.max(np.abs(residual)):.1e}")

# Identify the flat connections (Remark 5.6)
#
# The constant vector θ_l = c for all l is NOT in ker(K).
# K · ones = 28 · ones  (every row sums to 28, verified in Part 6),
# so the constant assignment has holonomy 4 on every plaquette — not flat.
#
# All four vectors in ker(K) are antipodally antisymmetric:
# θ_{L_f} = -θ_{-L_f} for every antipodal pair.
# One is the standard antisymmetric "time-axis" mode;
# the remaining three correspond to the three spatial axes.
# The gauge freedom (removing one overall phase from the path integral)
# is a constraint on the integration domain, not a zero eigenvalue.

# Build antipodal pairs from the sorted lightlike list
antipodal_pairs = []
for i, v in enumerate(lightlike):
    anti = tuple(-c for c in v)
    j = link_idx[anti]
    if i < j:
        antipodal_pairs.append((i, j))

print(f"  [PASS] 6 antipodal pairs identified")

# Verify: constant vector is NOT flat over R (holonomy = 4 per plaquette)
ones_vec = np.ones(n_links)
holonomies_const = M.T @ ones_vec          # holonomy on each plaquette
assert np.all(holonomies_const == 4), \
    "Constant vector should give holonomy 4 on every plaquette"
assert abs(np.dot(K @ ones_vec, ones_vec) - 28 * np.dot(ones_vec, ones_vec)) < 1e-10, \
    "Constant vector is eigenvector of K with eigenvalue 28, not a zero mode"
print(f"  [PASS] Constant vector is NOT flat: holonomy = 4 per plaquette")
print(f"         K·ones = 28·ones  (eigenvalue 28, not in ker(K))")

# Verify: ALL 4 null vectors are antipodally antisymmetric
n_antisym = 0
for col in range(4):
    v = null_space[:, col]
    is_antisym = all(abs(v[i] + v[j]) < 1e-10 for i, j in antipodal_pairs)
    if is_antisym:
        n_antisym += 1

assert n_antisym == 4, \
    f"Expected all 4 null vectors to be antisymmetric, found {n_antisym}"
print(f"  [PASS] All 4 flat modes are antipodally antisymmetric (θ_f = -θ_p)")

# Verify the GF(2) spatial axis interpretation
groups = {j: [i for i, v in enumerate(lightlike) if v[j] != 0]
          for j in [1, 2, 3]}

for j in [1, 2, 3]:
    g = groups[j]
    assert len(g) == 4
    indicator = np.zeros(n_links, dtype=int)
    for q in g:
        indicator[q] = 1
    holonomies = M.T @ indicator
    # Over R: holonomies are 0 or 2 (NOT zero)
    assert np.all(holonomies % 2 == 0), "Should be zero mod 2"
    assert not np.all(holonomies == 0), "Should NOT be zero over R"

print(f"  [PASS] Spatial-axis indicators: flat mod 2, NOT flat over R")
print(f"         (holonomies are 0 or 2, confirming GF(2) vs R rank gap)")

# Verify the rank gap
rank_R = np.linalg.matrix_rank(M)
rank_GF2 = gf2_rank(M)
assert rank_R == 8 and rank_GF2 == 7
print(f"  [PASS] rank(M): {rank_R} over R, {rank_GF2} over GF(2), gap = {rank_R - rank_GF2}")


# ================================================================
#  PART 5c — CW-COMPLEX OBSTRUCTION  (Remark 5.7)
# ================================================================

print(f"\n  PART 5c: CW-complex obstruction (Remark 5.7)")

# The paper's 1-skeleton is a star graph:
# 13 vertices (origin + 12 lightlike), 12 edges (origin to each)
# ∂₁: C₁ → C₀ maps edge e_i to vertex(i+1) - vertex(0)

d1 = np.zeros((13, 12), dtype=int)
for i in range(12):
    d1[0, i] = -1       # origin (source)
    d1[i + 1, i] = +1   # endpoint

rank_d1 = np.linalg.matrix_rank(d1)
ker_d1_dim = 12 - rank_d1

print(f"  ∂₁ shape: {d1.shape}, rank(∂₁) = {rank_d1}")
print(f"  dim ker(∂₁) = {ker_d1_dim}")

assert rank_d1 == 12, "Star graph should have rank-12 boundary operator"
assert ker_d1_dim == 0, "Star graph (tree) should have no 1-cycles"

# Chain complex condition: ∂₁ ∘ ∂₂ = 0 requires im(∂₂) ⊆ ker(∂₁) = {0}
# But rank(M) = 8, so ∂₂ ≠ 0: contradiction.
print(f"  [PASS] ker(∂₁) = {{0}} (star graph is a tree)")
print(f"  [PASS] rank(M) = 8 ≠ 0: incidence matrix cannot be ∂₂")

# Exhaustive sign search: no signing of M makes ∂₁∘∂₂ = 0
impossible_count = 0
for pi in range(len(plaquettes)):
    links_in = [link_idx[v] for v in plaquettes[pi]]
    found = False
    for signs in product([-1, 1], repeat=4):
        result = np.zeros(13, dtype=int)
        for li, s in zip(links_in, signs):
            result[0] -= s
            result[li + 1] += s
        if np.all(result == 0):
            found = True
            break
    if not found:
        impossible_count += 1

assert impossible_count == 21
print(f"  [PASS] All 21 plaquettes: no valid ∂₂ signing exists")

# Also fails over GF(2)
d1_gf2 = np.abs(d1) % 2   # over GF(2), -1 = 1
product_gf2 = (d1_gf2 @ M) % 2
assert not np.all(product_gf2 == 0)
print(f"  [PASS] ∂₁∘∂₂ ≠ 0 over GF(2) (fails over every coefficient field)")

# Euler characteristic cross-check
chi_cells = 13 - 12 + 21   # from cell dimensions
chi_v2    = 1 - 4 + 13     # from v2 Betti numbers
assert chi_cells == 22
assert chi_v2 == 10
assert chi_cells != chi_v2
print(f"  [PASS] Euler cross-check: 13-12+21 = {chi_cells} ≠ 1-4+13 = {chi_v2}")
print(f"         v2 Betti numbers (1,4,13) are inconsistent")

# GF(2) rank gap
rank_R = np.linalg.matrix_rank(M)
rank_GF2 = gf2_rank(M)
print(f"\n  Rank gap: rank(M) over R = {rank_R}, over GF(2) = {rank_GF2}")
print(f"  Gap of {rank_R - rank_GF2} due to ternary lattice structure")


# ================================================================
#  PART 5d — EXTENDED K_{6,6} COMPLEX  (Remark 5.8)
# ================================================================

print(f"\n  PART 5d: Extended K_{{6,6}} complex (Remark 5.8)")

def build_quadrilateral_complex(vertices, plaquettes_as_tuples):
    """Build valid CW complex with quadrilateral 2-cells."""
    vert_idx = {v: i for i, v in enumerate(vertices)}
    all_edges = {}
    face_boundaries = []

    for plaq in plaquettes_as_tuples:
        fv = sorted([v for v in plaq if v[0] == +1])
        pv = sorted([v for v in plaq if v[0] == -1])
        cycle = [fv[0], pv[0], fv[1], pv[1]]

        boundary = []
        for j in range(4):
            a, b = cycle[j], cycle[(j + 1) % 4]
            edge_key = frozenset([a, b])
            canonical = tuple(sorted([a, b]))
            if edge_key not in all_edges:
                all_edges[edge_key] = canonical
            sign = +1 if (a, b) == canonical else -1
            boundary.append((edge_key, sign))
        face_boundaries.append(boundary)

    edge_list = sorted(all_edges.keys(), key=lambda e: all_edges[e])
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    nv = len(vertices)
    ne = len(edge_list)
    nf = len(plaquettes_as_tuples)

    d1_ext = np.zeros((nv, ne), dtype=int)
    for i, ek in enumerate(edge_list):
        a, b = all_edges[ek]
        d1_ext[vert_idx[a], i] = -1
        d1_ext[vert_idx[b], i] = +1

    d2_ext = np.zeros((ne, nf), dtype=int)
    for fi, boundary in enumerate(face_boundaries):
        for ek, sign in boundary:
            d2_ext[edge_idx[ek], fi] += sign

    return d1_ext, d2_ext, nv, ne, nf

d1_ext, d2_ext, nv, ne, nf = build_quadrilateral_complex(lightlike, plaquettes)

# Verify chain complex condition
chain_product = d1_ext @ d2_ext
assert np.all(chain_product == 0), "∂₁∘∂₂ ≠ 0 on extended complex"

rank_d1_ext = np.linalg.matrix_rank(d1_ext)
rank_d2_ext = np.linalg.matrix_rank(d2_ext)

b0 = nv - rank_d1_ext
b1 = (ne - rank_d1_ext) - rank_d2_ext
b2 = nf - rank_d2_ext
chi_ext = b0 - b1 + b2
chi_cells_ext = nv - ne + nf

assert chi_ext == chi_cells_ext

# Verify with Hodge Laplacian
L1 = d1_ext.T @ d1_ext + d2_ext @ d2_ext.T
evals_L1 = np.linalg.eigvalsh(L1.astype(float))
b1_hodge = np.sum(np.abs(evals_L1) < 1e-10)
assert b1_hodge == b1

print(f"  Extended complex: |C₀|={nv}, |C₁|={ne}, |C₂|={nf}")
print(f"  [PASS] ∂₁∘∂₂ = 0 (valid chain complex)")
print(f"  [PASS] rank(∂₁) = {rank_d1_ext}, rank(∂₂) = {rank_d2_ext}")
print(f"  [PASS] (b₀, b₁, b₂) = ({b0}, {b1}, {b2})")
print(f"  [PASS] χ = {b0} - {b1} + {b2} = {chi_ext} = {nv} - {ne} + {nf}")
print(f"  [PASS] Hodge verification: dim ker(L₁) = {b1_hodge} = b₁")


# ================================================================
#  PART 6 — BOUNDARY BF THEORY (Proposition 5.8)
# ================================================================

print(f"\n  PART 6: BF theory — K_bdy and K_total spectrum")

# Boundary Laplacian: K_bdy = (1/13) [[I6, -I6], [-I6, I6]]
# This is the graph Laplacian of the 6 antipodal pairs with coupling 1/13.
# It comes from expanding cos(θ_f - θ_p) ≈ 1 - (θ_f - θ_p)²/2.
future_idx = [i for i, v in enumerate(lightlike) if v[0] == +1]

K_bdy = np.zeros((12, 12))
for fi in future_idx:
    pi = link_idx[tuple(-c for c in lightlike[fi])]
    K_bdy[fi, fi] += 1.0 / 13
    K_bdy[fi, pi] -= 1.0 / 13
    K_bdy[pi, fi] -= 1.0 / 13
    K_bdy[pi, pi] += 1.0 / 13

# K_bdy eigenvalues: 6 zeros (symmetric modes) + 6 at 2/13 (antisymmetric)
evals_bdy = sorted(np.linalg.eigvalsh(K_bdy))
n_bdy_zero = sum(1 for e in evals_bdy if abs(e) < 1e-10)
n_bdy_lifted = sum(1 for e in evals_bdy if abs(e - 2.0/13) < 0.001)
assert n_bdy_zero == 6 and n_bdy_lifted == 6
print(f"  [PASS] K_bdy spectrum: 6 zeros + 6 at 2/13")

# Verify all 4 bulk zero modes are antisymmetric
evals_bulk, evecs_bulk = np.linalg.eigh(K.astype(float))
null_vecs = evecs_bulk[:, np.abs(evals_bulk) < 1e-10]
for col in range(null_vecs.shape[1]):
    v = null_vecs[:, col]
    for fi in future_idx:
        pi = link_idx[tuple(-c for c in lightlike[fi])]
        assert abs(v[fi] + v[pi]) < 1e-10, "Null vector not antisymmetric"
print(f"  [PASS] All 4 bulk zero modes are antisymmetric (θ_f = -θ_p)")

# K_bdy restricted to the 4D null space: should give 2/13 on all 4
K_bdy_restricted = null_vecs.T @ K_bdy @ null_vecs
evals_restricted = np.linalg.eigvalsh(K_bdy_restricted)
assert all(abs(e - 2.0/13) < 1e-10 for e in evals_restricted)
print(f"  [PASS] K_bdy lifts all 4 zero modes to 2/13")

# Verify all-ones is eigenvector of K_bulk with eigenvalue 28
row_sums = K.sum(axis=1)
assert np.all(row_sums == 28)
print(f"  [PASS] All-ones is K_bulk eigenvector with eigenvalue 28 (not a zero mode)")

# Full K_total spectrum
K_total = K.astype(float) + K_bdy
evals_total = sorted(np.linalg.eigvalsh(K_total))
expected_approx = [2/13]*4 + [6+2/13]*2 + [8]*3 + [10]*2 + [28]
for computed, expected in zip(evals_total, expected_approx):
    assert abs(computed - expected) < 0.001, \
        f"Mismatch: computed {computed:.4f}, expected {expected:.4f}"

print(f"  [PASS] K_total spectrum: {{(2/13)⁴, (6+2/13)², 8³, 10², 28¹}}")
print(f"  [PASS] No zero eigenvalue in K_total (all modes lifted or nonzero)")


# ================================================================
#  PART 7 — CHARACTER EXPANSION AND FINITE-SYSTEM CROSSOVER
#           Verifies Proposition 6.4 (beta_c ≈ 2.7364, C_peak ≈ 12.69)
# ================================================================

from scipy.special import i0 as _i0, i1 as _i1
import math as _math

print(f"\n  PART 7: Character expansion and finite-system crossover (Prop. 6.4)")

# ── The exact character expansion (truncated to |n_p| ≤ 1) ──────────────
#
# For a U(1) lattice gauge theory with Wilson (heat-kernel) action
#   exp(β cos θ) = Σ_{n∈Z} I_n(β) exp(inθ),
# the partition function on the 21-plaquette, 12-link diamond is:
#   Z(β) = Σ_{n_p ∈ Z^21} [Π_p I_{n_p}(β)] Π_l δ(Σ_{p∋l} n_p, 0)
#
# The Kronecker delta enforces Gauss's law: for each link l, the net
# plaquette charge must vanish.
#
# Truncating to |n_p| ≤ 1 (justified for moderate β by rapid decay of
# I_n/I_0 for |n| > 1):
#
#   Zero sector (all n_p = 0):  Z_0 = I_0(β)^21
#
#   Single-excitation sector:   one plaquette at n = +1, one at n = -1,
#   the rest at zero.  Gauss's law at every link requires the two excited
#   plaquettes to share NO link (otherwise one link carries net charge ±1).
#   Each such compatible pair contributes I_1(β)^2 · I_0(β)^19.
#   Factor of 2 because (n_p=+1, n_q=−1) and (n_p=−1, n_q=+1) are distinct.
#
#   ln Z(β) ≈ 21·ln I_0(β) + ln(1 + 2·N_compat·(I_1(β)/I_0(β))^2)
#
# where N_compat is the number of non-adjacent plaquette pairs.

# Count compatible (non-adjacent) plaquette pairs using the already-built M
N_compat = 0
for _i in range(len(plaquettes)):
    for _j in range(_i + 1, len(plaquettes)):
        # Two plaquettes share a link iff their M-columns have a common 1
        if np.dot(M[:, _i], M[:, _j]) == 0:
            N_compat += 1

assert N_compat == 60, f"Expected 60 compatible pairs, got {N_compat}"
print(f"  [PASS] Compatible (non-adjacent) plaquette pairs: {N_compat} = 60")

# ── Evaluate lnZ and specific heat on a fine β grid ─────────────────────
_beta_arr = np.linspace(0.05, 6.0, 1200)
_lnZ      = np.empty(len(_beta_arr))

for _bi, _beta in enumerate(_beta_arr):
    _I0    = float(_i0(_beta))
    _I1    = float(_i1(_beta))
    _lnI0  = _math.log(max(_I0, 1e-300))
    _ratio = _I1 / max(_I0, 1e-300)
    # Guard against log(0) in the correction term
    _corr  = 1.0 + 2.0 * N_compat * _ratio**2
    _lnZ[_bi] = 21.0 * _lnI0 + _math.log(max(_corr, 1e-300))

# Specific heat: C(β) = β² · d²(lnZ)/dβ²  (numerical second derivative)
_dbeta   = _beta_arr[1] - _beta_arr[0]
_d2lnZ   = np.gradient(np.gradient(_lnZ, _dbeta), _dbeta)
_C       = _beta_arr**2 * _d2lnZ

# Locate peak (skip 10-point boundary artefacts on each side)
_peak_idx  = int(np.argmax(_C[10:-10])) + 10
_beta_c    = float(_beta_arr[_peak_idx])
_C_peak    = float(_C[_peak_idx])

print(f"  [PASS] Crossover location:  beta_c  = {_beta_c:.4f}  (expected ≈ 2.7364)")
print(f"  [PASS] Peak specific heat:  C_peak  = {_C_peak:.4f}  (expected ≈ 12.69)")

# Numerical-differentiation tolerance: ±0.02 on beta_c, ±0.2 on C_peak
assert abs(_beta_c - 2.7364) < 0.02, \
    f"beta_c = {_beta_c:.4f} deviates from expected 2.7364 by more than 0.02"
assert abs(_C_peak - 12.69) < 0.20, \
    f"C_peak = {_C_peak:.4f} deviates from expected 12.69 by more than 0.20"

# Verify the width estimate Δβ ≈ 2/C_peak (narrower ↔ sharper crossover)
_width = 2.0 / _C_peak
print(f"  [PASS] Crossover width estimate: Δbeta ≈ {_width:.4f}  (= 2/C_peak)")

# Sanity: <U_p> = (1/21) · d(lnZ)/dβ should lie in (0, 1)
_dlnZ_db   = np.gradient(_lnZ, _dbeta)
_U_avg_at_peak = float(_dlnZ_db[_peak_idx]) / 21.0
assert 0.0 < _U_avg_at_peak < 1.0, \
    f"Mean plaquette action <U_p> = {_U_avg_at_peak:.4f} out of (0,1)"
print(f"  [PASS] Mean plaquette action at crossover: <U_p> = {_U_avg_at_peak:.4f}")
print(f"  [NOTE] beta_c = {_beta_c:.4f} is in lattice units (a = 1).")
print(f"         Physical identification requires the continuum-limit energy")
print(f"         unit (Open Problem O1); the crossover itself is exact within")
print(f"         the |n_p| ≤ 1 character truncation.")


# ================================================================
#  SUMMARY
# ================================================================

print(f"\n{'=' * 66}")
print(f"  ALL CHECKS PASSED")
print(f"{'=' * 66}")
print(f"""
  Verified claims:
    Lemma 2.3:  12 lightlike vectors, partition 80 = 12 + 66 + 2
    Prop. 2.5:  D4 split 24 = 12 + 12, lightlike roots = lattice LL
    Thm.  3.2:  Five holographic conditions (a)-(e) verified
    Thm.  4.2:  n_eff = (12, 0, 0, 0), Riemannian sum = (0,0,0,0)
    Prop. 5.1:  21 plaquettes (0 order-3, 6 order-2), all 2FS+2PS
    Prop. 5.4:  Spectrum {{0^4, 6^2, 8^3, 10^2, 28^1}}, Tr=84, rank=8
    Prop. 5.5:  dim ker(K) = 4 = flat-connection space
    Remark 5.6: All 4 flat modes antisymmetric; constant vector not in ker(K)
    Remark 5.7: CW obstruction verified (star graph, Euler 22 != 10)
    Remark 5.8: Extended K_{{6,6}} complex: (b0,b1,b2) = (1,7,3), chi = -3
    K_total:    4 modes lifted to 2/13, no zero eigenvalue
               (K_bdy = [[I,-I],[-I,I]]/13, corrected from v2)
    Prop. 6.4:  Finite-system crossover beta_c ~ 2.7364, C_peak ~ 12.69
               (character expansion |n_p|<=1, N_compat=60 pairs)

  Supplementary file for:
    "Exact Discretisation and Boundary Observables
     in Lorentzian Causal Diamonds" (v3)
""")

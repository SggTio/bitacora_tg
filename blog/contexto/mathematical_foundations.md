# Mathematical Foundations for Topology-Aware Neural Segmentation

## Preamble

This document provides the mathematical framework underpinning the integration of algebraic topology and deep learning into neural network architectures for medical image segmentation and survival prediction. It is structured as a pre-LaTeX draft suitable for formalization into thesis chapters. Each section identifies definitions, propositions, theorems, and open questions, with examples and analogies provided throughout to make the material accessible. Results marked [TO FORMALIZE] indicate statements that require rigorous proof or precise citation for the final document.

---

# Chapter 1: Algebraic Topology on Discrete Image Domains

## 1.1 The Ill-Posed Problem of Images as Topological Spaces

### 1.1.1 The Continuous Ground Truth

We begin with the mathematical object we wish to study. A brain tumor in a patient's anatomy occupies a region Ω ⊂ ℝ³ (or Ω ⊂ ℝ² for a single MRI slice). This region is a compact subset of Euclidean space with (ideally) piecewise smooth boundary ∂Ω. The topological properties of Ω — its connected components, holes, and cavities — are captured by its singular homology groups H_k(Ω; ℤ) for k = 0, 1, 2, ....

**Definition 1.1 (Betti Numbers).** The k-th Betti number of Ω is β_k(Ω) := rank(H_k(Ω; ℤ)). Concretely:
- β₀(Ω) = number of connected components
- β₁(Ω) = number of independent 1-cycles (tunnels in 3D, holes in 2D)
- β₂(Ω) = number of independent 2-cycles (enclosed cavities in 3D)

**Example 1.1 (Familiar Shapes).** For a solid disk D² ⊂ ℝ²: β₀ = 1, β₁ = 0. For an annulus (donut cross-section) A ⊂ ℝ²: β₀ = 1, β₁ = 1. For a solid ball B³ ⊂ ℝ³: β₀ = 1, β₁ = 0, β₂ = 0. For a hollow sphere S² ⊂ ℝ³: β₀ = 1, β₁ = 0, β₂ = 1.

**Analogy 1.1 (Betti Numbers as Structural Features).** Think of Betti numbers as an architect's structural report for a building. β₀ counts how many separate buildings there are. β₁ counts how many archways or tunnels you could walk through without leaving the structure. β₂ counts how many sealed rooms (enclosed voids) exist inside. A conventional photo (Dice score) tells you how much building there is, but the architect's report tells you about its *structure* — and structure is what matters for whether the building stands or falls.

**Clinical relevance.** A glioblastoma with a necrotic core has β₂ = 1 (the cavity). If segmentation merges the necrotic core with background, β₂ drops to 0 — a topological error invisible to Dice score but clinically meaningful (necrosis extent affects prognosis and treatment planning). Conversely, if the algorithm hallucinates a hole that doesn't exist, β₁ increases spuriously, potentially misleading treatment decisions.

### 1.1.2 The Discretization Problem

An MRI scanner does not give us Ω. It gives us a discrete intensity function:

**Definition 1.2 (Digital Image).** A digital image is a function I: G → ℝ where G = {1, ..., N₁} × {1, ..., N₂} × {1, ..., N₃} is a finite cubic lattice (the voxel grid). Each point p ∈ G is a voxel with intensity I(p) ∈ ℝ.

**Definition 1.3 (Binary Segmentation Mask).** A binary segmentation mask is a function M: G → {0, 1}. The foreground is supp(M) = {p ∈ G : M(p) = 1}.

**The fundamental question:** Given the continuous region Ω and its digitization M (the binary mask obtained by sampling Ω on the grid G), under what conditions does the topology of M "match" the topology of Ω?

This is NOT a question about homeomorphism. The discrete set supp(M) with the subspace topology inherited from ℝ³ (or even the discrete topology) has trivial algebraic topology — it is a finite disjoint union of points. Instead, we must construct an intermediate topological space from M that admits meaningful homology computation.

**Analogy 1.2 (Pixels vs. Topology).** Imagine you have a mosaic of colored tiles representing the Mona Lisa. Each tile is just a square of uniform color — individually meaningless. But when you glue them together according to their adjacency pattern, you recover (approximately) the continuous painting. The cubical complex construction below is exactly this "gluing" process for binary images: it turns a discrete set of 0s and 1s into a genuine topological space whose holes and connectivity we can study.

### 1.1.3 Cubical Complexes: The Bridge Between Discrete and Continuous

**Definition 1.4 (Elementary Interval).** An elementary interval is either a degenerate interval [l, l] for some l ∈ ℤ, or a non-degenerate interval [l, l+1] for some l ∈ ℤ.

**Definition 1.5 (Elementary Cube).** An elementary cube is a product of elementary intervals: σ = I₁ × I₂ × ... × I_d ⊂ ℝ^d. The dimension of σ is the number of non-degenerate intervals in the product.

**Definition 1.6 (Cubical Complex).** A cubical complex K is a finite collection of elementary cubes such that for every σ ∈ K, all faces of σ also belong to K. The underlying topological space |K| = ⋃_{σ ∈ K} σ ⊂ ℝ^d inherits the subspace topology from ℝ^d.

**Construction 1.1 (From Binary Image to Cubical Complex).** Given a binary mask M: G → {0, 1}, we construct the cubical complex K(M) as follows: for each voxel p = (i, j, k) with M(p) = 1, include the 3-cube [i, i+1] × [j, j+1] × [k, k+1] and all its faces (2-faces = squares, 1-faces = edges, 0-faces = vertices). The resulting K(M) is a cubical complex whose underlying space |K(M)| is a compact subset of ℝ³.

**Key insight:** |K(M)| is a genuine topological space — a CW complex — whose singular homology is well-defined and computable. This is the space whose topology we compute when we "compute the topology of a binary image."

**Example 1.2 (A 2×2 L-shape).** Consider a 4×4 grid where the L-shaped region consists of voxels (1,1), (1,2), (2,1). The cubical complex K(M) contains three 2-cubes (unit squares), their shared edges, and their vertices. The underlying space |K(M)| is homotopy equivalent to a point (contractible), so β₀ = 1, β₁ = 0. If instead we had a ring of voxels forming a square frame, the cubical complex would have β₁ = 1, detecting the hole in the center.

### 1.1.4 When Does Digitization Preserve Topology?

**Definition 1.7 (Reach of a Compact Set).** The reach of a compact set Ω ⊂ ℝ^d, denoted reach(Ω), is the supremum of all r > 0 such that every point within distance r of Ω has a unique nearest point in Ω.

Intuitively, reach(Ω) is the smallest radius of curvature or the smallest distance between distinct boundary components. A smooth disk has reach equal to its radius; a very thin annulus has reach equal to half its wall thickness.

**Analogy 1.3 (Reach as "Wiggle Room").** Think of reach as how much you can inflate a balloon inside the shape before it either pops through a thin wall (small feature) or gets pinched between two close boundaries. A thick donut has high reach; a paper-thin donut has reach approaching zero. When you digitize a shape on a grid, the grid spacing must be much smaller than the reach — otherwise the grid is too coarse to "see" the fine features, and their topology gets lost.

**Definition 1.8 (Digitization at Resolution r).** The digitization of Ω at resolution r > 0 is D_r(Ω) = {p ∈ rℤ^d : p ∈ Ω}, the set of grid points (at spacing r) that lie inside Ω.

**Theorem 1.1 (Topological Faithfulness of Digitization).** [Latecki et al., 2000; cf. Edelsbrunner & Harer, 2010] Let Ω ⊂ ℝ^d be a compact set with reach(Ω) > ρ > 0. There exists a constant C_d (depending only on dimension) such that if the grid resolution r < ρ / C_d, then:

H_k(|K(D_r(Ω))|; ℤ) ≅ H_k(Ω; ℤ) for all k ≥ 0.

That is, the cubical homology of the digitized image matches the singular homology of the continuous shape.

**Proof sketch.** The result follows from the Nerve Theorem applied to the cover of Ω by the Voronoi cells of grid points. When the Voronoi cells are sufficiently small relative to the reach, each cell and every intersection of cells is contractible, and the nerve (which captures the combinatorial intersection pattern) has the same homology as Ω. The cubical complex K(D_r(Ω)) is a geometric realization of this nerve.

**Corollary 1.1.** For our synthetic dataset with disks (reach ≈ radius ≈ 20-35 pixels) and annuli (reach ≈ wall thickness ≈ 5-12 pixels) on a 128×128 grid (resolution r = 1 pixel), the condition r ≪ reach(Ω) is satisfied for disks but only marginally for thin annuli. This predicts that thin donut shapes may exhibit topological artifacts in their digitization.

**Example 1.3 (Clinical Implication).** Consider a brain tumor with two separate foci that are 3 voxels apart in an MRI with 1mm isotropic resolution. The "gap" between foci has reach ≈ 1.5mm. Since the grid resolution (1mm) is not much smaller than the reach, the digitization may merge the two foci into one connected component (β₀ drops from 2 to 1) — a clinically meaningful error that is predicted by Theorem 1.1's resolution condition.

**Remark 1.1 (This is NOT a Homeomorphism).** The digitized cubical complex |K(M)| is homotopy equivalent to Ω (under the conditions of Theorem 1.1), which is stronger than having isomorphic homology but weaker than homeomorphism. The annulus A = {x ∈ ℝ² : r₁ ≤ |x| ≤ r₂} is homeomorphic to S¹ × [0,1], while its cubical complex approximation is a finite CW complex homotopy equivalent to S¹. They have the same homology (H₀ ≅ ℤ, H₁ ≅ ℤ) and the same fundamental group (π₁ ≅ ℤ), but they are not homeomorphic (one is a manifold with boundary, the other is a polyhedral complex). The relationship is:

Ω ←homotopy equiv.→ |K(D_r(Ω))| (under sufficient resolution)

This is a **discretization with bounded topological information loss** — specifically, zero homological information loss when the resolution condition is met. However, finer geometric information (smooth structure, curvature, exact boundary shape) IS lost.

## 1.2 Persistent Homology and Filtrations

### 1.2.1 Motivation: From Binary to Graded Topology

A binary mask M provides a single topological snapshot. But the MRI intensity image I: G → ℝ contains richer information — features appear and disappear at different intensity thresholds. Persistent homology captures this multi-scale topological structure.

**Analogy 1.4 (The Flood Analogy).** Imagine a mountainous landscape being slowly flooded from below. As the water rises, islands (connected components) appear at the peaks and gradually merge as valleys are submerged. Some islands have lakes (holes) that form and fill. The persistence diagram is a logbook recording when each island appeared (birth) and when it merged with another island or when each lake formed and filled (death). Features that persist through a large range of water levels are "real" mountains; features that appear and vanish quickly are just noise ripples.

**Definition 1.9 (Sublevel Set Filtration).** Given a function f: |K| → ℝ on a cubical complex K, the sublevel set filtration is the nested family of subcomplexes:

K_t = {σ ∈ K : f(σ) ≤ t} for t ∈ ℝ

where f(σ) = max_{v ∈ vertices(σ)} f(v). As t increases from -∞ to +∞, K_t grows from ∅ to K, and topological features (components, holes, cavities) are born and die.

**Definition 1.10 (Persistence Diagram).** The persistence diagram Dgm_k(f) of a filtration is the multiset of points (b_i, d_i) ∈ ℝ² where b_i is the birth time and d_i is the death time of the i-th k-dimensional homological feature. Points far from the diagonal (high persistence |d_i - b_i|) represent robust topological features; points near the diagonal represent noise.

**Example 1.4 (Persistence of a Mountain Range).** Consider a 1D function f(x) = sin(x) + 0.1·sin(10x) representing a mountain range with small ripples. The large sinusoidal peaks produce persistence points far from the diagonal (persistence ≈ 2, the amplitude of sin(x)). The small ripples produce points near the diagonal (persistence ≈ 0.2). The persistence diagram cleanly separates "real" mountain features from "noise" ripples, at any scale, without any thresholding decisions.

**Definition 1.11 (Euclidean Distance Transform Filtration).** Given a binary mask M, the Euclidean Distance Transform (EDT) is:

EDT_M(p) = min_{q ∈ supp(M)} ||p - q||₂  (for p ∉ supp(M))
EDT_M(p) = min_{q ∉ supp(M)} ||p - q||₂  (for p ∈ supp(M))

The sublevel set filtration of -EDT_M (negated, so the interior has the lowest values) naturally captures the multi-scale structure: the core of the object appears first, and boundary details appear later.

**Example 1.5 (EDT Filtration of a Donut).** For an annulus (2D donut), the EDT assigns highest values to the medial circle (the center of the ring wall). The sublevel filtration of -EDT first reveals this medial circle (birth of H₁ ≅ one loop), then gradually thickens it to fill the ring. The persistence of this H₁ feature equals the half-thickness of the ring wall — directly measuring a geometric property of the shape through purely topological means.

### 1.2.2 Stability: The Foundational Theorem

**Theorem 1.2 (Stability of Persistent Homology, Cohen-Steiner, Edelsbrunner, Harer, 2007).** Let f, g: |K| → ℝ be two tame functions on a triangulable topological space. Then:

d_B(Dgm_k(f), Dgm_k(g)) ≤ ||f - g||_∞

where d_B is the bottleneck distance between persistence diagrams, defined as:

d_B(Dgm(f), Dgm(g)) = inf_γ sup_p ||p - γ(p)||_∞

where γ ranges over all bijections between diagrams (including the diagonal).

**Proof Roadmap.** The proof proceeds in four steps:

*Step 1 — The Box Lemma (Interleaving of Sublevel Sets).* If ||f - g||_∞ ≤ ε, then the sublevel sets satisfy the nesting:

f⁻¹(-∞, t-ε] ⊆ g⁻¹(-∞, t] ⊆ f⁻¹(-∞, t+ε]

for all t ∈ ℝ. This is the geometric foundation: the sublevel sets of f and g are "ε-close" in the inclusion sense.

*Step 2 — Induced Maps on Homology.* The inclusions from Step 1 induce maps on homology groups via functoriality. These maps create an "ε-interleaving" of persistence modules: the persistence module of f and the persistence module of g are algebraically interleaved at distance ε.

*Step 3 — From Interleaving to Matching.* The key technical lemma (made precise by Bauer & Lesnick, 2015) shows that an ε-interleaving of persistence modules implies an ε-matching of their barcodes/diagrams. This is proved using the structure theorem for persistence modules (decomposition into interval modules) and a careful combinatorial matching argument: each interval in the barcode of f must be matched to an interval in the barcode of g that differs by at most ε in its endpoints.

*Step 4 — Optimality.* The bound is tight: there exist functions f, g with d_B(Dgm(f), Dgm(g)) = ||f - g||_∞ exactly. For instance, shifting a function by a constant: g = f + ε moves every persistence point by exactly ε.

**Analogy 1.5 (Topographic Map Stability).** Imagine you draw a topographic map of a mountain range. Each peak is a "born" feature, and the saddle point where it merges with a higher peak is its "death." The persistence diagram records (elevation_born, elevation_merged) for each peak. Now imagine it rains, raising all water levels by ε meters. The stability theorem says the new topographic map's peaks and saddles shift by at most ε meters — no peak can teleport or appear/disappear suddenly. Small perturbations in the landscape cause small perturbations in the topographic summary. But if you drew a contour line at a fixed elevation (thresholding), the contour could change dramatically — islands might merge or split with a tiny water level change. This is why thresholding is dangerous for topology.

**Significance:** Small perturbations in the input function (e.g., noise in the image) produce small perturbations in the persistence diagram. This is why persistent homology is a robust topological descriptor — it is Lipschitz continuous with constant 1 in the bottleneck metric.

**Theorem 1.2b (Wasserstein Stability Extension).** The stability result extends to the p-Wasserstein distance between persistence diagrams:

W_p(Dgm_k(f), Dgm_k(g)) ≤ C_p · ||f - g||_∞

for appropriate constants C_p depending on the complexity of the underlying space. The Wasserstein distance is defined as:

W_p(D₁, D₂) = (inf_γ Σ_i ||p_i - γ(p_i)||_∞^p)^{1/p}

where γ ranges over all bijections. Unlike the bottleneck distance (which captures only the single worst-matched feature), the Wasserstein distance accumulates all matching costs. This is directly relevant because topological loss functions (§1.3) typically use W₂² (the squared 2-Wasserstein distance), and Wasserstein stability ensures that these losses are well-behaved under small input perturbations.

**Corollary 1.2.** The persistence diagram of a noisy image I + ε (where ||ε||_∞ ≤ δ) differs from the persistence diagram of the clean image I by at most δ in bottleneck distance. However, this does NOT mean the thresholded mask (I + ε > 0.5) has similar topology to (I > 0.5) — thresholding is a discontinuous operation that can introduce arbitrary topological changes. This is precisely why computing TDA on thresholded noisy images (as in the original toy model) produces unreliable results.

**Example 1.6 (Why Thresholding Breaks Topology).** Consider two pixels in a noisy image with values 0.499 and 0.501. Thresholding at 0.5 assigns them to different classes. Adding noise of magnitude 0.002 can flip both pixels, potentially creating or destroying a connected component. The perturbation is tiny (||ε||_∞ = 0.002), but the topological change is catastrophic (Δβ₀ = ±1). The persistence diagram of the full image changes by at most 0.002 (stable), but the topology of the *thresholded* image changes arbitrarily. This is the key insight: work with the filtration (stable), not with a single threshold (unstable).

### 1.2.3 Vectorization: Persistence Images

Persistence diagrams are multisets of variable cardinality — they do not live in a vector space and cannot be directly input to neural networks. Vectorization maps them to fixed-dimensional representations.

**Definition 1.12 (Persistence Image, Adams et al., 2017).** Given a persistence diagram D = {(b_i, d_i)}, define the transformed diagram D' = {(b_i, d_i - b_i)} (birth vs. persistence). The persistence image is:

PI(x, y) = Σ_{(b_i, p_i) ∈ D'} w(b_i, p_i) · φ_{(b_i, p_i)}(x, y)

where φ_{μ}(x, y) is a Gaussian kernel centered at μ with bandwidth σ, and w(b, p) is a weighting function (typically w(b, p) = p, weighting by persistence to suppress noise).

**Analogy 1.6 (Persistence Images as Heat Maps).** Think of each point in the persistence diagram as a small campfire, with the fire's intensity proportional to the feature's persistence (lifespan). The persistence image is the thermal image you'd see looking down from above: a smooth heat map where intense, long-lived features produce bright hot spots and short-lived noise produces barely visible warmth. The Gaussian kernel is the heat spreading from each campfire. This heat map can be discretized on any grid, producing a fixed-size image that neural networks can consume.

**Proposition 1.1.** The persistence image map PI: Dgm → ℝ^{n×n} is:
1. Well-defined (independent of diagram ordering)
2. Stable: ||PI(D₁) - PI(D₂)||_∞ ≤ C · d_W^1(D₁, D₂) for some constant C depending on the bandwidth σ and weighting function w
3. Injective in the limit of infinite resolution and zero bandwidth [TO FORMALIZE: precise conditions require that the resolution of the discretization grid exceeds the separation between distinct persistence points]

**Remark 1.2.** For our application, we compute persistence images at the same resolution as the input image (128×128). This means the persistence image can be interpreted as a "topological density map" over the image domain, where high-valued regions indicate concentrations of topological features. This spatial correspondence is WHY persistence images can be used as additional input channels — they are naturally co-registered with the image.

## 1.3 The Topological Loss Landscape

### 1.3.1 Differentiability of Persistent Homology

**Theorem 1.3 (Differentiability, Brüel-Gabrielsson et al., 2020; cf. Leygonie et al., 2022).** Let f_θ: |K| → ℝ be a parametric family of filtration functions, where θ ∈ ℝ^p are parameters. If the persistence diagram Dgm_k(f_θ) has no points on the diagonal (all features have positive persistence) and no two birth or death values coincide (genericity condition), then the map θ ↦ Dgm_k(f_θ) is differentiable, and:

∂(b_i, d_i)/∂θ depends only on ∂f_θ/∂θ evaluated at the critical simplices responsible for the birth and death of feature i.

**Significance:** This theorem enables backpropagation through topological loss functions. The gradient of a topological loss with respect to network parameters flows through the critical pixels (birth/death locations of topological features), providing spatially localized supervision. Unlike Dice loss (which distributes gradients across all foreground pixels), topological gradients concentrate at the exact locations where topological features are created or destroyed.

**Example 1.7 (Critical Pixel Gradient).** Suppose a neural network predicts a nearly closed ring, with a single pixel gap at position (42, 73). This gap means β₁ = 0 (no hole), when the ground truth has β₁ = 1. The topological loss identifies (42, 73) as a critical pixel — the death location of the H₁ feature. The gradient pushes this single pixel's prediction toward 1, closing the gap and restoring the correct topology. The Dice loss, in contrast, barely notices this pixel among thousands.

### 1.3.2 The Wasserstein Topological Loss (TopoLoss)

**Definition 1.13 (Topological Loss via Wasserstein Matching, Hu et al., 2019).** Given a prediction likelihood map f: Ω → [0,1] and ground truth g (viewed as a step function), the topological loss is:

L_topo(f, g) = Σ_d W_q^q(Dgm_d(f), Dgm_d(g))

where d ranges over homological dimensions, W_q is the q-Wasserstein distance (typically q = 2), and the persistence diagrams are computed from the superlevel set filtrations (as threshold decreases from 1 to 0, the thresholded set grows).

The total training loss combines pixel-wise and topological terms:

L = L_BCE + λ · L_topo

**Theorem 1.4 (Correctness Guarantee for TopoLoss, Hu et al., 2019).** If L_topo(f, g) = 0 for prediction f and ground truth g, then for any generic threshold α ∈ (0,1), the binarized prediction f^α = {x : f(x) ≥ α} has the same Betti numbers as g^α.

**Proof Roadmap.**

*Step 1.* The Wasserstein distance W_q(Dgm(f), Dgm(g)) = 0 if and only if Dgm(f) = Dgm(g) as multisets (by definition of a metric).

*Step 2.* If two functions have identical persistence diagrams, they have the same number of homological features born and dying at each threshold pair. In particular, for any generic threshold α (one where no feature is exactly born or dying), the Betti numbers β_k(f^α) = β_k(g^α).

*Step 3.* The gradient computation identifies, for each unmatched or mismatched persistence point, the critical pixel responsible for the birth or death of that feature. The gradient directs the network to modify these specific pixels to bring the persistence diagrams into alignment.

**Key insight:** The persistence diagram encodes ALL possible thresholdings simultaneously. By matching persistence diagrams, you ensure topological correctness across all thresholds, not just α = 0.5. This is fundamentally more powerful than checking topology at a single threshold.

**Analogy 1.7 (The Music Critic).** Standard segmentation losses (Dice) are like a spelling checker for a musical score — they count how many notes are correct. The topological loss is like a music critic who listens for the overall structure: Are all the themes present? Do the melodies connect properly? Are there phantom themes that shouldn't exist? If the critic says the two performances have identical structure (L_topo = 0), then at every possible tempo (threshold), the musical themes match.

### 1.3.3 The Degenerate Flat Landscape and the Bootstrapping Problem

**Proposition 1.2 (Topological Degeneracy of Uniform Predictions).** Let P: G → [0, 1] be a near-constant prediction map, P(p) ≈ c for all p ∈ G. Then the sublevel set filtration of P produces a persistence diagram concentrated within distance ε of the diagonal, where ε = max_p P(p) - min_p P(p).

*Proof sketch.* If P varies by at most ε, then all topological features born at threshold t die by threshold t + ε, giving persistence at most ε. As ε → 0, all points approach the diagonal, and gradients of any topological loss vanish.

**Corollary 1.3 (The Bootstrapping Problem).** This proposition reveals a fundamental obstacle: at the beginning of training, when the network produces near-random or near-uniform predictions, the persistence diagram is concentrated near the diagonal and the topological loss provides near-zero gradients. The topological loss is only useful AFTER the network has developed some spatial structure in its predictions. This creates a chicken-and-egg problem: the network needs topology-aware supervision to learn correct topology, but topology-aware supervision requires structured predictions to produce meaningful gradients.

**Example 1.8 (The Dead Zone).** Consider a randomly initialized network predicting P(p) ≈ 0.5 ± 0.02 everywhere. The persistence diagram has all points within distance 0.02 of the diagonal — effectively trivial. The topological loss is essentially zero, and its gradient is essentially zero. Adding this loss to training changes nothing. Only after Dice loss (or BCE) has pushed predictions away from 0.5 — creating distinct foreground/background regions — does the topological loss "wake up" and start correcting topological errors. This is the mathematical justification for curriculum training (§2.3.2).

### 1.3.4 Betti Matching Loss

**Definition 1.14 (Induced Matching via Inclusion, Stucki et al., 2023).** Given prediction P and ground truth G (both as filtration functions on the same cubical complex K), define the comparison function C = max(P, G). The inclusion maps:

ι_P: K_t^P ↪ K_t^C   and   ι_G: K_t^G ↪ K_t^C

induce maps on persistent homology. A feature (b, d) in Dgm(P) is **matched** to a feature (b', d') in Dgm(G) if they both map to the same feature in Dgm(C). Unmatched features are topological errors.

**Definition 1.15 (Betti Matching Loss).** The Betti matching loss is:

L_BM(P, G) = Σ_{unmatched (b,d) in Dgm(P)} (d - b)² + Σ_{unmatched (b',d') in Dgm(G)} (d' - b')²

Matched features contribute zero loss; unmatched features are penalized by their squared persistence.

**Theorem 1.5 (Spatial Correctness of Induced Matching, Stucki et al., 2023).** The induced matching from the inclusion maps guarantees that matched features correspond to the same spatial location in the image. This is strictly stronger than Wasserstein matching of persistence diagrams, which can pair spatially distant features.

**Proof Roadmap.**

*Step 1 — Construction of C.* C = max(P, G) ensures that for superlevel sets, C^t ⊇ P^t and C^t ⊇ G^t for all thresholds t. This gives well-defined inclusion maps P^t ↪ C^t and G^t ↪ C^t.

*Step 2 — Functoriality.* By functoriality of homology, the inclusions induce maps on homology: H_k(P^t) → H_k(C^t) and H_k(G^t) → H_k(C^t). These extend to maps on persistence modules.

*Step 3 — The Matching Criterion.* A feature (b_P, d_P) in Dgm(P) is matched to (b_G, d_G) in Dgm(G) if both are mapped to the SAME feature in Dgm(C) by the induced maps. Since C = max(P, G), features that are spatially disjoint in P and G will correspond to DIFFERENT features in C (because the comparison image preserves spatial separation). Therefore, only spatially co-located features can be matched.

*Step 4 — Differentiability.* The Betti matching loss differentiates through the persistence computation using the critical cell gradient (as in Theorem 1.3), but restricted to unmatched features only. Matched features contribute zero loss, correctly ignoring topologically correct predictions.

**Analogy 1.8 (Overlaying Maps of an Archipelago).** Imagine two cartographers draw maps of the same archipelago. The Wasserstein matching pairs islands by their size (persistence), so if one map has a large island in the north and another large island in the south, they might be incorrectly paired purely because their sizes happen to match. The induced matching works by overlaying both maps on the same physical sheet (the comparison image C). An island in map A is matched to an island in map B only if they physically overlap on the overlaid sheet. This is the correct way to match geographic features — by location, not by size.

**Example 1.9 (Why Spatial Correctness Matters).** Suppose the ground truth has two tumor foci: one large focus in the left hemisphere and one small focus in the right. The prediction correctly identifies both but makes the left focus slightly too small and the right focus slightly too large. Wasserstein matching (TopoLoss) might pair the large predicted focus with the large ground truth focus even if it's in the wrong hemisphere, producing misleading gradients. Betti matching guarantees that each predicted focus is compared to the spatially overlapping ground truth focus, producing correct gradients that fix local errors.

### 1.3.5 The Betti Number Prior Approach

**Definition 1.16 (Betti Number Prior Loss, Clough et al., 2020).** Rather than matching persistence diagrams to ground truth, the user specifies the desired Betti numbers β = (β₀, β₁, ...) of the target structure. For each dimension d: if the prediction has more d-dimensional features than β_d, the loss penalizes the most persistent excess features (pushing them toward zero persistence, i.e., destroying them). If it has fewer, the loss penalizes the least persistent missing features (encouraging them to form).

**Example 1.10 (Semi-Supervised Topology).** For the left ventricular myocardium in cardiac MRI, the topology is known a priori: β₀ = 1 (one connected component) and β₁ = 1 (one hole — the ventricular cavity). Without any pixel-wise labels, a network can be trained with just this topological prior: "your prediction should have exactly one ring." The network learns to segment the myocardium from this structural knowledge alone. This is like telling an art student "draw exactly one island with exactly one lake" — they can learn from structural feedback without seeing the answer key.

**Remark 1.3.** The Betti number prior is a weaker form of topological supervision than Betti matching — it enforces the correct COUNT of features but not their spatial locations. A prediction with one component in the wrong place still satisfies β₀ = 1. However, for applications where the target topology is known but pixel-wise labels are scarce or expensive, this semi-supervised approach is powerful.

## 1.4 Discrete Morse Theory and the DMT-Loss

### 1.4.1 Motivation

Persistent homology identifies topological features (how many components, holes, cavities) but does not directly identify the SPATIAL structures — the skeletons, membranes, and critical points — that carry the topological information. Discrete Morse Theory (DMT) bridges this gap, localizing the exact cells in a complex that determine its topology.

### 1.4.2 Core Definitions

**Definition 1.17 (Discrete Morse Function, Forman 1998).** A function f on the cells of a CW complex K is a discrete Morse function if each cell σ has at most one higher-dimensional coface τ with f(τ) ≤ f(σ), and at most one lower-dimensional face ρ with f(ρ) ≥ f(σ).

**Definition 1.18 (Critical Cells).** A cell σ is critical if it violates both conditions in Definition 1.17 — it has no lower face with higher value AND no higher coface with lower value. Critical cells are local extrema in the Morse-theoretic sense. In a 2D image:
- Critical 0-cells correspond to local maxima (birth of H₀ features — connected components appear here)
- Critical 1-cells correspond to saddle points (death of H₀ features where components merge, or birth of H₁ features where loops form)
- Critical 2-cells correspond to local minima (death of H₁ features where loops fill in)

**Analogy 1.9 (The Conducting Teacher).** If the topological loss (Hu et al. 2019) is like a music critic saying "your performance has the wrong number of themes," then DMT-loss is like a conducting teacher pointing to the EXACT bars in the score where the themes begin and end, saying "focus your practice HERE — these bars determine the musical structure." DMT does not just count features; it localizes the specific cells that create and destroy them.

### 1.4.3 The DMT-Loss

**Definition 1.19 (DMT-Loss, Hu et al., 2021).** The DMT-loss identifies topologically critical structures from a discrete Morse decomposition:
- 1D critical structures (skeletons) connecting critical 0-cells and 1-cells — these determine the connectivity of curvilinear objects like vessels
- 2D critical structures (membranes) at critical 2-cells — these determine the boundary integrity of regions

The loss applies focused, weighted supervision on these critical structures:

L_DMT = Σ_{x ∈ critical structures} w(x) · ℓ(f_θ(x), g(x))

where w(x) is a weight reflecting the topological importance of cell x and ℓ is a pointwise loss (e.g., binary cross-entropy).

**Key insight:** The Morse complex decomposes the filtration into ascending and descending manifolds of critical cells. By the Morse Lemma, the topology of sublevel sets changes only at critical values, and the change is determined by the index of the critical cell (0 = new component, 1 = handle attachment, etc.). DMT-loss exploits this to focus supervision precisely where it matters for topology.

**Proposition 1.3 (DMT-Loss vs. TopoLoss Performance).** Hu et al. (2021) demonstrated that DMT-loss achieves Betti error of 0.982 on the CREMI neuronal membrane dataset, versus 3.016 for standard U-Net and 1.113 for TopoLoss (Hu et al. 2019). The improvement comes from directly supervising the spatially identified critical structures rather than indirectly optimizing persistence diagram distances.

**Example 1.11 (Vessels and Membranes).** In retinal vessel segmentation, the critical 1D structures are the vessel centerlines — the "skeleton" of the vascular tree. A single broken pixel on the centerline changes β₀ (connectivity), but Dice barely notices. DMT-loss identifies this centerline as the critical structure and applies heavy supervision there, ensuring vessel connectivity is preserved even if some peripheral pixels are misclassified.

**Remark 1.4 (Scope Limitations).** DMT-loss is most effective for structures with clear 1D skeletons (vessels, neurons, roads). For compact regions like brain tumors, where the "skeleton" may be a single point or a trivial structure, DMT-loss is less natural than persistence-based approaches. The identification of critical structures also depends on the choice of Morse function, which must align with the topological features of interest.

## 1.5 clDice: Skeleton-Based Topology Preservation

### 1.5.1 Definitions

**Definition 1.20 (Morphological Skeleton).** The skeleton S(V) of a binary mask V is the centerline or medial axis, obtained via iterative morphological thinning — repeatedly eroding the boundary until only a 1-pixel-wide structure remains that preserves the connectivity of V.

**Definition 1.21 (clDice, Shit et al., 2021).** Given prediction mask V_P with skeleton S_P and ground truth mask V_L with skeleton S_L:
- Topology Precision: Tprec(S_P, V_L) = |S_P ∩ V_L| / |S_P| — what fraction of the predicted skeleton lies within the ground truth?
- Topology Sensitivity: Tsens(S_L, V_P) = |S_L ∩ V_P| / |S_L| — what fraction of the ground truth skeleton lies within the prediction?
- clDice = 2 · Tprec · Tsens / (Tprec + Tsens) (harmonic mean)

### 1.5.2 The Homotopy Equivalence Theorem

**Theorem 1.6 (Homotopy Equivalence, Shit et al., 2021).** Let V_L and V_P be two binary masks, each admitting foreground and background skeletons, and assume the foreground is connected and admits a deformation retraction onto its skeleton. If:
1. S(V_L) ⊆ V_P (the ground truth skeleton lies inside the prediction), AND
2. S(V_P) ⊆ V_L (the predicted skeleton lies inside the ground truth),

then V_L and V_P are homotopy equivalent.

**Corollary 1.4.** clDice = 1 implies conditions (1) and (2), and therefore implies homotopy equivalence for binary 2D and 3D segmentations.

**Proof Roadmap.**

*Step 1.* The skeleton S(V) is a deformation retract of V: there exists a continuous map that progressively shrinks V onto S(V) without tearing or gluing. This means V and S(V) are homotopy equivalent.

*Step 2.* Given S(V_L) ⊆ V_P ⊆ K(V_L) (where K(V_L) is a thickened version of V_L), and the inclusion S(V_L) ↪ V_L is a homotopy equivalence, we have a chain of inclusions: S(V_L) ↪ V_P ↪ K(V_L). By the Whitehead theorem, since S(V_L) ↪ V_L induces isomorphisms on all homotopy groups, the inclusion S(V_L) ↪ V_P is also a homotopy equivalence.

*Step 3.* Similarly, from S(V_P) ⊆ V_L. Therefore, V_L ≃ S(V_L) ≃ V_P: the two masks are homotopy equivalent.

**Analogy 1.10 (Verifying Road Networks).** Imagine verifying that two road networks connect the same cities. Instead of comparing every piece of asphalt, you check: does the centerline of each road network lie within the asphalt of the other? If yes, the two networks must connect the same cities in the same way — they are topologically equivalent. The centerline is the skeleton, and "lying within" is the overlap check.

### 1.5.3 Soft-clDice and Computational Efficiency

**Definition 1.22 (Soft-clDice).** The differentiable version replaces exact morphological thinning with iterative max-pooling operations (approximating erosion). Starting from the soft probability map, K iterations of max-pooling progressively peel away the boundary, leaving an approximate soft skeleton. This is computed in O(n) time using standard pooling operations — dramatically cheaper than O(n³) persistent homology.

**Remark 1.5 (Scope).** clDice is primarily designed for tubular structures (vessels, neurons, roads) where the skeleton carries the essential topological information. For compact regions like brain tumors, the skeleton degenerates to a point or trivial structure, and clDice reduces to standard overlap. Thus, for brain tumor segmentation, clDice is relevant for the vascular component (if vessels within the tumor are modeled) but not for the tumor body itself, where persistent homology-based methods are more appropriate.

## 1.6 Train-Free TDA Segmentation

### 1.6.1 Motivation

A natural question is: how much segmentation can topology alone achieve, without any neural network? François and Tinarrage (2024) answered this by demonstrating that cubical persistent homology alone can segment brain tumors, providing a theoretical baseline that establishes the power of TDA as a standalone tool.

### 1.6.2 The Method

**Definition 1.23 (Three-Module TDA Pipeline, François & Tinarrage, 2024).** Given a 3D MRI volume:
1. **Whole object identification:** Automatic thresholding of the MRI intensity image to identify the approximate tumor region.
2. **Topologically distinctive subset detection:** Compute cubical persistent homology on the 3D volume. The H₂ persistence diagram reveals spherical cavities. The most persistent H₂ feature corresponds to the necrotic core (enclosed cavity within the tumor).
3. **Geometric deduction:** Extract the representative cycle of this H₂ feature — the 2-cycle bounding the cavity — to localize the enhancing rim. Remaining tumor regions (edema, non-enhancing core) are deduced from the geometric relationship between the enhancing rim and the whole-brain segmentation.

**Example 1.12 (Finding a Hollow Ball).** Imagine finding a hollow ball in a room full of objects using only a "void detector" (persistent homology). You don't need training — you know what hollow balls look like topologically: they enclose a void (β₂ = 1). You scan the room, find the void, trace its boundary, and you've found the ball. This is exactly what train-free PH segmentation does for glioblastomas with necrotic cores.

### 1.6.3 Theoretical Guarantee

**Proposition 1.4.** If the topological prior is correct — i.e., the enhancing tumor genuinely forms a shell enclosing a necrotic cavity with H₂ ≈ 1 — then the representative cycle computation correctly localizes the boundary of that cavity. The output topology is guaranteed to match the prior.

**Remark 1.6 (Limitations).** This method requires STRONG topological priors that may not hold for all tumor subtypes. Low-grade gliomas (solid masses with β₀ = 1, β₁ = 0, β₂ = 0) have no distinctive topology to exploit. Performance is lower than supervised methods on well-annotated datasets. The representative cycle computation is not unique — different algorithms may produce different (homologous) cycles, affecting precise localization. Nevertheless, this method demonstrates that the topological content of brain tumor MRI is mathematically rich enough to drive segmentation without any learning, establishing a non-trivial lower bound for what topology alone can achieve.

---

# Chapter 2: Neural Networks as Vector-Valued Functions

## 2.1 The Function Space Framework

### 2.1.1 Neural Networks as Parameterized Function Families

**Definition 2.1 (Neural Network as a Function).** A neural network with parameters θ ∈ Θ ⊂ ℝ^p defines a function:

f_θ: X → Y

where X is the input space and Y is the output space. For image segmentation:
- X = ℝ^{C_in × H × W} (input images with C_in channels, height H, width W)
- Y = [0, 1]^{C_out × H × W} (pixel-wise probability maps with C_out classes)
- Θ ⊂ ℝ^p is the parameter space (all weights and biases)

The network f_θ is a composition of layers: f_θ = σ_out ∘ L_n ∘ ... ∘ L_2 ∘ L_1, where each layer L_i is an affine map followed by a pointwise nonlinearity.

**Analogy 2.1 (Neural Network as a Factory Assembly Line).** A neural network is like a factory assembly line where each station (layer) performs a specific transformation on the product (feature map). The raw material enters (input image), and each station adds, reshapes, or refines features until the final product (pixel-wise predictions) emerges. The factory's configuration (parameters θ) determines what the assembly line produces. Training the network is like calibrating every station's settings to produce the desired output.

### 2.1.2 Convolutional Layers as Structured Linear Operators

**Definition 2.2 (Discrete 2D Convolution).** A convolutional layer with kernel w ∈ ℝ^{C_out × C_in × k × k} and bias b ∈ ℝ^{C_out} is the operator:

(Conv_w * x)[c, i, j] = Σ_{c'=1}^{C_in} Σ_{s=-⌊k/2⌋}^{⌊k/2⌋} Σ_{t=-⌊k/2⌋}^{⌊k/2⌋} w[c, c', s+⌊k/2⌋, t+⌊k/2⌋] · x[c', i+s, j+t] + b[c]

for output channel c ∈ {1, ..., C_out} and spatial position (i, j).

**Proposition 2.1 (Convolution as a Structured Linear Map).** The convolution operator Conv_w: ℝ^{C_in × H × W} → ℝ^{C_out × H × W} is a linear map (for fixed w). As a matrix, it is a block-Toeplitz matrix with Toeplitz blocks (doubly-block-circulant with appropriate boundary conditions). The key structural constraint is weight sharing: the same kernel w is applied at every spatial position.

**Remark 2.1 (Translation Equivariance).** Weight sharing endows convolutions with a fundamental symmetry property. Let T_v denote the translation operator (T_v x)[c, i, j] = x[c, i-v₁, j-v₂]. Then:

Conv_w ∘ T_v = T_v ∘ Conv_w

for all translations v. This is the mathematical statement that convolutions are translation-equivariant: shifting the input shifts the output identically. This property is why CNNs can recognize objects regardless of position.

**Example 2.1 (Why Translation Equivariance Matters for Tumors).** If a CNN can correctly segment a tumor in the left hemisphere, translation equivariance guarantees it produces the same segmentation if the tumor appears in the right hemisphere (modulo boundary effects). Without this property, the network would need separate training examples for every possible tumor location — an exponential increase in data requirements.

### 2.1.3 The U-Net as a Specific Function

**Definition 2.3 (U-Net Architecture).** A U-Net with L encoder levels defines:

f_θ^{UNet}: ℝ^{C_in × H × W} → [0, 1]^{C_out × H × W}

as the composition:

**Encoder path:** For l = 1, ..., L:
  e_l = DoubleConv_l ∘ Pool_l (e_{l-1})    where e_0 = x (input)

**Bottleneck:**
  b = DoubleConv_{L+1}(e_L)

**Decoder path:** For l = L, ..., 1:
  d_l = DoubleConv_l' ∘ Cat(Up_l(d_{l+1}), e_l)    where d_{L+1} = b

**Output:**
  f_θ(x) = σ ∘ Conv_{1×1}(d_1)

where Pool is max-pooling (2× spatial downsampling), Up is transposed convolution or bilinear upsampling (2× upsampling), Cat is channel-wise concatenation, DoubleConv is (Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU), and σ is the sigmoid function.

**Analogy 2.2 (The U-Net as Zoom-and-Enhance).** The U-Net encoder is like progressively zooming out of an image to understand the big picture (global context), while the decoder zooms back in to make precise local decisions. The skip connections are like keeping notes from each zoom level — when you zoom back in, you consult your notes to remember the fine details. The "U" shape comes from this zoom-out-then-zoom-in architecture, and the skip connections are the bridge that prevents information loss during the process.

**Proposition 2.2 (Domain and Codomain Consistency).** For the U-Net to be well-defined, the following dimension constraints must hold:

1. After l pooling operations: spatial resolution is H/2^l × W/2^l. This requires H and W to be divisible by 2^L.
2. Skip connections require encoder feature maps e_l and upsampled decoder maps Up_l(d_{l+1}) to have identical spatial dimensions.
3. Channel-wise concatenation: Cat(Up_l(d_{l+1}), e_l) has C_{up} + C_{enc} channels, which must match the input channel count of DoubleConv_l'.

### 2.1.4 The 1×1 Convolution as a Pointwise Linear Map

**Definition 2.4 (1×1 Convolution).** A 1×1 convolution with weight W ∈ ℝ^{C_out × C_in} and bias b ∈ ℝ^{C_out} is:

(Conv_{1×1} * x)[c, i, j] = Σ_{c'=1}^{C_in} W[c, c'] · x[c', i, j] + b[c]

**Proposition 2.3.** The 1×1 convolution is equivalent to applying the SAME linear map W: ℝ^{C_in} → ℝ^{C_out} independently at every spatial position. That is, if we denote x_{ij} = (x[1,i,j], ..., x[C_in,i,j])ᵀ ∈ ℝ^{C_in}, then:

(Conv_{1×1} * x)_{ij} = W · x_{ij} + b   ∀ (i,j)

*Proof.* This follows directly from the definition with k=1 — there is no spatial neighborhood, only channel interaction.

**Corollary 2.1 (Fusion via 1×1 Convolution).** Given two feature maps A ∈ ℝ^{C_A × H × W} and B ∈ ℝ^{C_B × H × W}, the operation:

F = Conv_{1×1}(Cat(A, B)) ∈ ℝ^{C_out × H × W}

applies a learned linear combination of A and B at each spatial position independently. Specifically:

F_{ij} = W_A · A_{ij} + W_B · B_{ij} + b

where W = [W_A | W_B] is partitioned according to the concatenation. This is strictly more expressive than A + B (addition), which is the special case W_A = W_B = I, b = 0.

**Proposition 2.4 (Addition is a Degenerate Special Case of Concatenation Fusion).** The additive fusion F = A + B constrains the fusion operator to the identity: each output channel is the sum of the corresponding channels in A and B. Concatenation + 1×1 conv allows:
- Weighted combinations: emphasizing spatial over topological features (or vice versa) at each position
- Cross-modal interactions: output channel c can depend on ANY combination of spatial and topological channels
- Position-dependent weighting: when combined with spatial context (via preceding conv layers)

This provides the mathematical justification for replacing additive fusion with concatenation fusion: the latter spans a strictly larger function space.

## 2.2 Multi-Modal Input Processing

### 2.2.1 Input-Channel Fusion: Formal Description

**Definition 2.5 (Input-Channel Fusion).** Given a spatial image I ∈ ℝ^{C_s × H × W} and topological features T ∈ ℝ^{C_t × H × W}, the input-channel fusion model is:

f_θ^{IF}(I, T) = UNet_θ(Cat(I, T))

where UNet_θ: ℝ^{(C_s + C_t) × H × W} → [0,1]^{C_out × H × W} is a standard U-Net with C_s + C_t input channels.

**Proposition 2.5 (First-Layer Cross-Modal Interaction).** In input-channel fusion, the first convolutional layer has kernels w ∈ ℝ^{C_1 × (C_s + C_t) × k × k}. Each output feature at position (i,j) is:

h₁[c, i, j] = Σ_{c'=1}^{C_s} (w * I)[c, c', i, j] + Σ_{c'=C_s+1}^{C_s+C_t} (w * T)[c, c', i, j]

This means the network learns JOINT spatial-topological features from the very first layer. A 3×3 kernel simultaneously "sees" a 3×3 patch of image intensity and the corresponding 3×3 patch of persistence image density.

**Analogy 2.3 (X-ray Vision).** Input-channel fusion is like giving a radiologist both the standard MRI and an "X-ray" showing the internal topological structure (persistence image) side by side from the very first glance. From the beginning of their analysis, they can correlate intensity patterns with structural patterns. Bottleneck fusion, by contrast, is like having one specialist analyze the MRI and another analyze the topology separately, meeting only at the end to compare notes.

**Proposition 2.6 (Universal Approximation for Input Fusion).** By the universal approximation theorem for CNNs (Zhou, 2020), for any continuous function g: ℝ^{(C_s+C_t) × H × W} → [0,1]^{C_out × H × W} and any ε > 0, there exists a CNN f_θ with (C_s+C_t) input channels and sufficiently many parameters such that ||f_θ - g||_∞ < ε. Therefore, input-channel fusion can in principle learn ANY continuous mapping from (image + topology) to segmentation.

### 2.2.2 Bottleneck Fusion: Formal Description

**Definition 2.6 (Bottleneck Fusion with Separate Encoders).** Let Enc_θ^S: ℝ^{C_s × H × W} → ℝ^{C_b × H' × W'} be the spatial encoder and Enc_θ^T: ℝ^{C_t × H × W} → ℝ^{C_b × H' × W'} be the topological encoder (both mapping to the same bottleneck spatial dimensions H' = H/2^L, W' = W/2^L). The bottleneck fusion model is:

f_θ^{BF}(I, T) = Dec_θ(Fuse(Enc_θ^S(I), Enc_θ^T(T)))

where Fuse: ℝ^{C_b × H' × W'} × ℝ^{C_b × H' × W'} → ℝ^{C_b × H' × W'} is the fusion operator, and Dec_θ is the decoder (with skip connections from the spatial encoder only).

**Proposition 2.7 (Bottleneck Fusion Restricts Early Cross-Modal Flow).** In bottleneck fusion, the spatial and topological features interact ONLY at resolution H' × W' (the bottleneck). Cross-modal information at the original resolution H × W must be captured entirely by the spatial encoder alone. This is equivalent to the assumption:

f_θ^{BF}(I, T) = g_θ(I, h_θ(T))

where h_θ: ℝ^{C_t × H × W} → ℝ^{C_b × H' × W'} compresses topological information and g_θ uses it as global context. The topological features modulate the decoding but cannot influence fine-grained spatial decisions at the original resolution.

**Theorem 2.1 (Expressiveness Comparison).** [TO FORMALIZE] The function class of input-channel fusion {f_θ^{IF}} is strictly larger than the function class of bottleneck fusion {f_θ^{BF}} (when both have the same total parameter count), because input fusion allows cross-modal interactions at all spatial resolutions while bottleneck fusion restricts them to the coarsest resolution.

However, the **sample complexity** of input fusion may be higher (more parameters to learn from data), creating a bias-variance tradeoff: bottleneck fusion has stronger inductive bias (fewer learnable cross-modal parameters), which may lead to better generalization on small datasets.

### 2.2.3 The Additive Fusion Pathology

**Proposition 2.8 (Why Additive Fusion Fails).** The original toy model uses:

F(A, B) = A + B   where A ∈ ℝ^{256 × 32 × 32}, B ∈ ℝ^{256 × 32 × 32}

where B = expand(MLP(flatten(T))) broadcasts a single 256-dimensional vector to all spatial positions. This imposes two constraints:

1. **Spatial uniformity**: B[c, i, j] = B[c, i', j'] for all (i,j), (i',j'). The topological modulation is identical at every spatial position.

2. **Channel-wise addition**: F[c, i, j] = A[c, i, j] + B[c]. Each output channel is shifted by a constant, independent of other channels.

The combined effect: the topological branch can only add a global bias to each feature channel. If the spatial encoder already produces reasonable features, the optimal B is near zero (adding bias increases loss), so the topological information is effectively ignored.

**Example 2.2 (The Volume Knob Analogy).** Additive fusion is like giving a sound engineer two tracks (spatial and topological) but only allowing them to adjust the volume of the topological track globally and add it to the spatial track. They can't make the topology louder in certain parts and quieter in others; they can't mix specific topological channels with specific spatial channels. Concatenation fusion is like giving them a full mixing board where every channel combination is independently controllable.

## 2.3 The Training Objective as Optimization on Function Space

### 2.3.1 The Composite Loss Functional

**Definition 2.7 (Composite Segmentation Loss).** The training objective is a functional L: Θ → ℝ defined as:

L(θ) = 𝔼_{(I,T,M) ~ D} [λ_dice · L_Dice(f_θ(I,T), M) + λ_topo · L_BM(f_θ(I,T), M)]

where D is the data distribution, L_Dice is the soft Dice loss, and L_BM is the Betti matching loss.

**Definition 2.8 (Soft Dice Loss).** For prediction P ∈ [0,1]^{H×W} and ground truth M ∈ {0,1}^{H×W}:

L_Dice(P, M) = 1 - (2 Σ_{i,j} P_{ij} M_{ij} + ε) / (Σ_{i,j} P_{ij} + Σ_{i,j} M_{ij} + ε)

This is differentiable everywhere (ε > 0 prevents division by zero) and provides gradient signal proportional to volumetric overlap.

### 2.3.2 The Curriculum Principle

**Proposition 2.9 (Curriculum Training Justification).** Consider the optimization landscape of L(θ) = λ_d · L_Dice(θ) + λ_t · L_BM(θ). At initialization (random θ):
- L_Dice provides strong, dense gradients (every pixel contributes)
- L_BM provides near-zero gradients (Proposition 1.2: near-uniform predictions have trivial topology)

As training progresses and predictions develop spatial structure:
- L_Dice gradients diminish (approaching local minimum for volumetric overlap)
- L_BM gradients emerge (predictions have non-trivial persistence diagrams with features to correct)

Therefore, the schedule λ_t(epoch) = 0 → 1 (increasing topological weight) aligns the loss landscape with the optimization trajectory. This is a form of **continuation method** — solving a sequence of progressively harder optimization problems.

**Analogy 2.4 (Learning to Draw).** Teaching a child to draw follows a curriculum: first learn basic shapes (Dice loss — get the rough outline right), then refine the structure (topological loss — make sure the eyes have pupils, the mouth doesn't float separately, the body is connected). If you demanded structural perfection from the start, the child would be overwhelmed by a signal they can't yet interpret. The curriculum principle recognizes that structural feedback is only useful once a basic sketch exists.

**[TO FORMALIZE]** Formal statement as a convergence result: under what conditions does curriculum training converge to a lower-loss solution than constant-weight training?

---

# Chapter 3: Synthesis — The TDA-SegUNet Architecture

## 3.1 The Complete Pipeline as a Composition of Mathematical Objects

### 3.1.1 The Full Computational Graph

The TDA-SegUNet pipeline involves the following mathematical objects and maps:

```
INPUT SPACE:
  Image: I ∈ ℝ^{C_s × H × W}      (e.g., C_s=1 for synthetic, C_s=4 for BraTS)
  Mask:  M ∈ {0,1}^{H × W}          (ground truth, available during training)

TDA PREPROCESSING (operates on M during training, on predictions during inference):
  EDT:     M ↦ EDT_M ∈ ℝ^{H × W}                    (Euclidean distance transform)
  Cubical: EDT_M ↦ K(EDT_M)                           (cubical complex construction)
  PH:      K(EDT_M) ↦ (Dgm_0, Dgm_1)                (persistent homology computation)
  PI:      (Dgm_0, Dgm_1) ↦ (PI_0, PI_1) ∈ ℝ^{2 × H × W}  (persistence image vectorization)

NEURAL NETWORK:
  Input Fusion:      f_θ: ℝ^{(C_s+2) × H × W} → [0,1]^{1 × H × W}
  Bottleneck Fusion: f_θ: ℝ^{C_s × H × W} × ℝ^{2 × H × W} → [0,1]^{1 × H × W}

LOSS COMPUTATION (during training):
  L_Dice: [0,1]^{H×W} × {0,1}^{H×W} → ℝ₊         (volumetric overlap)
  L_BM:   [0,1]^{H×W} × {0,1}^{H×W} → ℝ₊         (topological fidelity)

TOTAL: L(θ) = λ_d · L_Dice(f_θ(I, PI(M)), M) + λ_t · L_BM(f_θ(I, PI(M)), M)
```

**Analogy 3.1 (A Medical Imaging Pipeline).** The full pipeline is like a diagnostic workflow: the MRI machine produces raw images (input), a radiologist's assistant prepares structural annotations (TDA preprocessing → persistence images), the radiologist examines both the images and annotations simultaneously (neural network with fusion), and two quality checks verify the result — one checking that enough tumor is identified (Dice loss) and one checking that the tumor's structure makes anatomical sense (Betti matching loss).

### 3.1.2 Well-Definedness of the Pipeline

**Proposition 3.1.** The complete pipeline is well-defined under the following conditions:

1. **EDT well-definedness**: M must have both foreground and background pixels (otherwise EDT is degenerate). This is satisfied whenever the tumor does not fill the entire image.

2. **Cubical complex well-definedness**: Always well-defined for any function on a grid.

3. **Persistent homology computability**: Cubical persistent homology on an N×N grid is computable in O(N² · α(N²)) for H₀ and O(N⁶) worst-case for H₁ (cubic in number of cells), but typical medical images admit much faster computation via the matrix reduction algorithm.

4. **Persistence image well-definedness**: Requires removing infinite-persistence points from H₀ (the single essential class). The PI vectorizer must handle empty diagrams (H₁ of a disk) by returning zeros.

5. **Network domain matching**: Input tensor dimensions must satisfy the U-Net divisibility constraints (H, W divisible by 2^L where L is the number of encoder levels).

6. **Loss differentiability**: L_Dice is differentiable everywhere (with ε-smoothing). L_BM is differentiable at generic points (Theorem 1.3) but may have measure-zero non-differentiable points.

### 3.1.3 The Inference Gap

**Remark 3.1 (Training vs. Inference Asymmetry).** During training, TDA features are computed from the clean ground truth mask M. During inference, M is unknown — it is precisely what we are trying to predict. Two strategies address this:

**Strategy A (Two-Pass Inference):**
1. First pass: compute f_θ(I, 0) with zero topological features → rough prediction P₁
2. Compute TDA features from thresholded P₁
3. Second pass: compute f_θ(I, PI(P₁ > 0.5)) → refined prediction P₂

**Strategy B (Inference without TDA):** Train the model to be robust to the presence/absence of topological features (e.g., randomly zero-out topo channels during training as a form of dropout). At inference, use only spatial features.

**Analogy 3.2 (The Two-Pass Strategy).** Strategy A is like writing a first draft (pass 1), then going back to analyze the draft's structure and outline (TDA on P₁), and using that outline to write an improved second draft (pass 2). The first draft doesn't need to be perfect — it just needs to be structured enough to produce a useful outline.

**[TO FORMALIZE]** Under what conditions does Strategy A converge? Is P₂ guaranteed to be "better" than P₁ in topological metrics?

## 3.2 Theoretical Guarantees

### 3.2.1 Topological Correctness Guarantee

**Theorem 3.1 (adapted from Hu et al., 2019).** If the Betti matching loss satisfies L_BM(P, M) = 0 for a prediction P ∈ [0,1]^{H×W} and ground truth M, then for any generic threshold t ∈ (0,1), the binarized prediction P_t = (P > t) satisfies:

β_k(|K(P_t)|) = β_k(|K(M)|)   for all k

That is, zero topological loss guarantees that the thresholded prediction has the same Betti numbers as the ground truth.

**Remark 3.2.** This is a necessary but not sufficient condition for full topological correctness. Having the same Betti numbers does NOT imply the same topology. For instance, a torus and a genus-1 surface with a handle have different topologies but both have β₁ = 2. However, for segmentation purposes, matching Betti numbers at the correct spatial locations (which Betti matching ensures via Theorem 1.5) is sufficient for clinical applications — the spatial matching guarantees that each detected component, hole, or cavity corresponds to a real anatomical structure.

### 3.2.2 The Role of Each Component

**Theorem 3.2 (Informal Ablation Theorem).** [TO FORMALIZE] Consider four models trained on the same data:

1. U-Net with Dice loss only (baseline)
2. U-Net with Dice + topological loss (topological supervision only)
3. U-Net with TDA input channels and Dice loss (topological features only)
4. U-Net with TDA input channels and Dice + topological loss (full model)

Under mild conditions:
- Model 2 ≥ Model 1 in topological metrics (topological loss improves topology)
- Model 3 ≥ Model 1 in Dice score on samples where topology is discriminative
- Model 4 ≥ max(Model 2, Model 3) in BOTH metrics (the effects are complementary)

The topological input features help the network LEARN correct topology. The topological loss ENFORCES correct topology. They operate at different points in the pipeline and their benefits should compound.

**Analogy 3.3 (Seeing vs. Being Graded).** Having TDA input channels is like giving a student a textbook about shape (they can *see* the topological information). Having a topological loss is like grading them on structural correctness (they are *incentivized* to produce correct topology). The best student both reads the textbook AND is graded on structure.

## 3.3 Open Questions for Further Development

1. **Optimal Filtration Choice:** Is EDT the best filtration for brain tumors? Alternatives: intensity-based filtration, Rips complex on point clouds, Delaunay filtration. Each captures different geometric aspects.

2. **3D Extension:** All results in this document are stated for 2D. The 3D extension introduces H₂ (cavities), which is computationally expensive but clinically relevant (necrotic cores are H₂ features). What are the complexity bounds for 3D cubical PH on medical image volumes?

3. **Multi-Scale Persistence:** Brain tumors have features at multiple scales (fine peritumoral infiltration, medium enhancing rim, large edema). Can multi-scale persistence (zigzag persistence, multi-parameter persistence) capture this hierarchy?

4. **Convergence of Two-Pass Inference:** Under what conditions on the trained model does iterating TDA extraction → prediction converge to a fixed point?

---

# Chapter 4: Geometric Deep Learning — The Symmetry Framework

## 4.1 Motivation: Why Symmetry Matters

The translation equivariance of CNNs (Remark 2.1) is a specific instance of a much broader principle: neural network architectures should respect the symmetries of their input domains. Bronstein et al. (2021) established a unifying framework — the "5G" framework — that organizes all major neural network architectures through this lens.

**Analogy 4.1 (The Rules of a Board Game).** Every board game has rules about how pieces can move. In chess, a rook moves in straight lines; in checkers, pieces move diagonally. These movement rules are symmetries: they define what transformations are "legal." Similarly, every data domain has symmetries that a neural network should respect: images have translations, molecules have rotations, social networks have node permutations. A network that violates these symmetries is like a chess player who ignores the rules — it might occasionally make a good move, but it wastes most of its effort on illegal (and therefore useless) possibilities.

## 4.2 The Five Gs

**Definition 4.1 (The 5G Framework, Bronstein et al., 2021).** Neural network architectures can be organized by the symmetry group of their input domain:

1. **Grids** — Translation symmetry → Convolutional Neural Networks (CNNs). The input domain is ℤ^d (a regular grid). The symmetry group is the translation group ℤ^d. Weight sharing (applying the same kernel everywhere) is the architectural implementation of translation equivariance. This is the foundation of all image-processing neural networks, including U-Net.

2. **Groups** — Rotation/scale symmetry → Group-Equivariant CNNs. Beyond translations, images may have rotational or scale symmetry. A brain tumor looks the same regardless of the patient's head orientation. Group-equivariant CNNs extend weight sharing to rotation groups (e.g., the cyclic group C_n or the continuous rotation group SO(2)), ensuring that rotated inputs produce correspondingly rotated outputs.

3. **Graphs** — Permutation symmetry → Graph Neural Networks (GNNs). Brain connectivity networks are graphs where nodes are brain regions and edges are connections. The labeling of nodes is arbitrary (a symmetry), so the network must be permutation-equivariant: relabeling nodes should relabel outputs identically. Message-passing neural networks implement this via neighborhood aggregation.

4. **Geodesics** — Intrinsic operations on Riemannian manifolds → Manifold Neural Networks. The cortical surface of the brain is a curved 2D manifold embedded in 3D space. Operations on this surface should be defined intrinsically (independent of the embedding), using geodesic distances and parallel transport rather than Euclidean distances.

5. **Gauges** — Local symmetry transformations on fiber bundles → Gauge Equivariant CNNs. On a general manifold, there is no global coordinate system for defining filter orientations. Feature maps become sections of associated vector bundles, and the network must be equivariant to local gauge transformations (changes of reference frame). This is the most general and mathematically sophisticated level.

**Example 4.1 (Relevance to Brain Imaging).** Each of the 5G levels appears in brain tumor analysis:
- **Grids**: Standard MRI voxel grids → U-Net for segmentation
- **Groups**: Tumors can appear at any orientation → rotation-equivariant convolutions
- **Graphs**: Brain connectivity networks → GNNs for functional analysis (e.g., Bhattacharya et al. 2024 using PH on connectome graphs)
- **Geodesics**: Cortical surface analysis → intrinsic operations for detecting cortical tumor infiltration
- **Gauges**: Diffusion MRI on the cortical manifold → gauge equivariant CNNs (Hussain & Khan 2025, achieving angular resolution upsampling with 5-15 training subjects vs. 20-40 for baselines)

## 4.3 From Translation Equivariance to General Equivariance

**Definition 4.2 (Equivariant Map).** Let G be a group acting on spaces X and Y via representations ρ_X and ρ_Y. A map f: X → Y is G-equivariant if:

f(ρ_X(g) · x) = ρ_Y(g) · f(x)   ∀ g ∈ G, x ∈ X

In words: transforming the input and then applying f gives the same result as applying f and then transforming the output.

**Example 4.2 (Equivariance Hierarchy).** Consider a 2D image of a brain tumor:
- Translation equivariance (CNN): shifting the image shifts the segmentation identically
- Rotation equivariance (Group-CNN): rotating the image by 90° rotates the segmentation by 90°
- Scale equivariance: zooming in produces a zoomed-in segmentation

Standard CNNs have translation equivariance by construction but NOT rotation or scale equivariance. This means they must learn rotation invariance from data augmentation — a less efficient approach than building it into the architecture.

**Proposition 4.1 (Equivariance Reduces Sample Complexity).** If a function class F is constrained to be G-equivariant for a group G of size |G|, the effective hypothesis space is reduced by a factor of approximately |G|. Intuitively, each training example simultaneously teaches the network about |G| transformed versions of itself. This explains why CNNs (exploiting the enormous translation group) require far fewer parameters and examples than fully connected networks for image tasks.

**Remark 4.1 (Practical Impact).** Gauge equivariant CNNs (Cohen et al., 2019) applied to diffusion MRI achieve comparable angular resolution upsampling with 5-15 training subjects versus 20-40 for baselines — a 2-4× reduction in data requirements, directly attributable to exploiting the gauge symmetry of the diffusion signal on the cortical manifold.

## 4.4 Implications for Topology-Aware Architectures

The geometric deep learning framework provides two key insights for our work:

1. **The topology-aware U-Net respects grid symmetry** (translation equivariance), which is appropriate for voxel-based MRI analysis. The persistence images, being computed on the same grid, are naturally compatible with this symmetry.

2. **Future extensions** to cortical surface analysis (manifold data) or brain connectivity networks (graph data) would require moving to higher levels of the 5G hierarchy. Simplicial Neural Networks (using Hodge Laplacians) and Cell Complex Neural Networks provide principled frameworks for learning on these higher-order topological structures, though their application to brain tumor analysis remains an open research direction.

---

# Chapter 5: Survival Analysis and Cox Proportional Hazards

## 5.1 Motivation

The ultimate clinical goal of brain tumor analysis is not just accurate segmentation — it is predicting patient outcomes. Survival analysis provides the statistical framework for this prediction, and when combined with topological features from Chapter 1, it creates a powerful bridge from tumor shape to prognosis.

## 5.2 Mathematical Foundations of the Cox Model

### 5.2.1 The Hazard Function

**Definition 5.1 (Hazard Function).** For a non-negative random variable T (time to event, e.g., death), the hazard function is:

h(t) = lim_{Δt→0} P(t ≤ T < t + Δt | T ≥ t) / Δt

Intuitively, h(t)·Δt is the probability of the event occurring in the next small interval [t, t+Δt), given survival up to time t.

**Analogy 5.1 (The Hazard as Instantaneous Risk).** Think of the hazard function as the "danger level" at each moment. If you're driving a car, h(t) is how dangerous the road is at mile marker t, given that you've safely made it that far. A mountain pass (tumor progression) might have high hazard; a straight highway (remission period) has low hazard. The hazard can change over time, reflecting different phases of disease.

### 5.2.2 The Proportional Hazards Assumption

**Definition 5.2 (Cox Proportional Hazards Model).** The Cox model defines a patient's hazard as:

h(t | x) = h₀(t) · exp(f(x))

where h₀(t) is an unspecified baseline hazard (common to all patients) and f(x) is a log-risk function that depends on patient covariates x. The key assumption is that covariates act multiplicatively on the hazard — they shift the baseline risk up or down by a constant factor, regardless of time.

In classical CoxPH, f(x) = βᵀx is a linear function. In deep extensions (DeepSurv, SCNN), f(x) = g_θ(x) is a neural network.

**Example 5.1 (Proportional Hazards Intuition).** If patient A has exp(f(x_A)) = 2 and patient B has exp(f(x_B)) = 1, then patient A's hazard is always twice patient B's at every time point. The "shape" of the hazard curve is the same (determined by h₀(t)), but A's curve is scaled up. This is like two cars on the same road: A drives twice as fast (twice the risk), but both encounter the same sequence of mountain passes and highways.

### 5.2.3 The Cox Partial Likelihood

**Definition 5.3 (Cox Partial Likelihood).** The remarkable property of the Cox model is that the baseline hazard h₀(t) can be eliminated from the likelihood. Given n patients with event/censoring times t₁, ..., t_n and event indicators δ₁, ..., δ_n (δ_i = 1 if patient i experienced the event), the partial likelihood is:

L(θ) = ∏_{i: δᵢ=1} [exp(f(xᵢ)) / Σ_{j∈R(tᵢ)} exp(f(xⱼ))]

where R(tᵢ) = {j : t_j ≥ t_i} is the risk set at time tᵢ (all patients still at risk just before time tᵢ).

**Analogy 5.2 (The Horse Race).** Each observed event is like a horse race where we know which horse won (which patient experienced the event) and which horses were still in the race (the risk set). The partial likelihood says: among all patients who could have experienced the event at time tᵢ, what is the probability that it was patient i specifically? The model assigns higher probability to the patient with the highest predicted risk exp(f(xᵢ)), normalized by the sum of all risks in the race.

The negative log partial likelihood serves as a fully differentiable loss function:

ℓ(θ) = -Σ_{i: δᵢ=1} [f(xᵢ) - log(Σ_{j∈R(tᵢ)} exp(f(xⱼ)))]

enabling backpropagation through the entire network.

**Remark 5.1 (Risk Set Computation).** A critical practical consideration: the partial likelihood requires computing over the ENTIRE risk set for each event time, making standard mini-batch SGD non-trivial. Implementations typically either process the full training set per gradient update (feasible for survival datasets which are typically hundreds to thousands of patients, not millions) or use careful approximations (e.g., Efron's or Breslow's methods for tied event times).

### 5.2.4 Evaluation: The Concordance Index

**Definition 5.4 (Concordance Index).** The C-index measures the fraction of all comparable patient pairs where the model correctly identifies the patient who experiences the event first:

C = P(f(x_i) > f(x_j) | t_i < t_j, δ_i = 1)

A C-index of 0.5 is random guessing (no discrimination); 1.0 is perfect ranking.

**Example 5.2 (Clinical C-index Values).** For brain tumor survival:
- Random baseline: C = 0.5
- Clinical WHO paradigm (histologic grading + molecular markers): C ≈ 0.747
- DeepSurv on tabular clinical data: C ≈ 0.68-0.72
- SCNN/GSCNN (CNN + Cox on histopathology): C ≈ 0.801 (surpassing WHO)
- FCoxPH with persistent homology features: significant survival association (p < 0.001)
- BraTS OS prediction (segmentation + radiomics + ML): C ≈ 0.55-0.70

## 5.3 DeepSurv: Neural Cox Models

**Definition 5.5 (DeepSurv, Katzman et al., 2018).** DeepSurv replaces the linear predictor βᵀx in CoxPH with a deep feed-forward network:

f_θ(x) = FC_L ∘ σ ∘ ... ∘ FC_2 ∘ σ ∘ FC_1(x)

where FC_l are fully connected layers with batch normalization and dropout, and σ is a nonlinear activation (ReLU or SELU). The output is a single scalar log-risk score. Training uses average negative log partial likelihood with L₂ regularization.

**Analogy 5.3 (From Linear to Nonlinear Risk).** Traditional survival analysis predicts marathon finish times with a simple formula: time = base_pace × (age_factor + weight_factor). DeepSurv replaces this with a network that learns complex interactions: "this runner's age penalty is actually offset by their specific training history in THIS particular way." The prediction is no longer a simple sum of independent factors but a learned nonlinear function of all characteristics simultaneously.

## 5.4 End-to-End Imaging + Survival: The SCNN

**Definition 5.6 (Survival CNN, Mobadersany et al., 2018).** The SCNN passes histology image patches through VGG-like convolutional layers → fully connected layers → single Cox proportional hazards output node, trained end-to-end with negative log partial likelihood:

ℓ(θ) = -Σ_{i: δᵢ=1} [g_θ(Image_i) - log(Σ_{j∈R(tᵢ)} exp(g_θ(Image_j)))]

The ENTIRE CNN learns survival-relevant visual features directly, without hand-crafted feature engineering.

**Example 5.3 (What the SCNN Sees).** Heat map visualizations confirm that the SCNN learns to recognize clinically meaningful structures: microvascular proliferation (associated with aggressive tumors), necrosis extent, cell density, and nuclear morphology. These are the same features pathologists use for grading — the network rediscovers pathological knowledge from survival supervision alone.

The Genomic SCNN (GSCNN) incorporates molecular variables (IDH mutation, 1p/19q codeletion) at the fully connected layers, achieving C-index ≈ 0.801 on TCGA gliomas — surpassing the WHO clinical paradigm.

**Analogy 5.4 (The Self-Taught Building Inspector).** The SCNN is like training a building inspector by showing them thousands of photos of building foundations alongside the building's actual lifespan. Instead of giving them a checklist, you let them figure out what visual patterns predict structural longevity. The GSCNN additionally gives them the architect's blueprints (genomic data), combining visual inspection with structural knowledge.

## 5.5 Functional Cox PH with Topological Features: The Bridge

The most direct synthesis of algebraic topology and survival analysis is the **Functional Cox Proportional Hazards (FCoxPH) model** (Moon et al., 2023), which uses persistent homology features as functional predictors.

**Definition 5.7 (FCoxPH Pipeline).** The pipeline proceeds:

1. **Segmentation:** Tumor segmented from MRI → binary mask M
2. **Distance Transform:** M ↦ EDT_M
3. **Persistent Homology:** EDT_M ↦ (Dgm_0, Dgm_1, Dgm_2) via cubical complex PH
4. **Functional Representation:** Persistence diagrams represented as functional predictors (e.g., Betti curves, persistence landscapes)
5. **Functional Cox Regression:** h(t | PH_features) = h₀(t) · exp(∫ β(s) · PH(s) ds + interactions)

**Example 5.4 (What Topology Predicts).** Applied to 77 brain tumor patients, the FCoxPH model found that irregular, heterogeneous shape patterns captured by topological features are positively associated with survival hazards (p < 0.001). A tumor with many short-lived H₀ features (fragmented components that quickly merge) has a messier boundary — and this messiness predicts worse survival. This is information that standard radiomics (volume, surface area, mean intensity) cannot capture.

**Remark 5.2 (The Complete Circle).** The FCoxPH pipeline closes a complete circle:
- Chapter 1 provides the mathematical foundation for computing persistent homology from brain MRI
- Chapter 2 provides the neural network framework for producing segmentation masks
- Chapter 5 shows how the topological features of those masks predict patient survival
- Together, they form an end-to-end pipeline: MRI → segmentation (Ch. 2+3) → topological features (Ch. 1) → survival prediction (Ch. 5)

---

# Chapter 6: Evaluation Metrics — Measuring What Matters

## 6.1 Motivation: The Metric-Method Gap

A recurring theme throughout this document is the disconnect between what segmentation methods optimize and what clinicians need. Standard metrics (Dice, Hausdorff) measure pixel-level accuracy but are blind to topological structure. Topology-aware methods are designed to preserve structural integrity, but are typically evaluated with topology-blind metrics. This chapter formalizes the metric landscape and its implications.

## 6.2 Standard Metrics and Their Topological Blindness

**Definition 6.1 (Dice Similarity Coefficient).** For prediction P and ground truth G (both binary):

DSC = 2|P ∩ G| / (|P| + |G|)

DSC measures volumetric overlap. Soft Dice (Definition 2.8) is differentiable and is the primary metric across BraTS challenges.

**Definition 6.2 (Hausdorff Distance 95%).** HD95 is the 95th percentile of the set of directed Hausdorff distances between prediction and ground truth boundaries:

HD95 = max(d_95(∂P, ∂G), d_95(∂G, ∂P))

where d_95 is the 95th percentile of distances from one boundary to the nearest point on the other.

**Example 6.1 (Topological Blindness of Dice).** Consider a blood vessel segmentation where the prediction achieves Dice = 0.95, missing only 5% of vessel pixels. If those 5% happen to be at a critical junction, the vessel network may be topologically disconnected: one connected tree becomes three fragments (β₀ jumps from 1 to 3). A clinician planning a surgical approach based on vessel connectivity would be catastrophically misled. Dice sees 95% overlap (excellent); topology sees shattered connectivity (disastrous).

**Analogy 6.1 (Grading a Bridge).** Dice is like grading a bridge by the total weight of steel used (volumetric overlap). If 95% of the steel is in place, it gets an A grade. But if the missing 5% is the keystone at the center, the bridge collapses. Topological metrics are like checking whether the bridge actually connects the two sides of the river — the *structural* test that determines whether the bridge functions.

## 6.3 Topological Metrics

**Definition 6.3 (Betti Number Error).** The global Betti number error is:

β_err = Σ_d |β_d(P) - β_d(G)|

counting the mismatch in connected components (d=0), holes (d=1), and cavities (d=2). For β₀, computation is O(n·α(n)) via union-find; for higher dimensions, cubical complex computation is O(n³) worst-case.

**Remark 6.1 (Cancellation Problem).** Betti number error suffers from cancellation: if a prediction adds one spurious component and loses one real component, β₀ error = 0, masking two genuine topological errors. This is like a census that counts 100 people entering and 100 leaving a city — the net change is zero, but the city's population has completely turned over.

**Definition 6.4 (Betti Matching Error, Stucki et al., 2023).** Using the induced matching from Definition 1.14:

τ_err = |unmatched features in Dgm(P)| + |unmatched features in Dgm(G)|

Features that don't find a spatially corresponding partner are counted as topological errors. This resolves the cancellation problem because it counts features individually, not in aggregate.

**Example 6.2 (Betti Number Error vs. Betti Matching Error).** Suppose the ground truth has two tumor foci (A and B) and the prediction has two foci (A' and C), where A' spatially overlaps A but C is a hallucination in a completely different location. Betti number error: β₀(P) = 2 = β₀(G), so β₀ error = 0 (no error detected). Betti matching error: A↔A' is matched (zero error), B is unmatched in G (one error), C is unmatched in P (one error), so τ_err = 2. Betti matching correctly identifies both the missed focus and the hallucinated focus, while Betti number error sees nothing wrong.

**Definition 6.5 (Wasserstein Distance Between Persistence Diagrams).** The p-Wasserstein distance:

W_p(D₁, D₂) = (inf_γ Σ_i ||p_i - γ(p_i)||_∞^p)^{1/p}

measures the optimal total matching cost across all topological features. Used as the loss function in TopoLoss (Hu et al. 2019), it is differentiable but, as noted in §1.3.4, can be spatially incorrect.

**Definition 6.6 (clDice).** As defined in §1.5.1. Efficient O(n) computation via skeleton operations. Primarily meaningful for tubular structures.

## 6.4 The Connectivity Pitfall

**Proposition 6.1 (Connectivity Sensitivity, Berger et al., 2025).** The choice of pixel connectivity dramatically affects computed topology:
- In 2D: 4-connectivity (sharing edges only) versus 8-connectivity (sharing edges or corners)
- In 3D: 6-connectivity versus 26-connectivity

The same segmentation can have β₀ = 1 under 8-connectivity but β₀ = 3 under 4-connectivity. Published comparisons between methods using different connectivity conventions are INVALID.

**Example 6.3 (The Diagonal Pixel Problem).** Consider three pixels arranged diagonally: (1,1), (2,2), (3,3). Under 4-connectivity (only edge neighbors count), these are three separate components (β₀ = 3). Under 8-connectivity (corner neighbors also count), these form one connected component (β₀ = 1). A "topological improvement" reported in one paper might simply reflect a different connectivity convention, not a better algorithm.

**Remark 6.2 (Standardization).** Berger et al. (2025) further demonstrated that distributional metrics like Variation of Information (VOI) and Adjusted Rand Index (ARI) "irreversibly entangle topological and volumetric errors" — a method can improve VOI by fixing volume without fixing topology, and the metric cannot distinguish the two. They recommend Betti matching with explicit connectivity specification as the gold standard topological metric.

## 6.5 Practical Metric Selection

The following table summarizes the metric landscape with computational costs, differentiability, and topological sensitivity:

| Metric | Complexity | Differentiable | Captures Topology | Key Limitation |
|--------|-----------|----------------|-------------------|----------------|
| Dice/IoU | O(n) | Yes (soft) | No | Topologically blind |
| HD95 | O(n) with DT | Approximated | No | Sensitive to outliers |
| Betti number error | O(n³) for β₁+ | No | Partially | Cancellation; no spatial info |
| Betti matching error | O(n³) | Yes | Yes (spatially correct) | Computational cost |
| Wasserstein (PD) | O(k³) | Yes | Yes (not spatial) | Spatial mismatch |
| clDice | O(n) | Yes (soft) | Tubular only | Not for compact regions |
| Euler χ error | O(n) | No | Minimal | Cancellation (β₀ − β₁ + β₂) |

**Recommendation for Brain Tumor Segmentation.** For BraTS and brain tumor applications, the current standard (DSC, HD95) should be supplemented with: β₀ error (detecting merged/split tumors, minimal computational cost), Betti matching error (gold standard spatial topological metric, higher cost), and connectivity convention explicitly specified and standardized across comparisons.

---

## References

- Adams, H., et al. (2017). Persistence images: a stable vector representation of persistent homology. JMLR, 18(8), 1-35.
- Bauer, U., Lesnick, M. (2015). Induced matchings and the algebraic stability of persistence barcodes. J. Comput. Geom., 6(2), 162-191.
- Berger, C., Grönke, A., Stucki, N., Bauer, U. (2025). Pitfalls of topology-aware image segmentation. IPMI 2025. arXiv:2412.14619.
- Bronstein, M. M., Bruna, J., Cohen, T., Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv:2104.13478.
- Brüel-Gabrielsson, R., et al. (2020). A topology layer for machine learning. AISTATS.
- Clough, J. R., et al. (2020). A topological loss function for deep-learning based image segmentation using persistent homology. IEEE TPAMI. arXiv:1910.01877.
- Cohen, T. S., Weiler, M., Kicanaoglu, B., Welling, M. (2019). Gauge equivariant convolutional networks and the icosahedral CNN. ICML 2019.
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability of persistence diagrams. DCG, 37(1), 103-120.
- Crawford, L., Monod, A., Chen, A. X., Mukherjee, S., Rabadán, R. (2020). Predicting clinical outcomes in glioblastoma: An application of topological and functional data analysis. JASA.
- Edelsbrunner, H., Harer, J. (2010). Computational Topology: An Introduction. AMS.
- Forman, R. (1998). Morse theory for cell complexes. Advances in Mathematics, 134(1), 90-145.
- François, O., Tinarrage, R. (2024). Train-free segmentation in MRI with cubical persistent homology. arXiv:2401.01160.
- Hu, X., Li, F., Samaras, D., Chen, C. (2019). Topology-preserving deep image segmentation. NeurIPS.
- Hu, X., Wang, Y., Fuxin, L., Samaras, D., Chen, C. (2021). Topology-aware segmentation using discrete Morse theory. ICLR (Spotlight).
- Katzman, J., et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology, 18, 24.
- Latecki, L., et al. (2000). Well-composed sets. Computer Vision and Image Understanding, 61(1), 70-83.
- Leygonie, J., Oudot, S., Tillmann, U. (2022). A framework for differential calculus on persistence barcodes. FoCM, 22, 1069-1131.
- Mobadersany, P., et al. (2018). Predicting cancer outcomes from histology and genomics using convolutional networks. PNAS, 115(13), E2970-E2979.
- Moon, H., Yoon, J., Lazar, N. A., Huo, Y. (2023). Functional Cox proportional hazards model with persistent homology features. Annals of Applied Statistics.
- Shit, S., Paetzold, J. C., et al. (2021). clDice — A novel topology-preserving loss function for tubular structure segmentation. CVPR 2021.
- Stucki, N., Paetzold, J. C., Shit, S., Menze, B., Bauer, U. (2023). Topologically faithful image segmentation via induced matching of persistence barcodes. ICML 2023.
- Stucki, N., et al. (2024). Efficient Betti matching enables topology-aware 3D segmentation via persistent homology. arXiv:2407.04683.
- Stucki, N., et al. (2024). Topologically faithful multi-class segmentation in medical images. MICCAI 2024. arXiv:2403.11001.
- Zhou, D.-X. (2020). Universality of deep convolutional neural networks. Applied and Computational Harmonic Analysis, 48(2), 787-794.

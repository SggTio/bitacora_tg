# Detailed Paper Analysis — Section A

## Topological Data Analysis for Image Segmentation

---

### TIER 1 PAPER 1: Stability of Persistence Diagrams
**Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Discrete & Computational Geometry, 37(1), 103–120.**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Persistence Diagram:** Given a tame function f: X → ℝ on a topological space X, the persistence diagram Dgm(f) is the multiset of points (b_i, d_i) in the extended plane, where b_i and d_i are the birth and death times of homological features in the sublevel set filtration {f⁻¹(-∞, t]}.
- **Bottleneck Distance:** d_B(Dgm(f), Dgm(g)) = inf_γ sup_p ||p - γ(p)||_∞, where γ ranges over all bijections between diagrams (including the diagonal).
- **Tame Function:** A function f: X → ℝ such that homology groups of sublevel sets are finite-dimensional and changes occur at finitely many values.

**Main Theorem (Stability):** If f, g: X → ℝ are tame functions on a triangulable space X, then:

> d_B(Dgm(f), Dgm(g)) ≤ ||f - g||_∞

The bottleneck distance between persistence diagrams is bounded by the L∞ distance between the functions.

**Additional Result:** The theorem extends to the p-Wasserstein distance: W_p(Dgm(f), Dgm(g)) ≤ C_p · ||f - g||_∞ for appropriate constants.

#### 2. Problem, Scope, Prerequisites

**Problem:** Persistent homology provides topological summaries of data, but is this summary robust to perturbations? If you add small noise to a function, does the persistence diagram change drastically?

**Scope:** The result applies to any tame function on a triangulable topological space. This covers: images (functions on grids), point cloud data (distance functions), manifolds (height functions), and essentially all cases encountered in TDA applications.

**Prerequisites:** Algebraic topology (homology groups, exact sequences), filtered complexes, persistence modules. Understanding the interleaving of sublevel sets is essential.

#### 3. Proof Roadmap

**Step 1 — The Box Lemma:** Consider two functions f, g with ||f - g||_∞ ≤ ε. Then the sublevel sets satisfy: f⁻¹(-∞, t-ε] ⊆ g⁻¹(-∞, t] ⊆ f⁻¹(-∞, t+ε]. This interleaving of sublevel sets is the geometric foundation.

**Step 2 — Induced Maps on Homology:** The inclusions from Step 1 induce maps on homology groups. These maps create a "ε-interleaving" of persistence modules: the persistence module of f and the persistence module of g are algebraically interleaved at distance ε.

**Step 3 — From Interleaving to Matching:** The key technical lemma (later made precise by Bauer-Lesnick 2015) shows that an ε-interleaving of persistence modules implies an ε-matching of their barcodes/diagrams. This is proved using the structure theorem for persistence modules (decomposition into interval modules) and a careful combinatorial matching argument.

**Step 4 — Optimality:** The bound is tight: there exist functions f, g with d_B(Dgm(f), Dgm(g)) = ||f - g||_∞ exactly.

**Mathematical Intuition:** The sublevel sets of f and g are "ε-close" in the inclusion sense. Since homology is functorial, this closeness propagates to homology groups. Topological features that persist in f must correspond to nearby features in g, and vice versa.

#### 4. New Problems Discovered

- The stability result is with respect to **bottleneck distance**, but applications often need **Wasserstein stability** (more discriminative but less stable).
- The bound is in terms of **L∞ norm** — very large local perturbations in small regions can dominate. This motivated work on Lipschitz stability with respect to other norms.
- **Thresholding is discontinuous:** Stability of persistence diagrams does NOT imply stability of thresholded binary images. This is exactly why computing TDA on thresholded noisy images (as in the original toy model) produces unreliable results.
- Multi-parameter persistence (functions f: X → ℝ^n) lacks a clean stability theory — this remains an active research area.

#### 5. Simple Analogy

Imagine you're drawing a topographic map of a mountain range. Each mountain peak represents a "born" topological feature, and the saddle point where it merges with a higher peak represents its "death." The persistence diagram records (elevation_born, elevation_merged) for each peak.

Now imagine it rains, raising all water levels by ε meters. The stability theorem says the new topographic map's peaks and saddles shift by at most ε meters — no peak can teleport to a distant location or appear/disappear suddenly. Small perturbations in the landscape cause small perturbations in the topographic summary.

But if you drew a contour line at a fixed elevation (thresholding), the contour could change dramatically — islands might merge or split with a tiny water level change. This is why thresholding is dangerous for topology.

---

### TIER 1 PAPER 2: Topology-Preserving Deep Image Segmentation (TopoLoss)
**Hu, X., Li, F., Samaras, D., Chen, C. (NeurIPS 2019). arXiv:1906.05404**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Likelihood Map:** A function f: Ω → [0,1] where Ω is the image domain, produced by a neural network. f(x) represents the probability that pixel x belongs to the foreground.
- **Superlevel Set Filtration:** As threshold α decreases from 1 to 0, the thresholded set f^α = {x ∈ Ω : f(x) ≥ α} grows monotonically, creating a filtration whose topology changes at critical thresholds.
- **Topological Loss:** Given prediction f and ground truth g (also viewed as a likelihood map), the topological loss is defined via the Wasserstein distance between their persistence diagrams:

> L_topo(f, g) = Σ_d W_q^q(Dgm_d(f), Dgm_d(g))

where d ranges over homological dimensions and q is typically 2.

**Main Result:** The total training loss is L = L_BCE + λ · L_topo, combining pixel-wise binary cross-entropy with the topological loss. The gradient of L_topo with respect to network parameters flows through **critical pixels** — the specific pixel locations where topological features are born or die in the filtration.

**Correctness Guarantee:** If L_topo(f, g) = 0, then for any generic threshold α, the binarized prediction f^α has the same Betti numbers as the ground truth g^α.

#### 2. Problem, Scope, Prerequisites

**Problem:** Standard segmentation losses (Dice, cross-entropy) optimize pixel overlap but are blind to topology. A single broken pixel in a vessel segmentation changes connectivity (β₀ increases by 1) but barely affects Dice score.

**Scope:** Binary 2D segmentation. Architecture-agnostic — the topological loss can be added to any network producing pixel-wise predictions. Demonstrated on neuron membrane segmentation (ISBI 2012) and retinal vessel segmentation.

**Prerequisites:** Persistent homology computation (cubical complexes for images), the notion of critical simplices in a filtration, differentiability of persistence (the critical pixel gradient).

#### 3. Proof Roadmap (Correctness Guarantee)

**Step 1:** The Wasserstein distance W_q(Dgm(f), Dgm(g)) = 0 if and only if Dgm(f) = Dgm(g) (as multisets).

**Step 2:** If two functions have identical persistence diagrams, they have the same number of features born and dying at each threshold pair. In particular, for any generic threshold α, the Betti numbers β_k(f^α) = β_k(g^α).

**Step 3:** The gradient computation identifies, for each unmatched or mismatched persistence point, the critical pixel responsible for the birth or death of that feature. The gradient directs the network to modify these specific pixels to bring the persistence diagrams into alignment.

**Mathematical Intuition:** The persistence diagram encodes ALL possible thresholdings simultaneously. By matching persistence diagrams, you ensure topological correctness across all thresholds, not just α = 0.5.

#### 4. New Problems Discovered

- **Wasserstein matching is not spatially correct:** Two components in different image regions can be matched based on their persistence values alone, even if they correspond to completely different spatial structures. (This is the fundamental limitation addressed by Stucki et al. 2023.)
- **Computational cost:** Persistence computation is O(n³) worst-case for n pixels. Practical implementation requires patch-based processing (64×64 patches), limiting the topological context to local regions.
- **Batch size constraint:** Topological loss computation for large images requires processing the full risk set, limiting effective batch size to 1 during topological loss phases.
- **Bootstrapping problem:** On random/uniform predictions, the persistence diagram is concentrated near the diagonal, providing near-zero gradients. The topological loss only helps once predictions have some spatial structure.

#### 5. Simple Analogy

Imagine you're a music teacher comparing two students' performances of the same symphony. Standard metrics (Dice) count how many notes they got right, like a spelling checker. The topological loss is like a music critic who listens for the overall structure — are all the themes present? Do the melodies connect properly? Are there phantom themes that shouldn't exist?

The Wasserstein matching pairs themes between the two performances based on their "sound profile" (persistence), but it might accidentally pair the violin theme with the cello theme if they happen to sound similar. This spatial mismatch is the key limitation.

---

### TIER 1 PAPER 3: Topologically Faithful Image Segmentation via Induced Matching (Betti Matching)
**Stucki, N., Paetzold, J.C., Shit, S., Menze, B., Bauer, U. (ICML 2023). arXiv:2211.15272**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Comparison Image:** Given likelihood map L ∈ [0,1]^{m×n} and ground truth G ∈ {0,1}^{m×n}, the comparison image is C = max(L, G) (pointwise maximum).
- **Induced Matching:** The inclusion maps L ↪ C and G ↪ C (where the inclusions are on sublevel/superlevel sets) induce maps on persistent homology. The induced matching pairs features in Dgm(L) and Dgm(G) that map to the SAME feature in Dgm(C). This is the "TopoMatch" matching.
- **Betti Matching Error:** τ_err = number of unmatched features in Dgm(L) + number of unmatched features in Dgm(G). Features that don't find a spatial correspondent are counted as topological errors.
- **Betti Matching Loss:** L_BM = Σ (persistence of unmatched features in L)² + Σ (persistence of unmatched features in G)². Differentiable with respect to L.

**Main Results:**
1. The induced matching is **spatially correct**: matched features correspond to the same spatial region in the image.
2. Betti matching error is strictly more informative than Betti number error (β₀ error can be zero when Betti matching error is nonzero, due to compensating errors).
3. Betti matching loss is differentiable and improves topological segmentation quality across six diverse datasets while preserving volumetric scores.

#### 2. Problem, Scope, Prerequisites

**Problem:** Prior topological losses (Hu et al. 2019) use Wasserstein matching of persistence diagrams, which matches features based on persistence values ONLY, ignoring spatial location. A component in the upper-left of the prediction can be matched to a component in the lower-right of the ground truth, producing incorrect gradient directions.

**Scope:** Binary 2D image segmentation. Extended to 3D (Stucki et al. 2024) and multi-class (MICCAI 2024). Architecture-agnostic loss function.

**Prerequisites:** Persistent homology, induced maps on persistence modules (from Bauer & Lesnick, 2015), the algebraic stability theorem for persistence barcodes.

#### 3. Proof Roadmap (Spatial Correctness)

**Step 1 — Construction of C:** C = max(L, G) ensures that C^t ⊇ L^t and C^t ⊇ G^t for all thresholds t (where these are superlevel sets). This gives well-defined inclusion maps.

**Step 2 — Induced Maps on Persistent Homology:** By functoriality of homology, the inclusions L^t ↪ C^t and G^t ↪ C^t induce maps on homology: H_k(L^t) → H_k(C^t) and H_k(G^t) → H_k(C^t). These extend to maps on persistence modules.

**Step 3 — The Matching:** A feature (b_L, d_L) in Dgm(L) is matched to (b_G, d_G) in Dgm(G) if both are mapped to the same feature in Dgm(C) by the induced maps. Since C = max(L, G), features that are spatially disjoint in L and G will correspond to different features in C, ensuring **spatial correctness**.

**Step 4 — Differentiability:** The Betti matching loss differentiates through the persistence computation using the critical cell gradient (as in Hu et al.), but restricted to unmatched features only. Matched features contribute zero loss.

**Mathematical Intuition:** The comparison image C acts as a "common reference frame." Embedding both L and G into C via natural inclusions creates a spatially grounded correspondence. Two features in L and G can only be matched if they overlap spatially in C.

#### 4. New Problems Discovered

- **Computational complexity:** O(n³) for persistence computation, limiting 3D applicability until the 2024 optimized C++ implementation.
- **Multi-class extension:** Requires projecting N-class problems to N binary problems because multi-parameter persistent homology is computationally intractable.
- **Connectivity sensitivity:** Berger et al. (IPMI 2025) showed that the choice of pixel connectivity (4- vs. 8-connected in 2D) dramatically affects Betti matching results, requiring standardization.
- **Hyperparameter α:** The weight balancing Betti matching loss with base loss (Dice) needs careful tuning per dataset.

#### 5. Simple Analogy

Imagine two maps of the same archipelago drawn by different cartographers. The Wasserstein matching pairs islands by their size (persistence), so if one map has a large island in the north and another large island in the south, they might be incorrectly paired.

The induced matching works by overlaying both maps on the same sheet (the comparison image C). An island in map A is matched to an island in map B only if they physically overlap on the overlaid sheet. This is obviously the correct way to match geographic features — by location, not by size.

---

### TIER 1 PAPER 4: A Topological Loss Function Using Persistent Homology (Clough et al.)
**Clough, J.R., et al. (IEEE TPAMI, 2020). arXiv:1910.01877**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Betti Number Prior:** Instead of matching persistence diagrams to ground truth, the user specifies the desired Betti numbers β = (β₀, β₁, ...) of the target structure. For the left ventricular myocardium: β₀ = 1 (one component), β₁ = 1 (one hole — the cavity).
- **Topological Loss via Betti Priors:** For each dimension d, if the current prediction has more d-dimensional features than β_d, the loss penalizes the most persistent excess features. If it has fewer, the loss penalizes the least persistent missing features.

**Main Result:** Topological constraints can be enforced **without pixel-wise ground truth labels** — only the Betti number priors are needed. This enables semi-supervised topological training. Demonstrated on cardiac MRI (UK Biobank, ACDC) and 3D placenta segmentation.

#### 2. Problem, Scope, Prerequisites

**Problem:** Many anatomical structures have known topology (heart has chambers, blood vessels form networks) but pixel-wise annotation is expensive. Can we use topological knowledge alone to supervise segmentation?

**Scope:** Binary and multi-structure 2D/3D segmentation where the target topology is known a priori. More general than methods requiring pixel-wise ground truth for computing PH.

**Prerequisites:** Cubical persistent homology, differentiability of PH, knowledge of target anatomy topology.

#### 3. Proof Roadmap

The approach does not prove a formal theorem but relies on the differentiability of persistent homology (Brüel-Gabrielsson et al. 2020). The gradient of the topological loss with respect to network weights identifies which pixels need to change to create or destroy topological features, using the critical simplex identification from the persistence computation.

#### 4. New Problems Discovered

- Betti number priors are a **weak** topological specification — they don't enforce spatial correctness.
- The approach can be sensitive to noise that creates many spurious short-lived topological features.
- Requires careful balance between topological and pixel-wise losses to prevent topological loss from dominating and producing degenerate solutions.

#### 5. Simple Analogy

Instead of giving a student an answer key (pixel-wise labels), you tell them: "Your drawing should have exactly one island with exactly one lake." They can use this structural knowledge to correct their work even without seeing the exact answer. This is semi-supervised topological learning.

---

### TIER 1 PAPER 5: Topology-Aware Segmentation Using Discrete Morse Theory (DMT-Loss)
**Hu, X., Wang, Y., Fuxin, L., Samaras, D., Chen, C. (ICLR 2021 Spotlight). arXiv:2103.09992**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Discrete Morse Function:** A function f on the cells of a CW complex such that each cell has at most one higher-dimensional coface with lower f-value and at most one lower-dimensional face with higher f-value.
- **Critical Cells:** Cells that violate both conditions — they are local extrema in the Morse-theoretic sense. Critical 0-cells are "peaks," critical 1-cells are "saddles" for 1D structures, etc.
- **DMT-Loss:** Identifies topologically critical 1D structures (skeletons for curvilinear objects like vessels) and 2D structures (membranes for region boundaries), then applies weighted loss on these structures.

**Main Result:** DMT-loss focuses training on the specific pixels that determine topology, achieving Betti error of 0.982 on CREMI versus 3.016 for standard U-Net and 1.113 for TopoLoss (Hu et al. 2019). Computationally more efficient than standard PH for identifying critical structures.

#### 2. Problem, Scope, Prerequisites

**Problem:** Persistent homology-based methods identify topological features but don't directly identify the SPATIAL structures (skeletons, membranes) that carry topological information. DMT bridges this gap.

**Scope:** 2D segmentation of curvilinear and membrane-like structures. Particularly effective for neurons, vessels, and cell boundaries.

**Prerequisites:** Discrete Morse theory (Forman 1998), critical cell theory, the relationship between Morse theory and homology via the Morse complex.

#### 3. Proof Roadmap

**Key Insight:** The Morse complex decomposes a filtered complex into ascending and descending manifolds of critical cells. For 2D images, critical 0-cells correspond to connected components (birth of H₀), critical 1-cells correspond to merging events (death of H₀) or birth of loops (H₁). The DMT-loss identifies the 1D skeleton connecting critical cells and applies focused supervision there.

The proof of the relationship between Morse theory and persistent homology relies on the Morse Lemma: the topology of sublevel sets changes only at critical values, and the change is determined by the index of the critical cell (0 = new component, 1 = handle attachment, etc.).

#### 4. New Problems Discovered

- DMT is most effective for structures with clear 1D skeletons; it's less natural for compact regions like tumors.
- The identification of critical structures depends on the Morse function, which must be chosen carefully.
- The method does not directly optimize persistence diagrams — it's a heuristic that focuses on topologically important regions.

#### 5. Simple Analogy

If the topological loss (Hu 2019) is like a music critic saying "your performance has the wrong number of themes," then DMT-loss is like a conducting teacher pointing to the EXACT bars in the score where the themes begin and end, saying "focus your practice HERE — these bars determine the musical structure."

---

### TIER 1 PAPER 6: clDice — A Novel Topology-Preserving Loss Function
**Shit, S., Paetzold, J.C., et al. (CVPR 2021). arXiv:2003.07311**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Morphological Skeleton:** S(V) = the centerline/medial axis of binary mask V, obtained via iterative morphological thinning.
- **Topology Precision:** Tprec(S_P, V_L) = |S_P ∩ V_L| / |S_P| — the fraction of predicted skeleton lying within the ground truth volume.
- **Topology Sensitivity:** Tsens(S_L, V_P) = |S_L ∩ V_P| / |S_L| — the fraction of ground truth skeleton lying within the prediction.
- **clDice:** Harmonic mean of Tprec and Tsens: clDice = 2 · Tprec · Tsens / (Tprec + Tsens).
- **Soft-clDice:** Differentiable version using iterative max-pooling for soft skeletonization, enabling use as a training loss.

**Main Theorem (Homotopy Equivalence):** Let V_L and V_P be two binary masks admitting foreground and background skeletons. If the foreground skeleton of V_L is included in the foreground of V_P, and vice versa, and similarly for the background, then the foregrounds of V_L and V_P are **homotopy equivalent**.

**Corollary:** clDice = 1 implies homotopy equivalence for binary 2D and 3D segmentation.

#### 2. Problem, Scope, Prerequisites

**Problem:** For tubular structures (vessels, neurons, roads), connectivity is the critical property. Standard Dice can be high even when connections are broken. Need a metric that specifically measures topological correctness for networks/tubes.

**Scope:** Tubular/network structures in 2D and 3D. Less meaningful for compact regions (disks, tumors) where the skeleton is a single point.

**Prerequisites:** Morphological operations (dilation, erosion, thinning), homotopy theory (homotopy equivalence, deformation retraction), cell complex theory.

#### 3. Proof Roadmap (Theorem 1: Homotopy Equivalence)

**Step 1:** Assume L_A ⊆ A ⊆ K_A and L_B ⊆ B ⊆ K_B are connected subcomplexes of a cell complex, where L denotes skeleton and K denotes the "thickened" version (full mask). Assume all inclusions are homotopy equivalences (skeleton is a deformation retract of the mask).

**Step 2:** If additionally L_A ⊆ B ⊆ K_A and L_B ⊆ A ⊆ K_B (each skeleton is contained in the other's thickened mask), then by the Whitehead theorem, an inclusion that induces isomorphisms on all homotopy groups is a homotopy equivalence.

**Step 3:** The chain of inclusions L_A ↪ B ↪ K_A, combined with L_A ↪ A (homotopy equivalence), gives L_A ↪ B is a homotopy equivalence. Similarly for the other direction. Therefore A and B are homotopy equivalent.

**Mathematical Intuition:** If the skeleton of your prediction lies inside the ground truth, and vice versa, then the two shapes must be "the same" topologically — you can continuously deform one into the other.

#### 4. New Problems Discovered

- Soft skeletonization via iterative max-pooling is an approximation — it doesn't produce exact morphological skeletons, leading to some discrepancy between soft-clDice (training) and hard-clDice (evaluation).
- The method requires structures to HAVE meaningful skeletons. For compact regions (tumors, organs), the skeleton is trivial and clDice reduces to standard overlap.
- The homotopy equivalence guarantee is for binary masks. The soft version (used in training) doesn't carry the same formal guarantee.
- Computational cost is O(n) — significantly cheaper than PH-based methods.

#### 5. Simple Analogy

Imagine verifying that two road networks connect the same cities. Instead of comparing every piece of asphalt, you check: does the centerline of each road network lie within the asphalt of the other? If yes, the two networks must connect the same cities in the same way — they're topologically equivalent. The centerline is the skeleton, and "lying within" is the overlap check.

---

### TIER 2 SUMMARIES — TDA Foundations

**Adams et al. (2017) — Persistence Images (JMLR)**
Introduces persistence images as a stable, vectorized representation of persistence diagrams. The key contribution is mapping variable-cardinality multisets (persistence diagrams) to fixed-dimensional vectors (images) via Gaussian kernel density estimation with persistence-based weighting. Stability is proved: ||PI(D₁) - PI(D₂)|| ≤ C · W₁(D₁, D₂). Directly used in TDA-SegUNet as the vectorization method. *Relevance: Core preprocessing step in the TDA feature extraction pipeline.*

**Leygonie, Oudot, Tillmann (2022) — Differentiability Framework (FoCM)**
Establishes the rigorous mathematical framework for differentiating through persistence barcodes. Proves that the map from filtration parameters to persistence diagrams is differentiable at generic points (when no two critical values coincide). Provides explicit formulas for the Jacobian. *Relevance: Theoretical foundation for all topological loss functions (Theorem 1.3 in our Mathematical Foundations document).*

**Brüel-Gabrielsson et al. (2020) — Topology Layer (AISTATS)**
Implements a differentiable persistence computation as a PyTorch/TensorFlow layer. Practical contribution enabling GPU-accelerated topological loss computation. *Relevance: Engineering prerequisite for implementing topological losses in practice.*

**Stucki et al. (2024) — Efficient Betti Matching 3D (arXiv:2407.04683)**
Extends Betti matching to 3D with an optimized C++ implementation that makes topological loss feasible for 3D medical volumes. Demonstrates significant improvements on 3D segmentation benchmarks. *Relevance: Direct prerequisite for applying Betti matching to BraTS 3D data.*

**Stucki et al. (MICCAI 2024) — Multi-class Topological Segmentation (arXiv:2403.11001)**
Extends Betti matching to N-class segmentation by projecting to N binary problems. Avoids computationally intractable multi-parameter persistent homology. Validated on four medical datasets. *Relevance: Directly needed for BraTS multi-class (WT, TC, ET) segmentation.*

**Berger et al. (IPMI 2025) — Pitfalls (arXiv:2412.14619)**
Critical analysis showing that connectivity choice (4-conn vs 8-conn in 2D) dramatically changes topological metrics, distributional metrics (VOI, ARI) entangle topological and volumetric errors irreversibly, and existing topology-aware methods need standardized evaluation. *Relevance: Methodological guide for experimental design and evaluation — must follow these recommendations in thesis experiments.*

---

### References — Section A

1. Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability of persistence diagrams. *DCG*, 37(1), 103–120.
2. Hu, X., Li, F., Samaras, D., Chen, C. (2019). Topology-preserving deep image segmentation. *NeurIPS 2019*. arXiv:1906.05404.
3. Stucki, N., Paetzold, J.C., Shit, S., Menze, B., Bauer, U. (2023). Topologically faithful image segmentation via induced matching of persistence barcodes. *ICML 2023*. arXiv:2211.15272.
4. Clough, J.R., et al. (2020). A topological loss function for deep-learning based image segmentation using persistent homology. *IEEE TPAMI*. arXiv:1910.01877.
5. Hu, X., Wang, Y., Fuxin, L., Samaras, D., Chen, C. (2021). Topology-aware segmentation using discrete Morse theory. *ICLR 2021*. arXiv:2103.09992.
6. Shit, S., Paetzold, J.C., et al. (2021). clDice — A novel topology-preserving loss function for tubular structure segmentation. *CVPR 2021*. arXiv:2003.07311.
7. Adams, H., et al. (2017). Persistence images. *JMLR*, 18(8), 1–35.
8. Leygonie, J., Oudot, S., Tillmann, U. (2022). A framework for differential calculus on persistence barcodes. *FoCM*, 22, 1069–1131.
9. Brüel-Gabrielsson, R., et al. (2020). A topology layer for machine learning. *AISTATS 2020*.
10. Stucki, N., et al. (2024). Efficient Betti matching enables topology-aware 3D segmentation. arXiv:2407.04683.
11. Berger, C., et al. (2025). Pitfalls of topology-aware image segmentation. *IPMI 2025*. arXiv:2412.14619.
12. Bauer, U., Lesnick, M. (2015). Induced matchings and the algebraic stability of persistence barcodes. *J. Comput. Geom.*, 6(2), 162–191.
# Detailed Paper Analysis — Section B

## TDA-SegUNet and Topology-Based Segmentation Architectures

---

### TIER 1 PAPER 7: TDA-SegUNet
**TDA SegUNet: Topological Data Analysis-Based Shape-Aware Brain Tumor Segmentation. IEEE JBHI (2025). DOI: 10.1109/JBHI.2025.3544702**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Persistence Image Channels:** For each MRI slice, the EDT of the binary tumor mask is computed, cubical persistent homology is applied, and the resulting H₀ and H₁ persistence diagrams are vectorized into persistence images (PIs) at the same spatial resolution as the MRI slice. These PIs serve as additional input channels.
- **Multi-Channel Input:** The U-Net receives [T1, T1ce, T2, FLAIR, PI₀, PI₁] — 4 MRI modalities plus 2 persistence image channels, for a total of 6 input channels.

**Main Results:**
- Evaluated on BraTS 2020 for segmenting enhancing tumor (ET), tumor core (TC), and whole tumor (WT).
- The TDA features (PI₀, PI₁) provide shape-aware information that standard intensity-based features miss.
- The architecture is a standard 2D U-Net with modified input channels — the innovation is in the feature engineering, not the network architecture.

#### 2. Problem, Scope, Prerequisites

**Problem:** Standard U-Net segmentation uses only intensity and local texture features. Tumor shape — number of components, presence of holes (necrotic cavities), boundary regularity — carries diagnostic information that intensity alone cannot capture.

**Scope:** 2D slice-wise brain tumor segmentation on BraTS. The method is limited to 2D because 3D persistence image computation is expensive and the paper processes individual slices.

**Prerequisites:** GUDHI or similar library for cubical PH computation, persistence image vectorization, standard U-Net architecture, BraTS data preprocessing pipeline.

#### 3. Proof Roadmap

This paper is primarily empirical — it does not prove formal theorems. The justification rests on:
- The stability theorem (Cohen-Steiner et al. 2007) ensures PIs are robust to small perturbations.
- The information-theoretic argument that PIs add non-redundant topological features orthogonal to intensity.

#### 4. New Problems Discovered

- **2D limitation:** Processing slices independently loses inter-slice topological information. A 3D spherical cavity (β₂ = 1) appears as a sequence of 2D annuli, but this global structure is not captured.
- **Inference gap:** During training, TDA features are computed from ground truth masks. During inference, the ground truth is unavailable — the paper addresses this by using initial segmentation predictions.
- **Computational overhead:** PH computation adds ~30-60s per volume of preprocessing time.
- **Input fusion simplicity:** Using PIs as input channels is the simplest fusion strategy but provides no architectural guarantee that the network will learn to USE the topological information.

#### 5. Simple Analogy

Imagine you're sorting photos of coffee cups (solid disks when viewed from above) and donuts (annuli). If you only look at the color and texture, you might confuse a brown disk with a brown annulus. But if someone hands you an X-ray showing the internal structure (the persistence image), you can immediately distinguish them. TDA-SegUNet gives the neural network both the photo AND the X-ray.

---

### TIER 1 PAPER 8: Train-Free Segmentation in MRI with Cubical Persistent Homology
**François, O., Tinarrage, R. (2024). arXiv:2401.01160**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Three-Module Pipeline:** (1) Identify the whole object via automatic thresholding, (2) Detect a topologically distinctive subset using persistent homology (localize representative cycles), (3) Deduce remaining regions from the geometric relationship between object and subset.
- **Representative Cycles:** For each feature in the persistence diagram, the representative cycle is the specific set of cells (pixels) that form the topological feature. For an H₁ feature (hole), the representative cycle is the boundary loop of the hole.
- **Topological Prior:** The method requires knowledge that the target structure has specific topology. For GBM enhancing tumor: approximately spherical with β₂ ≈ 1 (enclosing a necrotic cavity).

**Main Results:**
- Segments brain tumors on BraTS data using ONLY persistent homology — no neural network.
- The enhancing tumor is localized by finding the H₂ representative cycle (the sphere bounding the necrotic cavity).
- Fully interpretable: every step has a clear mathematical justification.
- Requires NO training data.

#### 2. Problem, Scope, Prerequisites

**Problem:** Can we perform medical image segmentation using topology alone, without any machine learning? This provides a theoretical lower bound and demonstrates the power of TDA as a standalone tool.

**Scope:** Limited to structures with strong, distinctive topological priors. Works well for GBM (spherical enhancing rim around necrotic cavity) but would struggle with topologically simple tumors (solid blobs with β₀=1, β₁=0, β₂=0).

**Prerequisites:** Cubical complexes (3D), representative cycle computation in persistent homology, understanding of the relationship between homology classes and geometric subsets.

#### 3. Proof Roadmap

**Step 1:** The intensity image is filtered to identify the MRI intensity range corresponding to enhancing tumor (using contrast-enhanced T1).

**Step 2:** Cubical persistent homology is computed on the 3D image volume. The H₂ persistence diagram reveals spherical cavities. The most persistent H₂ feature corresponds to the necrotic core.

**Step 3:** The representative cycle of this H₂ feature — the 2-cycle bounding the cavity — is extracted. This cycle localizes the enhancing rim of the tumor.

**Step 4:** The remaining tumor regions (edema, non-enhancing core) are deduced geometrically from the relationship between the enhancing rim and the whole-brain segmentation.

The mathematical guarantee is that if the topological prior is correct (the tumor has a single enclosed cavity), the method will correctly localize the boundary of that cavity.

#### 4. New Problems Discovered

- Requires **strong topological priors** that may not hold for all tumor subtypes.
- Cannot handle tumors without distinctive topology (solid masses).
- Performance is lower than supervised neural network methods on well-annotated datasets.
- The representative cycle computation is not unique — different algorithms may produce different (homologous) cycles, affecting the precise localization.

#### 5. Simple Analogy

Imagine finding a hollow ball in a room full of objects using only a metal detector (which detects enclosed cavities). You don't need any training — you know what hollow balls look like topologically (they enclose a void). You scan the room, find the void, trace its boundary, and you've found the ball. This is what train-free PH segmentation does for necrotic tumors.

---

### TIER 2 SUMMARIES — Brain Cancer TDA Applications

**Oyama et al. (2021) — PH Features for Lung and Brain Cancers (arXiv:2012.12102)**
Demonstrates that persistent homology features extracted from medical images can characterize tumor morphology for both lung and brain cancers. Persistence diagrams are vectorized and used as features for classification and survival prediction. Key finding: topological features capture shape information complementary to standard radiomics. *Relevance: Validates the TDA→feature→prediction pipeline used in our FCoxPH toy model.*

**Crawford et al. (2020) — SECT for GBM Prognosis (JASA)**
Introduces the Smooth Euler Characteristic Transform (SECT), a topological statistic with a well-defined inner product structure. SECT computes directional Euler characteristic curves from tumor boundaries. Applied to TCGA-GBM, SECT outperforms existing shape quantifications and molecular assays for predicting clinical outcomes. *Relevance: Alternative topological feature extraction for survival prediction, potentially superior to persistence images.*

**Bhattacharya et al. (2024) — Brain Tumor Connectomics (arXiv:2407.17938)**
Applies Vietoris-Rips persistent homology to diffusion MRI brain connectome data for differentiating meningiomas from gliomas. Achieves 88% accuracy using topological features of structural connectivity networks. *Relevance: Shows TDA applied to connectivity rather than intensity — a different modality.*

**TopoGBM (2026) — Topological Neural Networks for GBM (arXiv:2602.11234)**
Semi-supervised representation learning using a brain-inspired topological regularizer for GBM prognosis. Achieves C-index of 0.67 across unseen multi-institutional datasets. Approximately 50% of prognostic signal localized to tumor and peri-tumoral regions. *Relevance: Most recent work combining TDA with deep learning for GBM specifically.*

**Gracia-Tabuenca et al. (2025) — TDA for GBM Necrosis (arXiv:2503.17331)**
Introduces subcomplex lacunarity, a TDA-based shape descriptor for necrosis. Uses Persistent Homology Transform and persistence landscapes to identify four distinct GBM subtypes. *Relevance: TDA for tumor subtyping, which connects to survival prediction.*

---

### References — Section B

1. TDA-SegUNet (2025). IEEE JBHI. DOI:10.1109/JBHI.2025.3544702.
2. François, O., Tinarrage, R. (2024). Train-free segmentation in MRI with cubical persistent homology. arXiv:2401.01160.
3. Oyama, A., et al. (2021). arXiv:2012.12102.
4. Crawford, L., et al. (2020). JASA.
5. Bhattacharya, S., et al. (2024). arXiv:2407.17938.
6. arXiv:2602.11234 (2026).
7. Gracia-Tabuenca, Z., et al. (2025). arXiv:2503.17331.

---

# Detailed Paper Analysis — Section C

## Deep Survival Analysis and Cox Proportional Hazards

---

### TIER 1 PAPER 9: DeepSurv
**Katzman, J.L., et al. (BMC Medical Research Methodology, 2018). arXiv:1606.00931**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Cox Proportional Hazards Model:** h(t|x) = h₀(t) · exp(ĥ_θ(x)), where h₀(t) is an unspecified baseline hazard and ĥ_θ(x) is a risk function. In classical CoxPH, ĥ(x) = βᵀx (linear). In DeepSurv, ĥ_θ(x) = g_θ(x) is a deep neural network.
- **Negative Log Partial Likelihood:** The loss function:

> L(θ) = -Σ_{i:δᵢ=1} [ĥ_θ(xᵢ) - log(Σ_{j∈R(tᵢ)} exp(ĥ_θ(xⱼ)))]

where δᵢ=1 indicates an observed event and R(tᵢ) is the risk set at time tᵢ.
- **Concordance Index (C-index):** The fraction of all patient pairs where the model correctly identifies the patient who dies first. C=0.5 is random; C=1 is perfect.

**Main Results:**
- DeepSurv achieves higher C-index than linear CoxPH and Random Survival Forests on datasets with nonlinear covariate interactions.
- The network learns a nonlinear risk function that captures treatment interaction effects.
- Treatment recommendations based on DeepSurv's predicted risk significantly improve patient outcomes (validated by log-rank test).

#### 2. Problem, Scope, Prerequisites

**Problem:** Standard CoxPH assumes linear log-risk, which cannot model complex nonlinear interactions between covariates and treatment effects. Can a deep network learn these interactions?

**Scope:** Tabular survival data (clinical covariates). NOT image-based — inputs are feature vectors, not images. Extension to images requires CNN feature extraction as a preprocessing step.

**Prerequisites:** Survival analysis fundamentals (censoring, risk sets, hazard functions), neural network optimization, the Cox partial likelihood.

#### 3. Proof Roadmap

No formal theorem is proved. The approach relies on:
- **Universal approximation:** Deep networks can approximate any continuous function, so g_θ can model arbitrary nonlinear risk functions.
- **Partial likelihood consistency:** The Cox partial likelihood is a valid estimating equation even when h₀(t) is unspecified (Cox 1972, 1975).
- The composition of a deep network with the Cox partial likelihood loss creates a differentiable objective that can be optimized via standard backpropagation.

#### 4. New Problems Discovered

- The partial likelihood requires computing over the **entire risk set** for each event, making mini-batch SGD non-trivial. Implementations typically process the full training set per gradient update.
- The proportional hazards assumption may be violated in many clinical settings. Cox-Time (Kvamme et al. 2019) addresses this by making the risk function time-dependent.
- For imaging applications, the bottleneck is feature extraction: DeepSurv itself takes vectors, not images. End-to-end imaging+survival requires the SCNN approach (Mobadersany).

#### 5. Simple Analogy

Traditional survival analysis is like predicting marathon finish times using a simple formula: finish_time = base_pace × (age_factor + weight_factor). DeepSurv replaces this with a neural network that learns complex interactions: "this runner's age penalty is actually offset by their specific training history in THIS particular way." The prediction is no longer a simple sum of factors but a learned nonlinear function of all runner characteristics.

---

### TIER 1 PAPER 10: Predicting Cancer Outcomes from Histology and Genomics (SCNN/GSCNN)
**Mobadersany, P., et al. (PNAS, 2018). DOI:10.1073/pnas.1717139115**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Survival CNN (SCNN):** VGG-like convolutional layers → fully connected layers → single Cox proportional hazards output node. The ENTIRE CNN is trained end-to-end with negative log partial likelihood loss.
- **Genomic SCNN (GSCNN):** Extends SCNN by incorporating genomic variables (IDH mutation status, 1p/19q codeletion) at the fully connected layers. The network simultaneously learns from histology images and molecular biomarkers.

**Main Results:**
- SCNN achieves C-index ~0.754 on TCGA gliomas (histology alone).
- GSCNN achieves C-index ~0.801, **surpassing the WHO clinical paradigm** (~0.747) based on genomic subtype + histologic grading.
- Heat map visualizations confirm the network recognizes clinically meaningful structures: microvascular proliferation, necrosis, cell density, nuclear morphology.

#### 2. Problem, Scope, Prerequisites

**Problem:** Can deep learning predict patient survival directly from pathology images, bypassing the need for manual grading and hand-crafted features? Can imaging and genomics be integrated into a single predictive model?

**Scope:** Histopathology images (H&E stained tissue) from TCGA glioma cohort (1,061 images from 769 patients). NOT MRI — this is histopathology. Extension to MRI-based survival requires replacing the histology CNN with an MRI-based encoder.

**Prerequisites:** VGG/ResNet architectures, Cox partial likelihood, TCGA data access, understanding of glioma grading.

#### 3. Proof Roadmap

Primarily empirical. The key methodological contributions are:
- **Sampling strategy:** Address tumor heterogeneity by randomly sampling multiple patches per slide during training (each with survival label inherited from the patient).
- **Multi-modal fusion:** Genomic variables enter at the fully connected layer, where they interact with learned image features. This is analogous to the bottleneck fusion in our TDA-SegUNet framework.
- **Validation:** 15 independent train/test splits with randomized patient assignments.

#### 4. New Problems Discovered

- Interpretability remains limited despite heat maps — the network is still a black box.
- Requires large cohorts (769 patients) to avoid overfitting on high-dimensional image inputs.
- Transfer to different institutions/scanners requires domain adaptation.
- The approach does not directly model tumor topology — it learns whatever visual features correlate with survival, which may or may not include topological patterns.

#### 5. Simple Analogy

Imagine training an art critic to predict how long a building will last by showing them photos of the building's foundation. Instead of giving them a checklist ("look for cracks, measure rebar spacing"), you show them thousands of foundation photos with known building lifespans and let them figure out what matters. The SCNN is this self-taught inspector. The GSCNN additionally gets the architect's blueprints (genomic data), combining visual inspection with structural knowledge.

---

### TIER 2 SUMMARIES — Survival Analysis

**Ching et al. (2018) — Cox-nnet (PLOS Computational Biology)**
A two-layer neural network for survival prediction from high-dimensional omics data (~20,000 genes). Hidden layer activations reveal survival-relevant biological pathways. *Relevance: Shows how neural survival models can provide interpretable intermediate representations.*

**Chen et al. (2022) — Cox PH Denoising Autoencoder for GBM (Frontiers)**
Combines convolutional denoising autoencoder with CoxPH for GBM survival from multi-modal MRI. Achieves C-index 0.74 on BraTS 2019 by learning compressed feature representations from tumor regions. *Relevance: Direct BraTS survival prediction using MRI, not histopathology.*

**Feng et al. (2020) — BraTS 2018 OS Winner (Frontiers)**
Won first place in BraTS 2018 OS prediction. Two-stage: 3D U-Net ensemble segmentation → radiomic feature extraction → gradient boosting survival model. *Relevance: Establishes the benchmark pipeline that our topological approach aims to improve.*

**Moon et al. (2023) / Jang et al. (2025) — FCoxPH with PH Features**
Functional Cox PH model consuming persistent homology features as functional predictors. Irregular, heterogeneous shape patterns captured by TDA are positively associated with survival hazards (p < 0.001). *Relevance: The key bridge between TDA and survival — directly relevant to Toy Model C.*

---

### References — Section C

1. Katzman, J.L., et al. (2018). DeepSurv. *BMC Med. Res. Methodol.*, 18, 24.
2. Mobadersany, P., et al. (2018). PNAS, 115(13), E2970–E2979.
3. Ching, T., et al. (2018). Cox-nnet. *PLOS Comp. Biol.*
4. Chen, W., et al. (2022). Frontiers in Computational Neuroscience.
5. Feng, X., et al. (2020). Frontiers in Computational Neuroscience.
6. Moon, H., et al. (2023). Annals of Applied Statistics.
7. Jang, J., et al. (2025). arXiv:2512.05646.
8. Kvamme, H., et al. (2019). JMLR, 20(129), 1–30.

---

# Detailed Paper Analysis — Section D

## Diffeomorphic Registration and Differential Geometry

---

### TIER 1 PAPER 11: VoxelMorph
**Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V. (IEEE TMI, 2019). arXiv:1809.05231**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Deformation Field:** φ: ℝ³ → ℝ³, a dense spatial transformation mapping each voxel to its corresponding location in the target image.
- **Stationary Velocity Field (SVF):** v: ℝ³ → ℝ³, integrated via the ODE dφ/dt = v(φ(t)) over unit time. The solution φ = exp(v) is guaranteed to be a **diffeomorphism** (smooth, invertible, topology-preserving).
- **Scaling-and-Squaring:** Efficient integration: φ = exp(v) ≈ (exp(v/2^T))^{2^T}. Starting from φ₀ = Id + v/2^T, repeatedly compose with itself T times.
- **Registration Loss:** L(θ) = L_sim(I_fixed, I_moving ∘ φ_θ) + λ · L_smooth(φ_θ), combining image similarity (NCC or MSE) with a smoothness regularizer (diffusion or bending energy of the deformation field).

**Main Results:**
- VoxelMorph achieves registration accuracy (Dice on anatomical labels) comparable to state-of-the-art ANTs SyN while being **100-1000× faster** (seconds vs. hours).
- The U-Net encoder-decoder takes concatenated [I_fixed, I_moving] as input (2-channel 3D volume) and outputs the 3D displacement/velocity field.
- The diffeomorphic variant (VoxelMorph-diff, Dalca et al. 2019) provides topology-preservation guarantees via SVF integration.

#### 2. Problem, Scope, Prerequisites

**Problem:** Classical deformable registration (ANTs, LDDMM) optimizes a new objective for EACH image pair, taking minutes to hours. Can we amortize this cost by learning a registration function?

**Scope:** Brain MRI atlas-based and pairwise registration. Extended to cardiac, lung, and multi-organ registration by the community.

**Prerequisites:** Understanding of diffeomorphisms as elements of an infinite-dimensional Lie group, the exponential map (exp: Lie algebra → Lie group), U-Net architecture.

#### 3. Proof Roadmap

**Diffeomorphic Guarantee (for VoxelMorph-diff):**
- **Step 1:** The network outputs a stationary velocity field v ∈ ℝ^{H×W×D×3}.
- **Step 2:** Integration via scaling-and-squaring: start with φ₀ = Id + v/2^T (T integration steps). Since v/2^T is a small displacement, φ₀ is a diffeomorphism for sufficiently large T (by the inverse function theorem — if the Jacobian is everywhere positive, the map is locally invertible).
- **Step 3:** Squaring: φ_{k+1} = φ_k ∘ φ_k. The composition of diffeomorphisms is a diffeomorphism (diffeomorphisms form a group).
- **Step 4:** Therefore φ = φ_T is a diffeomorphism (positive Jacobian determinant everywhere).

**Caveat:** In practice, with finite grid resolution and discrete integration, the Jacobian can become negative at some voxels ("folding"). The number of folding voxels is a quality metric.

#### 4. New Problems Discovered

- **Folding voxels:** Even the diffeomorphic variant can produce a small number of negative Jacobian voxels due to discrete approximation. MDReg-Net (Li & Fan 2022) achieves 0.089 average folding voxels versus 3.68 for VoxelMorph.
- **Brain tumors break diffeomorphism:** Tumors represent topology-changing events (tissue creation, not deformation). Standard diffeomorphic registration cannot handle this — requires joint registration-segmentation or biophysical models.
- **Hyperparameter sensitivity:** The smoothness weight λ trades off accuracy versus regularity and requires dataset-specific tuning. GradICON (Tian et al. 2023) eliminates this via implicit regularization.
- **Atlas choice:** Registration quality depends on the atlas — computed atlases may not represent pathological anatomy well.

#### 5. Simple Analogy

Traditional registration is like having a tailor custom-fit every single suit from scratch — perfect fit, but takes forever. VoxelMorph trains an AI tailor who has seen thousands of body types and learned the general patterns. When a new customer arrives, the AI tailor produces a well-fitting suit in seconds by applying learned alterations. The diffeomorphic guarantee ensures the suit never tears (topology is preserved) — no holes appear or seams break.

---

### TIER 2 SUMMARIES — Differential Geometry

**GradICON (Tian et al., CVPR 2023, arXiv:2206.05897)**
Achieves approximate diffeomorphism via gradient inverse consistency — penalizing deviation of the composed forward-backward Jacobian from identity. No explicit smoothness penalty needed. A single hyperparameter set works across brain, knee, and lung datasets. *Relevance: Eliminates tuning, potentially simpler for BraTS registration.*

**FireANTs (Jena et al., 2024, arXiv:2404.01249)**
Training-free optimization on the Lie group of diffeomorphisms using Riemannian Adam. 200-1200× faster than ANTs, 10× less memory than learning-based methods. Exploits Lie algebra structure: compute descent in tangent space, map back via exponential. *Relevance: Could replace VoxelMorph for atlas construction without GPU training.*

**MDReg-Net (Li & Fan, 2022)**
Multi-resolution diffeomorphic registration achieving 0.089 average folding voxels. Progressive coarse-to-fine velocity field estimation. *Relevance: Superior diffeomorphic quality for precision volume measurements.*

**Estienne et al. (2020) — Joint Registration-Segmentation**
Concurrent brain registration and tumor segmentation, addressing the chicken-and-egg problem that registration needs segmentation masks and segmentation benefits from atlas alignment. *Relevance: Direct application to the brain tumor pipeline.*

**Scheufele et al. (2019) — SIBIA (Brain-Tumor Biophysical Models + Diffeomorphic Registration)**
Jointly estimates tumor growth parameters (diffusion, proliferation) and diffeomorphic maps by coupling a reaction-diffusion PDE with image registration. *Relevance: The most mathematically sophisticated approach to brain tumor analysis, connecting PDEs, differential geometry, and imaging.*

**Bronstein et al. (2021) — Geometric Deep Learning (arXiv:2104.13478)**
The "5G" unifying framework: Grids, Groups, Graphs, Geodesics, Gauges. Establishes symmetry as the organizing principle for all neural network architectures. *Relevance: Theoretical foundation for understanding why CNNs work (translation equivariance) and how to extend to manifold data.*

**Louis et al. (IPMI 2019) — Riemannian Geometry for Disease Progression**
Learns a Riemannian metric on image space such that disease progressions become geodesics. Applied to Alzheimer's. *Relevance: Shows how Riemannian geometry can model disease evolution on brain imaging data.*

---

### References — Section D

1. Balakrishnan, G., et al. (2019). VoxelMorph. *IEEE TMI*. arXiv:1809.05231.
2. Dalca, A.V., et al. (2019). Unsupervised learning of probabilistic diffeomorphic registration. *Medical Image Analysis*.
3. Tian, L., et al. (2023). GradICON. *CVPR 2023*. arXiv:2206.05897.
4. Jena, R., et al. (2024). FireANTs. arXiv:2404.01249.
5. Li, B., Fan, Y. (2022). MDReg-Net. *Human Brain Mapping*.
6. Estienne, T., et al. (2020). Frontiers in Computational Neuroscience.
7. Scheufele, K., et al. (2019). CMAME.
8. Bronstein, M.M., et al. (2021). arXiv:2104.13478.
9. Louis, M., et al. (2019). IPMI 2019.

---

# Detailed Paper Analysis — Section E

## Evaluation Metrics and Critical Analysis

---

### TIER 1 PAPER 12: Pitfalls of Topology-Aware Image Segmentation
**Berger, C., Grönke, A., Stucki, N., Bauer, U. (IPMI 2025). arXiv:2412.14619**

#### 1. Definitions, Ideas, Main Results

**Key Definitions:**
- **Connectivity:** In 2D, 4-connectivity (sharing edges only) versus 8-connectivity (sharing edges or corners). In 3D, 6-connectivity versus 26-connectivity. The choice dramatically affects computed topology.
- **Distributional Metrics:** Metrics like Variation of Information (VOI) and Adjusted Rand Index (ARI) that measure agreement between two segmentations at the component level.

**Main Results:**
1. **Connectivity changes everything:** The same segmentation can have β₀ = 1 under 8-connectivity but β₀ = 3 under 4-connectivity. Published comparisons between methods using different connectivity conventions are INVALID.
2. **Distributional metrics entangle errors:** VOI and ARI "irreversibly entangle topological and volumetric errors" — a method can improve VOI by fixing volume without fixing topology, or vice versa, and VOI cannot distinguish the two.
3. **Betti number error has cancellation:** If a prediction adds one spurious component and loses one real component, β₀ error = 0, masking two errors.
4. **Betti matching is recommended** as the gold standard topological metric because it is spatially correct and does not suffer from cancellation.

#### 2. Problem, Scope, Prerequisites

**Problem:** The topology-aware segmentation community lacks standardized evaluation. Papers use different connectivity conventions, different metrics, and different definitions of "topological correctness." This makes method comparison unreliable.

**Scope:** All topology-aware segmentation methods, across all datasets and modalities. This is a methodological paper, not a new method.

**Prerequisites:** Understanding of all existing topological metrics (Betti numbers, Betti matching, clDice, VOI, ARI, Euler characteristic).

#### 3. Proof Roadmap

The paper proves its claims by **construction** — exhibiting specific examples where:
- Connectivity choice flips the Betti number from correct to incorrect
- VOI improves while topology worsens
- Betti number error is zero while Betti matching error reveals real topological errors

These are concrete counterexamples, not abstract theorems.

#### 4. New Problems Discovered

- No consensus exists on which connectivity to use, and the "correct" choice may depend on the application.
- Even Betti matching is affected by connectivity choice — the paper calls for community standardization.
- Multi-class topological metrics are underdeveloped.
- Runtime comparisons between topological losses are rarely fair (different implementations, hardware).

#### 5. Simple Analogy

Imagine two referees scoring the same gymnastics routine using different rulebooks. One referee counts a "stuck landing" as perfect; the other counts it as a minor deduction. If you compare gymnasts scored by different referees, the comparison is meaningless. This paper is like an independent auditor discovering that the topology-aware segmentation community has been using incompatible rulebooks (connectivity conventions) and points metrics (Betti numbers with cancellation) and demands everyone adopt a standardized scoring system (Betti matching with explicit connectivity).

---

### TIER 2 SUMMARIES — Metrics and Evaluation

**Chung et al. (2023) — Wasserstein Distance for Brain Networks**
Unified topological inference framework using Wasserstein distance between persistence diagrams for brain network analysis. Proved superior to graph-theoretic features for temporal lobe epilepsy discrimination. Model-free and distribution-free. *Relevance: Shows Wasserstein distance as a statistical test, not just a loss function.*

**Morand et al. (2025) — Smooth clDice**
Addresses discontinuity issues in standard clDice metric. Proposes a smoothed version that is more robust to small perturbations in the skeleton. *Relevance: Improved evaluation metric for tubular structures.*

**Rieck et al. (2009) — Brain Tumor Geometric Invariants**
Uses discrete Gauss curvature and mean curvature on tumor surfaces reconstructed from MRI. Verifies Gauss-Bonnet theorem (total Gauss curvature = 4π for genus-0 tumors). Shows Gauss curvature is stable while mean curvature is resolution-sensitive. *Relevance: Discrete differential geometry for tumor shape analysis, connecting to the differential geometry branch of the thesis.*

---

### TIER 2 SUMMARIES — Topological and Geometric Neural Networks

**Hansen & Gebhart (2020) — Sheaf Neural Networks**
Generalizes GNNs using cellular sheaf Laplacians. Each edge carries a linear map between node feature spaces. *Relevance: Theoretical framework for modeling multi-modal data on graphs with topology.*

**Bodnar et al. (NeurIPS 2022) — Neural Sheaf Diffusion**
Non-trivial sheaves provide control over asymptotic behavior of diffusion on graphs, enabling separation in heterophilic settings. *Relevance: Could model tumor-brain connectivity where tumor nodes have different properties.*

**Ebli et al. (2020) — Simplicial Neural Networks**
Extends GNNs to simplicial complexes using Hodge Laplacians L_k = B_k^T B_k + B_{k+1} B_{k+1}^T. Kernel of k-Laplacian ≅ k-th cohomology (Hodge theorem). *Relevance: Principled framework for learning on higher-order topological structures.*

**Hajij et al. (2020) — Cell Complex Neural Networks**
Unifying framework for deep learning on cell complexes with inter-cellular message passing. *Relevance: Most general algebraic topology-based neural network framework.*

---

### References — Section E

1. Berger, C., et al. (2025). Pitfalls of topology-aware image segmentation. *IPMI 2025*. arXiv:2412.14619.
2. Chung, M.K., et al. (2023). *NeuroImage*. arXiv:2302.06673.
3. Morand, O., et al. (2025). EPITA Technical Report.
4. Rieck, B., et al. (2009). *Int. J. Biomedical Imaging*. PMC2659777.
5. Hansen, J., Gebhart, T. (2020). arXiv:2012.06333.
6. Bodnar, C., et al. (2022). *NeurIPS 2022*.
7. Ebli, S., et al. (2020). arXiv:2010.03633.
8. Hajij, M., et al. (2020). *NeurIPS 2020 TDA Workshop*.
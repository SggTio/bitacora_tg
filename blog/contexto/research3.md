# Algebraic topology and differential geometry meet deep learning for brain tumor analysis

**The intersection of algebraic topology, differential geometry, and deep learning has produced a rapidly maturing toolkit for brain tumor segmentation, survival prediction, and diagnostics that goes far beyond conventional pixel-wise methods.** Topological data analysis (TDA) enters segmentation pipelines at every stage—as preprocessing features, differentiable loss functions, within-network layers, and even as standalone methods that replace neural networks entirely. Differential geometry contributes through diffeomorphic registration networks that guarantee topology-preserving transformations, atlas-based tumor analysis, and geometric deep learning on manifolds. Survival prediction bridges these domains through functional Cox proportional hazards models that consume topological features extracted from tumor shapes. Together, these methods address a fundamental limitation of standard approaches: conventional metrics like Dice score are topologically blind, unable to detect broken vessels, merged tumor foci, or spurious cavities that carry clinical significance.

This report surveys five interconnected research areas—topology-inspired segmentation architectures, deep survival analysis with Cox models, differential geometric architectures, broader algebraic topology and differential geometry approaches, and topological evaluation metrics—providing the mathematical foundations, implementation details, computational requirements, and key references needed for a master's thesis at this intersection.

---

## 1. TDA-SegUNet and topology-inspired segmentation architectures

### 1.1 TDA-SegUNet: persistence images as auxiliary input channels

TDA-SegUNet is a **2D U-Net-based architecture that integrates topological data analysis through a preprocessing pipeline** constructing persistence images from MRI scans. Published in IEEE (2025), it computes 0-dimensional and 1-dimensional persistent homology from each MRI slice. The **0-dimensional persistence images** capture connected components (blob-like local structures), while **1-dimensional persistence images** capture loops and holes (ring-like global structures). These persistence images serve as additional input channels alongside the original MRI modalities, enriching the U-Net with shape-aware topological information.

The mathematical pipeline proceeds as follows: given an MRI slice, a sublevel set filtration tracks how topological features (connected components, loops) appear and disappear as the intensity threshold increases. Each feature's birth and death values form a persistence diagram. This diagram is then converted into a persistence image—a stable, vectorized 2D representation created by placing weighted Gaussian kernels at each birth-death coordinate and discretizing onto a grid. The resulting persistence images are resolution-matched to the MRI slices and concatenated as input channels.

TDA-SegUNet was evaluated on the **BraTS20 dataset** for segmenting enhancing tumor (ET), tumor core (TC), and whole tumor (WT) regions. The architecture operates on 2D patches, making it computationally lighter than full 3D approaches but potentially sacrificing inter-slice topological information. Requirements include multi-modal MRI input (T1, T1ce, T2, FLAIR) and a preprocessing step to compute persistence images for each slice.

A related architecture, **TDAConvAttentionNet**, extends this paradigm by combining convolutional layers for local features, attention mechanisms for global context, and persistent homology-derived features. It reports Dice scores of **89.36% (WT), 87.36% (TC), and 89.98% (ET)** on BraTS validation data, specifically targeting challenges of patchy patterns and unrealistic topological structures in brain tumor segmentation.

### 1.2 Topological loss functions: the dominant paradigm

The most influential approach to integrating topology into segmentation is through **differentiable topological loss functions** computed during training. These are architecture-agnostic—applicable to any backbone that produces pixel-wise probability maps.

**Hu et al. (NeurIPS 2019)** introduced the foundational TopoLoss, which computes persistent homology on the predicted likelihood map and the ground truth, then penalizes differences between their persistence diagrams. The total loss combines cross-entropy with a weighted topological term: *L = L_CE + λ · L_topo*. The topological loss uses Wasserstein-like matching between persistence diagrams, and its gradient identifies **critical pixels**—the birth and death locations of topological features—pushing them to match the ground truth topology. A key theoretical guarantee states that when the topological loss reaches zero, the thresholded segmentation provably has the same Betti numbers as the ground truth. The computational cost of persistence computation is O(n³) worst-case for n pixels, necessitating patch-based processing (typically 64×64 patches) and batch size of 1 during topological loss computation. Code is available at github.com/HuXiaoling/TopoLoss.

**Clough et al. (IEEE TPAMI, 2020)** took a different approach using **cubical persistent homology with Betti number priors**. Rather than matching persistence diagrams to ground truth, users specify the desired Betti numbers of the target structure—for instance, β₀=1 and β₁=1 for a ring-shaped myocardium. The loss penalizes persistence bars beyond the desired count, making it **semi-supervised**: it can provide a training signal using only topological priors without pixel-wise labels. This was demonstrated on cardiac MRI segmentation (UK Biobank, ACDC) and 3D placenta segmentation from ultrasound.

**Stucki et al. (ICML 2023)** identified a critical flaw in Wasserstein-based matching: it can match topological features that are spatially distant, pairing a component in the upper-left with one in the lower-right. Their **Betti matching** loss resolves this through induced matchings from algebraic topology. Given prediction L and ground truth G, a comparison image C = max(L, G) is constructed, and the inclusion maps L → C ← G induce natural matchings on persistent homology that guarantee **spatially correct correspondences**. This was extended to 3D in 2024 with a highly optimized C++ implementation (github.com/nstucki/Betti-Matching-3D) and to multi-class segmentation at MICCAI 2024, where N-class problems are projected into N single-class problems to avoid computationally infeasible multi-parameter persistent homology.

**Hu et al. (ICLR 2021, Spotlight)** introduced **discrete Morse theory (DMT)** as an alternative to persistent homology for identifying topologically critical structures. DMT identifies 1D skeletons (for curvilinear structures like vessels) and 2D patches (for membranes) that are critical for topological accuracy. The DMT-loss focuses training on these critical structures, achieving Betti error of **0.982 on CREMI** versus 3.016 for standard U-Net and 1.113 for TopoLoss. Code: github.com/HuXiaoling/DMT_loss.

### 1.3 clDice: topology preservation for tubular structures

**Shit et al. (CVPR 2021)** introduced clDice (centerline Dice), which preserves topology through **soft-skeletonization** rather than persistent homology. It computes the fraction of predicted skeleton within the ground truth (topology precision) and the fraction of ground truth skeleton within the prediction (topology sensitivity), then takes their harmonic mean. A theoretical guarantee establishes that clDice=1 implies homotopy equivalence for binary 2D/3D segmentations. The soft-clDice variant is differentiable, using iterative max-pooling for skeletonization, and computationally efficient—using only standard pooling operations with O(n) complexity. It is primarily designed for **tubular structures** (vessels, neurons, roads) and is less meaningful for compact regions like tumors.

### 1.4 Train-free segmentation: TDA without neural networks

François and Tinarrage (arXiv:2401.01160, 2024) demonstrated that **cubical persistent homology alone can segment medical images without any neural network**. Their three-module pipeline identifies the whole object via automatic thresholding, detects a topologically distinctive subset using persistent homology (localizing representative cycles from the persistence diagram), and deduces remaining regions from the geometric object. For glioblastoma on BraTS data, the enhancing tumor has approximately spherical topology (H₂ ≈ 1), which serves as the geometric prior. The method requires **no training data**, is fully interpretable, and guarantees the output topology. However, it requires strong prior knowledge of the target structure's topology and may underperform supervised methods on well-annotated datasets.

### 1.5 Summary of integration strategies

| Integration strategy | Examples | Advantages | Limitations |
|---|---|---|---|
| Preprocessing (TDA features as input) | TDA-SegUNet, TDAConvAttentionNet | Easy to implement; adds shape info without modifying training | Topology not directly optimized; preprocessing cost |
| Topological loss function | TopoLoss, Betti matching, DMT-loss, clDice | End-to-end trainable; provable guarantees; architecture-agnostic | O(n³) for PH; slower training; patch-based |
| Post-processing correction | Byrne et al. CMR correction | Does not affect base model; fixes errors | Not end-to-end; limited correction capability |
| Standalone TDA (no network) | François & Tinarrage (2024) | No training data; interpretable; guaranteed topology | Requires strong priors; lower performance ceiling |

Key software dependencies span **GUDHI**, **Ripser**, **Cubical Ripser**, and **giotto-tda** for persistent homology computation across these methods.

---

## 2. Deep survival analysis with Cox proportional hazards models

### 2.1 Mathematical foundations of neural Cox models

The Cox proportional hazards model defines a patient's hazard as h(t|x) = h₀(t) · exp(f(x)), where h₀(t) is an unspecified baseline hazard and f(x) is a log-risk function. In classical CoxPH, f(x) = βᵀx is linear; in deep extensions, f(x) = g_θ(x) is a neural network. Training maximizes the **Cox partial likelihood**, which does not depend on the baseline hazard:

**L(θ) = ∏_{i: δᵢ=1} [exp(f(xᵢ)) / Σ_{j∈R(tᵢ)} exp(f(xⱼ))]**

where δᵢ indicates whether patient i experienced the event and R(tᵢ) is the risk set at time tᵢ. The negative log partial likelihood serves as a fully differentiable loss function, enabling backpropagation through the entire network. A critical practical consideration is that this loss requires computing over the **entire risk set** for each event time, making standard mini-batch SGD non-trivial—implementations either process the full training set per update or use approximations. The primary evaluation metric is the **concordance index (C-index)**, measuring whether predicted risk scores correctly rank patient survival times.

### 2.2 DeepSurv and Cox-nnet: foundational deep survival models

**DeepSurv** (Katzman et al., BMC Medical Research Methodology, 2018) replaces the linear predictor in CoxPH with a deep feed-forward network. The architecture takes patient covariates as input, passes them through fully connected layers with batch normalization, nonlinear activation (ReLU/SELU), and dropout, and outputs a single log-risk score. Training uses average negative log partial likelihood with L₂ regularization. DeepSurv outperforms linear CoxPH and Random Survival Forests on datasets with nonlinear covariate effects. Multiple implementations exist: the original (github.com/jaredleekatzman/DeepSurv in Theano), **pycox** (PyTorch, github.com/havakv/pycox), **PySurvival**, and **TorchSurv** (opensource.nibr.com/torchsurv/).

**Cox-nnet** (Ching et al., PLOS Computational Biology, 2018) is a two-layer neural network optimized for high-dimensional omics data (~20,000 gene features). Its hidden layer activations serve as survival-sensitive dimensionality reduction, revealing richer biological information at pathway and gene levels. **Cox-nnet v2.0** (Wang et al., Bioinformatics, 2021) achieved 32-fold training speedups and extended to histopathology imaging data with pre-extracted features combined with gene expression data. Code: github.com/lanagarmire/cox-nnet.

### 2.3 End-to-end imaging + Cox architectures

The landmark paper integrating CNNs directly with Cox survival models is by **Mobadersany et al. (PNAS, 2018)**, introducing the **Survival CNN (SCNN)**. The architecture passes histology image patches through VGG-like convolutional layers, then fully connected layers, terminating in a Cox proportional hazards output node trained with negative log partial likelihood. The entire CNN is trained end-to-end, so **convolutional filters learn survival-relevant visual features directly**. The genomic variant (GSCNN) incorporates molecular variables (IDH mutation, 1p/19q codeletion) at the fully connected layers, achieving C-index of ~0.801 on TCGA gliomas—**surpassing the WHO clinical paradigm** (~0.747). Heat map visualizations confirm the network recognizes clinically meaningful structures like microvascular proliferation and necrosis.

The precursor **DeepConvSurv** (Zhu et al., IEEE BIBM, 2016) was the first to develop a deep CNN for survival analysis directly from pathological images, replacing hand-crafted radiomic features with learned deep features and training with Cox partial likelihood loss.

### 2.4 The dominant BraTS pipeline: segmentation then survival

The BraTS challenge (2017–2020) included overall survival prediction, establishing a **two-stage paradigm** that dominates brain tumor survival analysis:

**Stage 1** uses 3D U-Net variants (nnU-Net, Attention U-Net, V-Net) to segment multi-modal MRI (T1, T1ce, T2, FLAIR) into enhancing tumor, tumor core, and whole tumor/edema regions. **Stage 2** extracts radiomic features from segmented regions—volume, surface area, shape descriptors, texture features (GLCM, GLRLM), histogram statistics, location features—combines them with clinical variables (age, resection status), and feeds them into survival models (LASSO-Cox, Random Forest, XGBoost, or gradient boosting). Classification targets short-term (<10 months), mid-term (10–15 months), and long-term (>15 months) survivors. Typical performance: segmentation Dice **~0.88–0.95** for whole tumor, survival C-index **~0.55–0.70**. Feng et al. (Frontiers in Computational Neuroscience, 2020) won first place in BraTS 2018 OS prediction using this paradigm.

Chen et al. (Frontiers in Computational Neuroscience, 2023) demonstrated a **convolutional denoising autoencoder + CoxPH** pipeline for GBM, extracting features from multi-modal MRI tumor regions and combining them with clinical data (age, sex, KPS, treatment status) for survival analysis, achieving **C-index of 0.74** on BraTS 2019.

### 2.5 Functional Cox PH with topological features: bridging TDA and survival

A particularly notable connection to algebraic topology comes from **Moon et al. (Annals of Applied Statistics, 2023)**, who developed the **Functional Cox Proportional Hazards (FCoxPH) model** that uses persistent homology features as functional predictors for survival analysis. The pipeline proceeds: tumor segmentation → distance transform of segmented mask → cubical complex persistent homology → persistence diagrams represented as functions → functional Cox regression with interaction terms for tumor location. Applied to **77 brain tumor patients and 133 lung cancer patients**, the method found that irregular, heterogeneous shape patterns captured by topological features are positively associated with survival hazards (p < 0.001). Jang et al. (arXiv:2512.05646, 2025) extended this approach with lobe-specific interaction modeling for gliomas, confirming topological features as strong survival predictors beyond conventional radiomics.

This FCoxPH framework represents the most direct synthesis of Topics 1 and 2: algebraic topology (persistent homology) generates the features, and a Cox proportional hazards model consumes them for survival prediction.

---

## 3. Diffeomorphic registration networks and differential geometric architectures

### 3.1 VoxelMorph: the foundational learning-based registration framework

**VoxelMorph** (Balakrishnan et al., IEEE TMI, 2019; Dalca et al., MICCAI 2018) established the paradigm for learning-based diffeomorphic registration. The architecture is a U-Net encoder-decoder that takes concatenated moving and fixed 3D images as input and outputs a dense deformation field. In the diffeomorphic variant (VoxelMorph-diff), the network outputs a **stationary velocity field (SVF)** that is integrated via scaling-and-squaring to produce a diffeomorphic (smooth, invertible, topology-preserving) transformation: **φ = exp(v)**. The loss combines image similarity (NCC or mutual information) with a smoothness regularizer on deformation gradients.

On brain MRI, VoxelMorph achieves comparable Dice to classical ANTs SyN registration but is **100–1000× faster** (seconds versus hours). Preprocessing requires FreeSurfer processing, skull stripping, and affine normalization. Code: github.com/voxelmorph/voxelmorph (PyTorch and TensorFlow).

### 3.2 IConDiffNet: inverse consistency meets diffeomorphism

**IConDiffNet** (Liao et al., Physics in Medicine & Biology, 2025) introduces a novel unsupervised architecture that systematically enforces both diffeomorphism and **inverse consistency**—the requirement that forward and backward transformations are exact inverses. The architecture incorporates an energy constraint minimizing total deformation energy, ensuring physically plausible transformations. Evaluated on **375 subjects** for 3D inter-patient brain MRI registration, IConDiffNet outperforms VoxelMorph-Diff, SYMNet, and ANTs-SyN in Dice, Hausdorff distance, and total deformation energy. The mathematical foundation lies in diffeomorphic transformations as smooth invertible mappings where the energy functional constrains the solution space to anatomically plausible deformations.

### 3.3 MDReg-Net and multi-resolution approaches

**MDReg-Net** (Li and Fan, Human Brain Mapping, 2022) uses a multi-resolution fully convolutional framework where sub-networks at each resolution level estimate velocity fields. Coarser velocity fields warp the moving image before finer-resolution refinement, enabling progressive handling of large deformations. The key result: MDReg-Net produces near-zero non-positive Jacobian determinant voxels (**0.089 average**), compared to **9,571 for ANTs** and **3.68 for VoxelMorph**, indicating far superior diffeomorphic quality. Evaluated on PING, MALC, and Mindboggle-101 brain datasets.

### 3.4 GradICON: implicit diffeomorphism through inverse consistency

**GradICON** (Tian et al., CVPR 2023) takes a radically different approach: rather than explicitly constraining transformations to be diffeomorphic, it penalizes deviations of the **Jacobian of the composed forward-backward map** from the identity matrix. This gradient inverse-consistency loss implicitly regularizes transformations toward diffeomorphisms without any explicit smoothness penalty—regularity emerges as a consequence. A single hyperparameter set works across brain MRI (OASIS), knee MRI, and lung CT without dataset-specific tuning. Code: github.com/uncbiag/ICON.

### 3.5 FireANTs: Riemannian optimization on the diffeomorphism group

**FireANTs** (Jena et al., arXiv:2404.01249, 2024) is not a neural network but an optimization-based method that directly exploits the **Lie group structure of diffeomorphisms**. It generalizes adaptive optimization (Adam/RMSProp) from Euclidean spaces to the non-Euclidean manifold of diffeomorphisms, computing descent directions from the identity transform using the Lie algebra (tangent space at identity) and the exponential map. FireANTs achieves state-of-the-art on multiple benchmarks while being **200–1200× faster than ANTs**, requiring **10× less memory** than deep learning methods, and being completely training-free. Code: github.com/rohitrango/FireANTs.

### 3.6 How differential geometry serves brain tumor analysis

Differential geometry enters brain tumor analysis through several distinct pathways:

**Atlas-based segmentation** uses diffeomorphic registration to map brain atlases to patient space for label transfer. For tumor patients, special handling is needed because tumors violate the diffeomorphic assumption (tissue is created, not just deformed). Solutions include tumor masking during registration, joint registration-segmentation (Estienne et al., Frontiers in Computational Neuroscience, 2020), and low-rank-plus-sparse decomposition that separates "healthy" from "pathological" image components before registration (Liu et al., PMC4707015, 2015).

**Biophysical tumor modeling** couples diffeomorphic registration with reaction-diffusion tumor growth models. The **SIBIA framework** (Scheufele et al., CMAME, 2019) jointly estimates tumor growth parameters (diffusion coefficient, proliferation rate) and diffeomorphic registration maps, using a healthy atlas as proxy for the pre-tumor brain state. The tumor growth PDE is ∂c/∂t = ∇·(D∇c) + ρc(1-c), where c represents tumor cell concentration.

**Volume calculation and mass effect** quantification uses the **Jacobian determinant** of the deformation field from diffeomorphic registration. Local volume changes between atlas and patient space directly quantify tumor-induced displacement and compression. Gooya et al. demonstrated that normalized Jacobian determinants correlate with expert visual scores of tumor mass effect.

**Longitudinal tracking** is addressed by the BraTS-Reg challenge (ISBI/MICCAI 2022), benchmarking deformable registration between pre-operative and follow-up brain glioma scans with dramatic tissue appearance changes.

| Method | How geometry is used | Key advantage | Brain tumor application |
|---|---|---|---|
| VoxelMorph-diff | SVF → exp map → diffeomorphism | 100-1000× faster than classical | Atlas registration |
| IConDiffNet | Diffeomorphism + inverse consistency | Superior alignment quality | Inter-patient registration |
| MDReg-Net | Multi-resolution velocity fields | Near-zero folding voxels | Brain structure alignment |
| GradICON | Jacobian-based implicit regularization | No tuning across datasets | Multi-organ registration |
| FireANTs | Riemannian Adam on Lie algebra | Training-free, memory-efficient | Atlas construction |
| SIBIA/CLAIRE | Diffeomorphic + PDE tumor model | Joint registration-biophysics | Tumor parameter estimation |

---

## 4. Gauge equivariance, geometric deep learning, and Riemannian neural networks

### 4.1 Geometric deep learning: the unifying framework

**Bronstein et al. (arXiv:2104.13478, 2021)** established a unifying mathematical framework for neural network architectures through the lens of symmetry groups—the "5G" framework: Grids (translation equivariance → CNNs), Groups (rotation/scale equivariance), Graphs (permutation equivariance → GNNs), Geodesics (intrinsic operations on Riemannian manifolds), and Gauges (local symmetry transformations on fiber bundles). This framework is directly relevant to brain imaging: functional connectivity networks are graphs, cortical surfaces are Riemannian manifolds, and brain atlas operations involve diffeomorphisms.

**Gauge equivariant CNNs** (Cohen et al., ICML 2019) extend equivariance from global symmetries to local gauge transformations on manifolds. On a manifold, there is no preferred frame for positioning filters, so networks must be equivariant to gauge changes. Feature maps become sections of associated vector bundles, with convolution defined intrinsically. Hussain and Khan (Scientific Reports, 2025) applied gauge equivariant CNNs to diffusion MRI, achieving comparable angular resolution upsampling with **5–15 training subjects** versus the 20–40 required by baselines. Ru et al. (Neurocomputing, 2025) used anisotropic gauge equivariant convolutions on 3D meshes for **intracranial aneurysm segmentation**.

### 4.2 Riemannian geometry for disease progression and brain connectivity

**Louis et al. (IPMI 2019)** used deep generative networks to learn both a submanifold of observations and a Riemannian metric such that observed disease progressions become geodesics. Applied to Alzheimer's disease, the learned metric revealed that hippocampal atrophy progresses faster in females and occurs earlier in APOE4 carriers. The mathematical foundation rests on the theorem that given smooth curves on a manifold, there exists a Riemannian metric making them geodesics (via the tubular neighborhood theorem and Nash embedding theorem).

**SPD manifold networks** operate on the Riemannian manifold of symmetric positive definite matrices for brain functional connectivity analysis. These networks use Log-Euclidean algebra for projecting SPD matrices to tangent space, with BiMap and ReEig layers replacing standard convolution and ReLU. Applied to brain state change detection from fMRI, they capture the intrinsic geometry of correlation matrices.

The comprehensive reference for Riemannian methods in medical imaging is **Pennec, Sommer, and Fletcher (eds.), "Riemannian Geometric Statistics in Medical Image Analysis" (Elsevier/MICCAI Society, 2019)**, covering Fréchet means, parallel transport, Levi-Civita connections, and applications to diffusion tensor tractography and computational anatomy.

---

## 5. Broader algebraic topology and differential geometry approaches for brain cancer

### 5.1 The Smooth Euler Characteristic Transform for GBM prognosis

**Crawford et al. (Journal of the American Statistical Association, 2020)** introduced the Smooth Euler Characteristic Transform (SECT), a topological statistic with a well-defined inner product structure (unlike persistence diagrams) that quantifies MRI tumor shapes. The SECT constructs directional Euler characteristic curves from tumor boundaries and smooths them for use in functional regression. Applied to the **TCGA-GBM cohort**, SECT outperformed both existing tumor shape quantifications and common molecular assays as a predictor of clinical outcomes. The pipeline operates post-segmentation: segmented tumor → SECT feature extraction → survival modeling.

### 5.2 Persistent homology for brain tumor classification and connectomics

**Bhattacharya et al. (arXiv:2407.17938, 2024)** applied Vietoris-Rips persistent homology to DWI brain connectome data (84 ROIs from the Desikan-Killiany atlas) for differentiating meningiomas from gliomas from healthy controls, achieving **88% accuracy** (healthy vs. meningioma) and **80%** (glioma vs. meningioma). Persistent homology reveals topological alterations in whole-brain structural connectivity that are specific to tumor subtypes—a fundamentally different information source than imaging intensity.

For GBM heterogeneity analysis, a 2025 preprint (arXiv:2503.17331) introduced **subcomplex lacunarity**, a TDA-based shape descriptor that quantifies geometric characteristics of necrosis. Using the Persistent Homology Transform and persistence landscapes, the method identified **four distinct GBM subtypes** based on necrotic cell aggregation patterns in 93 TCGA-GBM patients.

**TopoGBM** (arXiv:2602.11234, February 2026) represents the cutting edge: a semi-supervised representation learning approach using a brain-inspired topological regularizer (TopoLoss) on an encoder for GBM prognosis. It achieves C-index of **0.67 and 0.58** across unseen multi-institutional datasets (UPENN, RHUH, UCSF), with approximately 50% of the prognostic signal localized to tumor and peri-tumoral regions via occlusion attribution.

### 5.3 TDA for GBM detection and monitoring

A comprehensive TDA approach for GBM from FLAIR MRI (MDPI Mathematics, 2020) combined persistent homology analysis of tumor growth models, **persistent entropy** for monitoring temporal progression, and topological+textural features with interpretable ML classifiers, achieving **97% classification accuracy** on TCIA data. Enhanced brain tumor detection via TDA and low-rank Tucker decomposition (ScienceDirect, 2024) achieved **97.28% overall accuracy** across pituitary, meningioma, and glioma tumors.

### 5.4 Discrete curvature invariants for tumor evolution

Discrete differential geometry contributes through **tumor surface geometric invariants** (International Journal of Biomedical Imaging, PMC2659777). Reconstructed tumor surfaces from MRI segmentation yield discrete Gauss curvature (angle defect formula) and discrete mean curvature (dihedral angles). The Gauss-Bonnet theorem confirms Gauss curvature equals **4π for genus-0 (sphere-topology) tumors**, providing a validation criterion for segmentation quality. Longitudinal analysis of low-grade gliomas over 2–3 years showed Gauss curvature is stable while mean curvature is sensitive to mesh resolution—an important practical consideration for volume tracking.

### 5.5 Sheaf theory and higher-order topological neural networks

Several advanced algebraic structures have been adapted for neural network architectures with potential medical imaging applications:

**Sheaf Neural Networks** (Hansen and Gebhart, arXiv:2012.06333, 2020) generalize graph convolutional networks using cellular sheaf Laplacians, assigning vector spaces and linear maps to nodes and edges for richer relational encoding. **Neural Sheaf Diffusion** (Bodnar et al., NeurIPS 2022) showed that non-trivial sheaves provide greater control over asymptotic diffusion behavior, enabling linear class separation in heterophilic graph settings—relevant for brain tumor networks where tumor-induced connectivity changes create heterophilic structures.

**Simplicial Neural Networks** (Ebli et al., NeurIPS 2020 TDA Workshop) extend GNNs to simplicial complexes using Hodge Laplacians: L_k = B_k^T B_k + B_{k+1} B_{k+1}^T, where the kernel of the k-Laplacian is isomorphic to k-cohomology by the Hodge theorem. **Cell Complex Neural Networks** (Hajij et al., NeurIPS 2020 TDA Workshop) provide a unifying framework for deep learning on cell complexes with inter-cellular message passing respecting topology. While these are not yet widely applied to brain tumor analysis specifically, they provide principled frameworks for brain mesh analysis and tumor surface characterization.

### 5.6 Wasserstein distance for brain network discrimination

**Chung et al. (NeuroImage, 2023)** developed a unified topological inference framework using Wasserstein distance between persistence diagrams of brain networks. Applied to temporal lobe epilepsy discrimination from resting-state fMRI, the method proved superior to graph-theoretic features and bottleneck distance, successfully localizing discriminative brain regions. The framework is model-free and distribution-free, robust to sex and site variations. Code: github.com/laplcebeltrami/PH-STAT.

### 5.7 Comprehensive reviews anchoring the field

Two major 2025 reviews systematize this landscape. **"Advancing Precision Medicine: Algebraic Topology and Differential Geometry in Radiology and Computational Pathology"** (Laboratory Investigation, 2025) covers how Betti curves, persistence landscapes, and curvature features augment radiomics, pathomics, and multiomics. **"Enhancing Brain Tumor Diagnosis: The Role of Topological Features in Deep Learning Approaches"** (Biomedical Signal Processing and Control, 2025) provides the first systematic benchmark of DL-TDA hybrid models, proposing a taxonomy of early fusion (topology-informed inputs), intermediate fusion (TDA-embedded bottlenecks), and topological loss-based supervision.

---

## 6. Topological versus standard evaluation metrics

### 6.1 Standard metrics and their topological blindness

**Dice Similarity Coefficient** (DSC = 2|P∩G|/(|P|+|G|)) measures volumetric overlap in O(n) time and is the primary metric across BraTS challenges and most segmentation benchmarks. Soft Dice loss is differentiable and standard in nnU-Net. However, Dice is **completely insensitive to topology**: two segmentations with identical Dice can have dramatically different connectivity, holes, and structural integrity. A single-pixel gap breaking a vessel changes β₀ by +1 and β₁ by −1 but changes Dice by approximately 0.001.

**Hausdorff Distance 95%** (HD95) captures worst-case boundary deviation but is sensitive to single outlier voxels, non-differentiable in standard form, and purely geometric—it ignores structural integrity entirely. **IoU** (Jaccard index) is monotonically related to Dice (DSC = 2·IoU/(1+IoU)) and shares the same topological blindness.

### 6.2 Topological metrics: what they capture and at what cost

**Betti number error** (β_err = Σ_d |β_d(P) − β_d(G)|) counts global topological complexity mismatch—whether the prediction has the correct number of connected components, loops, and cavities. For β₀, computation is O(n·α(n)) via union-find; for higher dimensions, cubical complex computation is O(n³) worst-case. Betti number error is **not spatially aware**: a prediction could have the correct component count but in entirely wrong locations. Compensating errors cancel out (one extra hole + one missing hole = β₁ error of 0). It is not differentiable.

**Betti matching error** (τ_err), introduced by Stucki et al. (ICML 2023), resolves the spatial awareness problem. Using induced matchings from algebraic topology through inclusion maps on persistent homology, it counts features in prediction and ground truth that do not spatially correspond. It is strictly more sensitive than Betti number error and, crucially, **differentiable**—enabling its use as both metric and loss function. Computational cost is O(n³), which limited 3D applicability until the optimized C++ implementation of 2024.

**Wasserstein distance between persistence diagrams** (W_p) measures the total optimal matching cost across all topological features, weighting by persistence. It is differentiable and serves as the loss function in Hu et al.'s TopoLoss (NeurIPS 2019). However, Stucki et al. demonstrated that **Wasserstein matching can be spatially incorrect**, pairing topological features that are distant in the image purely based on persistence diagram coordinates. **Bottleneck distance** (W_∞) captures only the single largest topological discrepancy, analogous to Hausdorff distance in persistence diagram space. Both have O(n³) and O(n^{1.5} log n) complexity respectively.

**clDice** offers efficient topology-awareness at O(n) cost through skeleton-based evaluation, with a theoretical guarantee of homotopy equivalence. **ccDice** (Connected Component Dice, Rougé et al., MICCAI 2024 Workshop) generalizes Dice to the connected component scale. **Euler characteristic error** (χ = β₀ − β₁ + β₂) provides an O(n) topological scalar but suffers from cancellation effects.

### 6.3 When metrics disagree and what this means clinically

The most instructive cases of disagreement come from tubular structures. Shit et al. (CVPR 2021) showed that two 3D brain vessel segmentations with **identical Dice scores** differed dramatically in small vessel connectivity—only clDice and topological metrics distinguished them. For brain tumors, standard metrics cannot distinguish a prediction that correctly identifies **separate tumor foci** from one that merges them, nor can they detect spurious holes within predicted tumor regions.

Conversely, topological metrics ignore volumetric accuracy: a topologically correct segmentation can have poor pixel overlap. A dilated or eroded segmentation preserving connectivity has perfect Betti numbers but degraded Dice and Hausdorff distance.

**Berger et al. (IPMI 2025)** identified critical pitfalls: distributional metrics like VOI and ARI "irreversibly entangle topological and volumetric errors," and the choice of **connectivity** (4- vs. 8-connectivity in 2D, 6- vs. 26-connectivity in 3D) dramatically affects topological metrics, requiring standardization.

### 6.4 Computational cost comparison and practical recommendations

| Metric | Complexity | Relative cost | Differentiable | Captures topology |
|---|---|---|---|---|
| Dice/IoU | O(n) | 1× | Yes (soft) | No |
| HD95 | O(n) with DT | 2–5× | Approximated | No |
| Betti number error | O(n³) for β₁+ | 50–500× | No | Partially (no spatial) |
| Betti matching error | O(n³) | 100–1000× | Yes | Yes (spatially correct) |
| Wasserstein (PD) | O(k³) | 100–1000× | Yes | Yes (not spatially correct) |
| clDice | O(n) | 5–10× | Yes (soft) | Tubular only |
| ccDice | O(n log n) | 3–5× | No | Component-level |
| Euler χ error | O(n) | 1–2× | No | Minimal (cancellation) |

All topological metrics are **architecture-agnostic** in evaluation: they can be computed on any segmentation output regardless of the producing model. Betti matching, Wasserstein distance, and clDice operate on soft probability maps (requiring access to network outputs, not just binary masks), while Betti number error and ccDice work on binary segmentations.

For BraTS and brain tumor segmentation specifically, current challenge metrics (DSC, HD95, sensitivity, specificity, lesion-wise DSC from 2023 onward) do not include any topological metrics. Adding **β₀ error** (component counting for detecting merged/split tumors) and **ccDice** would capture clinically relevant topological information at minimal computational cost. For detailed topological analysis, Betti matching error with standardized connectivity is the current gold standard.

---

## 7. Cross-cutting themes and synthesis across domains

### 7.1 Three mathematical pillars, one clinical goal

The methods surveyed here converge on brain tumor analysis from three mathematical directions that are more complementary than competing. **Persistent homology** (from algebraic topology) captures global shape features—connected components, loops, cavities—that characterize tumor morphology, detect segmentation errors, and predict survival. **Diffeomorphic registration** (from differential geometry) enables atlas-based analysis, volume quantification through Jacobian determinants, and longitudinal tracking while preserving brain topology. **Cox proportional hazards models** (from survival statistics) provide the inferential framework connecting image-derived features to patient outcomes.

The most powerful approaches combine these pillars. The **FCoxPH pipeline** (Moon et al., 2023) extracts persistent homology features from segmented tumors and feeds them into functional Cox regression for survival prediction. The **SIBIA framework** (Scheufele et al., 2019) couples diffeomorphic registration with biophysical tumor growth models. The **SECT** (Crawford et al., 2020) uses Euler characteristic transforms—a topological construction—within a statistical framework for GBM prognosis.

### 7.2 Maturity and readiness of each approach

Topology-inspired loss functions have reached a level of maturity suitable for practical deployment, with the **Betti matching** line of work (Stucki, Paetzold, Bauer at TUM) representing the current state-of-the-art through optimized 3D implementations and multi-class extensions. Diffeomorphic registration networks are well-established, with **VoxelMorph** serving as a standard tool and newer methods like GradICON eliminating the need for dataset-specific tuning. Deep survival analysis with Cox models is mature for tabular and histopathology data but remains largely a two-stage process for brain MRI, with end-to-end imaging-to-survival architectures being an active research frontier.

The least mature but most mathematically rich approaches—sheaf neural networks, simplicial neural networks, gauge equivariant CNNs for brain imaging—remain largely theoretical or demonstrated only on non-tumor tasks. Their application to brain tumor analysis represents an open research opportunity.

### 7.3 Practical guidance for implementation

For a brain tumor analysis pipeline incorporating these methods, the following practical considerations apply. **Data**: BraTS 2021+ provides the standard benchmark (2,040 patients, 4 MRI modalities, expert segmentation labels). **Compute**: Persistent homology adds O(n³) overhead per patch; 3D Betti matching requires the optimized C++ backend. Diffeomorphic registration networks require a single GPU (NVIDIA V100 or equivalent). **Software stack**: giotto-tda or GUDHI for persistent homology, VoxelMorph or GradICON for registration, pycox or TorchSurv for survival analysis, and the Betti-Matching-3D library for topological loss and evaluation.

---

## Conclusion

The integration of algebraic topology and differential geometry into deep learning for brain tumor analysis has progressed from theoretical curiosity to practical methodology in under a decade. Three developments stand out as particularly consequential. First, **differentiable topological loss functions**—especially Betti matching (Stucki et al., ICML 2023)—have solved the spatial correspondence problem that plagued earlier Wasserstein-based approaches, enabling topology-aware training that is both mathematically rigorous and computationally feasible in 3D. Second, **functional Cox proportional hazards models** consuming persistent homology features (Moon et al., 2023; Crawford et al., 2020) demonstrate that topological shape descriptors carry genuine prognostic information beyond conventional radiomics—irregular and heterogeneous patterns captured only by TDA predict worse survival outcomes. Third, **implicit diffeomorphic regularization** through inverse consistency (GradICON, CVPR 2023) eliminates the need for explicit smoothness penalties while producing approximately diffeomorphic transformations, dramatically simplifying the registration component of tumor analysis pipelines.

The critical gap remains evaluation: BraTS challenges still rely exclusively on Dice and Hausdorff metrics that are provably blind to the topological features these methods are designed to preserve. Adopting Betti matching error and connected component Dice as standard evaluation metrics would align assessment with the structural properties that matter clinically—whether tumor foci are correctly separated, whether vessel connectivity is preserved, and whether necrotic cavities are faithfully represented. For a master's thesis at this intersection, the most impactful contribution would likely combine a topologically-informed segmentation architecture (Betti matching loss on a 3D U-Net backbone) with topological feature extraction for survival prediction (persistent homology → FCoxPH), evaluated on BraTS data using both standard and topological metrics to quantify what is gained by respecting the mathematics of shape.
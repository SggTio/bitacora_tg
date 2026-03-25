# Differential geometry meets engineering: a map of active research frontiers

**The intersection of differential geometry, algebraic topology, and engineering has crystallized into at least a dozen sustained research communities, each developing new mathematical theory driven by application needs.** These are not isolated efforts — they share infrastructure, conferences, software, and increasingly, researchers. For a prospective PhD student in applied mathematics, these communities offer unusually rich opportunities: the mathematical problems are deep and genuinely open, the applications are consequential, and the fields are growing rapidly. This report maps the landscape, organized by research area, identifying the mathematical theories under development, the communities sustaining them, and the connections between them.

---

## Computational anatomy: where infinite-dimensional geometry became an applied science

Computational anatomy is perhaps the most mathematically mature example of differential geometry in engineering. Its core framework, **Large Deformation Diffeomorphic Metric Mapping (LDDMM)**, places a right-invariant Riemannian metric on the infinite-dimensional Lie group of diffeomorphisms. Optimal deformations between anatomical shapes are geodesics on this group, governed by the **Euler-Poincaré equation (EPDiff)** — the same equation that governs ideal fluid flow in Arnold's 1966 formulation, but with Sobolev-class rather than L² metrics. The initial momentum fully determines the geodesic (geodesic shooting), yielding a finite-dimensional parameterization of inherently infinite-dimensional deformations.

This is a genuine community, not a collection of isolated papers. The Johns Hopkins Center for Imaging Science (Michael Miller, Laurent Younes, Nicolas Charon) developed the foundational theory and maintains active software. Alain Trouvé (ENS Paris-Saclay) co-developed the optimal control formulations. Darryl Holm (Imperial College) bridges geometric mechanics and computational anatomy through his work on Euler-Poincaré theory, metamorphosis, and stochastic variational principles. Stefan Sommer (Copenhagen) has developed stochastic LDDMM and sub-Riemannian frame bundle methods. Xavier Pennec (INRIA) pioneered intrinsic Riemannian statistics for transformation groups. The landmark textbook *Riemannian Geometric Statistics in Medical Image Analysis* (Pennec, Sommer, Fletcher, 2020) unifies the field across **636 pages** with contributions from dozens of researchers.

The field underwent a major paradigm shift from 2019 onward with deep learning integration. Adrian Dalca's **VoxelMorph** (MIT/Harvard) predicts diffeomorphic deformation fields in seconds rather than hours. Marc Niethammer (now UCSD) developed Region-specific Diffeomorphic Metric Mapping generalizing EPDiff with spatially-varying regularizers, and NePhi (ECCV 2024) for neural deformation fields. A March 2025 paper extends LDDMM from Lie groups to **Lie groupoids** to handle discontinuous sliding motion — a genuinely new mathematical construction motivated by application needs. The mathematical frontier here includes stochastic EPDiff equations with Eulerian noise preserving momentum map structure, neural ODEs interpreted through the LDDMM lens, and implicit neural representations for resolution-independent diffeomorphic registration.

Key venues include MICCAI, IPMI, the MFCA workshop (Mathematical Foundations of Computational Anatomy), SIAM Journal on Imaging Sciences, and Medical Image Analysis. The community publishes across pure mathematics (Journal of Nonlinear Science, Foundations of Computational Mathematics) and applied venues alike.

---

## Shape spaces and manifold statistics are building a new statistical theory

Closely allied to computational anatomy, **statistical shape analysis on manifolds** develops the mathematical infrastructure for doing statistics when data lives on curved spaces. The core problem: anatomical shapes, curves, and surfaces form infinite-dimensional Riemannian manifolds, not vector spaces. Standard statistical tools (means, regression, PCA) must be rebuilt from scratch.

The **elastic shape analysis** framework, developed principally by Anuj Srivastava and Eric Klassen (Florida State), uses the Square-Root Velocity (SRV) transform to convert the elastic metric on curve spaces into the L² metric, making the pre-shape space a unit Hilbert sphere where geodesics and means become computable. A major 2024 advance by Martin Bauer (Vienna), Charon, Klassen and collaborators introduced the **F_{a,b} transform**, fully generalizing SRV to elastic metrics for all parameter choices and all dimensions — unifying two decades of work and enabling metric learning for specific applications like cell shape analysis.

For surfaces, the story is richer and harder. Michor and Mumford's startling 2005 discovery that the L² metric on spaces of immersions has **vanishing geodesic distance** showed that infinite-dimensional geometry behaves very differently from finite-dimensional geometry. Higher-order Sobolev metrics resolve this, and comprehensive numerical frameworks for second-order Sobolev metrics on 3D surfaces were published in IJCV in 2023 by Bauer, Charon, and collaborators.

Xavier Pennec's framework of **intrinsic statistics on Riemannian manifolds** (Fréchet means, exponential barycenters, principal geodesic analysis) provides the statistical backbone. Tom Fletcher (Virginia) developed geodesic regression and principal geodesic analysis. Recent theoretical surprises include the **smeary central limit theorem** (Eltzner et al., Annals of Statistics, 2019), revealing non-standard asymptotic behavior near cut loci — a genuinely new phenomenon with no Euclidean analogue. Sebastian Kurtek (Ohio State) applies elastic shape analysis to brain structures, tumor shapes, and cell biology, while the **Geomstats** library (Nina Miolane, UCSB) provides computational infrastructure supporting NumPy, PyTorch, and TensorFlow backends.

This area is particularly attractive for PhD research because it combines deep geometric questions (what is the right metric on shape space?) with statistical theory (what happens to CLTs on manifolds?) and immediate biomedical applications.

---

## Geometric deep learning grounds neural architectures in differential geometry

Geometric deep learning (GDL) has become one of the fastest-growing intersections of differential geometry and computation. The framework, systematized by Michael Bronstein (Oxford), Joan Bruna (NYU), Taco Cohen, and Petar Veličković in their "5 Gs" framework (Grids, Groups, Graphs, Geodesics, Gauges), derives neural network architectures from symmetry principles. The full MIT Press textbook is being released chapter by chapter in 2025.

The deepest geometry enters through **gauge equivariant convolutional networks**. Maurice Weiler's 2024 monograph formulates CNNs on manifolds as a literal gauge field theory: feature maps are sections of associated fiber bundles, convolution kernels are G-steerable kernels constrained by the structure group, and parallel transport defines how features are compared across the manifold. This is not a metaphor — it is the precise mathematical content. Cohen, Geiger, and Weiler proved that G-equivariant convolutions with steerable kernels are the most general equivariant linear maps, establishing G-CNNs as a universal architecture class.

**Diffusion models on manifolds** represent a particularly active 2022–2025 frontier. Riemannian score-based generative models (De Bortoli et al., NeurIPS 2022) extend score matching to manifolds using heat kernels and the Laplace-Beltrami operator. Riemannian Flow Matching (Chen and Lipman, 2024) extends flow matching using conditional vector fields on manifolds. A 2025 paper introduces a Riemannian metric derived from diffusion model score functions to characterize data manifold geometry. These developments require new theory at the interface of stochastic analysis on manifolds, spectral geometry, and generative modeling.

The practical impact is enormous. **AlphaFold** uses SE(3)-equivariant architectures. Drug discovery, molecular modeling, climate science on spherical data, and robotics all benefit directly. The GRaM workshop (Geometry-grounded Representation Learning and Generative Modeling) was the most popular workshop at ICML 2024.

---

## Riemannian optimization has become essential computational infrastructure

Optimization on manifolds reformulates constrained optimization as unconstrained optimization on smooth manifolds — Stiefel, Grassmann, SPD, fixed-rank, and oblique manifolds being the workhorses. The mathematical core uses Riemannian gradients, retractions (efficient approximations to exponential maps), vector transport, and geodesic convexity.

Nicolas Boumal's 2023 Cambridge University Press textbook and his **Manopt** ecosystem (MATLAB, Python via Pymanopt, Julia via Manopt.jl) have become standard infrastructure. Boumal holds an ERC Starting Grant (GEOSYM, 2021–2027) for this work. P.-A. Absil's foundational 2008 textbook with Mahony and Sepulchre established the field. Silvère Bonnabel's 2013 paper on Riemannian stochastic gradient descent enabled large-scale applications.

A striking application is **brain-computer interfaces via SPD manifold methods**. EEG signals represented as spatial covariance matrices become points on the SPD manifold. Alexandre Barachant's Minimum Distance to Riemannian Mean classifier won **five international BCI competitions** (2014–2016) using the affine-invariant Riemannian metric. The advantage is not marginal — Riemannian methods offer robustness to noise, invariance to electrode re-referencing, and small training sample requirements that Euclidean methods cannot match.

Recent theoretical advances include Riemannian versions of Adam (Becigneul et al., 2019), accelerated methods on manifolds (Alimisis et al., AISTATS 2021), landing methods combining projection and intrinsic strategies (Ablin, 2022), and a novel **iso-Riemannian optimization** framework (Diepeveen, 2025) using non-Levi-Civita connections for optimization on learned data manifolds.

---

## Information geometry is experiencing a renaissance

Information geometry, founded by Shun-ichi Amari, studies probability distribution spaces using the Fisher information metric and dual α-connections. After decades as a somewhat specialized field, it is experiencing a renaissance driven by connections to deep learning optimization, generative modeling, and optimal transport.

The **natural gradient** — Riemannian gradient descent with the Fisher metric — is the bridge to deep learning. K-FAC (Martens and Grosse, 2015) made practical natural gradient feasible for deep networks through Kronecker-factored approximate curvature. The Bayesian Learning Rule (Khan and Rue, 2024) tweaks Adam to approximate Bayesian inference via natural gradients. The connection between Fisher information, Gauss-Newton, and generalized Gauss-Newton matrices (Martens, JMLR 2020) provides theoretical depth.

A vibrant emerging direction connects information geometry to **optimal transport**. Gabe Khan and Jun Zhang's survey "When Optimal Transport Meets Information Geometry" maps two distinct geometries on probability spaces: Fisher-Rao (local, infinitesimal) versus Wasserstein (global, transport-based). The Jordan-Kinderlehrer-Otto scheme reinterprets the Fokker-Planck equation as gradient flow of KL-divergence in Wasserstein space. Wasserstein natural gradients pull back the L²-Wasserstein metric to parameter space, behaving like Newton's method asymptotically.

The **Information Geometry** journal (Springer, launched 2018) and the biannual GSI conferences (Geometric Science of Information) provide dedicated venues. Alice Le Brigant's information geometry module in Geomstats and Frank Nielsen's computational methods work make the theory accessible. The March 2025 FDIG conference at the University of Tokyo signals ongoing community vitality.

---

## Topological data analysis spans neuroscience, materials, and biomedicine

TDA has grown from a mathematical curiosity into a field with distinct application communities, each developing specialized mathematical tools. The unifying thread is **persistent homology** — tracking topological features across scales — but the specific constructions vary dramatically by application.

In **neuroscience**, the Blue Brain Project collaboration between Kathryn Hess (EPFL) and Ran Levi (Aberdeen) demonstrated that directed cliques in neural networks form geometric structures up to 11 dimensions, revealing organization invisible to conventional analysis. Their 2019 work solved the neuron classification problem using persistent homology barcodes of dendritic trees. Carina Curto (Penn State) develops algebraic tools for neural codes, including the "neural ring" — her 2025 Annual Review of Neuroscience survey with Sanderson represents field maturation. Robert Ghrist's group (Penn) tracks neural manifold topology across populations using persistent homology (PNAS, 2024).

In **materials science**, Yasuaki Hiraoka's group (Kyoto/RIKEN) leads globally. Their 2016 PNAS paper revealed medium-range order in amorphous silica invisible to conventional methods. The HomCloud software and subsequent work on granular materials, polymers, and iron ore sinters established a pipeline from persistent homology through machine learning to materials properties. A 2025 Nature Communications paper showed persistent homology elucidates hierarchical structures responsible for mechanical properties in covalent amorphous solids.

In **computational biology**, Guowei Wei (Michigan State) introduced topological deep learning in 2017 and developed **persistent Laplacians** — recovering topological invariants through harmonic spectra while providing additional non-topological information through nonharmonic spectra. His group accurately predicted SARS-CoV-2 variant emergence approximately two months ahead using algebraic topology combined with deep learning.

The mathematical frontier is **multi-parameter persistent homology**, where no canonical complete descriptor analogous to the barcode exists. Oudot, Scoccola, and collaborators developed differentiability and optimization frameworks for multiparameter descriptors (ICML 2024). Signed barcodes via rank decompositions (Botnan, Oppermann, Oudot; Foundations of Computational Mathematics, 2024) and connections to quiver representation theory make this an area where pure algebra meets computation directly.

---

## Geometric mechanics and control theory anchor the robotics frontier

Lie group formulations of mechanics provide the mathematical backbone for modern geometric robotics. Frank Park's textbook *Modern Robotics* (2017) frames kinematics and dynamics entirely on SE(3) and SO(3). Francesco Bullo and Andrew Lewis's *Geometric Control of Mechanical Systems* (2005) formalized controllability analysis using affine connections. Sub-Riemannian geometry naturally describes nonholonomic motion planning — car-like robots, rolling contact, and snake locomotion all involve distributions (constrained velocity directions) rather than full tangent spaces.

The explosive recent growth area is **learning on Lie groups for robotics**. SE(3)-equivariant neural networks for imitation and reinforcement learning were surveyed in a 2025 tutorial (Seo et al., UC Berkeley). **Lie Neurons** (Lin et al., ICML 2024, Michigan) are adjoint-equivariant neural networks for semisimple Lie algebras. **LieFVIN** (Duruisseaux, Leok, Atanasov; L4DC 2023) combines Lie group variational integrators with deep learning, preserving both Lie group structure and symplecticity while learning controlled dynamics. The ICRA 2024 tutorial on geometry in robotics drew substantial attendance. This intersection of geometric mechanics with deep learning is where some of the most active hiring is occurring.

**Symplectic and geometric numerical integration** provides the computational substrate. Melvin Leok (UCSD) leads variational integrator development on Lie groups, with recent work on collision integrators for hybrid systems, spectral variational integrators of arbitrary order, and Type II Hamiltonian variational principles for adjoint sensitivity analysis. Klas Modin (Chalmers) connects geometric hydrodynamics, Lie-Poisson integrators, and information geometry of diffeomorphism groups. Ernst Hairer's definitive monograph *Geometric Numerical Integration* (2006) remains the standard reference.

**Contact geometry in thermodynamics** is a smaller but rapidly growing community. The thermodynamic phase space is naturally a contact manifold — the Gibbs relation dS − ΣTᵢdXᵢ is a contact form. Alessandro Bravetti (UNAM) has shown that thermodynamic entropy emerges as a Noether invariant of contact Hamiltonian time-translation symmetry. Yoshimura and Gay-Balmaz developed Dirac structure formulations for nonequilibrium thermodynamics. The **port-Hamiltonian systems** framework (van der Schaft, Maschke) uses Dirac structures for power-conserving interconnections, recently extended to continuum mechanics in a comprehensive 2025 paper in the Journal of Nonlinear Science. The Irreversible Port-Hamiltonian Systems framework (Ramírez, Le Gorrec) encodes the second law as a structural property. Entov and Polterovich's 2023 paper bringing modern contact topology tools to non-equilibrium thermodynamics signals that pure topologists are now engaging with this applied direction.

---

## Discrete exterior calculus and geometric flows provide computational foundations

Discrete exterior calculus (DEC) discretizes differential forms on simplicial complexes, preserving the de Rham complex, Stokes' theorem, and Hodge decomposition at the discrete level. The field is reaching a critical maturity milestone: the **IMSI workshop on DEC (September 2025, University of Chicago)** is the first dedicated workshop bringing together a community that has "often worked independently." A major 2025 breakthrough by Guzmán and Potu (Brown University) established comprehensive convergence theory for DEC approximations of Hodge-Laplacians, resolving a longstanding open problem.

Keenan Crane (CMU) is the dominant figure in discrete differential geometry for geometry processing, ranked top-10 worldwide in computer graphics for 2020–2025. Anil Hirani (UIUC), who founded DEC in his 2003 Caltech thesis, continues developing convergence analysis and organizing the community. Douglas Arnold's **Finite Element Exterior Calculus** (FEEC) provides the rigorous Hilbert space framework. A particularly exciting 2024 development by Braune et al. (ETH Zürich) extends DEC to **bundle-valued differential forms**, discretizing the exterior covariant derivative with convergence guarantees — the first rigorous treatment connecting DEC to gauge theory.

The **Decapodes.jl** software (Fairbanks, U. Florida) represents a novel approach: combining Applied Category Theory with DEC for composable multiphysics simulations. This categorical perspective on structure-preserving discretization is genuinely new mathematical theory motivated by computational needs.

**Geometric flows** in surface processing form an allied community. Xianfeng Gu (Stony Brook) pioneered discrete surface Ricci flow for conformal parameterization, with applications in brain mapping, virtual colonoscopy, and mesh generation. Feng Luo (Rutgers) proved the discrete uniformization theorem — every piecewise linear metric is discrete conformal to a unique constant curvature metric. Crane's conformal curvature flow (SIGGRAPH 2013) reformulates Willmore flow in curvature space, enabling large stable time steps. Bai and Li's 2024 breakthrough in Foundations of Computational Mathematics provides new convergence theory for parametric FEM approximations of mean curvature flow and Willmore flow through a "projected distance" framework.

A 2025 frontier is **neural geometric flows** — neural networks approximating mean curvature in phase field formulations, and neural ODEs for brain surface reconstruction (V2C-Flow, Bongratz et al., 2024). The convergence of classical geometric analysis with deep learning creates opportunities for mathematical analysis of learned geometric operators.

---

## How these twelve areas connect into a coherent landscape

The areas described above are not independent — they form a densely connected network sharing mathematical infrastructure, researchers, and computational tools.

The deepest structural connection runs through **Lie groups and Euler-Poincaré theory**. LDDMM's geodesic equations on diffeomorphism groups, Arnold's equations for ideal fluids, and Lie group formulations for robotics are all instances of the same Euler-Poincaré reduction framework developed by Marsden, Ratiu, and Holm. Modin's 2024 work on information geometry of diffeomorphism groups (with Khesin and Misiolek) directly connects computational anatomy to information geometry. Bravetti's 2023 paper connects contact geometry to Fisher information matrices and statistical manifolds.

**Software ecosystems serve as connective tissue.** Geomstats bridges shape analysis, Riemannian optimization, information geometry, and geometric deep learning with a unified API. Manopt provides manifold optimization used across machine learning and signal processing. The DEC/geometry-processing tools (PyDEC, libigl, geometry-central) underpin both geometric flows research and geometric deep learning on meshes. DiffusionNet and DeltaConv use DEC-discretized Laplacians as building blocks for learning on surfaces.

The **structure-preserving philosophy** unites DEC, FEEC, symplectic integrators, and port-Hamiltonian discretization: preserve the mathematical structure (complexes, symplectic forms, conservation laws, Hodge decompositions) rather than merely approximating equations pointwise. Brugnoli's 2022 dual-field port-Hamiltonian discretization via FEEC exemplifies this synthesis.

**Persistent homology connects to differential geometry** through multiple channels: persistent homology detects curvature (Bubenik et al., Inverse Problems, 2020), Wei's evolutionary de Rham-Hodge method connects differential forms to persistent homology for molecular analysis, and Morse theory provides the theoretical bridge between gradient flows on manifolds and the algebraic topology of sublevel sets.

---

## Choosing a PhD direction: where the mathematical opportunities are richest

For a prospective PhD student in applied mathematics, several considerations should guide the choice among these areas.

**Areas where foundational mathematical theory is still being built** offer the strongest opportunities for theoretical contributions. Multi-parameter persistent homology lacks a canonical complete descriptor — this is an open algebraic problem with immediate computational consequences. Shape spaces of surfaces require understanding vanishing geodesic distance phenomena, completeness of Sobolev metrics, and non-standard limit theorems. Bundle-valued DEC was only rigorously formulated in 2024. Contact geometry for thermodynamics is developing its basic vocabulary.

**Areas where deep learning creates new mathematical questions** offer the strongest career positioning. Geometric deep learning needs rigorous approximation theorems for equivariant architectures. Diffeomorphic registration needs theoretical guarantees for neural approaches. Riemannian diffusion models need analysis of convergence on manifolds with curvature. These are areas where mathematical rigor has not caught up with empirical practice.

**Areas with the most active hiring and funding** include geometric deep learning (driven by drug discovery and molecular modeling), Riemannian optimization (driven by ML with structured constraints), and computational anatomy (driven by precision medicine). Port-Hamiltonian systems are growing rapidly in control engineering. TDA in materials science connects to the substantial materials genome initiative funding.

The strongest position for a PhD student would be at an intersection of two or more of these areas — for instance, Riemannian optimization meets geometric deep learning (training equivariant networks on manifolds), or computational anatomy meets geometric deep learning (learning diffeomorphic registrations with geometric guarantees), or TDA meets information geometry (topological methods for understanding statistical manifold structure), or DEC meets port-Hamiltonian systems (structure-preserving discretization of thermomechanical systems). These intersections are where the most novel mathematics is emerging, where collaboration opportunities are richest, and where a student can develop a distinctive research identity.

The key venues that span multiple areas — GSI (Geometric Science of Information), the SGP graduate school, FoCM (Foundations of Computational Mathematics), and the ICML geometry workshops — are worth attending early in a PhD to map the community and identify collaborators. The Geomstats and Manopt communities actively welcome new contributors and provide natural entry points into the research network.
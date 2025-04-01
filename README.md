
# Refactor Checklist
- [x] move independent utility functions related to data saving/loading from the old utils.utils.py
- [ ] update everything to use the new data loading utilities
- [ ] extricate all code related to modeling the features into more modular classes to establish the "backend"
- [ ] move all old preprocessing code to new utility files
- [ ] export only the most relevant plotting functions that can be reused into a new plotting utils file (try to leave the ones that are overly specific)





# Architectural Design Plan

## Pattern Descriptions

1. Strategy Pattern for Swappable Algorithms
- **Feature Modeling:** might have multiple ways of extracting or transforming features (e.g., "raw embeddings," "KDE-based features," "manifold embeddings," etc.).
- **Metrics:** Different metrics (e.g., IoU, MMD, Mahalanobis) can be treated as separate strategies that compute a single numerical score from data.
- We can swap out or add new feature modeling methods or metrics without changing the rest of the pipeline. Any new approach just implements the same interface.

2. Template Method for "Evaluation Pipelines"
- For a skeleton of a multi-step "evaluation" where each step can be slightly changed or specialized by subclasses.
- Define a base class that lays out a fixed sequence of steps in an evaluate() method, but calls virtual hooks for each step
- Ensures a consistent overall "flow" for evaluation but allows developers to override certain steps. It's especially nice for a standard "evaluate" interface that runs the same pipeline in multiple contexts.

3. Factory (or Abstract Factory) for Creating "Test" or "Metric" Objects
- If we load config files or user input that says "use MMD with `sigma=2.0`" or "use KS test with `alpha=0.05`," a factory can produce the correct objects.
- Clean separation between the code that decides which test to run (like a config specifying "test": "ks") and the code that implements each test.
- Abstract Factory is more relevant if you need to create a family of objects consistently (like "clustering approach + distance measure + post-hoc aggregator" all in a bundle).
  - In this scenario, one "factory" method can produce multiple cooperating objects.


4. “Pipeline” (Pipe & Filter) for Data Flows
- e.g. to chain transformations:
  - Load dataset -> Preprocess embeddings -> Run manifold dimension reduction -> Compute metrics or hypothesis tests -> Write results
- Each step is a filter that transforms data from the previous step, or a stage that can be replaced or extended.


## Combined
A realistic approach could combine these patterns:
1. Strategy for Feature Extraction: Let users plug in a FeatureExtractionStrategy.
2. Template Method for Evaluate: Your Evaluator class has a run() or evaluate() method that outlines the pipeline: load → extract → do metrics → do hypothesis test → produce final logs.
3. Factory for Metrics & Tests**: If a config says {"metric": "MMD", "test": "KS", "sigma": 2.0}, the factory yields the correct classes.
4. Pipeline: Possibly to handle each step as a modular stage.





```bash
feature_eval_toolkit/
├── data/
│   ├── datasets.py           # FeatureDataset, LogitsDataset, etc.
│   ├── loaders.py            # create_feature_dataloader, create_logit_dataloader
│   └── io_utils.py           # Save/load .pt, .h5, .npy, JSON metadata, etc.
│   └── pipe_factory.py       # NEW: multiprocessing-driven wrappers and factory methods for piping model features/logits to the new Datasets/DataLoaders
│
├── evaluation/
│   ├── evaluator_base.py     # Template method pattern - primarily code pasted from an older rewrite suggested by ChatGPT
│   ├── metric_evaluators.py  # MMD/Entropy/Feature evaluation classes
│   ├── ood_evaluators.py     # OOD-specific evaluators for energy, entropy, KL
│   └── runner.py             # Optional orchestrator
│
├── models/
│   ├── modelers/
│   │   ├── kde.py            # ClusteredKDEModel, cluster sampling
│   │   ├── kernel_pca.py     # unfinished pytorch KPCA implementation for a dimensionality reduction approach I was considering
│   │   ├── spectral.py       # Diffusion Maps, Laplacian Eigenmaps
│   │   └── tsne.py           # TSNE feature modeling with complementary clustering methods
│   ├── clustering_utils.py   # basic clustering methods, Convex Hulls, Silhouette Scores
│   ├── modeler_base.py       # Abstract feature modeling base class, etc
│   └── modeler_factory.py    # feature modeling factory methods for interfacing with the hypothesis testing code
│
├── hypothesis/
│   └── hypothesis_base.py    # KSTest, WilcoxonTest, BootstrappedTest, HypothesisTestRunner, etc implemented in Strategy design pattern
│
├── metrics/
│   ├── distances.py          # MMD, Mahalanobis, Gaussian/JS/Wasserstein kernels
│   ├── metric_strategies.py  # Metric interface (for evaluation)
│   └── ood_scores.py         # metrics defined for out-of-domain detection and distribution divergence - uses logit-based scoring, median/CI thresholds
│
├── pipelines/
│   └── staging.py            # PipelineStage + composed pipeline logic
│
├── utils/
│   └── utils.py              # Generic I/O, formatting, filtering
```
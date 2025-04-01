# FeatureSpaceOdyssey - a neural network feature modeling and evaluation framework

A modular Python library for evaluating, analyzing, and testing feature representations extracted from deep neural networks. Designed with flexibility in mind, this toolkit supports evaluation workflows that rely on precomputed features or real-time streaming of probabilities/logits and features from external pipelines.

This library emphasizes:
- Decoupled design with clean abstractions
- Model-agnostic feature modeling
- Strategy-based hypothesis testing
- Pluggable evaluation logic via templates
- Compatibility with real-time and offline data sources


### Data Ingestion
The framework will support two modes for ingesting feature/score data:
1. **Offline Mode** - Load precomputed feature tensors or logits from .npy, .pt, .h5, etc.
2. **Streaming / Pipe Mode** - Use `pipe_factory.py` to build streaming datasets that receive logits/features from a model pipeline in real-time.


### Feature Modeling
All feature modeling methods must subclass ModelerBase. This allows models like:
- `ClusteredKDEModeler`: KDE + clustering-based modeling
- `DiffusionMapModeler`: Nonlinear manifold modeling
- `KernelPCAModeler`: Kernel PCA embedding

The `modeler_factory.py` will allow instantiating models with factory methods by string ID and config, making experimentation and CLI-based scripting easy.

##### Planned additions:
- GMMs, DPMMs
- Tangent-space models
- Spectral Graph Embeddings (Laplacian)
- Normalizing Flows
- Convex Hull models

Each modeler will be fit on features, score new samples, and optionally expose cluster assignments, latent embeddings, or summary statistics.

### Hypothesis Testing
All statistical comparison logic is designed using a strategy pattern:
- Base interface: `HypothesisTest`
- Tests currently include: `KSTest`, `WilcoxonTest`, and a higher-level `BootstrappedTest` that wraps other tests for boot-strapped testing
  - future extensions will likely be non-parametric hypothesis tests as well

A future `hypothesis_runner.py` will handle:
- Factory creation by name
- Parameter specification (e.g. alpha=0.05)
- Comparison of source vs. test feature scores


### Evaluator Templates
Evaluation logic uses a template pattern (`EvaluatorBase`) to define multi-step workflows:
```python
class KDEEvaluator(EvaluatorBase):
    def evaluate(self, source_data, test_data):
        model.fit(source_data)
        scores = model.score_samples(test_data)
        return test.run_test(source_scores, test_scores)
```
This ensures consistency across different modeling or test choices and facilitates flexible benchmarking.

##### Planned additions:
- OOD Evaluator (entropy vs. energy)
- Cluster purity evaluators
- Agreement vs. uncertainty
- Time-series and stability evaluators

### Examples & Usage
**Examples coming soon...**








# Architectural Design Plan

**NOTE: older planning notes - may be deprecated; should still describe the same high-level ideas**

1. Strategy Pattern for Swappable Algorithms
- **Feature Modeling:** might have multiple ways of extracting or transforming features (e.g., "raw embeddings," "KDE-based features," "manifold embeddings," etc.).
- **Metrics:** Different metrics (e.g., IoU, MMD, Mahalanobis) can be treated as separate strategies that compute a single numerical score from data.
- We can swap out or add new feature modeling methods or metrics without changing the rest of the pipeline. Any new approach just implements the same interface.

1. Template Method for "Evaluation Pipelines"
- For a skeleton of a multi-step "evaluation" where each step can be slightly changed or specialized by subclasses.
- Define a base class that lays out a fixed sequence of steps in an evaluate() method, but calls virtual hooks for each step
- Ensures a consistent overall "flow" for evaluation but allows developers to override certain steps. It's especially nice for a standard "evaluate" interface that runs the same pipeline in multiple contexts.

1. Factory (or Abstract Factory) for Creating "Test" or "Metric" Objects
- If we load config files or user input that says "use MMD with `sigma=2.0`" or "use KS test with `alpha=0.05`," a factory can produce the correct objects.
- Clean separation between the code that decides which test to run (like a config specifying "test": "ks") and the code that implements each test.
- Abstract Factory is more relevant if you need to create a family of objects consistently (like "clustering approach + distance measure + post-hoc aggregator" all in a bundle).
  - In this scenario, one "factory" method can produce multiple cooperating objects.

1. “Pipeline” (Pipe & Filter) for Data Flows
- e.g. to chain transformations:
  - Load dataset -> Preprocess embeddings -> Run manifold dimension reduction -> Compute metrics or hypothesis tests -> Write results
- Each step is a filter that transforms data from the previous step, or a stage that can be replaced or extended.






```bash
feature_eval_toolkit/
├── data/                 # Datasets and streaming-aware data loaders
│   ├── datasets.py       # FeatureDataset, LogitDataset, PipeDataset variants
│   └── loaders.py        # get_loader() utility for tensors, files, or callables
├── evaluation/
│   ├── evaluator_base.py     # Template method pattern - primarily code pasted from an older rewrite suggested by ChatGPT
│   ├── metric_evaluators.py  # MMD/Entropy/Feature evaluation classes
│   └── runner.py             # Optional orchestrator
├── models/
│   ├── modelers/
│   │   ├── kde.py            # ClusteredKDEModel, cluster sampling
│   │   ├── kernel_pca.py     # pytorch-based KPCA implementation for a kernel dimensionality reduction
│   │   ├── spectral.py       # Diffusion Maps, Laplacian Eigenmaps
│   │   └── tsne.py           # TSNE feature modeling with complementary clustering methods
│   ├── clustering_utils.py   # basic clustering methods, Convex Hulls, Silhouette Scores
│   ├── modeler_base.py       # Abstract feature modeling base class, etc
│   └── modeler_factory.py    # feature modeling factory methods for interfacing with the hypothesis testing code
├── hypothesis/
│   ├── hypothesis_base.py    # KSTest, WilcoxonTest, BootstrappedTest, HypothesisTestRunner, etc implemented in Strategy design pattern
│   └── hypothesis_runner.py  # (Planned) factory + test strategy executor
├── metrics/
│   ├── distances.py          # MMD, Mahalanobis, etc.
│   ├── ood_scores.py         # Energy, entropy, KL divergence scores - i.e. out-of-domain detection and distribution divergence
│   └── metric_strategies.py  # Swappable metric implementations
├── pipeline/
│   ├── staging.py            # PipelineStage base and chainable logic
│   └── pipe_factory.py       # Runtime pipelines for multiprocessing-driven streaming of features/logits
├── utils/
│   ├── io_utils.py           # Save/load .pt, .h5, .npy, JSON metadata, etc.
│   ├── utils.py              # General-purpose utilities
│   └── clustering_utils.py   # basic clustering methods, Convex Hulls, Silhouette Scores
└── README.md
```



# Refactor Checklist
- [x] move independent utility functions related to data saving/loading from the old `utils/utils.py`
- [ ] update everything to use the new data loading utilities
- [ ] extricate all code related to modeling the features into more modular classes to establish the "backend"
- [ ] move all old preprocessing code to new utility files
- [ ] export only the most relevant plotting functions that can be reused into a new plotting utils file (try to leave the ones that are overly specific)
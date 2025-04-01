

## 🔁 Mapping: Old File → New Modular Placement

| Old File | Status | New Location | Notes |
|----------|--------|---------------|-------|
| `evaluate.py` | 🚫 Should not be used directly | N/A | Monolithic control flow; replace with clean `evaluate_runner.py` or script in old project calling the modular new project. |
| `clustering.py` | 🔃 Partially relevant | `models/spectral.py`<br>`models/clustering_utils.py` | Keep `find_optimal_clusters`, `incremental_clustering`, `convex_hull_intersection`. Refactor all loading, saving, and plotting logic out. |
| `distance_measures.py` | 🔁 Merge needed | `metrics/distances.py` | Move functions like `gaussian_kernel`, `compute_mmd`, `compute_mahalanobis`. Delete file after de-duping. |
| `evaluator_base.py` | ✅ Base for refactoring | `evaluation/base_evaluator.py` | Keep `EvaluatorBase`, remove `ModelEvaluator`, port evaluation-specific hooks into separate `Evaluator` classes like `KDEEvaluator`. |
| `hypothesis_base.py` | ✅ Mostly fine | `hypothesis/hypothesis_base.py` | Interfaces are solid. Standardize names (`run_test`, etc.). Consider merging some convenience logic into `hypothesis_runner.py`. |
| `hypothesis_testing.py` | 🔃 Should be decomposed | `hypothesis/hypothesis_runner.py` | This file contains test-running logic tied to KDE; refactor into strategy-compatible test runners using `Modeler + HypothesisTest`. |
| `kde_modeling.py` | 🔃 Refactor as modeler | `models/kde.py` | `ClusteredKDEModel` should be wrapped in a class conforming to `BaseModeler`. Break out utility logic into `kde_utils.py` if needed. |
| `manifold_modeling.py` | 🔃 Partial refactor | `models/manifold.py` | DiffusionMap and KernelPCA classes should subclass `BaseModeler`. Move sklearn model-specific logic into separate utilities if necessary. |
| `ood_detection.py` | ⚠️ Some parts portable | `metrics/ood_scores.py` + `evaluation/ood_evaluators.py` | Keep scoring functions (`energy_score`, `entropy_score`, `kl_divergence`) that take logits. Remove any code dependent on PyTorch models. |

---

## 📦 Specific Refactor Notes by File

### 🧹 `evaluate.py` (OLD)

- **Discard** or move to old project (for glue logic only).
- Do **not** carry over any direct `torch.nn.Module` references.
- Many one-off scripts and pipeline setups should become:
  - `EvaluatorRunner`
  - Config-driven `evaluate_config.py`

---

### 🧠 `clustering.py`

- Port:
  - `find_optimal_clusters()` → `models/clustering_utils.py`
  - `incremental_clustering()` → same
  - `convex_hull_intersection()` → same or separate `geometry_utils.py`
- Discard:
  - File loading, saving
  - Plotting

---

### 🔍 `distance_measures.py`

- Confirm overlap with `metrics/distances.py`.
- Keep:
  - `gaussian_kernel`
  - `compute_mmd`
  - `compute_mahalanobis`
- Consider moving Mahalanobis to a `Metric` strategy.

---

### 📊 `evaluator_base.py`

- Keep `EvaluatorBase`, adapt to `Template Method` style (e.g., `load_data_hook`, `extract_features_hook`, `compute_results_hook`).
- Move each old evaluator into its own subclass:
  - `EnergyEvaluator`, `KDEEvaluator`, `EntropyEvaluator`, etc.
- Remove legacy `ModelEvaluator`.

---

### 🧪 `hypothesis_testing.py`

- Refactor:
  - `run_kde_mmd_test()` → `HypothesisTestRunner.run(model_A, model_B)`
  - Move cluster comparison logic into `cluster_test_runner.py` or unify under `HypothesisTestRunner`.

---

### 📈 `kde_modeling.py`

- Wrap `ClusteredKDEModel` in a `BaseModeler`-compliant class.
- Remove plotting, I/O.
- Move `score_features_from_kde()` or similar methods into evaluation logic, not modeling.

---

### 🌀 `manifold_modeling.py`

- Keep `DiffusionMap`, `KernelPCA`.
- Wrap each with a `BaseModeler` interface.
- Move sklearn-specific setup (e.g., `fit_transform`) into `fit()` and `transform()` methods.

---

### 🔍 `ood_detection.py`

- Keep:
  - `compute_energy_score()`
  - `compute_entropy_score()`
  - `compute_kl_divergence()`
- Remove:
  - Any direct `.cuda()` or PyTorch model inference
- Place scoring in `metrics/ood_scores.py`.
- OOD evaluation logic (e.g., running tests over multiple checkpoints) should live in `evaluation/ood_evaluators.py`.

---

## ✅ Conclusion

You're in a great position to finish cleaning and modularizing. You can now:

### 🟢 Safely Copy and Refactor:
- `ClusteredKDEModel`, `DiffusionMap`, `KernelPCA`
- Distance metrics
- Hypothesis test base and strategy classes
- Evaluator templates and hooks

### 🔴 Fully Remove or Rewrite:
- `evaluate.py` (old)
- Model-forward logic
- PyTorch-dependent OOD evaluation

---

Consider
1. A refactored `evaluate_runner.py` (that replaces the old monolithic `evaluate.py`)
2. One `Evaluator` subclass (e.g., `KDEEvaluator`) and corresponding test runner hookup
3. A cleanup PR-style list of files to delete or mark as legacy




import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy.optimize import minimize
from typing import Callable, List, Dict, Any, Optional, Tuple
import cvxpy as cp

class MACIAdaptiveConformal:
    def __init__(
        self,
        score_function: Callable,
        random_state: Optional[int] = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.score_function = score_function
        self.random_state = random_state
        self.eps = float(eps)
        self.tau_hat: Optional[float] = None
        self._rng = np.random.default_rng(self.random_state)

    def _process_raw_scores(self, raw_scores: List, data: List[Dict]) -> List[np.ndarray]:
        if raw_scores and isinstance(raw_scores[0], np.ndarray):
            return [np.asarray(s, dtype=float) for s in raw_scores]
        per_sample_scores: List[np.ndarray] = []
        samples = [d.get('sample', d) for d in data]
        for i, s_i in enumerate(raw_scores):
            n_claims = len(samples[i].get("atomic_facts", []))
            s_arr = np.asarray(list(s_i), dtype=float)[:n_claims]
            per_sample_scores.append(np.nan_to_num(s_arr, nan=0.0))
        return per_sample_scores

    def _compute_nonconformity_score(self, sample: dict, scores_i: np.ndarray) -> float:
        atomic_facts = sample.get("atomic_facts", [])
        if not atomic_facts or scores_i.size == 0: return 0.0
        labels = np.asarray([af.get("is_supported", False) for af in atomic_facts], dtype=bool)
        s_raw = np.asarray(scores_i, dtype=float)
        s_raw = np.nan_to_num(s_raw, nan=0.0, posinf=1.0, neginf=0.0)
        s = np.clip(s_raw, 0.0, 1.0 - self.eps)
        idx = np.argsort(s, kind='mergesort')
        s_sorted_asc, labels_asc = s[idx], labels[idx]
        false_positions = np.where(~labels_asc)[0]
        if not false_positions.size: return 0.0
        k_star = int(false_positions.max())
        costs = -np.log(1.0 - s_sorted_asc)
        return float(np.sum(costs[:k_star + 1]))

    def fit_on_calib(self, calib_data: List[dict], alpha: float = 0.1) -> "MACIAdaptiveConformal":
        raw_scores = self.score_function(calib_data)
        per_sample_scores = self._process_raw_scores(raw_scores, calib_data)
        calib_samples = [entry.get('sample', entry) for entry in calib_data]
        s_values = [self._compute_nonconformity_score(s, sc) for s, sc in zip(calib_samples, per_sample_scores)]
        
        logging.info(f"    - Calibration set size: {len(calib_data)} samples")
        if not s_values:
            raise ValueError("Cannot compute scores from calibration data.")
        
        logging.info(f"    - Nonconformity stats: min={min(s_values):.4f}, max={max(s_values):.4f}, mean={np.mean(s_values):.4f}")
        
        n = len(s_values)
        quantile_index = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
        quantile_index = min(quantile_index, n - 1)
        
        sorted_s_values = np.sort(s_values)
        self.tau_hat = sorted_s_values[quantile_index]
            
        logging.info(f"    - Assigned tau_hat: {self.tau_hat:.4f}")
        return self

    def predict(self, data: List[dict]) -> Tuple[List[dict], List[float]]:
        if self.tau_hat is None: raise ValueError("Model is not calibrated.")
        raw_scores = self.score_function(data)
        per_sample_scores = self._process_raw_scores(raw_scores, data)
        samples = [d.get('sample', d) for d in data]

        filtered_data, retention_rates = [], []
        for sample, s_raw in zip(samples, per_sample_scores):
            atomic_facts = sample.get("atomic_facts", [])
            new_sample = dict(sample)
            if not atomic_facts or s_raw.size == 0:
                new_sample["filtered_claims"] = []
                retention_rates.append(1.0 if not atomic_facts else 0.0)
            else:
                s_tmp = np.asarray(s_raw, dtype=float)
                s_tmp = np.nan_to_num(s_tmp, nan=0.0, posinf=1.0, neginf=0.0)
                s = np.clip(s_tmp, 0.0, 1.0 - self.eps)
                indexed_items = sorted(list(zip(s, atomic_facts)), key=lambda x: x[0])
                s_sorted_asc = np.array([item[0] for item in indexed_items])
                costs = -np.log(1.0 - s_sorted_asc)
                cumulative_costs = np.concatenate(([0.0], np.cumsum(costs)))
                possible_K_indices = np.where(cumulative_costs <= self.tau_hat)[0]
                K = int(possible_K_indices.max()) if possible_K_indices.size > 0 else 0
                # Boundary randomization: with probability proportional to leftover budget,
                # include one more boundary item (i.e., increase K by 1) if feasible.
                # This randomization reduces discretization bias at the threshold.
                if K < len(costs):
                    leftover = float(self.tau_hat - cumulative_costs[K])
                    next_cost = float(costs[K])  # cost of the (K)-th item in sorted order
                    if np.isfinite(next_cost) and next_cost > 0.0 and leftover > 0.0:
                        p = float(np.clip(leftover / next_cost, 0.0, 1.0))
                        if self._rng.uniform(0.0, 1.0) < p:
                            K = K + 1
                new_sample["filtered_claims"] = [item[1] for item in indexed_items[K:]]
                retention_rates.append(len(new_sample["filtered_claims"]) / len(atomic_facts))
            filtered_data.append(new_sample)
        return filtered_data, retention_rates

class SubgroupOptimizedMACI:
    def __init__(self, model_names: List[str], grouper: Any, n_bins: int = 3, **kwargs):
        self.model_names, self.grouper, self.n_bins, self.kwargs = model_names, grouper, n_bins, kwargs
        self.weights, self.conformal_models = {}, {}
        self.fallback_weights, self.bin_edges = None, None
        self.bin_labels = ['low', 'medium', 'high'] if n_bins == 3 else [f'group_{i}' for i in range(n_bins)]
        # Timing accumulators
        self._timing: Dict[str, float] = {
            'weight_optimization_s': 0.0,
            'calibration_s': 0.0
        }

    def _get_subgroup_label(self, value: float) -> str:
        if self.bin_edges is None or not np.isfinite(value):
            return self.bin_labels[0]
        bin_index = np.digitize(value, self.bin_edges)
        return self.bin_labels[min(bin_index, len(self.bin_labels) - 1)]

    def _group_data_by_bins(self, data: List[Dict], bin_edges: np.ndarray) -> Dict[str, List[Dict]]:
        grouped_data = defaultdict(list)
        values = self.grouper.compute_values([d['sample'] for d in data])
        for item, value in zip(data, values):
            label = self._get_subgroup_label(value)
            grouped_data[label].append(item)
        return grouped_data
    def _learn_robust_weights_by_retention(self, training_data: List[Dict], target_tpr: float = 0.95) -> np.ndarray:
        """
        Stable convex program for learning ensemble weights on the probability simplex.

        Uses an epigraph reformulation with explicit nonnegative slack variables and
        Tikhonov regularization to improve numerical stability across solvers.
        """
        all_scores, all_labels = [], []
        for entry in training_data:
            sample, scores_dict = entry.get('sample', {}), entry.get('scores', {})
            labels = [af.get("is_supported", False) for af in sample.get("atomic_facts", [])]
            scores_per_model = [scores_dict.get(m, []) for m in self.model_names]
            min_len = min(len(labels), *[len(s) for s in scores_per_model])
            if min_len == 0:
                continue
            for i in range(min_len):
                all_labels.append(labels[i])
                all_scores.append([s[i] for s in scores_per_model])

        if len(all_labels) < 2 or len(np.unique(all_labels)) < 2:
            logging.warning("Skipping weight optimization: insufficient or single-class labels.")
            return np.ones(len(self.model_names)) / len(self.model_names)

        scores_matrix = np.nan_to_num(np.array(all_scores, dtype=float))
        labels_array = np.array(all_labels, dtype=int)
        n_models = scores_matrix.shape[1]

        pos = scores_matrix[labels_array == 1]
        neg = scores_matrix[labels_array == 0]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            logging.warning("Skipping weight optimization: missing positive or negative samples.")
            return np.ones(len(self.model_names)) / len(self.model_names)

        neg_proxy = np.mean(neg, axis=1)
        neg_w = np.clip(neg_proxy, 0.0, 1.0) ** 2
        neg_w = neg_w / (np.mean(neg_w) + 1e-12)

        pos_w = np.ones(pos.shape[0], dtype=float)
        sum_pos = np.sum(pos_w)
        sum_neg = np.sum(neg_w)
        if sum_pos > 0 and sum_neg > 0:
            scale = sum_pos / sum_neg
            neg_w = neg_w * scale

        alpha = 1.0
        beta = 5.0 * (target_tpr / max(1.0 - target_tpr, 1e-6))

        def solve_with(ridge: float, eps_w: float, solver_name: str) -> Optional[np.ndarray]:
            try:
                w = cp.Variable(n_models)
                t = cp.Variable()
                slack_neg = cp.Variable(neg.shape[0], nonneg=True)
                slack_pos = cp.Variable(pos.shape[0], nonneg=True)

                constraints = [
                    neg @ w - t <= slack_neg,
                    t - pos @ w <= slack_pos,
                    w >= eps_w,
                    cp.sum(w) == 1,
                    t >= 0,
                    t <= 1
                ]
                objective = (
                    alpha * cp.sum(cp.multiply(neg_w, slack_neg)) +
                    beta * cp.sum(cp.multiply(pos_w, slack_pos)) +
                    ridge * cp.sum_squares(w)
                )
                prob = cp.Problem(cp.Minimize(objective), constraints)

                if solver_name == 'osqp':
                    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polishing=True, linsys_solver='qdldl')
                elif solver_name == 'ecos':
                    prob.solve(solver=cp.ECOS, verbose=False, max_iters=200000, abstol=1e-7, reltol=1e-7, feastol=1e-7)
                elif solver_name == 'scs':
                    prob.solve(solver=cp.SCS, verbose=False, max_iters=300000, eps=2e-5, acceleration_lookback=20)
                else:
                    return None

                if w.value is None:
                    return None

                w_val = np.array(w.value, dtype=float).reshape(-1)
                if not np.all(np.isfinite(w_val)):
                    return None
                w_val = np.clip(w_val, 0.0, None)
                s = np.sum(w_val)
                if s <= 1e-12:
                    return None
                w_val = w_val / s
                logging.info("    - Weight optimization completed")
                return w_val
            except Exception as e:
                logging.debug(f"{solver_name.upper()} attempt failed (ridge={ridge}, eps_w={eps_w}): {e}")
                return None

        solver_order = []
        solver_pref = (self.kwargs or {}).get('solver', 'auto')
        if solver_pref in ('osqp', 'ecos', 'scs'):
            solver_order = [solver_pref] + [s for s in ('osqp', 'ecos', 'scs') if s != solver_pref]
        else:
            solver_order = ['osqp', 'ecos', 'scs']

        for ridge in (5e-3, 5e-2, 1e-1, 5e-1):
            for eps_w in (0.0, 1e-6, 1e-4):
                for slv in solver_order:
                    sol = solve_with(ridge=ridge, eps_w=eps_w, solver_name=slv)
                    if sol is not None:
                        return sol

        logging.warning("CVXPY solvers failed repeatedly; falling back to AUC-based SLSQP optimizer as last resort.")
        return self._learn_robust_weights(training_data)

    def _learn_robust_weights(self, training_data: List[Dict]) -> np.ndarray:
        all_scores, all_labels = [], []
        for entry in training_data:
            sample, scores_dict = entry.get('sample', {}), entry.get('scores', {})
            labels = [af.get("is_supported", False) for af in sample.get("atomic_facts", [])]
            if not all(m in scores_dict for m in self.model_names): continue
            scores_per_model = [scores_dict.get(m, []) for m in self.model_names]
            min_len = min(len(labels), *[len(s) for s in scores_per_model])
            if min_len == 0: continue
            for i in range(min_len):
                all_labels.append(labels[i])
                all_scores.append([s[i] for s in scores_per_model])
        
        if len(all_labels) < 2 or len(np.unique(all_labels)) < 2:
            return np.ones(len(self.model_names)) / len(self.model_names)

        scores_matrix = np.nan_to_num(np.array(all_scores, dtype=float))
        labels_array = np.array(all_labels, dtype=int)
        n_models = scores_matrix.shape[1]

        def objective_fn(weights: np.ndarray) -> float:
            w = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            ensemble_scores = scores_matrix @ w
            try: return -roc_auc_score(labels_array, ensemble_scores)
            except ValueError: return 0.0

        best_score, best_weights = -1.0, np.ones(n_models) / n_models
        for _ in range(10):
            w0 = np.random.dirichlet(np.ones(n_models))
            res = minimize(objective_fn, w0, method='SLSQP', bounds=[(0, 1)] * n_models, constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}))
            if res.success and -res.fun > best_score:
                best_score, best_weights = -res.fun, res.x / np.sum(res.x)
        return best_weights

    def get_budgets(self):
        return {subgroup: model.tau_hat for subgroup, model in self.conformal_models.items()}

    def get_weights(self):
        return {
            'subgroup_weights': self.weights,
            'fallback_weights': self.fallback_weights,
            'bin_edges': None if self.bin_edges is None else np.asarray(self.bin_edges).tolist(),
            'bin_labels': list(self.bin_labels) if self.bin_labels is not None else None,
        }

    def _compute_ensemble_scores(self, data: List[Dict], subgroup_label: str) -> List[np.ndarray]:
        subgroup_weights = self.weights.get(subgroup_label, self.fallback_weights)
        if subgroup_weights is None:
            raise RuntimeError(f"Weights not learned for subgroup '{subgroup_label}'.")
        
        final_scores = []
        for entry in data:
            scores_dict = entry.get('scores', {})
            scores_per_model = [scores_dict.get(m, []) for m in self.model_names]
            min_len = min(len(entry['sample']['atomic_facts']), *[len(s) for s in scores_per_model])
            if min_len == 0:
                final_scores.append(np.array([]))
            else:
                scores_matrix = np.array([np.nan_to_num(s[:min_len]) for s in scores_per_model]).T
                final_scores.append(scores_matrix @ subgroup_weights)
        return final_scores

    def fit(self, data: List[dict], alpha: float = 0.1, ensemble_train_ratio: float = 0.5, target_tpr: float = 0.95):
        """Learn subgroup-specific ensemble weights and conformal thresholds."""
        random_state = self.kwargs.get("random_state")
        grouper_name = self.grouper.__class__.__name__
        logging.info(f"SubgroupOptimizedMACI training started (grouper: '{grouper_name}')")
        
        ensemble_train_data, calib_data = train_test_split(
            data, 
            test_size=1.0 - ensemble_train_ratio, 
            random_state=random_state
        )
        logging.info(f"  - Data split: ensemble training {len(ensemble_train_data)} / conformal calibration {len(calib_data)}")

        logging.info(f"  - Learning bin edges by '{grouper_name}' values...")
        train_values = self.grouper.compute_values([d['sample'] for d in ensemble_train_data])
        finite_train_values = train_values[np.isfinite(train_values)]
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bin_edges = np.quantile(finite_train_values, quantiles) if len(finite_train_values) > 0 else np.array([])
        logging.info(f"  - Learned bin edges: {self.bin_edges}")

        grouped_ensemble_data = self._group_data_by_bins(ensemble_train_data, self.bin_edges)
        grouped_calib_data = self._group_data_by_bins(calib_data, self.bin_edges)

        for label in self.bin_labels:
            logging.info(f"--- Processing group '{label}' ---")
            sub_ensemble_data = grouped_ensemble_data.get(label, [])
            sub_calib_data = grouped_calib_data.get(label, [])
            
            if not sub_ensemble_data or not sub_calib_data:
                logging.warning(f"Skipping group '{label}' due to insufficient data.")
                continue
                
            logging.info(f"  - Learning ensemble weights (n={len(sub_ensemble_data)})...")
            _t0 = __import__('time').perf_counter()
            self.weights[label] = self._learn_robust_weights_by_retention(sub_ensemble_data, target_tpr=target_tpr)
            self._timing['weight_optimization_s'] += __import__('time').perf_counter() - _t0
            
            logging.info(f"  - Calibrating threshold (n={len(sub_calib_data)})...")
            score_func = lambda data, l=label: self._compute_ensemble_scores(data, l)
            
            conformal_model = MACIAdaptiveConformal(score_function=score_func, **self.kwargs)
            _t1 = __import__('time').perf_counter()
            conformal_model.fit_on_calib(sub_calib_data, alpha)
            self._timing['calibration_s'] += __import__('time').perf_counter() - _t1
            self.conformal_models[label] = conformal_model

        logging.info("--- Training fallback model on all data ---")
        self.fallback_weights = self._learn_robust_weights_by_retention(ensemble_train_data, target_tpr=target_tpr)
        
        logging.info("✅ Training complete.")
        return self

    def get_timing(self) -> Dict[str, float]:
        return dict(self._timing)

    def predict(self, data: List[dict]) -> Tuple[List[dict], List[float]]:
        if not self.conformal_models: raise ValueError("모델이 학습되지 않았습니다.")
        
        grouped_data_with_indices = defaultdict(list)
        values = self.grouper.compute_values([d['sample'] for d in data])
        for i, (item, value) in enumerate(zip(data, values)):
            label = self._get_subgroup_label(value)
            grouped_data_with_indices[label].append((i, item))
        
        results_placeholder = [None] * len(data)
        rates_placeholder = [None] * len(data)
        
        for label, indexed_subgroup_data in grouped_data_with_indices.items():
            if not indexed_subgroup_data: continue
            original_indices = [item[0] for item in indexed_subgroup_data]
            subgroup_data = [item[1] for item in indexed_subgroup_data]
            model = self.conformal_models.get(label)
            
            if model:
                logging.info(f"  - Predicting for group '{label}' (n={len(subgroup_data)})...")
                predicted_samples, rates = model.predict(subgroup_data)

                for i, original_item, predicted_sample, rate in zip(original_indices, subgroup_data, predicted_samples, rates):
                    new_result_item = original_item.copy()
                    new_result_item['sample'] = predicted_sample
                    results_placeholder[i] = new_result_item
                    rates_placeholder[i] = rate
            else:
                logging.warning(f"No trained model for group '{label}'. Using fallback weights for prediction.")
                fallback_score_func = lambda data_list: self._compute_ensemble_scores(data_list, label)
                fallback_model = MACIAdaptiveConformal(score_function=fallback_score_func, **self.kwargs)
                fallback_model.tau_hat = 0.0 
                predicted_samples, rates = fallback_model.predict(subgroup_data)
                for i, original_item, predicted_sample, rate in zip(original_indices, subgroup_data, predicted_samples, rates):
                    new_result_item = original_item.copy()
                    new_result_item['sample'] = predicted_sample
                    results_placeholder[i] = new_result_item
                    rates_placeholder[i] = rate

        return results_placeholder, rates_placeholder
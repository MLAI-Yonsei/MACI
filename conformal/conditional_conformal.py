import numpy as np
from typing import List, Tuple, Optional, Callable
import torch
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from functools import lru_cache
import sys
import os

# Add conditional-conformal path to Python path (local vendor copy) using repo-relative path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
vendor_path = os.path.join(repo_root, 'conditional-conformal', 'src')
if vendor_path not in sys.path:
    sys.path.append(vendor_path)
from conditionalconformal import CondConf

# ==============================================================================
# === Step 1: Classes and Helper Functions for Boosting ===
# ==============================================================================

def as_tensor(x, dtype, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)

def get_current_basis(primals, duals, Phi, S, quantile):
    """Helper function to find a stable basis from LP solution"""
    interp_bools = np.logical_and(~np.isclose(duals, quantile - 1), ~np.isclose(duals, quantile))
    if np.sum(interp_bools) == Phi.shape[1]:
        return interp_bools
    preds = (Phi @ primals).flatten()
    active_indices = np.where(interp_bools)[0]
    interp_indices = np.argsort(np.abs(S - preds))[:Phi.shape[1]]
    diff_indices = np.setdiff1d(interp_indices, active_indices)
    num_missing = Phi.shape[1] - np.sum(interp_bools)
    
    if num_missing < len(diff_indices):
        from itertools import combinations
        for cand_indices in combinations(diff_indices, num_missing):
            cand_phi = Phi[np.concatenate((active_indices, cand_indices))]
            if np.isfinite(np.linalg.cond(cand_phi)):
                interp_bools[np.asarray(cand_indices)] = True
                break
    else:
        interp_bools[diff_indices] = True
    return interp_bools

def _choose_full_rank_rows(Phi: np.ndarray) -> np.ndarray:
    """Greedy row selection for full-rank basis"""
    d = Phi.shape[1]
    chosen = []
    cur = np.empty((0, d))
    for i in range(Phi.shape[0]):
        cand = np.vstack([cur, Phi[i:i+1]])
        if np.linalg.matrix_rank(cand) > np.linalg.matrix_rank(cur):
            chosen.append(i)
            cur = cand
        if len(chosen) == d:
            break
    if len(chosen) < d:
        chosen = list(range(Phi.shape[0]-d, Phi.shape[0]))
    return np.asarray(chosen, dtype=int)

def solve_qr_for_boosting(Phi: np.ndarray, s: torch.Tensor, q: float, dtype: torch.dtype) -> torch.Tensor:
    """Differentiable tau calculation function for boosting - robust fallback included"""
    S_np = s.detach().cpu().numpy().reshape(-1)
    assert Phi.shape[0] == S_np.shape[0], "Phi rows must match len(s)"
    assert 0.0 < q < 1.0, "q must be in (0,1)"

    b_eq = np.zeros(Phi.shape[1])
    bounds = [(q - 1.0, q)] * len(S_np)

    res = None
    try:
        res = linprog(-S_np, A_eq=Phi.T, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        res = None

    tau_initial = None
    duals = None
    if res is not None and getattr(res, "success", False):
        marg = None
        if hasattr(res, "eqlin") and res.eqlin is not None and hasattr(res.eqlin, "marginals") and res.eqlin.marginals is not None:
            marg = res.eqlin.marginals
        elif hasattr(res, "dual_eq") and res.dual_eq is not None:
            marg = res.dual_eq

        if marg is not None:
            tau_initial = -np.asarray(marg, dtype=float)
        if hasattr(res, "x") and res.x is not None:
            duals = np.asarray(res.x, dtype=float)

    try:
        if tau_initial is not None and duals is not None:
            basis_mask = get_current_basis(tau_initial, duals, Phi, S_np, q)
            basis_idx = np.where(basis_mask)[0]
            if basis_idx.size != Phi.shape[1]:
                basis_idx = _choose_full_rank_rows(Phi)
        else:
            basis_idx = _choose_full_rank_rows(Phi)

        Phi_basis = Phi[basis_idx]
        s_basis = s[basis_idx]

        tau_sol = torch.linalg.lstsq(as_tensor(Phi_basis, dtype), s_basis).solution
        tau = tau_sol
    except Exception:
        tau = torch.zeros((Phi.shape[1],), dtype=dtype)

    return tau.reshape(-1, 1)

def torch_score_func_sample_level(features: List[np.ndarray], annotations: List[np.ndarray], beta: torch.Tensor) -> torch.Tensor:
    """sample-level score (max_false_score) calculation"""
    scores = as_tensor(np.zeros((len(features),)), dtype=beta.dtype)
    for i, (f, a) in enumerate(zip(features, annotations)):
        cs = -as_tensor(f, dtype=beta.dtype) @ beta
        at = as_tensor(a, dtype=torch.bool)
        scores[i] = torch.sort(cs[~at], descending=True)[0][0] if torch.sum(~at) > 0 else torch.tensor(1e9, dtype=beta.dtype)
    return scores

def cond_score_loss(beta: torch.Tensor, dataset: Tuple, z_processed: np.ndarray, random_seed: int, q: float) -> torch.Tensor:
    """Claim-level loss function for boosting"""
    indices = np.arange(len(dataset[0]))
    ind_train, ind_calib = train_test_split(indices, test_size=0.5, random_state=random_seed)
    
    x_train, y_train = [dataset[0][i] for i in ind_train], [dataset[1][i] for i in ind_train]
    x_calib, y_calib = [dataset[0][i] for i in ind_calib], [dataset[1][i] for i in ind_calib]
    z_train, z_calib = z_processed[ind_train], z_processed[ind_calib]

    scores_train_sample = torch_score_func_sample_level(x_train, y_train, beta)
    tau = solve_qr_for_boosting(z_train, scores_train_sample, q, beta.dtype)
    
    cutoffs = (as_tensor(z_calib, dtype=beta.dtype) @ tau).flatten()
    
    total_loss = torch.tensor(0.0, dtype=beta.dtype, requires_grad=True)
    count = 0
    for i, (f_c, a_c) in enumerate(zip(x_calib, y_calib)):
        claim_scores = -(as_tensor(f_c, dtype=beta.dtype) @ beta)
        perc = torch.sigmoid(cutoffs[i] - claim_scores)
        total_loss = total_loss + torch.mean(perc)
        count += 1
        
    total_loss = total_loss / count if count > 0 else total_loss
    return -total_loss

class ConditionalConformalBoosting:
    def __init__(self, random_state: int = 0):
        self.rng = np.random.default_rng(random_state)
        self.beta: Optional[np.ndarray] = None
        self.z_projector: Optional[np.ndarray] = None

    def _extract_features_for_boosting(self, data: List[dict]) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
        basic_features = [d['features_4d'] for d in data]
        annotations = [d['annotations'] for d in data]
        conditional_features = []
        for d in data:
            sample = d.get('sample', {})
            scores_dict = d.get('scores', {})
            base_features = d.get('prompt_features', [])
            logprob_scores = scores_dict.get('logprob', np.array([]))
            logprob_mean = np.mean(logprob_scores) if logprob_scores.size > 0 else 0.0
            logprob_std = np.std(logprob_scores) if logprob_scores.size > 1 else 0.0
            claim_count = len(sample.get('atomic_facts', []))
            combined_features = np.concatenate([base_features, [logprob_mean, logprob_std, claim_count]])
            conditional_features.append(combined_features)
        z = np.array(conditional_features, dtype=float)
        if not np.isfinite(z).all():
            z = np.nan_to_num(z, nan=np.nanmean(z, axis=0))
        
        return basic_features, z, annotations
    
    def _preprocess_z(self, z: np.ndarray) -> np.ndarray:
        intercept = np.ones((z.shape[0], 1))
        z_aug = np.hstack([z, intercept])
        try:
            _, s, Vt = np.linalg.svd(z_aug, full_matrices=False)
            rank = np.sum(s > 1e-10)
            self.z_projector = Vt.T[:, :rank]
        except np.linalg.LinAlgError:
            self.z_projector = np.eye(z_aug.shape[1])
        return z_aug @ self.z_projector

    def fit(self, data: List[dict], alpha: float = 0.1, boosting_epochs: int = 1000, boosting_lr: float = 0.005) -> np.ndarray:
        basic_features, z, annotations = self._extract_features_for_boosting(data)
        dataset_boost = (basic_features, annotations)
        z_processed = self._preprocess_z(z)
        
        
        feature_dim = basic_features[0].shape[1]
        beta_tensor = torch.tensor([0.25] * feature_dim, dtype=torch.float, requires_grad=True)
        optimizer = torch.optim.Adam([beta_tensor], lr=boosting_lr)
        
        for epoch in range(boosting_epochs):
            optimizer.zero_grad()
            seed_epoch = self.rng.integers(1e7)
            loss = cond_score_loss(beta_tensor, dataset_boost, z_processed, seed_epoch, q=1 - alpha)
            if torch.isnan(loss) or torch.isinf(loss): break
            loss.backward()
            if beta_tensor.grad is not None and torch.isfinite(beta_tensor.grad).all():
                optimizer.step()

        self.beta = beta_tensor.detach().cpu().numpy()
        #
        return self.beta

# ==============================================================================
# === Step 2: Classes and Helper Functions for Calibration and Prediction ===
# ==============================================================================


class ConditionalConformalInference:
    def __init__(self, random_state: int = 0):
        self.rng = np.random.default_rng(random_state)
        self.alpha: Optional[float] = None
        self.beta: Optional[np.ndarray] = None
        self.model: Optional[CondConf] = None
        # Adaptive alpha components
        self.adaptive_enabled: bool = False
        self.retention_target: Optional[float] = None
        self.quantile_theta: Optional[np.ndarray] = None  # parameters for linear quantile_fn
        self._z_proj_for_quantile: Optional[np.ndarray] = None  # projector used for z in quantile fit
        
    def _make_z_only(self, data: List[dict]) -> np.ndarray:
        """z generation - same structure as boosting: [prompt_features..., logprob_mean, logprob_std, claim_count]"""
        max_base_len = 0
        for d in data:
            base = d.get('prompt_features', np.array([]))
            try:
                base_len = int(np.asarray(base).size)
            except Exception:
                base_len = 0
            if base_len > max_base_len:
                max_base_len = base_len

        cond_feats: List[np.ndarray] = []
        for d in data:
            sample = d.get('sample', {})
            scores_dict = d.get('scores', {})

            base = np.asarray(d.get('prompt_features', np.array([])), dtype=float).ravel()
            if base.size < max_base_len:
                pad = np.zeros(max_base_len - base.size, dtype=float)
                base = np.concatenate([base, pad])
            elif base.size > max_base_len and max_base_len > 0:
                base = base[:max_base_len]

            logprob_scores = np.asarray(scores_dict.get('logprob', np.array([])), dtype=float).ravel()
            logprob_mean = float(np.mean(logprob_scores)) if logprob_scores.size > 0 else 0.0
            logprob_std = float(np.std(logprob_scores)) if logprob_scores.size > 1 else 0.0

            claim_count = float(len(sample.get('atomic_facts', [])))

            combined = np.concatenate([base, np.array([logprob_mean, logprob_std, claim_count], dtype=float)])
            cond_feats.append(combined)

        result = np.asarray(cond_feats, dtype=float)
        return result

    def _make_yz_for_calib(self, data: List[dict], beta: np.ndarray, eps: float = 0.0):
        z = self._make_z_only(data)
        y_list = []
        for d in data:
            feats = d['features_4d']
            ann = np.asarray(d['annotations'], dtype=bool)
            s = -(feats @ beta)
            false_s = s[~ann]
            if false_s.size > 0:
                y_list.append(np.min(false_s) - eps)
            else:
                y_list.append((np.max(s) if s.size > 0 else 0.0))
        y = np.asarray(y_list, dtype=float)
        mask = np.isfinite(y)
        return y[mask], z[mask], mask

    def fit(self, calib_data: List[dict], alpha: float, beta: np.ndarray,
            adaptive_alpha: bool = False, retention_target: float = 0.7):
        """Set up and calibrate CondConf model"""

        self.alpha = alpha
        self.beta = beta
        self.adaptive_enabled = bool(adaptive_alpha)
        self.retention_target = float(retention_target) if adaptive_alpha else None
        if not self.adaptive_enabled:
            self.quantile_theta = None
        
        
        y_calib, z_calib, mask = self._make_yz_for_calib(calib_data, beta)
        self._last_calib_mask = mask
        
        self.model = CondConf(score_fn=lambda x, y: y, Phi_fn=lambda x: x, seed=self.rng.integers(1e6))
        self.model.setup_problem(x_calib=z_calib, y_calib=y_calib)
        
        
        if self.adaptive_enabled:
            try:
                self._fit_adaptive_quantile_fn(calib_data, z_calib, mask)
                
            except Exception as e:
                
                self.adaptive_enabled = False
        return self

    def predict(self, test_data: List[dict]) -> List[dict]:
        if not self.model or self.beta is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        z_test = self._make_z_only(test_data)
        out = []

        for i, d in enumerate(test_data):
            sample = dict(d.get('sample', {}))
            claims = sample.get('atomic_facts', [])
            if not claims:
                sample['filtered_claims'] = []
                out.append(sample)
                continue

            feats = d['features_4d']
            scores = -(feats @ self.beta)
            z_i = z_test[i:i+1]

            get_threshold_fn = lambda threshold, x: threshold

            try:
                if self.adaptive_enabled and self.quantile_theta is not None:
                    q_i = float(self._quantile_fn(z_i))
                else:
                    q_i = float(self.alpha)

                thr = self.model.predict(
                    quantile=q_i,
                    x_test=z_i,
                    score_inv_fn=get_threshold_fn,
                    randomize=True,
                    exact=True
                )
                thr = float(np.squeeze(thr))
                s_min = float(np.min(scores)) if scores.size > 0 else -np.inf
                s_max = float(np.max(scores)) if scores.size > 0 else np.inf
                if not np.isfinite(thr):
                    thr = s_max
                else:
                    thr = float(np.clip(thr, s_min, s_max))
                sample['filtered_claims'] = [c for j, c in enumerate(claims) if scores[j] <= thr]
            except Exception:
                sample['filtered_claims'] = []

            out.append(sample)

        return out

    # ------------------------------------------------------------------
    # Adaptive alpha utilities
    # ------------------------------------------------------------------
    def _get_claim_scores_list(self, data: List[dict], beta: np.ndarray) -> List[np.ndarray]:
        scores_list = []
        for d in data:
            feats = d['features_4d']
            s = -(feats @ beta)
            scores_list.append(s)
        return scores_list

    def _compute_retention_given_threshold(self, claim_scores: np.ndarray, threshold: float) -> float:
        if claim_scores.size == 0:
            return 0.0
        return float(np.mean(claim_scores <= threshold))

    def _fit_adaptive_quantile_fn(self, calib_data: List[dict], z_calib: np.ndarray, mask: np.ndarray):
        assert self.model is not None and self.beta is not None and self.retention_target is not None

        calib_data_masked = [calib_data[i] for i, m in enumerate(mask) if m]
        claim_scores_list = self._get_claim_scores_list(calib_data_masked, self.beta)
        quantile_grid = np.linspace(0.01, 0.99, 31)
        q_star = np.zeros(len(z_calib), dtype=float)
        for i in range(len(z_calib)):
            z_i = z_calib[i:i+1]
            best_q = None
            best_r = -1.0
            best_q_near = None
            for q in quantile_grid:
                try:
                    cutoff = self.model.predict(
                        quantile=float(q),
                        x_test=z_i,
                        score_inv_fn=lambda c, x: c,
                        randomize=True,
                        exact=True
                    )
                    T = float(np.asarray(cutoff).reshape(-1)[0])
                except Exception:
                    continue
                if not np.isfinite(T):
                    continue
                r = self._compute_retention_given_threshold(claim_scores_list[i], T)
                if r >= self.retention_target:
                    best_q = float(q)
                    break
                if r > best_r:
                    best_r = r
                    best_q_near = float(q)
            q_star[i] = float(best_q if best_q is not None else (best_q_near if best_q_near is not None else quantile_grid[-1]))

        def phi_alpha(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            ones = np.ones((x.shape[0], 1))
            return np.concatenate([ones, x, x**2], axis=1)

        Phi = phi_alpha(z_calib)
        ridge = 1e-6
        theta = np.linalg.pinv(Phi.T @ Phi + ridge * np.eye(Phi.shape[1])) @ (Phi.T @ q_star)
        self.quantile_theta = theta
        self._z_proj_for_quantile = None

    def _quantile_fn(self, z_row: np.ndarray) -> float:
        """Given single-row z (1 x d), return clipped quantile using phi_alpha (1, z, z^2)."""
        assert self.quantile_theta is not None
        z = np.asarray(z_row)
        phi = np.concatenate([np.ones((z.shape[0], 1)), z, z**2], axis=1)
        q = float(phi @ self.quantile_theta)
        return float(np.clip(q, 0.01, 0.99))

    
    def evaluate_auroc(self, test_data: List[dict]) -> dict:
        if not self.model or self.beta is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        all_scores = []
        all_labels = []
        
        for sample_data in test_data:
            features = sample_data['features_4d']
            annotations = np.array(sample_data['annotations'])
            
            nonconformity_scores = -features @ self.beta
            
            all_scores.extend(nonconformity_scores)
            all_labels.extend((~annotations.astype(bool)).astype(int))
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        try:
            auroc = roc_auc_score(all_labels, all_scores)
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            
            results = {
                'auroc': auroc,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'n_samples': len(all_scores),
                'n_false_claims': np.sum(all_labels),
                'n_true_claims': len(all_labels) - np.sum(all_labels)
            }
            
            
            
            return results
            
        except ValueError as e:
            
            return {
                'auroc': np.nan,
                'error': str(e),
                'n_samples': len(all_scores),
                'n_false_claims': np.sum(all_labels),
                'n_true_claims': len(all_labels) - np.sum(all_labels)
            }
    
    def get_claim_scores(self, test_data: List[dict]) -> List[dict]:
        """Return claim-level scores for each sample"""
        if not self.model or self.beta is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        results = []
        for sample_data in test_data:
            features = sample_data['features_4d']
            annotations = np.array(sample_data['annotations'])
            claims = sample_data.get('sample', {}).get('atomic_facts', [])
            
            nonconformity_scores = -features @ self.beta
            
            sample_result = {
                'sample_id': sample_data.get('sample_id', 'unknown'),
                'claims': claims,
                'nonconformity_scores': nonconformity_scores.tolist(),
                'annotations': annotations.tolist(),
                'is_false': (~annotations.astype(bool)).tolist()
            }
            results.append(sample_result)
        
        return results

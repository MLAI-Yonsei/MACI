"""
Basic Conformal Implementation for Factuality Assessment

This module implements a basic conformal prediction method for assessing
the factuality of generated text by filtering claims based on conformity scores.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class BasicConformal:
    def __init__(
        self, 
        score_function: Callable,
        random_state: Optional[int] = None
    ):
        self.score_function = score_function
        self.random_state = random_state
        self.calibration_scores = None
        self.threshold = None
        self._rng = np.random.default_rng(random_state)
        self._tie_gamma_keep: float = 1.0

    def fit_on_calib(self, calib_data: List, alpha: float = 0.1) -> 'BasicConformal':
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        raw_scores = self.score_function(calib_data)
        per_sample_scores: List[List[float]] = []
        if len(raw_scores) == len(calib_data) and hasattr(raw_scores[0], "__iter__") and not isinstance(raw_scores[0], (str, bytes)):
            for i, sample in enumerate(calib_data):
                if 'atomic_facts' in sample:
                    s_i = np.asarray(list(raw_scores[i]), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())
        else:
            if len(raw_scores) != len(calib_data):
                raise ValueError("score_function must return one score per sample or a per-claim score list per sample")
            for i, sample in enumerate(calib_data):
                if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                    s_i = np.asarray([float(raw_scores[i])] * len(sample['atomic_facts']), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())

        S_values: List[float] = []
        for sample, scores_i in zip(calib_data, per_sample_scores):
            if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                false_scores = [s for s, fact in zip(scores_i, sample['atomic_facts']) if not fact.get('is_supported', False)]
                if len(false_scores) == 0:
                    S_values.append(float('-inf'))
                else:
                    vals = np.asarray(false_scores, dtype=float)
                    S_values.append(float(np.nanmax(vals)) if vals.size > 0 else float('-inf'))
            else:
                vals = np.asarray(scores_i, dtype=float)
                if vals.size == 0:
                    S_values.append(float('-inf'))
                else:
                    S_values.append(float(np.nanmax(vals)))

        self.calibration_scores = np.array(S_values, dtype=float)
        n = len(self.calibration_scores)
        if n == 0:
            raise ValueError("No calibration samples available to compute threshold")
        quantile = 1 - alpha
        try:
            self.threshold = np.quantile(self.calibration_scores, quantile, method='higher')
        except TypeError:
            self.threshold = np.quantile(self.calibration_scores, quantile)

        sorted_scores = np.sort(self.calibration_scores)
        k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
        k = min(max(k, 0), n - 1)
        t_star = float(sorted_scores[k])
        n_lt = int(np.sum(self.calibration_scores < t_star))
        n_eq = int(np.sum(np.isclose(self.calibration_scores, t_star)))
        if n_eq <= 0:
            gamma_standard = 0.0
        else:
            gamma_standard = ((1.0 - alpha) * (n + 1) - n_lt) / n_eq
            gamma_standard = float(np.clip(gamma_standard, 0.0, 1.0))
        self._tie_gamma_keep = 1.0 - gamma_standard
        return self

    def predict(self, data: List) -> Tuple[List, List]:
        if self.threshold is None:
            raise ValueError("Model must be fitted before prediction")
        raw_scores = self.score_function(data)
        per_sample_scores: List[List[float]] = []
        if len(raw_scores) == len(data) and hasattr(raw_scores[0], "__iter__") and not isinstance(raw_scores[0], (str, bytes)):
            for i, sample in enumerate(data):
                if 'atomic_facts' in sample:
                    s_i = np.asarray(list(raw_scores[i]), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())
        else:
            if len(raw_scores) != len(data):
                raise ValueError("score_function must return one score per sample or per-claim score lists per sample")
            for i, sample in enumerate(data):
                if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                    s_i = np.asarray([float(raw_scores[i])] * len(sample['atomic_facts']), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())

        filtered_data: List = []
        retention_rates: List[float] = []
        for sample, scores_i in zip(data, per_sample_scores):
            if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                filtered_claims = []
                for claim, s in zip(sample['atomic_facts'], scores_i):
                    if s > self.threshold:
                        filtered_claims.append(claim)
                    elif np.isclose(s, self.threshold):
                        if self._rng.uniform() < self._tie_gamma_keep:
                            filtered_claims.append(claim)
                sample = dict(sample)
                sample['filtered_claims'] = filtered_claims
                retention_rate = len(filtered_claims) / len(sample['atomic_facts'])
            elif 'atomic_facts' in sample and len(sample['atomic_facts']) == 0:
                sample = dict(sample)
                sample['filtered_claims'] = []
                retention_rate = 0.0
            else:
                sample = dict(sample)
                if len(scores_i) == 0:
                    sample['is_retained'] = False
                    retention_rate = 0.0
                else:
                    s = float(scores_i[0])
                    sample['is_retained'] = (s > self.threshold) or (np.isclose(s, self.threshold) and self._rng.uniform() < self._tie_gamma_keep)
                    retention_rate = 1.0 if sample['is_retained'] else 0.0
            filtered_data.append(sample)
            retention_rates.append(retention_rate)
        return filtered_data, retention_rates

    def get_coverage(self, data: List) -> float:
        if self.threshold is None:
            raise ValueError("Model must be fitted before computing coverage")
        raw_scores = self.score_function(data)
        per_sample_scores: List[List[float]] = []
        if len(raw_scores) == len(data) and hasattr(raw_scores[0], "__iter__") and not isinstance(raw_scores[0], (str, bytes)):
            for i, sample in enumerate(data):
                if 'atomic_facts' in sample:
                    s_i = np.asarray(list(raw_scores[i]), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())
        else:
            if len(raw_scores) != len(data):
                raise ValueError("score_function must return one score per sample or per-claim score lists per sample")
            for i, sample in enumerate(data):
                if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                    s_i = np.asarray([float(raw_scores[i])] * len(sample['atomic_facts']), dtype=float)
                else:
                    s_i = np.asarray([float(raw_scores[i])], dtype=float)
                s_i = np.where(np.isnan(s_i), -np.inf, s_i)
                per_sample_scores.append(s_i.tolist())
        indicators = []
        for sample, scores_i in zip(data, per_sample_scores):
            if 'atomic_facts' in sample and len(sample['atomic_facts']) > 0:
                false_scores = [s for s, fact in zip(scores_i, sample['atomic_facts']) if not fact.get('is_supported', False)]
                if len(false_scores) == 0:
                    indicators.append(1.0)
                else:
                    vals = np.asarray(false_scores, dtype=float)
                    max_false = float(np.nanmax(vals)) if vals.size > 0 else float('-inf')
                    indicators.append(1.0 if max_false <= self.threshold else 0.0)
            else:
                vals = np.asarray(scores_i, dtype=float)
                if vals.size == 0:
                    indicators.append(1.0)
                else:
                    indicators.append(1.0 if float(np.nanmax(vals)) <= self.threshold else 0.0)
        return float(np.mean(indicators))

    def get_threshold(self) -> float:
        return self.threshold



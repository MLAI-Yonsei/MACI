#!/usr/bin/env python3
"""
Flexible conditional grouping utilities for subgroup analysis.
"""

import numpy as np
import pandas as pd
import re
import warnings
import json
import os
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

warnings.filterwarnings('default')
np.seterr(all='warn')


class ConditionalGrouper(ABC):
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute_values(self, dataset: List[Dict[str, Any]], **kwargs) -> np.ndarray:
        pass
    
    def create_bins(self, values: np.ndarray, method: str = 'quartiles', 
                   custom_bins: List[float] = None) -> List[Tuple[float, float]]:
        finite_values = values[np.isfinite(values)]
        
        if len(finite_values) == 0:
            return [(float(np.min(values)), float(np.max(values)))]
        
        if method == 'quartiles':
            quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        elif method == 'quintiles':
            quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        elif method == 'deciles':
            quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        elif method == 'tertiles':
            quantiles = [0.0, 0.33, 0.67, 1.0]
        elif method == 'median_split':
            quantiles = [0.0, 0.5, 1.0]
        elif method == 'custom' and custom_bins:
            qs = np.array(custom_bins)
        else:
            quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]  
        
        if method != 'custom':
            qs = np.quantile(finite_values, quantiles)
        
        bins = [(float(qs[i]), float(qs[i+1])) for i in range(len(qs)-1)]
        return bins
    
    def get_group_info(self, dataset: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        values = self.compute_values(dataset, **kwargs)
        finite_values = values[np.isfinite(values)]
        
        return {
            'name': self.name,
            'description': self.description,
            'total_samples': len(values),
            'valid_samples': len(finite_values),
            'min_value': float(np.min(finite_values)) if len(finite_values) > 0 else np.nan,
            'max_value': float(np.max(finite_values)) if len(finite_values) > 0 else np.nan,
            'mean_value': float(np.mean(finite_values)) if len(finite_values) > 0 else np.nan,
            'std_value': float(np.std(finite_values)) if len(finite_values) > 0 else np.nan,
        }


# View metadata configuration (globally overridable)
def _default_view_csv_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(repo_root, 'data', 'wiki_scores', 'wikibio_final.csv')

GLOBAL_VIEW_METADATA_CSV = _default_view_csv_path()

def set_view_metadata_csv(csv_path: str):
    global GLOBAL_VIEW_METADATA_CSV
    if isinstance(csv_path, str) and len(csv_path) > 0:
        GLOBAL_VIEW_METADATA_CSV = csv_path


class ViewCountGrouper(ConditionalGrouper):

    def __init__(self):
        super().__init__(
            name="view_count",
            description="Wikipedia view count (from wikibio_final.csv)"
        )
        self._loaded = False
        self._csv_path = None
        self._name_to_views = {}
        self._global_min_count = 0.0

    @staticmethod
    def _parse_name_from_prompt(prompt: str) -> str:
        if not isinstance(prompt, str):
            try:
                prompt = str(prompt)
            except Exception:
                return ""
        txt = prompt.strip()
        # Typical pattern: "Please write one biographical paragraph about {NAME}."
        import re
        m = re.search(r"about\s+(.+?)(?:[\.]|\n|$)", txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: try after 'about '
        if 'about ' in txt:
            return txt.split('about ', 1)[-1].strip().rstrip('.').strip()
        return txt

    def _ensure_loaded(self):
        # Lazy-load and refresh if global path changed
        if (not self._loaded) or (self._csv_path != GLOBAL_VIEW_METADATA_CSV):
            try:
                df = pd.read_csv(GLOBAL_VIEW_METADATA_CSV)
                name_col = 'Name' if 'Name' in df.columns else None
                views_col = 'Views' if 'Views' in df.columns else None
                maxc_col = 'max_counts' if 'max_counts' in df.columns else None
                mapping = {}
                values_for_min = []
                if name_col and (views_col or maxc_col):
                    for _, row in df.iterrows():
                        name = str(row[name_col]).strip()
                        v = np.nan
                        # Per-row preference: Views if finite, else max_counts
                        if views_col is not None:
                            try:
                                vv = float(row[views_col])
                                if np.isfinite(vv):
                                    v = vv
                            except Exception:
                                pass
                        if (not np.isfinite(v)) and maxc_col is not None:
                            try:
                                mv = float(row[maxc_col])
                                if np.isfinite(mv):
                                    v = mv
                            except Exception:
                                pass
                        mapping[name] = v
                        if np.isfinite(v):
                            values_for_min.append(v)
                self._name_to_views = mapping
                self._csv_path = GLOBAL_VIEW_METADATA_CSV
                # Global minimum over available finite counts; default to 0.0 if none
                self._global_min_count = float(np.min(values_for_min)) if len(values_for_min) > 0 else 0.0
                self._loaded = True
            except Exception:
                # If loading fails, mark as loaded with empty mapping
                self._name_to_views = {}
                self._csv_path = GLOBAL_VIEW_METADATA_CSV
                self._global_min_count = 0.0
                self._loaded = True

    def compute_values(self, dataset: List[Dict[str, Any]], **kwargs) -> np.ndarray:
        self._ensure_loaded()
        values = []
        for sample in dataset:
            prompt = sample.get('prompt', '')
            name = self._parse_name_from_prompt(prompt)
            # Direct match first
            val = self._name_to_views.get(name)
            if val is None:
                # Try naive normalization: collapse spaces
                key2 = " ".join(name.split())
                val = self._name_to_views.get(key2, np.nan)
            # Fallback: global min count if missing or NaN
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                val = self._global_min_count
            values.append(float(val))
        return np.array(values, dtype=float)


class FalseClaimRiskGrouper(ConditionalGrouper):

    def __init__(self):
        super().__init__(
            name="false_claim_risk",
            description="Text-based false-claim risk index (higher → more risk)"
        )
        self.abs_terms = [
            'always', 'never', 'guarantee', 'guaranteed', 'cure', 'proven',
            'will', 'must', 'definitely', 'certainly', 'undoubtedly', 'no doubt'
        ]
        self.enum_keywords = [
            'symptom', 'symptoms', 'signs', 'causes', 'cause', 'types', 'treatments',
            'treatment', 'risk factors', 'complications', 'side effects', 'prevention'
        ]
        self.citation_patterns = [
            r'according\s+to', r'based\s+on', r'research\s+(?:shows?|indicates?|suggests?)',
            r'studies?\s+(?:show|indicate|suggest|reveal|demonstrate)', r'\(\d{4}\)', r'\[[\d,\s-]+\]'
        ]
        self.compiled_cite = [re.compile(p, re.IGNORECASE) for p in self.citation_patterns]

    @staticmethod
    def _num_sentences(text: str) -> int:
        if not text:
            return 0
        return max(1, text.count('.') + text.count('!') + text.count('?') + text.count('\n'))

    @staticmethod
    def _listiness(text: str) -> int:
        if not text:
            return 0
        markers = [',', ';', '\n', '-', '*', '•']
        count = sum(text.count(m) for m in markers)
        # Enumerations like "1.", "2)", "(3)"
        count += len(re.findall(r'(?:(?<=\s)|^)(?:\d{1,2}[\.)\]])', text))
        return count

    def _citation_density(self, text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        matches = 0
        low = text.lower()
        for pat in self.compiled_cite:
            matches += len(pat.findall(low))
        return matches / max(1, len(words))

    def _absolute_density(self, text: str) -> float:
        if not text:
            return 0.0
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        abs_cnt = sum(1 for w in words if w in self.abs_terms)
        return abs_cnt / max(1, len(words))

    def _enum_keyword_score(self, prompt: str, response: str) -> float:
        txt = f"{prompt} {response}".lower()
        return float(sum(1 for k in self.enum_keywords if k in txt))

    def compute_values(self, dataset: List[Dict[str, Any]], **kwargs) -> np.ndarray:
        vals = []
        for sample in dataset:
            prompt = sample.get('prompt', '') or ''
            response = sample.get('response', '') or ''
            resp = str(response)

            # Features
            num_words = len(resp.split())
            len_norm = min(1.0, num_words / 400.0)
            sent_norm = min(1.0, self._num_sentences(resp) / 12.0)
            list_norm = min(1.0, self._listiness(resp) / 40.0)
            num_density = (sum(ch.isdigit() for ch in resp) / max(1, len(resp)))
            abs_density = self._absolute_density(resp)
            cite_density = self._citation_density(resp)
            enum_score = min(1.0, self._enum_keyword_score(str(prompt), resp) / 4.0)

            # Composite risk (clipped to [0,1])
            risk = (
                0.30 * len_norm +
                0.15 * sent_norm +
                0.20 * list_norm +
                0.10 * num_density +
                0.15 * abs_density +
                0.10 * enum_score -
                0.10 * cite_density
            )
            vals.append(float(np.clip(risk, 0.0, 1.0)))
        return np.array(vals, dtype=float)


class MedicalContentGrouper(ConditionalGrouper):
    def __init__(self):
        super().__init__(
            name="medical_content",
            description="Medical content (Information/Interpretation/Action)"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        return " ".join(text.strip().lower().split())

    def _classify(self, prompt: str) -> int:
        p = self._normalize(prompt)

        # Heuristic keyword sets
        info_kw = [
            "what is", "what are", "definition", "define", "symptom", "signs", "cause", "why",
            "prognosis", "life expectancy", "effect", "does .* do", "means?", "treatment", "therapy",
            "disease", "syndrome", "disorder", "cancer", "diabetes", "ards", "tay-sachs", "paget",
            "thalassemia", "psp", "rosacea", "empyema"
        ]
        drug_kw = [
            "drug", "medication", "medicine", "dose", "dosage", "tablet", "pill", "mg", "patch",
            "paxlovid", "zoloft", "lexapro", "meloxicam", "naproxen", "fentanyl", "celexa", "restoril",
            "calcitonin", "latanoprost", "aldactazide", "nicoderm"
        ]
        symptom_kw = [
            "pain", "ache", "swelling", "lump", "dark urine", "dizziness", "lightheaded", "fatigue",
            "muscle aches", "discharge", "sunburn", "hoarder", "smell"
        ]
        interpret_kw = [
            "what does it mean", "what does .* mean", "when should you worry", "should i worry",
        ]
        action_kw = [
            "should i", "do i need", "is it okay", "can i", "how to", "how do i", "stop", "start",
            "continue", "switch", "swap", "get tested", "try", "take", "drink", "use"
        ]

        def contains_any(keys: List[str]) -> bool:
            for k in keys:
                if " .* " in k or ".*" in k:
                    import re
                    if re.search(k, p):
                        return True
                if k in p:
                    return True
            return False

        # Action-seeking first (high precision phrases)
        if contains_any(action_kw):
            return 2

        # Information-seeking: has disease/drug entity cues and info-type query words
        if (contains_any(info_kw) or contains_any(drug_kw)) and ("?" in prompt or contains_any(["what", "why", "signs", "symptom", "life expectancy", "treatment"])):
            return 0

        # Interpretation-seeking: general symptom phrases or interpret patterns
        if contains_any(interpret_kw) or contains_any(symptom_kw):
            return 1

        # Fallback: map generic questions with what/why to information
        if contains_any(["what", "why"]):
            return 0

        # Otherwise treat as action if imperative-like
        if contains_any(["how to", "how do i"]):
            return 2

        # Default to interpretation
        return 1

    def compute_values(self, dataset: List[Dict[str, Any]], **kwargs) -> np.ndarray:
        values = []
        for sample in dataset:
            prompt = sample.get('prompt', '')
            values.append(self._classify(prompt))
        return np.array(values, dtype=float)

    def create_bins(self, values: np.ndarray, method: str = 'ignored', custom_bins: List[float] = None) -> List[Tuple[float, float]]:
        return [(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5)]


class ExpertQAFieldGrouper(ConditionalGrouper):
    """ExpertQA official metadata.field based 3-group classifier

    - 0: Biology/Medicine (Biology, Chemistry, Psychology, Environmental Science, etc.)
    - 1: Engineering/Technology (Engineering and Technology, Physics and Astronomy, Architecture, etc.)
    - 2: Other (All other fields)

    The mapping is loaded from '/expertqa_prompt_to_field.json' by default.
    If the file does not exist, all samples are classified as Other(2).
    The values are integer labels, and create_bins is fixed to discrete intervals.
    """

    def __init__(self, mapping_path: str = "/expertqa_prompt_to_field.json"):
        super().__init__(
            name="expertqa_field",
            description="ExpertQA metadata.field → {Bio/Med, Eng/Tech, Other}"
        )
        self.mapping_path = mapping_path
        self._loaded = False
        self._prompt_to_field = {}

        self.bio_med_fields = set([
            "Healthcare / Medicine",
            "Biology",
            "Chemistry",
            "Psychology",
            "Environmental Science",
        ])
        self.eng_tech_fields = set([
            "Engineering and Technology",
            "Physics and Astronomy",
            "Architecture",
        ])

    @staticmethod
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        return " ".join(text.strip().split())

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            if os.path.exists(self.mapping_path):
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._prompt_to_field = {self._normalize(k): v for k, v in data.items()}
            else:
                self._prompt_to_field = {}
        except Exception:
            self._prompt_to_field = {}
        finally:
            self._loaded = True

    def _field_to_group(self, field: str) -> int:
        if not isinstance(field, str):
            return 2
        f = field.strip()
        if f in self.bio_med_fields:
            return 0
        if f in self.eng_tech_fields:
            return 1
        return 2

    def compute_values(self, dataset: List[Dict[str, Any]], **kwargs) -> np.ndarray:
        self._ensure_loaded()
        labels = []
        for sample in dataset:
            prompt = sample.get('prompt', '')
            p_key = self._normalize(prompt)
            field = self._prompt_to_field.get(p_key)
            if field is None:
                q = sample.get('question', '')
                q_key = self._normalize(q)
                field = self._prompt_to_field.get(q_key)
            group_id = self._field_to_group(field)
            labels.append(float(group_id))
        return np.array(labels, dtype=float)

    def create_bins(self, values: np.ndarray, method: str = 'ignored', custom_bins: List[float] = None) -> List[Tuple[float, float]]:
        return [(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5)]


def get_available_groupers() -> Dict[str, ConditionalGrouper]:
    return {
        'view_count': ViewCountGrouper(),
        'medical_content': MedicalContentGrouper(),
        'false_claim_risk': FalseClaimRiskGrouper(),
    }


def compute_conditional_coverage_by_grouper(
    filtered_dataset: List[Dict[str, Any]], 
    grouping_values: np.ndarray, 
    bins: List[Tuple[float, float]]
) -> List[float]:
    """Calculate conditional coverage by a specific grouper"""
    
    def compute_marginal_coverage(sub_dataset: List[Dict[str, Any]]) -> float:
        """Calculate marginal coverage from a given subset"""
        indicators = []
        for d in sub_dataset:
            retained = d.get('filtered_claims', [])
            has_false = any([not c.get('is_supported', False) for c in retained])
            indicators.append(0.0 if has_false else 1.0)
        return float(np.mean(indicators)) if indicators else 0.0
    
    coverage_results = []
    
    for bin_min, bin_max in bins:
        mask = []
        for i, value in enumerate(grouping_values):
            if np.isfinite(value):
                mask.append(bin_min <= value <= bin_max)
            else:
                mask.append(False)
        
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            coverage_results.append(np.nan)
            continue
        
        subset = [filtered_dataset[i] for i in indices]
        coverage = compute_marginal_coverage(subset)
        coverage_results.append(coverage)
    
    return coverage_results


def compute_retention_by_grouper(
    filtered_dataset: List[Dict[str, Any]], 
    grouping_values: np.ndarray, 
    bins: List[Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """Calculate retention rate by a specific grouper"""
    
    retention_results = []
    
    for bin_min, bin_max in bins:
        mask = []
        for i, value in enumerate(grouping_values):
            if np.isfinite(value):
                mask.append(bin_min <= value <= bin_max)
            else:
                mask.append(False)
        
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            retention_results.append({
                'bin': (float(bin_min), float(bin_max)),
                'samples': 0,
                'retained': 0,
                'total': 0,
                'rate': np.nan,
            })
            continue
        
        total_claims = 0
        retained_claims = 0
        sample_count = len(indices)
        
        for idx in indices:
            d = filtered_dataset[idx]
            afs = d.get('atomic_facts', [])
            total_claims += len(afs)
            retained_claims += len(d.get('filtered_claims', []))
        
        rate = (retained_claims / total_claims) if total_claims > 0 else np.nan
        
        retention_results.append({
            'bin': (float(bin_min), float(bin_max)),
            'samples': sample_count,
            'retained': int(retained_claims),
            'total': int(total_claims),
            'rate': float(rate) if not np.isnan(rate) else np.nan,
        })
    
    return retention_results

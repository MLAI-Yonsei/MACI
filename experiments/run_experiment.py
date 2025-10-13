import numpy as np
import pickle
import os
import sys
import json
import argparse
import time
import logging
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import defaultdict
warnings.filterwarnings('default')
warnings.simplefilter('ignore', category=FutureWarning)
np.seterr(all='warn')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conformal.basic_conformal import BasicConformal
from conformal.adaptive_conformal import MACIAdaptiveConformal, SubgroupOptimizedMACI
from conditional_groupers import get_available_groupers
from conditional_groupers import set_view_metadata_csv

MODEL_NAMES = ['qwen-2.5-72b-instruct', 'deepseek-chat-v3-0324', 'llama-3.3-70b-instruct']

def setup_logging(log_dir: str):
    """Sets up logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"experiment_log_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    logging.info(f"📝 Logging to {log_filename}")


def load_1000_samples(data_dir: str, scores_dir: Optional[str] = None, dataset_type: str = "auto", limit_samples: int = 1000):
    """Load up to `limit_samples` samples and attach LLM scores."""
    logging.info(f"📁 Loading up to {limit_samples} samples with provided scores...")
    
    if dataset_type == "auto":
        wikibio_path = os.path.join(data_dir, "wiki_scores", "wikibio_final_dataset.pkl")
        medlfqa_path = os.path.join(data_dir, "med_scores", "medlfqa_dataset.pkl")
        
        if os.path.exists(wikibio_path):
            dataset_type = "wikibio"
            logging.info(f"  🔍 Auto-detected dataset type: {dataset_type}")
        elif os.path.exists(medlfqa_path):
            dataset_type = "medlfqa"
            logging.info(f"  🔍 Auto-detected dataset type: {dataset_type}")
        else:
            raise FileNotFoundError(f"Could not find dataset files in {data_dir}")
    
    if dataset_type == "wikibio":
        dataset_path = os.path.join(data_dir, "wiki_scores", "wikibio_final_dataset.pkl")
        base_scores_dir = os.path.join(data_dir, "wiki_scores")
        score_prefix = "wikibio_scores"
        basic_scores = {
            'frequencies': os.path.join(base_scores_dir, "wikibio_final_frequencies.npz"),
            'logprobs': os.path.join(base_scores_dir, "wikibio_final_logprobs.npz"),
            'selfevals': os.path.join(base_scores_dir, "wikibio_final_self_evals.npz")
        }
    elif dataset_type == "medlfqa":
        dataset_path = os.path.join(data_dir, "med_scores", "medlfqa_dataset.pkl")
        base_scores_dir = os.path.join(data_dir, "med_scores")
        score_prefix = "medlfqa_scores"
        basic_scores = {
            'frequencies': os.path.join(base_scores_dir, "medlfqa_frequencies.npz"),
            'logprobs': os.path.join(base_scores_dir, "medlfqa_logprobs.npz"),
            'selfevals': os.path.join(base_scores_dir, "medlfqa_selfevals.npz")
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    logging.info(f"  📊 Dataset: {dataset_path}")
    logging.info(f"  🎯 Score prefix: {score_prefix}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    dataset_1000 = dataset[:limit_samples]
    
    frequencies = {}
    logprobs = {}
    selfevals = {}
    
    for score_type, score_path in basic_scores.items():
        try:
            if score_type == 'frequencies':
                frequencies = np.load(score_path, allow_pickle=True)
            elif score_type == 'logprobs':
                logprobs = np.load(score_path, allow_pickle=True)
            elif score_type == 'selfevals':
                selfevals = np.load(score_path, allow_pickle=True)
            logging.info(f"  ✅ Loaded {score_type}: {score_path}")
        except FileNotFoundError:
            logging.warning(f"  ⚠️ {score_type} not found: {score_path}")
    
    if scores_dir is not None and os.path.isdir(scores_dir):
        score_files_dir = scores_dir
    else:
        score_files_dir = base_scores_dir
    
    logging.info(f"  🎯 Score files directory: {score_files_dir}")
    
    import glob
    all_npz_files = sorted(glob.glob(os.path.join(score_files_dir, f"{score_prefix}_*.npz")))
    def find_by_tokens(token_options: List[List[str]]):
        for tokens in token_options:
            for fp in all_npz_files:
                name = os.path.basename(fp).lower()
                if all(t in name for t in tokens):
                    return fp
        return None

    score_files = {
        'qwen-2.5-72b-instruct': find_by_tokens([
            ['qwen-2.5-72b','instruct'], ['qwen','instruct'], ['qwen']
        ]),
        'deepseek-chat-v3-0324': find_by_tokens([
            ['deepseek','chat','v3'], ['deepseek','chat'], ['deepseek']
        ]),
        'llama-3.3-70b-instruct': find_by_tokens([
            ['llama-3.3-70b','instruct'], ['llama-3.3','instruct'], ['llama']
        ]),
    }

    llm_scores = {}
    for model_name, filename in score_files.items():
        try:
            model_data = np.load(filename, allow_pickle=True)
            model_prompts = model_data['prompts'].tolist()
            model_scores_list = model_data['scores_list'].tolist()
            llm_scores[model_name] = {p: s for p, s in zip(model_prompts, model_scores_list)}
            logging.info(f"  ✅ Loaded {model_name} scores")
        except (FileNotFoundError, TypeError):
            logging.warning(f"  ⚠️ {model_name} scores not found or invalid: {filename}")
            llm_scores[model_name] = {}
    
    aligned_data = []
    for i, sample in enumerate(dataset_1000):
        prompt = sample['prompt']
        atomic_facts = sample.get('atomic_facts', [])
        n_claims = len(atomic_facts)
        
        if n_claims == 0:
            continue
            
        if prompt in selfevals:
            selfeval_vals = selfevals[prompt]
            if hasattr(selfeval_vals, 'ndim') and selfeval_vals.ndim == 1:
                if np.allclose(selfeval_vals, -1):
                    continue
            elif np.allclose(selfeval_vals, -1):
                continue
            
        annotations = np.array([af.get('is_supported', False) for af in atomic_facts])
        
        freq_scores = np.zeros(n_claims)
        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in frequencies:
                freq_vals = frequencies[key]
                if hasattr(freq_vals, 'ndim') and freq_vals.ndim == 1:
                    freq_scores = freq_vals[:n_claims]
                else:
                    freq_scores = np.full(n_claims, freq_vals.item() if hasattr(freq_vals, 'item') else freq_vals)
                freq_scores = np.nan_to_num(freq_scores, nan=0.0)
        else:
            if prompt in frequencies:
                freq_vals = frequencies[prompt]
                if hasattr(freq_vals, 'ndim') and freq_vals.ndim == 1:
                    freq_scores = freq_vals[:n_claims]
                else:
                    freq_val = freq_vals.item() if hasattr(freq_vals, 'item') else freq_vals
                    freq_val = 0.0 if np.isnan(freq_val) else freq_val
                    freq_scores = np.full(n_claims, freq_val)
                freq_scores = np.nan_to_num(freq_scores, nan=0.0)

        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in logprobs:
                lp_vals = logprobs[key]
                if hasattr(lp_vals, 'ndim') and lp_vals.ndim == 1:
                    logprob_scores = np.nan_to_num(lp_vals[:n_claims], nan=0.0)
                else:
                    v = lp_vals.item() if hasattr(lp_vals, 'item') else lp_vals
                    v = 0.0 if np.isnan(v) else v
                    logprob_scores = np.full(n_claims, v)
            else:
                logprob_scores = np.zeros(n_claims)
        else:
            if prompt in logprobs:
                logprob_vals = logprobs[prompt]
                if hasattr(logprob_vals, 'ndim') and logprob_vals.ndim == 1:
                    logprob_scores = logprob_vals[:n_claims]
                    logprob_scores = np.nan_to_num(logprob_scores, nan=0.0)
                else:
                    logprob_val = logprob_vals.item() if hasattr(logprob_vals, 'item') else logprob_vals
                    logprob_val = 0.0 if np.isnan(logprob_val) else logprob_val
                    logprob_scores = np.full(n_claims, logprob_val)
            else:
                logprob_scores = np.zeros(n_claims)
            
        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in selfevals:
                se_vals = selfevals[key]
                if hasattr(se_vals, 'ndim') and se_vals.ndim == 1:
                    selfeval_scores = np.nan_to_num(se_vals[:n_claims], nan=0.0)
                else:
                    v = se_vals.item() if hasattr(se_vals, 'item') else se_vals
                    v = 0.0 if np.isnan(v) else v
                    selfeval_scores = np.full(n_claims, v)
            else:
                selfeval_scores = np.zeros(n_claims)
        else:
            if prompt in selfevals:
                selfeval_vals = selfevals[prompt]
                if hasattr(selfeval_vals, 'ndim') and selfeval_vals.ndim == 1:
                    selfeval_scores = selfeval_vals[:n_claims]
                    selfeval_scores = np.nan_to_num(selfeval_scores, nan=0.0)
                else:
                    selfeval_val = selfeval_vals.item() if hasattr(selfeval_vals, 'item') else selfeval_vals
                    selfeval_val = 0.0 if np.isnan(selfeval_val) else selfeval_val
                    selfeval_scores = np.full(n_claims, selfeval_val)
            else:
                selfeval_scores = np.zeros(n_claims)
            
        ordinal_scores = np.arange(n_claims)
        if n_claims > 1:
            ordinal_scores = ordinal_scores / (n_claims - 1)
        else:
            ordinal_scores = np.array([0.5])
        
        scores_dict = {}
        for model_name, model_data in llm_scores.items():
            if prompt in model_data:
                scores_dict[model_name] = np.array(model_data[prompt][:n_claims])
                scores_dict[model_name] = np.clip(scores_dict[model_name], 0.0, 1.0)
            else:
                scores_dict[model_name] = np.full(n_claims, 0.5)
        
        valid_llm_scores = []
        for model_name in MODEL_NAMES:
            if model_name in scores_dict:
                valid_llm_scores.append(scores_dict[model_name])
        
        if valid_llm_scores:
            ensemble_mean = np.mean(valid_llm_scores, axis=0)
            ensemble_std = np.std(valid_llm_scores, axis=0)
            lambda_uncertainty = 0.0
            ensemble_scores = ensemble_mean - lambda_uncertainty * ensemble_std
            ensemble_scores = np.clip(ensemble_scores, 0.0, 1.0)
        else:
            ensemble_scores = np.full(n_claims, 0.5)
        
        features_4d = np.concatenate((
            freq_scores.reshape(-1, 1),
            selfeval_scores.reshape(-1, 1),
            (logprob_scores / (np.max(logprob_scores) + 1e-8)).reshape(-1, 1),
            ordinal_scores.reshape(-1, 1)
        ), axis=1)
        
        aligned_data.append({
            'sample': sample,
            'annotations': annotations,
            'scores': {
                'frequency': freq_scores,
                'selfeval': selfeval_scores,
                'logprob': logprob_scores,
                'ensemble': ensemble_scores,
                **scores_dict
            },
            'features_4d': features_4d,
            'prompt_features': np.array([1.0, len(sample.get('response', '')), len(prompt)])
        })
    
    logging.info(f"✅ Loaded {len(aligned_data)} valid samples")
    return aligned_data


def create_splits(data, calib_ratio=0.7, test_ratio=0.3, random_seed=42):
    """Create calibration and test splits based on ratios with random shuffling"""
    total_size = len(data)
    calib_size = int(total_size * calib_ratio)
    test_size = int(total_size * test_ratio)
    
    if calib_size + test_size > total_size:
        test_size = total_size - calib_size
    
    logging.info(f"📊 Creating splits: {calib_size} calib ({calib_ratio*100:.0f}%), {test_size} test ({test_ratio*100:.0f}%)")
    
    np.random.seed(random_seed)
    indices = np.random.permutation(total_size)
    
    calib_idx = indices[:calib_size]
    test_idx = indices[calib_size:calib_size + test_size]
    
    calib_data = [data[i] for i in calib_idx]
    test_data = [data[i] for i in test_idx]
    
    logging.info(f"🎲 Random split with seed {random_seed}: calib indices {calib_idx[:5]}..., test indices {test_idx[:5]}...")
    
    return calib_data, test_data, calib_idx, test_idx


def run_bcp_experiment(calib_data, test_data, score_type='frequency', alpha=0.1, **kwargs):
    """
    Run BCP (Split Conformal) experiment.
    [FIXED] Uses a unified score_function that relies on pre-aligned data.
    """
    logging.info(f"📈 Running BCI (Split Conformal) with {score_type} scores...")

    calib_samples = [item['sample'] for item in calib_data]
    test_samples = [item['sample'] for item in test_data]
    
    def score_function(samples):
        result = []
        sample_to_data = {item['sample']['prompt']: item for item in calib_data + test_data}
        
        for sample in samples:
            prompt = sample['prompt']
            if prompt in sample_to_data:
                scores = sample_to_data[prompt]['scores'].get(score_type)
                if scores is not None:
                    if score_type in ['frequency', 'selfeval', 'logprob']:
                        non_conformity_scores = 1.0 - scores
                    else:
                        non_conformity_scores = 1.0 - scores
                    result.append(non_conformity_scores)
                else:
                    
                    n_claims = len(sample.get('atomic_facts', []))
                    result.append(np.full(n_claims, 0.5))
            else:
                n_claims = len(sample.get('atomic_facts', []))
                result.append(np.full(n_claims, 0.5))
        return result
    
    basic_conformal = BasicConformal(score_function=score_function, random_state=0)
    basic_conformal.fit_on_calib(calib_samples, alpha=alpha)
    filtered_results, _ = basic_conformal.predict(test_samples)

    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "BCP")
    
    return {
        'coverage': coverage,
        'retention_rate': retention['overall_retention_rate'],
        'retained_claims': retention['retained_claims'],
        'total_claims': retention['total_claims'],
        'filtered_results': filtered_results
    }

def run_as_experiment(calib_data: List[Dict], test_data: List[Dict], 
                      model_names: List[str],
                      alpha: float, 
                      as_mode: str, 
                      subgroup_name: str, **kwargs) -> Dict:
    """Run MACI (Adaptive Subclaims) experiment for a given subgroup."""
    logging.info(f"📊 Running MACI experiment with mode: {as_mode} for subgroup: '{subgroup_name}'...")
    
    timing: Dict[str, float] = {}

    if as_mode == 'subgroup_optimized':
        available_groupers = get_available_groupers()
        if subgroup_name not in available_groupers:
            raise ValueError(f"Unknown subgroup: {subgroup_name}")
        grouper = available_groupers[subgroup_name]

        as_model = SubgroupOptimizedMACI(
            model_names=model_names,
            grouper=grouper,
            n_bins=3,
            random_state=kwargs.get('random_state', 0),
            solver='osqp',
        )
        t0 = time.perf_counter()
        as_model.fit(calib_data, alpha=alpha, ensemble_train_ratio=0.5, target_tpr=kwargs.get('target_tpr', 0.95))
        timing_details = as_model.get_timing()
        timing['maci_weight_optimization_s'] = timing_details.get('weight_optimization_s', 0.0)
        timing['maci_calibration_s'] = timing_details.get('calibration_s', 0.0)

        t1 = time.perf_counter()
        filtered_results, _ = as_model.predict(test_data)
        timing['maci_inference_s'] = time.perf_counter() - t1
        budgets = as_model.get_budgets()
        weights = as_model.get_weights()

    else:
        score_type = kwargs.get("as_score_type", "ensemble")
        def score_function(data_list: List[Dict]) -> List[np.ndarray]:
            scores_list = []
            for item in data_list:
                valid_scores = [item['scores'][m] for m in model_names if m in item['scores']]
                if valid_scores:
                    scores_list.append(np.mean(valid_scores, axis=0))
                else:
                    scores_list.append(np.array([0.5] * len(item.get('sample', {}).get('atomic_facts', []))))
            return scores_list

        as_model = MACIAdaptiveConformal(score_function=score_function, random_state=kwargs.get('random_state', 0))
        t0 = time.perf_counter()
        as_model.fit_on_calib(calib_data, alpha=alpha)
        timing['maci_calibration_s'] = time.perf_counter() - t0
        t1 = time.perf_counter()
        filtered_results, _ = as_model.predict(test_data)
        timing['maci_inference_s'] = time.perf_counter() - t1
        budgets = {'overall': as_model.tau_hat}
        weights = None
    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "MACI")
    return {
        'coverage': coverage,
        'retention_rate': retention['overall_retention_rate'],
        'retained_claims': retention['retained_claims'],
        'total_claims': retention['total_claims'],
        'budgets': budgets,
        'weights': weights,
        'filtered_results': filtered_results,
        'timing': timing
    }


def run_cci_experiment(
    calib_data,
    test_data,
    alpha=0.1,
    boosting_epochs=1000,
    boosting_lr=0.005,
    calib_split_for_boost=0.3,  
    random_seed=0,
    adaptive_alpha: bool = False,
    retention_target: float = 0.7
):
    """
    Two-stage CCI:
      - Stage 1 (Boosting): learn beta on a subset of calib_data
      - Stage 2 (CondConf): calibrate CondConf on the remaining calib_data using learned beta
      - Predict on test_data
    """
    logging.info("🎯 Running CCI (Boosting -> CondConf) with internal calib split...")

    try:
        from conformal.conditional_conformal import ConditionalConformalBoosting, ConditionalConformalInference
    except Exception as e:
        logging.error(f"CCI unavailable due to missing dependencies: {e}")
        return {
            "coverage": None,
            "retention_rate": None,
            "retained_claims": 0,
            "total_claims": 0,
            "filtered_results": [],
            "timing": {"cci_skipped": True, "error": str(e)}
        }

    rng = np.random.default_rng(random_seed)
    idx = np.arange(len(calib_data))
    rng.shuffle(idx)
    k = int(len(idx) * calib_split_for_boost)
    idx_boost, idx_conf = idx[:k], idx[k:]
    if len(idx_conf) == 0:
        idx_boost, idx_conf = idx[:-1], idx[-1:] [1]
    calib_boost = [calib_data[i] for i in idx_boost]
    calib_conf  = [calib_data[i] for i in idx_conf]
    logging.info(f"  🔧 calib split -> boost:{len(calib_boost)} | conf:{len(calib_conf)} (seed={random_seed})")

    booster = ConditionalConformalBoosting(random_state=random_seed)
    t_boost_0 = time.perf_counter()
    beta = booster.fit(
        calib_boost,
        boosting_epochs=boosting_epochs,
        boosting_lr=boosting_lr
    )
    t_boost_1 = time.perf_counter()

    infer = ConditionalConformalInference(random_state=random_seed)
    t_fit_0 = time.perf_counter()
    infer.fit(calib_conf, alpha=alpha, beta=beta, adaptive_alpha=adaptive_alpha, retention_target=retention_target)
    t_fit_1 = time.perf_counter()
    auroc_results = infer.evaluate_auroc(test_data)
    t_pred_0 = time.perf_counter()
    filtered_results = infer.predict(test_data)
    t_pred_1 = time.perf_counter()

    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "CCI")

    return {
        "coverage": coverage,
        "retention_rate": retention["overall_retention_rate"],
        "retained_claims": retention["retained_claims"],
        "total_claims": retention["total_claims"],
        "filtered_results": filtered_results,
        "beta": beta,
        "calib_sizes": {"boost": len(calib_boost), "conf": len(calib_conf)},
        "split_seed": random_seed,
        "timing": {
            "cci_boost_fit_s": t_boost_1 - t_boost_0,
            "cci_condconf_fit_s": t_fit_1 - t_fit_0,
            "cci_inference_s": t_pred_1 - t_pred_0,
            "cci_adaptive_alpha_enabled": bool(adaptive_alpha)
        }
    } 
    
def evaluate_retention(filtered_dataset: List[Dict], method_name: str = "") -> Dict:
    total_original_claims = 0
    total_retained_claims = 0

    if not filtered_dataset:
        return {'overall_retention_rate': 0.0, 'retained_claims': 0, 'total_claims': 0}

    for item in filtered_dataset:
        sample_dict = item.get('sample', item)
        if not isinstance(sample_dict, dict):
            logging.warning(f"Skipping invalid item in retention evaluation: {type(sample_dict)}")
            continue

        original_claims = sample_dict.get('atomic_facts', [])
        retained_claims = sample_dict.get('filtered_claims', [])

        total_original_claims += len(original_claims)
        total_retained_claims += len(retained_claims)

    if total_original_claims > 0:
        overall_retention_rate = total_retained_claims / total_original_claims
    else:
        overall_retention_rate = 0.0

    return {
        'overall_retention_rate': overall_retention_rate,
        'retained_claims': total_retained_claims,
        'total_claims': total_original_claims
    }

def compute_marginal_coverage(filtered_dataset: List[Dict]):
    indicators = []
    for item in filtered_dataset:
        sample_dict = item.get('sample', item)
        if not isinstance(sample_dict, dict):
            logging.warning(f"Skipping invalid item in coverage calculation: {type(sample_dict)}")
            continue
        
        retained = sample_dict.get('filtered_claims', [])
        
        if len(retained) == 0:
            indicators.append(1.0)
        else:
            has_false = any(not claim.get('is_supported', False) for claim in retained if isinstance(claim, dict))
            indicators.append(0.0 if has_false else 1.0)
            
    return np.mean(indicators) if indicators else 0.0

def compute_conditional_coverage(test_data, filtered_results, grouper, alpha=0.1, binning_method='quartiles'):
    """Compute conditional coverage for subgroups"""
    
    combined_data = []
    for orig_sample, filtered_sample in zip(test_data, filtered_results):
        combined_sample = dict(orig_sample['sample'])
        combined_sample['scores'] = orig_sample['scores']
        combined_sample['filtered_claims'] = filtered_sample.get('filtered_claims', [])
        combined_data.append(combined_sample)
    
    method_mapping = {
        'quantile': 'tertiles',
        'equal_width': 'tertiles',
        'quartiles': 'tertiles'
    }
    method = method_mapping.get(binning_method, 'tertiles')
    
    values = grouper.compute_values(combined_data)
    
    if len(values) == 0:
        logging.warning(f"    ⚠️ Warning: {grouper.__class__.__name__} returned no values")
        return {}
    
    if np.all(values == values[0]):
        logging.warning(f"    ⚠️ Warning: {grouper.__class__.__name__} all values identical ({values[0]:.4f})")
    
    bins = grouper.create_bins(values, method=method)
    
    groups = {}
    group_names = ['low', 'medium', 'high'] if len(bins) == 3 else [f'bin_{i}' for i in range(len(bins))]
    
    for i, (bin_min, bin_max) in enumerate(bins):
        if i == len(bins) - 1:
            mask = (values >= bin_min) & (values <= bin_max)
        else:
            mask = (values >= bin_min) & (values < bin_max)
        
        indices = np.where(mask)[0].tolist()
        bin_name = group_names[i] if i < len(group_names) else f'bin_{i}'
        groups[bin_name] = indices
    
    results = {}
    for group_name, indices in groups.items():
        if len(indices) == 0:
            continue
            
        group_indicators = []
        group_total_claims = 0
        group_retained_claims = 0
        
        for idx in indices:
            filtered_sample = filtered_results[idx]
            retained = filtered_sample.get('filtered_claims', [])
            original_claims = test_data[idx]['sample'].get('atomic_facts', [])
            
            has_false = any(not claim.get('is_supported', False) for claim in retained)
            group_indicators.append(0.0 if has_false else 1.0)
            
            group_total_claims += len(original_claims)
            group_retained_claims += len(retained)
        
        coverage = np.mean(group_indicators) if group_indicators else 0.0
        retention_rate = group_retained_claims / group_total_claims if group_total_claims > 0 else 0.0
        results[group_name] = {
            'size': len(indices),
            'coverage': coverage,
            'retention_rate': retention_rate,
            'retained_claims': group_retained_claims,
            'total_claims': group_total_claims,
            'target_coverage': 1 - alpha,
        }
    
    return results


def save_aggregated_results_to_json(results: Dict, args: argparse.Namespace):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_output_dir = os.path.join(repo_root, 'analysis', 'experiment_results')
    output_dir = getattr(args, 'time_out', None) or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    groups_str = "_".join(sorted(args.conditional_groups))
    filename = f"results_{args.dataset_type}_{args.model_set}_{groups_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    logging.info(f"\n💾 Saving aggregated results to {filepath}...")

    def convert_to_native_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    keys_to_exclude = {'filtered_results', 'beta', 'weights', 'budgets', 'calib_sizes', 'split_seed'}

    serializable_data = {}
    for method_name, method_data in results.items():
        serializable_data[method_name] = {}
        for key, value in method_data.items():
            if key in keys_to_exclude:
                continue
            

            try:
                cleaned_value = json.loads(json.dumps(value, default=convert_to_native_types))
                serializable_data[method_name][key] = cleaned_value
            except Exception as e:
                logging.warning(f"Could not serialize key '{key}' for method '{method_name}'. Skipping. Error: {e}")

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        logging.info(f"✅ Successfully saved results.")
    except Exception as e:
        logging.error(f"❌ Failed to save results to JSON: {e}")

def main():
    parser = argparse.ArgumentParser(description="Experiment with three conformal methods")
    parser.add_argument("--random-seed", type=int, default=123, help="Random seed")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (defaults to repo_root/data)")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save logs (defaults to repo_root/logs)")
    parser.add_argument("--dataset-type", type=str, default="auto", choices=["auto", "wikibio", "medlfqa"],
                       help="Dataset type (auto-detected if not specified)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Significance level (fixed if --adaptive-alpha is false)")
    parser.add_argument("--adaptive-alpha", action='store_true', help="Enable per-sample adaptive alpha (learn q*(z) for retention target)")
    parser.add_argument("--retention-target", type=float, default=0.4, help="Target retention used to learn adaptive alpha")
    parser.add_argument("--scores-dir", type=str, default=None, help="Directory containing final NPZ score files (optional)")
    parser.add_argument("--calib-ratio", type=float, default=0.75, help="Calibration set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--boosting-epochs", type=int, default=100, help="Boosting epochs")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of repeated runs with different random splits")
    parser.add_argument("--model-set", type=str, default="fixed", choices=["fixed"], help="Model set (fixed 3 models)")
    parser.add_argument("--bcp-score-type", type=str, default="frequency",
                       choices=['frequency', 'selfeval', 'logprob', 'ensemble'],
                       help="Score type for BCI")
    # --as-score-type removed; MACI uses ensemble by default
    parser.add_argument("--as-mode", type=str, default="subgroup_optimized", choices=["standard", "subgroup_optimized"], help="AS variant")
    parser.add_argument("--conditional-groups", type=str, nargs='*',
                       default=['false_claim_risk','medicalcontent','view_count'],
                       choices=['false_claim_risk','medicalcontent','view_count'],
                       help="Conditional groups to analyze")
    parser.add_argument("--view-metadata-csv", type=str, default=None,
                       help="Optional CSV for view_count grouper; defaults to repo-relative data path")
    parser.add_argument("--binning-method", type=str, default="quantile",
                       choices=['quantile', 'equal_width'],
                       help="Binning method for conditional groups")
    parser.add_argument("--limit-samples", type=int, default=2000, help="Max number of samples to load")
    parser.add_argument("--target-tpr", type=float, default=0.8, help="Target TPR for subgroup-optimized AS")

    parser.add_argument("--time-profile", action='store_true', help="Enable timing profile output")
    parser.add_argument("--time-out", type=str, default=None, help="Directory to save timing JSON (defaults to repo_root/analysis/experiment_results)")

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not args.data_dir:
        args.data_dir = os.path.join(repo_root, 'data')
    if not args.log_dir:
        args.log_dir = os.path.join(repo_root, 'logs')
    if not getattr(args, 'time_out', None):
        args.time_out = os.path.join(repo_root, 'analysis', 'experiment_results')

    setup_logging(args.log_dir)
    if args.view_metadata_csv:
        set_view_metadata_csv(args.view_metadata_csv)
    
    logging.info("=" * 80)
    logging.info(f"📊 Setup: {args.calib_ratio*100:.0f}% calibration + {args.test_ratio*100:.0f}% test, α={args.alpha}, adaptive={args.adaptive_alpha}")
    logging.info(f"🔄 Number of runs: {args.n_runs}")
    logging.info(f"🏷️  BCI Score: {args.bcp_score_type}")
    logging.info(f"🧠  CCI: enabled")
    logging.info(f"🎯 MACI: enabled")
    logging.info(f"📊 Conditional groups: {args.conditional_groups}")
    logging.info(f"🔧 Binning Method: {args.binning_method}")
    logging.info(f"🆕 Using provided scores and enhanced features")
    
    limit_samples = args.limit_samples
    data = load_1000_samples(args.data_dir, scores_dir=args.scores_dir, dataset_type=args.dataset_type, limit_samples=limit_samples)
    
    from collections import defaultdict
    all_runs_results = defaultdict(lambda: defaultdict(list))
    
    groupers = []
    available_groupers = get_available_groupers()
    for group_name in args.conditional_groups:
        if group_name in available_groupers:
            groupers.append(available_groupers[group_name])
        else:
            logging.warning(f"⚠️ Unknown grouper: {group_name}")
    
    detected_dataset_type = args.dataset_type
    if detected_dataset_type == "auto":
        if data and 'scores' in data[0] and isinstance(data[0]['scores'].get('frequency'), np.ndarray):
            detected_dataset_type = 'medlfqa'
        else:
            detected_dataset_type = 'wikibio'
    logging.info(f"➡️ Using detected dataset type: {detected_dataset_type}")
    
    factscore_npz_path = None
    if detected_dataset_type == 'wikibio':
        wikibio_npz_path = os.path.join(args.data_dir, "wiki_scores", "wikibio_final_frequencies.npz")

    model_names_to_use = MODEL_NAMES
    logging.info(f" MACI Models: {', '.join(model_names_to_use)}")

    for run_idx in range(args.n_runs):
        logging.info(f"\n🔄 Run {run_idx + 1}/{args.n_runs}")
        logging.info("-" * 50)
        
        random_seed = args.random_seed + run_idx
        calib_data, test_data, calib_idx, test_idx = create_splits(
            data, args.calib_ratio, args.test_ratio, random_seed=random_seed
        )
        
        logging.info(f"📊 Run {run_idx + 1} sizes: {len(calib_data)} calib, {len(test_data)} test (seed: {random_seed})")
        
        results = {}
        
        try:
            results['BCI'] = run_bcp_experiment(calib_data, test_data, score_type=args.bcp_score_type, alpha=args.alpha)
        except Exception as e:
            logging.error(f"❌ BCP failed: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            results['BCI'] = None
        
        try:
            results['CCI'] = run_cci_experiment(
                calib_data, test_data,
                alpha=args.alpha,
                boosting_epochs=args.boosting_epochs,
                adaptive_alpha=args.adaptive_alpha,
                retention_target=args.retention_target
            )
        except Exception as e:
            logging.error(f"❌ CCI failed: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            results['CCI'] = None
            
        results['MACI'] = {
            'coverage': [], 
            'retention_rate': [],
            'retained_claims': [],
            'total_claims': [],
            'subgroup_results': {}
        }

        logging.info("--- Starting MACI Experiments ---")
        mace_marginal_results_set = False

        for subgroup_name in args.conditional_groups:
            try:
                mace_subgroup_result = run_as_experiment(
                    calib_data, test_data,
                    model_names=model_names_to_use,
                    alpha=args.alpha, 
                    as_mode='subgroup_optimized', 
                    subgroup_name=subgroup_name,
                    random_state=random_seed,
                    target_tpr=args.target_tpr
                )

                if mace_subgroup_result and 'filtered_results' in mace_subgroup_result:
                    flat_filtered_results = []
                    for res in mace_subgroup_result['filtered_results']:
                        flat_item = dict(res.get('sample', {}))
                        flat_item['filtered_claims'] = res.get('sample', {}).get('filtered_claims', [])
                        flat_filtered_results.append(flat_item)
                    mace_subgroup_result['filtered_results'] = flat_filtered_results

                if not mace_marginal_results_set:
                    results['MACI']['coverage'] = mace_subgroup_result['coverage']
                    results['MACI']['retention_rate'] = mace_subgroup_result['retention_rate']
                    results['MACI']['retained_claims'] = mace_subgroup_result['retained_claims']
                    results['MACI']['total_claims'] = mace_subgroup_result['total_claims']
                    results['MACI']['filtered_results'] = mace_subgroup_result.get('filtered_results', [])
                    results['MACI']['timing'] = mace_subgroup_result.get('timing', {})

                    mace_marginal_results_set = True

                target_grouper = available_groupers.get(subgroup_name)
                if target_grouper:
                    try:
                        conditional_results = compute_conditional_coverage(
                            test_data, 
                            mace_subgroup_result['filtered_results'], 
                            target_grouper, 
                            args.alpha, 
                            args.binning_method
                        )
                        results['MACI']['subgroup_results'][target_grouper.__class__.__name__] = conditional_results
                    except Exception as e:
                        logging.error(f"    ❌ MACI subgroup analysis for {target_grouper.__class__.__name__} failed: {e}")

            except Exception as e:
                logging.error(f"❌ MACI ({subgroup_name}) failed: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")

                            
        for method_name, result in results.items():
            if not result or result.get('coverage') is None:
                continue

            all_runs_results[method_name]['coverage'].append(result['coverage'])
            all_runs_results[method_name]['retention_rate'].append(result['retention_rate'])
            all_runs_results[method_name]['retained_claims'].append(result['retained_claims'])
            all_runs_results[method_name]['total_claims'].append(result['total_claims'])
            
            run_subgroup_results = {}
            if method_name == 'MACI':
                run_subgroup_results = result.get('subgroup_results', {})
            else:
                for grouper in groupers:
                    try:
                        conditional_results = compute_conditional_coverage(
                            test_data, 
                            result['filtered_results'], 
                            grouper, 
                            args.alpha, 
                            args.binning_method
                        )
                        run_subgroup_results[grouper.__class__.__name__] = conditional_results
                    except Exception as e:
                        logging.error(f"    ❌ {grouper.__class__.__name__} failed for {method_name}: {e}")
    
            all_runs_results[method_name]['subgroup_results'].append(run_subgroup_results)
         
                       
        logging.info(f"\n📊 Run {run_idx + 1} Results:")
        for method_name, result in results.items():
            if not result or result.get('coverage') is None:
                logging.info(f"  {method_name}: ❌ FAILED or SKIPPED")
                continue
            logging.info(f"  {method_name}: Coverage={result['coverage']:.4f}, Retention={result['retention_rate']:.3f}, Claims={result['retained_claims']}/{result['total_claims']}")

        if args.time_profile:
            timing_payload = {
                'dataset_type': detected_dataset_type,
                'model_set': args.model_set,
                'boosting_epochs': args.boosting_epochs,
                'adaptive_alpha': args.adaptive_alpha,
                'retention_target': args.retention_target,
                'run_idx': run_idx,
                'CCI': results.get('CCI', {}).get('timing', {}),
                'MACI': {}
            }
            try:
                first_subgroup = next(iter(results['MACI'].get('subgroup_results', {}).keys()), None)
                if first_subgroup:
                    mace_timing = None
                    mace_timing = results['MACI'].get('timing')
                    timing_payload['MACI'] = mace_timing if mace_timing else {}
            except Exception:
                pass

            if not timing_payload['MACI']:
                try:
                    timing_payload['MACI'] = {}
                except Exception:
                    timing_payload['MACI'] = {}

            os.makedirs(args.time_out, exist_ok=True)
            tstamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            timing_path = os.path.join(args.time_out, f"time_profile_{detected_dataset_type}_{args.model_set}_{tstamp}.json")
            with open(timing_path, 'w', encoding='utf-8') as f:
                json.dump(timing_payload, f, indent=2, ensure_ascii=False)
            logging.info(f"⏱️ Saved timing profile to {timing_path}")


        if run_idx == 0 and getattr(args, 'show_sample_idx', None) is not None and args.show_sample_idx >= 0:
            idx = int(args.show_sample_idx)
            if 0 <= idx < len(test_data):
                def _get_claim_text(c: Dict[str, Any]) -> str:
                    if not isinstance(c, dict):
                        return str(c)
                    return c.get('atom') or c.get('text') or c.get('claim') or c.get('fact') or str(c)
                def _get_claim_support(c: Dict[str, Any]) -> str:
                    if isinstance(c, dict):
                        v = c.get('is_supported')
                        if isinstance(v, (bool, np.bool_)):
                            return 'T' if bool(v) else 'F'
                    return '?'

                sample = test_data[idx]['sample']
                prompt = sample.get('prompt', '')
                response = sample.get('response', '')
                original_claims = sample.get('atomic_facts', [])
                original_pairs = [(_get_claim_text(c), _get_claim_support(c)) for c in original_claims]

                bci_item = results.get('BCI', {}).get('filtered_results', [None]*len(test_data))[idx]
                cci_item = results.get('CCI', {}).get('filtered_results', [None]*len(test_data))[idx]
                mace_item = results.get('MACE', {}).get('filtered_results', [None]*len(test_data))[idx]

                def _filtered_claims(item):
                    if not item:
                        return []
                    claims = item.get('filtered_claims')
                    if claims is None and isinstance(item.get('sample'), dict):
                        claims = item['sample'].get('filtered_claims', [])
                    return [(_get_claim_text(c), _get_claim_support(c)) for c in (claims or [])]

                logging.info("\n=== SAMPLE CLAIMS DUMP ===")
                logging.info(f"[Test idx={idx}] Prompt: {prompt}")
                logging.info(f"Original claims ({len(original_pairs)}):")
                for i, (t, lab) in enumerate(original_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")

                bci_pairs = _filtered_claims(bci_item)
                cci_pairs = _filtered_claims(cci_item)
                mace_pairs = _filtered_claims(mace_item)

                logging.info(f"\n[BCI] filtered claims ({len(bci_pairs)}):")
                for i, (t, lab) in enumerate(bci_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")

                logging.info(f"\n[CCI] filtered claims ({len(cci_pairs)}):")
                for i, (t, lab) in enumerate(cci_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")

                logging.info(f"\n[MACI] filtered claims ({len(mace_pairs)}):")
                for i, (t, lab) in enumerate(mace_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")

        if run_idx == 0 and getattr(args, 'show_sample_count', 0) > 0:
            dump_n = min(int(args.show_sample_count), len(test_data))
            def _get_claim_text(c: Dict[str, Any]) -> str:
                if not isinstance(c, dict):
                    return str(c)
                return c.get('atom') or c.get('text') or c.get('claim') or c.get('fact') or str(c)
            def _get_claim_support(c: Dict[str, Any]) -> str:
                if isinstance(c, dict):
                    v = c.get('is_supported')
                    if isinstance(v, (bool, np.bool_)):
                        return 'T' if bool(v) else 'F'
                return '?'
            def _filtered_pairs(item):
                if not item:
                    return []
                claims = item.get('filtered_claims')
                if claims is None and isinstance(item.get('sample'), dict):
                    claims = item['sample'].get('filtered_claims', [])
                return [(_get_claim_text(c), _get_claim_support(c)) for c in (claims or [])]
            for idx in range(dump_n):
                sample = test_data[idx]['sample']
                prompt = sample.get('prompt', '')
                original_pairs = [(_get_claim_text(c), _get_claim_support(c)) for c in sample.get('atomic_facts', [])]
                bci_item = results.get('BCI', {}).get('filtered_results', [None]*len(test_data))[idx]
                cci_item = results.get('CCI', {}).get('filtered_results', [None]*len(test_data))[idx]
                mace_item = results.get('MACE', {}).get('filtered_results', [None]*len(test_data))[idx]
                logging.info("\n=== SAMPLE CLAIMS DUMP ===")
                logging.info(f"[Test idx={idx}] Prompt: {prompt}")
                logging.info(f"Original claims ({len(original_pairs)}):")
                for i, (t, lab) in enumerate(original_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")
                bci_pairs = _filtered_pairs(bci_item)
                cci_pairs = _filtered_pairs(cci_item)
                mace_pairs = _filtered_pairs(mace_item)
                logging.info(f"\n[BCI] filtered claims ({len(bci_pairs)}):")
                for i, (t, lab) in enumerate(bci_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")
                logging.info(f"\n[CCI] filtered claims ({len(cci_pairs)}):")
                for i, (t, lab) in enumerate(cci_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")
                logging.info(f"\n[MACI] filtered claims ({len(mace_pairs)}):")
                for i, (t, lab) in enumerate(mace_pairs, 1):
                    logging.info(f"  {i:2d}. [{lab}] {t}")
    
    logging.info("\n" + "=" * 100)
    logging.info("📊 AGGREGATED RESULTS (All Runs)")
    logging.info("=" * 100)
    for method_name in sorted(all_runs_results.keys()):
        method_results = all_runs_results[method_name]
        
        if not method_results['coverage']:
            logging.info(f"\n{method_name}: ❌ NO SUCCESSFUL RUNS")
            continue
        
        n_runs = len(method_results['coverage'])
        coverage_mean = np.mean(method_results['coverage'])
        coverage_std = np.std(method_results['coverage'])
        retention_mean = np.mean(method_results['retention_rate'])
        retention_std = np.std(method_results['retention_rate'])
        retained_claims_mean = np.mean(method_results['retained_claims'])
        retained_claims_std = np.std(method_results['retained_claims'])
        total_claims_mean = np.mean(method_results['total_claims'])
        
        logging.info(f"\n{'='*20} {method_name} ({n_runs} runs) {'='*20}")
        logging.info(f"📈 MARGINAL RESULTS:")
        logging.info(f"  Coverage: {coverage_mean:.4f} ± {coverage_std:.4f}")
        logging.info(f"  Retention Rate: {retention_mean:.3f} ± {retention_std:.3f}")
        logging.info(f"  Claims: {retained_claims_mean:.1f} ± {retained_claims_std:.1f}/{total_claims_mean:.1f}")
        
        if method_results['subgroup_results']:
            logging.info(f"\n📊 SUBGROUP RESULTS:")
            
            subgroup_data = {}
            for run_results in method_results['subgroup_results']:
                for grouper_name, grouper_results in run_results.items():
                    if grouper_name not in subgroup_data:
                        subgroup_data[grouper_name] = {}
                    
                    for group_name, group_result in grouper_results.items():
                        if group_name not in subgroup_data[grouper_name]:
                            subgroup_data[grouper_name][group_name] = {
                                'coverage': [], 'retention_rate': [], 'retained_claims': [],
                                'total_claims': [], 'size': []
                            }
                        
                        subgroup_data[grouper_name][group_name]['coverage'].append(group_result['coverage'])
                        subgroup_data[grouper_name][group_name]['retention_rate'].append(group_result['retention_rate'])
                        subgroup_data[grouper_name][group_name]['retained_claims'].append(group_result['retained_claims'])
                        subgroup_data[grouper_name][group_name]['total_claims'].append(group_result['total_claims'])
                        subgroup_data[grouper_name][group_name]['size'].append(group_result['size'])
            
            for grouper_name, groups in subgroup_data.items():
                logging.info(f"\n  🔍 {grouper_name}:")
                
                for group_name, group_data in groups.items():
                    if not group_data['coverage']:
                        continue
                    
                    group_coverage_mean = np.mean(group_data['coverage'])
                    group_coverage_std = np.std(group_data['coverage'])
                    group_retention_mean = np.mean(group_data['retention_rate'])
                    group_retention_std = np.std(group_data['retention_rate'])
                    group_retained_claims_mean = np.mean(group_data['retained_claims'])
                    group_retained_claims_std = np.std(group_data['retained_claims'])
                    group_total_claims_mean = np.mean(group_data['total_claims'])
                    group_size_mean = np.mean(group_data['size'])
                    
                    target_coverage = 1 - args.alpha
                    violation_marker = "⚠️ " if abs(group_coverage_mean - target_coverage) > 0.014 else "✅ "
                    
                    logging.info(f"    {violation_marker}{group_name}:")
                    logging.info(f"      Coverage: {group_coverage_mean:.3f} ± {group_coverage_std:.3f} (target: {target_coverage:.1f})")
                    logging.info(f"      Retention: {group_retention_mean:.3f} ± {group_retention_std:.3f}")
                    logging.info(f"      Claims: {group_retained_claims_mean:.1f} ± {group_retained_claims_std:.1f}/{group_total_claims_mean:.1f}")
                    logging.info(f"      Group size: {group_size_mean:.1f} samples")
                    logging.info(f"      Coverage gap: {group_coverage_mean - target_coverage:+.3f}")
        

    
    logging.info("\n" + "=" * 100)

    save_aggregated_results_to_json(all_runs_results, args)


if __name__ == "__main__":
    main()
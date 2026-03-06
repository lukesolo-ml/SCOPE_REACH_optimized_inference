"""
Engine initialization for SCOPE and REACH.

Reduces the risk of user error by calling SGLang engines for a given task
"""

import sglang as sgl

def standard_engine(model_path, max_len, use_time_horizon = False):
    return sgl.Engine(
        model_path=model_path,
        skip_tokenizer=True,
        disable_overlap_scheduler = use_time_horizon,
        enable_custom_logit_processor= use_time_horizon,
        context_length=max_len,
    )
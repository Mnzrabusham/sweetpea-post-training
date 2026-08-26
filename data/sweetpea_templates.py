"""
SweetPea Experiment Generator Templates.

Defines programmatic generators for diverse cognitive and perceptual experiment
paradigms in SweetPea DSL, parameterizable across factor names, level sets,
derivation predicates, and sequence constraints.
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class SweetPeaExample:
    prompt: str
    code: str
    task_family: str
    complexity: str
    constraint_types: List[str]


# Factor vocabularies
PERCEPTUAL_FACTORS = [
    ("color", ["red", "green", "blue", "yellow"]),
    ("shape", ["circle", "square", "triangle", "diamond"]),
    ("orientation", ["up", "down", "left", "right"]),
    ("size", ["small", "medium", "large"]),
    ("contrast", ["low", "medium", "high"]),
    ("spatial_frequency", ["low_freq", "mid_freq", "high_freq"]),
    ("motion_direction", ["leftward", "rightward", "upward", "downward"]),
    ("luminance", ["dark", "neutral", "bright"]),
]

COGNITIVE_FACTORS = [
    ("word_text", ["red", "green", "blue", "yellow"]),
    ("font_color", ["red", "green", "blue", "yellow"]),
    ("target_direction", ["left", "right"]),
    ("flanker_direction", ["left", "right"]),
    ("task_type", ["color_naming", "word_reading"]),
    ("modality", ["visual", "auditory"]),
    ("valence", ["positive", "neutral", "negative"]),
    ("arousal", ["low", "high"]),
    ("cue_validity", ["valid", "invalid", "neutral"]),
    ("stimulus_location", ["left_visual_field", "right_visual_field"]),
    ("response_hand", ["left_hand", "right_hand"]),
    ("memory_load", ["low_load", "high_load"]),
    ("feedback_type", ["reward", "punishment", "none"]),
    ("digit_value", ["one", "two", "three", "four"]),
    ("letter_case", ["uppercase", "lowercase"]),
]


# ==============================================================================
# 1. Factorial Crossings (2-way, 3-way, 4-way)
# ==============================================================================
def gen_factorial_crossing(rng: random.Random) -> SweetPeaExample:
    num_factors = rng.choice([2, 3, 4])
    selected_pools = rng.sample(PERCEPTUAL_FACTORS + COGNITIVE_FACTORS, num_factors)
    
    factor_defs = []
    factor_var_names = []
    factor_desc = []
    
    for fname, flevels in selected_pools:
        # Sample 2 to 4 levels
        k = rng.randint(2, min(4, len(flevels)))
        chosen_levels = rng.sample(flevels, k)
        var_name = fname.replace("_", "")
        factor_defs.append(f"{var_name} = sp.Factor(\"{fname}\", {chosen_levels})")
        factor_var_names.append(var_name)
        factor_desc.append(f"'{fname}' with levels {chosen_levels}")
        
    design_str = f"[{', '.join(factor_var_names)}]"
    crossing_str = f"[{', '.join(factor_var_names)}]"
    
    # Optional repetition or minimum trials
    has_min_trials = rng.random() < 0.4
    min_trials = rng.choice([16, 24, 32, 48]) if has_min_trials else None
    
    constraints_code = []
    constraints_desc = []
    constraint_types = ["crossing"]
    
    if has_min_trials:
        constraints_code.append(f"sp.MinimumTrials({min_trials})")
        constraints_desc.append(f"require at least {min_trials} trials")
        constraint_types.append("minimum_trials")
        
    constraints_str = f"[{', '.join(constraints_code)}]"
    
    code = f"""import sweetpea as sp

# Define factors
{chr(10).join(factor_defs)}

# Define design and crossing
design = {design_str}
crossing = {crossing_str}
constraints = {constraints_str}

# Build block and synthesize
block = sp.CrossBlock(design, crossing, constraints)
experiments = sp.synthesize_trials(block, 1)
"""

    prompt_templates = [
        f"Create a SweetPea experiment with a full factorial crossing of {', '.join(factor_desc)}." +
        (f" The design must {', '.join(constraints_desc)}." if constraints_desc else "") +
        " Synthesize 1 trial sequence.",
        f"Write a Python script using SweetPea to generate a factorial design across {len(selected_pools)} factors: {'; '.join(factor_desc)}." +
        (f" Include the constraint to {', '.join(constraints_desc)}." if constraints_desc else "") +
        " Define the CrossBlock and synthesize the trials.",
        f"Using SweetPea DSL, define factors for {', '.join(factor_desc)}, cross all factors completely," +
        (f" ensure {', '.join(constraints_desc)}," if constraints_desc else "") +
        " and run synthesize_trials."
    ]
    
    return SweetPeaExample(
        prompt=rng.choice(prompt_templates),
        code=code.strip(),
        task_family="factorial_crossing",
        complexity="simple" if num_factors == 2 else "medium",
        constraint_types=constraint_types
    )


# ==============================================================================
# 2. Stroop / Flanker / Congruency Paradigms (WithinTrial Derivations)
# ==============================================================================
def gen_congruency_paradigm(rng: random.Random) -> SweetPeaExample:
    paradigm_type = rng.choice(["stroop", "flanker", "simon"])
    
    if paradigm_type == "stroop":
        dim1_name, dim1_levels = "word", ["red", "green", "blue"]
        dim2_name, dim2_levels = "color", ["red", "green", "blue"]
        f1_var, f2_var = "word", "color"
        paradigm_name = "Stroop"
        f1_desc = f"text word ('{dim1_name}')"
        f2_desc = f"font color ('{dim2_name}')"
    elif paradigm_type == "flanker":
        dim1_name, dim1_levels = "target", ["left", "right"]
        dim2_name, dim2_levels = "flanker", ["left", "right"]
        f1_var, f2_var = "target", "flanker"
        paradigm_name = "Eriksen Flanker"
        f1_desc = f"central target direction ('{dim1_name}')"
        f2_desc = f"surrounding flanker direction ('{dim2_name}')"
    else:  # simon
        dim1_name, dim1_levels = "stimulus_side", ["left", "right"]
        dim2_name, dim2_levels = "response_key", ["left", "right"]
        f1_var, f2_var = "stimside", "respkey"
        paradigm_name = "Simon task"
        f1_desc = f"stimulus presentation side ('{dim1_name}')"
        f2_desc = f"correct response key ('{dim2_name}')"
        
    # Optional constraint: AtMostKInARow on congruent trials
    has_at_most_k = rng.random() < 0.6
    k_val = rng.choice([1, 2]) if has_at_most_k else None
    
    constraint_code = []
    constraint_desc = []
    constraint_types = ["within_trial_derivation"]
    
    if has_at_most_k:
        constraint_code.append(f"sp.AtMostKInARow({k_val}, congruency['congruent'])")
        constraint_desc.append(f"no more than {k_val} congruent trials in a row")
        constraint_types.append("at_most_k_in_a_row")
        
    constraints_str = f"[{', '.join(constraint_code)}]"
    
    code = f"""import sweetpea as sp

# Base factors
{f1_var} = sp.Factor("{dim1_name}", {dim1_levels})
{f2_var} = sp.Factor("{dim2_name}", {dim2_levels})

# Congruency derivation predicates
def is_congruent(f1, f2):
    return f1 == f2

def is_incongruent(f1, f2):
    return f1 != f2

congruent_level = sp.DerivedLevel("congruent", sp.WithinTrial(is_congruent, [{f1_var}, {f2_var}]))
incongruent_level = sp.DerivedLevel("incongruent", sp.WithinTrial(is_incongruent, [{f1_var}, {f2_var}]))
congruency = sp.Factor("congruency", [congruent_level, incongruent_level])

design = [{f1_var}, {f2_var}, congruency]
crossing = [{f1_var}, {f2_var}]
constraints = {constraints_str}

block = sp.CrossBlock(design, crossing, constraints)
experiments = sp.synthesize_trials(block, 1)
"""

    prompt_templates = [
        f"Implement a {paradigm_name} experiment in SweetPea DSL. Cross {f1_desc} ({dim1_levels}) with {f2_desc} ({dim2_levels}). Derive a 'congruency' factor with 'congruent' and 'incongruent' levels using a WithinTrial window." +
        (f" Add a constraint enforcing {', '.join(constraint_desc)}." if constraint_desc else "") +
        " Synthesize 1 trial sequence.",
        f"Write a SweetPea script for a {paradigm_name} paradigm. Define factor '{dim1_name}' and factor '{dim2_name}'. Create a derived congruency factor where congruent is f1 == f2 and incongruent is f1 != f2." +
        (f" Constrain the sequence so that {', '.join(constraint_desc)}." if constraint_desc else "") +
        " Synthesize the experiment block.",
    ]
    
    return SweetPeaExample(
        prompt=rng.choice(prompt_templates),
        code=code.strip(),
        task_family="stroop_congruency" if paradigm_type == "stroop" else "flanker_task",
        complexity="medium",
        constraint_types=constraint_types
    )


# ==============================================================================
# 3. Task-Switching Paradigms (Transition Derivations)
# ==============================================================================
def gen_task_switching_paradigm(rng: random.Random) -> SweetPeaExample:
    tasks = rng.choice([
        ("task", ["color_task", "shape_task"]),
        ("task", ["magnitude_judgment", "parity_judgment"]),
        ("task", ["vowel_consonant", "case_judgment"]),
    ])
    task_name, task_levels = tasks
    
    # Secondary stimulus factor
    stim_name, stim_levels = rng.choice([
        ("stimulus", ["letter_A", "letter_B"]),
        ("cue", ["left_cue", "right_cue"]),
        ("color", ["red", "blue"]),
    ])
    
    max_repeats = rng.choice([2, 3])
    min_trials = rng.choice([8, 12, 16])
    
    code = f"""import sweetpea as sp

# Factors
task = sp.Factor("{task_name}", {task_levels})
stim = sp.Factor("{stim_name}", {stim_levels})

# Transition derivation (between-trial dynamics)
def is_repeat(t):
    return t[-1] == t[0]

def is_switch(t):
    return t[-1] != t[0]

repeat_level = sp.DerivedLevel("repeat", sp.Transition(is_repeat, [task]))
switch_level = sp.DerivedLevel("switch", sp.Transition(is_switch, [task]))
transition_factor = sp.Factor("task_transition", [repeat_level, switch_level])

design = [task, stim, transition_factor]
crossing = [task, stim]
constraints = [
    sp.AtMostKInARow({max_repeats}, transition_factor["repeat"]),
    sp.MinimumTrials({min_trials})
]

block = sp.CrossBlock(design, crossing, constraints)
experiments = sp.synthesize_trials(block, 1)
"""

    prompt = (
        f"Design a task-switching experiment using SweetPea. Factor '{task_name}' has levels {task_levels} and factor '{stim_name}' has levels {stim_levels}. "
        f"Define a derived transition factor 'task_transition' with 'repeat' (task[-1] == task[0]) and 'switch' (task[-1] != task[0]) levels using Transition windows. "
        f"Constrain the experiment to have at most {max_repeats} repeat trials in a row and a minimum of {min_trials} trials. "
        f"Cross '{task_name}' and '{stim_name}', construct the CrossBlock, and synthesize 1 experiment."
    )
    
    return SweetPeaExample(
        prompt=prompt,
        code=code.strip(),
        task_family="task_switching_transition",
        complexity="complex",
        constraint_types=["transition_derivation", "at_most_k_in_a_row", "minimum_trials"]
    )


# ==============================================================================
# 4. N-Back / History Paradigms (Window Derivations)
# ==============================================================================
def gen_nback_paradigm(rng: random.Random) -> SweetPeaExample:
    stim_name, stim_levels = "letter", ["A", "B", "C"]
    n_back = rng.choice([1, 2])
    min_trials = rng.choice([12, 18])
    
    offset = -n_back
    window_width = n_back + 1
    
    code = f"""import sweetpea as sp

stimulus = sp.Factor("{stim_name}", {stim_levels})

def is_nback_match(s):
    return s[{offset}] == s[0]

def is_nback_mismatch(s):
    return s[{offset}] != s[0]

match_lvl = sp.DerivedLevel("match", sp.Window(is_nback_match, [stimulus], {window_width}, 1))
mismatch_lvl = sp.DerivedLevel("mismatch", sp.Window(is_nback_mismatch, [stimulus], {window_width}, 1))
nback_factor = sp.Factor("{n_back}_back", [match_lvl, mismatch_lvl])

design = [stimulus, nback_factor]
crossing = [stimulus]
constraints = [sp.MinimumTrials({min_trials})]

block = sp.CrossBlock(design, crossing, constraints)
experiments = sp.synthesize_trials(block, 1)
"""

    prompt = (
        f"Generate a {n_back}-back working memory experiment in SweetPea DSL. "
        f"Create a '{stim_name}' factor with levels {stim_levels}. "
        f"Derive a '{n_back}_back' factor with 'match' and 'mismatch' levels using a Window derivation of width {window_width} (stride 1) checking if stimulus[{offset}] == stimulus[0]. "
        f"Set MinimumTrials to {min_trials}, build the CrossBlock crossing '{stim_name}', and synthesize trials."
    )
    
    return SweetPeaExample(
        prompt=prompt,
        code=code.strip(),
        task_family="nback_window",
        complexity="complex",
        constraint_types=["window_derivation", "minimum_trials"]
    )


# ==============================================================================
# 5. Constrained Design with Exclude & Pin
# ==============================================================================
def gen_constrained_exclude_pin(rng: random.Random) -> SweetPeaExample:
    f1_name, f1_levels = "modality", ["visual", "auditory", "tactile"]
    f2_name, f2_levels = "intensity", ["low", "medium", "high"]
    
    exclude_target = rng.choice(f1_levels)
    pin_target = rng.choice(f2_levels)
    alt_mod = "visual" if exclude_target == "auditory" else "auditory"
    
    code = f"""import sweetpea as sp

modality = sp.Factor("{f1_name}", {f1_levels})
intensity = sp.Factor("{f2_name}", {f2_levels})

design = [modality, intensity]
crossing = [modality, intensity]
constraints = [
    sp.Exclude(modality["{exclude_target}"]),
    sp.Pin(1, intensity["{pin_target}"]),
    sp.AtMostKInARow(1, modality["{alt_mod}"])
]

# Set require_complete_crossing=False since levels are excluded
block = sp.CrossBlock(design, crossing, constraints, require_complete_crossing=False)
experiments = sp.synthesize_trials(block, 1)
"""

    alt_mod = "visual" if exclude_target == "auditory" else "auditory"
    prompt = (
        f"Create a SweetPea experiment crossing '{f1_name}' ({f1_levels}) and '{f2_name}' ({f2_levels}). "
        f"Apply the following constraints: exclude all trials with modality '{exclude_target}', "
        f"pin trial 1 to intensity '{pin_target}', and ensure at most 1 '{alt_mod}' modality trial in a row. "
        f"Construct the CrossBlock and synthesize the experiment."
    )
    
    return SweetPeaExample(
        prompt=prompt,
        code=code.strip(),
        task_family="constrained_design",
        complexity="medium",
        constraint_types=["exclude", "pin", "at_most_k_in_a_row"]
    )


# ==============================================================================
# 6. Partial Crossing with Additional Design Factors
# ==============================================================================
def gen_partial_crossing(rng: random.Random) -> SweetPeaExample:
    crossed_factors = rng.sample(PERCEPTUAL_FACTORS, 2)
    uncrossed_factor = rng.choice(COGNITIVE_FACTORS)
    
    cf1_name, cf1_levels = crossed_factors[0]
    cf2_name, cf2_levels = crossed_factors[1]
    uc_name, uc_levels = uncrossed_factor
    
    cf1_lev = rng.sample(cf1_levels, 2)
    cf2_lev = rng.sample(cf2_levels, 2)
    uc_lev = rng.sample(uc_levels, 2)
    
    v1, v2, v3 = cf1_name.replace("_", ""), cf2_name.replace("_", ""), uc_name.replace("_", "")
    
    code = f"""import sweetpea as sp

# Define primary crossed factors and uncrossed design factor
{v1} = sp.Factor("{cf1_name}", {cf1_lev})
{v2} = sp.Factor("{cf2_name}", {cf2_lev})
{v3} = sp.Factor("{uc_name}", {uc_lev})

design = [{v1}, {v2}, {v3}]
crossing = [{v1}, {v2}]  # Only {cf1_name} and {cf2_name} are fully crossed
constraints = [sp.AtMostKInARow(2, {v1}["{cf1_lev[0]}"])]

block = sp.CrossBlock(design, crossing, constraints)
experiments = sp.synthesize_trials(block, 1)
"""

    prompt = (
        f"Define a SweetPea experiment where '{cf1_name}' ({cf1_lev}) and '{cf2_name}' ({cf2_lev}) are in the crossing, "
        f"while '{uc_name}' ({uc_lev}) is included in the design but not the crossing. "
        f"Add a constraint that '{cf1_name}' level '{cf1_lev[0]}' appears at most 2 times in a row. "
        f"Build the CrossBlock and synthesize trials."
    )
    
    return SweetPeaExample(
        prompt=prompt,
        code=code.strip(),
        task_family="partial_crossing",
        complexity="medium",
        constraint_types=["partial_crossing", "at_most_k_in_a_row"]
    )


GENERATORS = [
    gen_factorial_crossing,
    gen_congruency_paradigm,
    gen_task_switching_paradigm,
    gen_nback_paradigm,
    gen_constrained_exclude_pin,
    gen_partial_crossing,
]


def generate_sample(rng: random.Random) -> SweetPeaExample:
    gen_fn = rng.choice(GENERATORS)
    return gen_fn(rng)


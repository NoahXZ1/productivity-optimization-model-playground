#!/usr/bin/env python3
"""
Paradigm Switch Dashboard
=========================

A deterministic command-line calculator for the log-skill / two-dimensional
paradigm-switching model.

Core model implemented
----------------------
1. Current within-paradigm productivity multiple:
       kappa >= 1
       x = ln(kappa)

2. Skill transfer:
       h in (0, 1]
       delta = -ln(h)        when alpha = 1
       x_after_switch = h * x_before
       kappa_after_switch = kappa_before ** h

3. Instantaneous switching ratio:
       V_plus / V_minus = Lambda * g * exp((h - 1) * x)
                         = Lambda * g * kappa ** (h - 1)

   where Lambda = c * S(tau), S(tau) = exp(-beta * tau),
   beta = ln(2) / half_life.

4. Long-horizon output uses the linear-productivity benchmark of the paper:
       kappa(t + s) = kappa(t) + learning_rate * s
       Integral output over duration D:
       O = multiplier * [kappa_start * D + 0.5 * learning_rate * D^2]

   This is the explicit benchmark version of the general path-dependent formula.
   It is not claimed to be the universal learning law; it is the safe closed-form
   default for a user-facing calculator.

5. Staged validation uses a two-state gain-transfer model:
       High state: (g_high, h_high), probability p
       Low state:  (g_low,  h_low),  probability 1 - p

   A noisy pilot has accuracy q, duration epsilon, direct cost C_e, and preserves
   rho fraction of old-paradigm production during the pilot.

Design principle
----------------
User inputs are simple. Mathematical computation is explicit, deterministic, and
free of LLM reasoning. No random sampling is used.

Run
---
Open this file in Visual Studio / VS Code / PyCharm and press Run.
The script will ask one question at a time:
    Please input current mastery...
    Please input expected gain...
    ...

No command-line argument bundle is required.

Author: generated for Noah Zhang's paradigm-switching model workflow.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

EPS = 1e-12
MAX_EXP_ARG = 709.0      # approximate safe upper bound for math.exp in double precision
MIN_EXP_ARG = -745.0     # below this, exp underflows to 0.0 in IEEE double precision


# -----------------------------------------------------------------------------
# User-facing presets
# -----------------------------------------------------------------------------

MASTERY_PRESETS: Dict[str, float] = {
    "novice": 1.0,
    "skilled": 2.0,
    "strong": 4.0,
    "expert": 8.0,
    "elite": 16.0,
}

GAIN_PRESETS: Dict[str, float] = {
    "worse": 0.8,
    "slightly_better": 1.2,
    "clearly_better": 1.5,
    "major": 2.0,
    "game_changing": 3.0,
    "explosive": 5.0,
}

TRANSFER_PRESETS: Dict[str, float] = {
    "almost_all": 0.90,
    "most": 0.75,
    "half": 0.50,
    "little": 0.25,
    "almost_none": 0.10,
}

WINDOW_HALF_LIFE_PRESETS: Dict[str, float] = {
    "not_urgent": 24.0,
    "slow": 12.0,
    "moderate": 6.0,
    "tight": 3.0,
    "very_tight": 1.0,
}

CONFIDENCE_PRESETS: Dict[str, float] = {
    "hype_only": 0.20,
    "possible": 0.40,
    "coin_flip": 0.50,
    "credible": 0.70,
    "high": 0.85,
    "verified": 0.95,
}


@dataclass(frozen=True)
class PilotPreset:
    """User-facing pilot-cost preset.

    epsilon: pilot duration in the same time unit as horizon, default months.
    direct_cost_months: direct cost expressed as equivalent months of current output speed.
    rho: fraction of old-paradigm production preserved during the pilot.
    """

    epsilon: float
    direct_cost_months: float
    rho: float


PILOT_COST_PRESETS: Dict[str, PilotPreset] = {
    "very_low": PilotPreset(epsilon=0.10, direct_cost_months=0.05, rho=0.90),
    "low": PilotPreset(epsilon=0.25, direct_cost_months=0.10, rho=0.85),
    "medium": PilotPreset(epsilon=0.75, direct_cost_months=0.35, rho=0.65),
    "high": PilotPreset(epsilon=1.50, direct_cost_months=0.75, rho=0.45),
    "very_high": PilotPreset(epsilon=3.00, direct_cost_months=1.50, rho=0.25),
}


# -----------------------------------------------------------------------------
# Dataclasses for model input and output
# -----------------------------------------------------------------------------

@dataclass
class ModelInputs:
    # Simple UI inputs
    kappa: float                         # current mastery multiple relative to novice
    g_high: float                        # high-state paradigm gain
    h_high: float                        # high-state skill-transfer exponent
    p_high: float                        # probability high state is true
    window_half_life: float              # time for niche coefficient S(t) to halve
    horizon: float                       # decision horizon, same unit as half-life
    pilot_cost_level: str = "medium"

    # Advanced parameters, all with safe defaults
    g_low: Optional[float] = None         # low-state gain; if None, derived conservatively
    h_low: Optional[float] = None         # low-state transfer; if None, derived conservatively
    current_delay: float = 0.0            # tau: delay since opportunity emerged
    coordination_efficiency: float = 1.0  # c in Lambda = c*S(tau)
    learning_rate: float = 1.0            # d kappa / dt in the linear benchmark
    pilot_accuracy: float = 0.75          # q: Pr(correct signal)
    pilot_duration: Optional[float] = None
    pilot_direct_cost: Optional[float] = None
    pilot_rho: Optional[float] = None
    parallel_share: float = 0.25          # capacity share allocated to new paradigm in parallel track


@dataclass
class DerivedQuantities:
    x: float
    delta_high: float
    delta_low: float
    beta: float
    niche_now: float
    lambda_now: float
    pilot_duration: float
    pilot_direct_cost: float
    pilot_rho: float
    pilot_decay_factor: float
    lambda_after_pilot: float
    g_low: float
    h_low: float


@dataclass
class PathValues:
    stay: float
    switch_now_expected: float
    switch_now_high: float
    switch_now_low: float
    pilot_first: float
    parallel_track: float
    no_experiment_best: float


@dataclass
class DecisionMetrics:
    immediate_pain_index_high: float
    immediate_pain_index_low: float
    no_pain_kappa_threshold_high: Optional[float]
    no_pain_x_threshold_high: Optional[float]
    critical_gain_for_no_pain: float
    critical_transfer_for_no_pain: Optional[float]
    belief_threshold: Optional[float]
    pilot_net_value_vs_no_experiment: float
    best_path: str
    recommendation: str
    next_action: str
    pain_label: str


@dataclass
class ModelResult:
    inputs: ModelInputs
    derived: DerivedQuantities
    path_values: PathValues
    relative_to_stay: Dict[str, float]
    metrics: DecisionMetrics
    warnings: List[str]


# -----------------------------------------------------------------------------
# Numeric utilities and validators
# -----------------------------------------------------------------------------

def fail(message: str) -> None:
    raise ValueError(message)


def ensure_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        fail(f"{name} must be finite; got {value!r}.")
    return value


def ensure_positive(value: float, name: str) -> float:
    ensure_finite(value, name)
    if value <= 0:
        fail(f"{name} must be positive; got {value!r}.")
    return value


def ensure_nonnegative(value: float, name: str) -> float:
    ensure_finite(value, name)
    if value < 0:
        fail(f"{name} must be nonnegative; got {value!r}.")
    return value


def ensure_probability(value: float, name: str, *, allow_zero_one: bool = True) -> float:
    ensure_finite(value, name)
    lo_ok = value >= 0.0 if allow_zero_one else value > 0.0
    hi_ok = value <= 1.0 if allow_zero_one else value < 1.0
    if not (lo_ok and hi_ok):
        bracket = "[0, 1]" if allow_zero_one else "(0, 1)"
        fail(f"{name} must be in {bracket}; got {value!r}.")
    return value




def ensure_fraction_positive(value: float, name: str) -> float:
    """Validate a fraction in (0, 1]."""
    ensure_finite(value, name)
    if value <= 0.0 or value > 1.0:
        fail(f"{name} must be in (0, 1]; got {value!r}.")
    return value


def safe_exp(z: float) -> float:
    """Overflow/underflow-safe exponential for reporting and ratios."""
    ensure_finite(z, "exponent")
    if z > MAX_EXP_ARG:
        return math.inf
    if z < MIN_EXP_ARG:
        return 0.0
    return math.exp(z)


def safe_div(num: float, den: float) -> float:
    if abs(den) < EPS:
        if abs(num) < EPS:
            return math.nan
        return math.inf if num > 0 else -math.inf
    return num / den


def fmt_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "∞" if x > 0 else "-∞"
    if abs(x) >= 10000 or (0 < abs(x) < 0.0001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


# -----------------------------------------------------------------------------
# Core mathematical functions
# -----------------------------------------------------------------------------

def log_skill(kappa: float) -> float:
    ensure_positive(kappa, "kappa")
    if kappa < 1.0:
        fail("kappa should be >= 1 because it is productivity relative to a novice baseline.")
    return math.log(kappa)


def delta_from_transfer(h: float, alpha: float = 1.0) -> float:
    ensure_fraction_positive(h, "skill transfer h")
    ensure_positive(alpha, "alpha")
    return -math.log(h) / alpha


def beta_from_half_life(half_life: float) -> float:
    ensure_positive(half_life, "window_half_life")
    return math.log(2.0) / half_life


def niche_coefficient(beta: float, tau: float) -> float:
    ensure_nonnegative(beta, "beta")
    ensure_nonnegative(tau, "tau")
    return safe_exp(-beta * tau)


def lambda_coefficient(c: float, beta: float, tau: float) -> float:
    ensure_fraction_positive(c, "coordination_efficiency c")
    return c * niche_coefficient(beta, tau)


def switching_ratio(kappa: float, g: float, h: float, lam: float = 1.0) -> float:
    """Immediate ratio V_plus / V_minus = Lambda*g*kappa^(h-1)."""
    ensure_positive(kappa, "kappa")
    ensure_positive(g, "gain g")
    ensure_fraction_positive(h, "transfer h")
    ensure_positive(lam, "Lambda")
    x = math.log(kappa)
    log_ratio = math.log(lam) + math.log(g) + (h - 1.0) * x
    return safe_exp(log_ratio)


def no_pain_threshold(g: float, h: float, lam: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
    """Return (kappa_star, x_star) for no-pain switching.

    We solve Lambda*g*exp((h-1)x) >= 1.
    If h == 1, the ratio is independent of x:
      - Lambda*g >= 1: every kappa is no-pain, return (inf, inf)
      - Lambda*g < 1: no kappa >= 1 is no-pain, return (0, -inf)
    """
    ensure_positive(g, "gain g")
    ensure_fraction_positive(h, "transfer h")
    ensure_positive(lam, "Lambda")
    effective_gain = lam * g
    if abs(1.0 - h) < EPS:
        if effective_gain >= 1.0:
            return math.inf, math.inf
        return 0.0, -math.inf
    x_star = math.log(effective_gain) / (1.0 - h)
    kappa_star = safe_exp(x_star)
    return kappa_star, x_star


def critical_gain_for_no_pain(kappa: float, h: float, lam: float = 1.0) -> float:
    """Minimum g needed for immediate ratio >= 1, holding kappa/h/Lambda fixed."""
    ensure_positive(kappa, "kappa")
    ensure_fraction_positive(h, "transfer h")
    ensure_positive(lam, "Lambda")
    # 1 <= Lambda*g*kappa^(h-1) => g >= kappa^(1-h)/Lambda
    return safe_exp((1.0 - h) * math.log(kappa) - math.log(lam))


def critical_transfer_for_no_pain(kappa: float, g: float, lam: float = 1.0) -> Optional[float]:
    """Minimum h needed for immediate ratio >= 1, holding kappa/g/Lambda fixed.

    Returns None when kappa == 1 because h then does not affect the ratio.
    Values >1 mean no feasible h in (0,1] can make switching no-pain.
    Values <=0 mean any feasible h is enough.
    """
    ensure_positive(kappa, "kappa")
    ensure_positive(g, "gain g")
    ensure_positive(lam, "Lambda")
    x = math.log(kappa)
    if abs(x) < EPS:
        return None
    return 1.0 - math.log(lam * g) / x


def kappa_after_learning(kappa_start: float, duration: float, learning_rate: float = 1.0) -> float:
    ensure_positive(kappa_start, "kappa_start")
    ensure_nonnegative(duration, "duration")
    ensure_nonnegative(learning_rate, "learning_rate")
    return kappa_start + learning_rate * duration


def kappa_after_switch(kappa_before: float, h: float) -> float:
    ensure_positive(kappa_before, "kappa_before")
    ensure_fraction_positive(h, "transfer h")
    return safe_exp(h * math.log(kappa_before))


def integrated_output_linear(
    kappa_start: float,
    duration: float,
    *,
    multiplier: float = 1.0,
    learning_rate: float = 1.0,
) -> float:
    """Closed-form output under kappa(t+s)=kappa_start+learning_rate*s.

    O = multiplier * ∫_0^D [kappa_start + learning_rate*s] ds
      = multiplier * [kappa_start*D + 0.5*learning_rate*D^2]
    """
    ensure_positive(kappa_start, "kappa_start")
    ensure_nonnegative(duration, "duration")
    ensure_nonnegative(multiplier, "multiplier")
    ensure_nonnegative(learning_rate, "learning_rate")
    return multiplier * (kappa_start * duration + 0.5 * learning_rate * duration * duration)


def switch_output_linear(
    kappa_before_switch: float,
    duration_after_switch: float,
    *,
    g: float,
    h: float,
    lam: float,
    learning_rate: float,
) -> float:
    """Integrated output if the agent switches now and then works for duration_after_switch."""
    ensure_positive(g, "gain g")
    ensure_fraction_positive(h, "transfer h")
    ensure_positive(lam, "Lambda")
    kappa_new = kappa_after_switch(kappa_before_switch, h)
    return integrated_output_linear(
        kappa_new,
        duration_after_switch,
        multiplier=lam * g,
        learning_rate=learning_rate,
    )


# -----------------------------------------------------------------------------
# Model assembly
# -----------------------------------------------------------------------------

def derive_low_state(g_high: float, h_high: float) -> Tuple[float, float]:
    """Conservative default for the low-payoff state.

    The simple UI gives one expected gain/transfer and a confidence value. To run
    a two-state real-options calculation, we need a low state. This function uses
    a transparent conservative default:
      - low gain is at most 1.0 and about 60% of high gain;
      - low transfer is lower than high transfer but never below 0.05.

    Advanced users can override both values from CLI.
    """
    ensure_positive(g_high, "g_high")
    ensure_fraction_positive(h_high, "h_high")
    g_low = max(0.05, min(1.0, 0.60 * g_high))
    h_low = max(0.05, min(h_high, 0.60 * h_high))
    return g_low, h_low


def prepare_inputs(raw: ModelInputs) -> Tuple[ModelInputs, DerivedQuantities, List[str]]:
    warnings: List[str] = []

    kappa = ensure_positive(raw.kappa, "kappa")
    if kappa < 1.0:
        fail("current mastery kappa must be >= 1.")
    g_high = ensure_positive(raw.g_high, "g_high")
    h_high = ensure_fraction_positive(raw.h_high, "h_high")
    p_high = ensure_probability(raw.p_high, "p_high")
    half_life = ensure_positive(raw.window_half_life, "window_half_life")
    horizon = ensure_positive(raw.horizon, "horizon")
    current_delay = ensure_nonnegative(raw.current_delay, "current_delay")
    c = ensure_fraction_positive(raw.coordination_efficiency, "coordination_efficiency")
    learning_rate = ensure_nonnegative(raw.learning_rate, "learning_rate")
    q = ensure_probability(raw.pilot_accuracy, "pilot_accuracy")
    if q < 0.5:
        fail("pilot_accuracy q must be >= 0.5. A signal worse than random should be inverted.")
    parallel_share = ensure_probability(raw.parallel_share, "parallel_share")

    if raw.g_low is None or raw.h_low is None:
        default_g_low, default_h_low = derive_low_state(g_high, h_high)
        g_low = default_g_low if raw.g_low is None else raw.g_low
        h_low = default_h_low if raw.h_low is None else raw.h_low
        warnings.append(
            "Low state was derived automatically. For serious use, calibrate --low-gain and --low-transfer from data."
        )
    else:
        g_low = raw.g_low
        h_low = raw.h_low

    g_low = ensure_positive(g_low, "g_low")
    h_low = ensure_fraction_positive(h_low, "h_low")

    if raw.pilot_cost_level not in PILOT_COST_PRESETS:
        fail(f"pilot_cost_level must be one of {list(PILOT_COST_PRESETS)}.")
    preset = PILOT_COST_PRESETS[raw.pilot_cost_level]

    pilot_duration = preset.epsilon if raw.pilot_duration is None else raw.pilot_duration
    pilot_duration = ensure_nonnegative(pilot_duration, "pilot_duration")
    if pilot_duration > horizon:
        warnings.append("Pilot duration exceeds horizon; it has been capped at the horizon.")
        pilot_duration = horizon

    pilot_rho = preset.rho if raw.pilot_rho is None else raw.pilot_rho
    pilot_rho = ensure_probability(pilot_rho, "pilot_rho")

    # Direct cost is measured in output units. By default, convert preset's
    # equivalent-month cost into output units using current baseline speed kappa.
    pilot_direct_cost = (
        preset.direct_cost_months * kappa if raw.pilot_direct_cost is None else raw.pilot_direct_cost
    )
    pilot_direct_cost = ensure_nonnegative(pilot_direct_cost, "pilot_direct_cost")

    beta = beta_from_half_life(half_life)
    niche_now = niche_coefficient(beta, current_delay)
    lam_now = c * niche_now
    pilot_decay_factor = niche_coefficient(beta, pilot_duration)
    lam_after_pilot = c * niche_coefficient(beta, current_delay + pilot_duration)

    x = log_skill(kappa)
    delta_high = delta_from_transfer(h_high)
    delta_low = delta_from_transfer(h_low)

    prepared = ModelInputs(
        kappa=kappa,
        g_high=g_high,
        h_high=h_high,
        p_high=p_high,
        window_half_life=half_life,
        horizon=horizon,
        pilot_cost_level=raw.pilot_cost_level,
        g_low=g_low,
        h_low=h_low,
        current_delay=current_delay,
        coordination_efficiency=c,
        learning_rate=learning_rate,
        pilot_accuracy=q,
        pilot_duration=pilot_duration,
        pilot_direct_cost=pilot_direct_cost,
        pilot_rho=pilot_rho,
        parallel_share=parallel_share,
    )

    derived = DerivedQuantities(
        x=x,
        delta_high=delta_high,
        delta_low=delta_low,
        beta=beta,
        niche_now=niche_now,
        lambda_now=lam_now,
        pilot_duration=pilot_duration,
        pilot_direct_cost=pilot_direct_cost,
        pilot_rho=pilot_rho,
        pilot_decay_factor=pilot_decay_factor,
        lambda_after_pilot=lam_after_pilot,
        g_low=g_low,
        h_low=h_low,
    )

    return prepared, derived, warnings


def compute_path_values(params: ModelInputs, derived: DerivedQuantities) -> PathValues:
    kappa = params.kappa
    H = params.horizon
    a = params.learning_rate
    p = params.p_high
    gH, hH = params.g_high, params.h_high
    gL, hL = derived.g_low, derived.h_low
    lam0 = derived.lambda_now

    stay = integrated_output_linear(kappa, H, multiplier=1.0, learning_rate=a)
    switch_H = switch_output_linear(kappa, H, g=gH, h=hH, lam=lam0, learning_rate=a)
    switch_L = switch_output_linear(kappa, H, g=gL, h=hL, lam=lam0, learning_rate=a)
    switch_expected = p * switch_H + (1.0 - p) * switch_L
    no_experiment_best = max(stay, switch_expected)

    pilot_value = compute_noisy_pilot_value(params, derived)
    parallel_value = compute_parallel_track_value(params, derived)

    return PathValues(
        stay=stay,
        switch_now_expected=switch_expected,
        switch_now_high=switch_H,
        switch_now_low=switch_L,
        pilot_first=pilot_value,
        parallel_track=parallel_value,
        no_experiment_best=no_experiment_best,
    )


def compute_noisy_pilot_value(params: ModelInputs, derived: DerivedQuantities) -> float:
    """Expected value of noisy staged validation using integrated payoffs.

    This implements the logic of the noisy pilot criterion, but uses the
    linear-productivity integrated payoff primitives instead of flat-speed payoffs.
    """
    kappa = params.kappa
    H = params.horizon
    eps = derived.pilot_duration
    a = params.learning_rate
    p = params.p_high
    q = params.pilot_accuracy
    rho = derived.pilot_rho
    Ce = derived.pilot_direct_cost

    gH, hH = params.g_high, params.h_high
    gL, hL = derived.g_low, derived.h_low

    remaining = max(0.0, H - eps)
    kappa_after_pilot_old = kappa_after_learning(kappa, eps, learning_rate=a)

    # Production retained during the pilot.
    pilot_production = rho * integrated_output_linear(kappa, eps, multiplier=1.0, learning_rate=a)

    # If after the pilot the agent stays, the old-paradigm state has continued to mature.
    stay_after_signal = integrated_output_linear(
        kappa_after_pilot_old, remaining, multiplier=1.0, learning_rate=a
    )

    # If after the pilot the agent switches, the switch happens after old skill has evolved
    # for epsilon and after the niche coefficient has decayed.
    switch_H_after = switch_output_linear(
        kappa_after_pilot_old,
        remaining,
        g=gH,
        h=hH,
        lam=derived.lambda_after_pilot,
        learning_rate=a,
    )
    switch_L_after = switch_output_linear(
        kappa_after_pilot_old,
        remaining,
        g=gL,
        h=hL,
        lam=derived.lambda_after_pilot,
        learning_rate=a,
    )

    # Signal probabilities and Bayesian posteriors.
    pi_H_signal = q * p + (1.0 - q) * (1.0 - p)
    pi_L_signal = (1.0 - q) * p + q * (1.0 - p)

    if pi_H_signal > EPS:
        posterior_if_H_signal = q * p / pi_H_signal
    else:
        posterior_if_H_signal = 0.0

    if pi_L_signal > EPS:
        posterior_if_L_signal = (1.0 - q) * p / pi_L_signal
    else:
        posterior_if_L_signal = 0.0

    expected_switch_after_H_signal = (
        posterior_if_H_signal * switch_H_after
        + (1.0 - posterior_if_H_signal) * switch_L_after
    )
    expected_switch_after_L_signal = (
        posterior_if_L_signal * switch_H_after
        + (1.0 - posterior_if_L_signal) * switch_L_after
    )

    best_after_H_signal = max(expected_switch_after_H_signal, stay_after_signal)
    best_after_L_signal = max(expected_switch_after_L_signal, stay_after_signal)

    return (
        pilot_production
        - Ce
        + pi_H_signal * best_after_H_signal
        + pi_L_signal * best_after_L_signal
    )


def compute_parallel_track_value(params: ModelInputs, derived: DerivedQuantities) -> float:
    """Capacity-split approximation for a parallel-track strategy.

    A fraction s of capacity is allocated to the new paradigm and 1-s remains on
    the old paradigm. This is not a new theorem; it is a transparent operational
    approximation for the user-facing dashboard.
    """
    s = params.parallel_share
    kappa = params.kappa
    H = params.horizon
    a = params.learning_rate
    p = params.p_high
    gH, hH = params.g_high, params.h_high
    gL, hL = derived.g_low, derived.h_low
    lam0 = derived.lambda_now

    old_part = integrated_output_linear(kappa, (1.0 - s) * H, multiplier=1.0, learning_rate=a)
    new_H = switch_output_linear(kappa, s * H, g=gH, h=hH, lam=lam0, learning_rate=a)
    new_L = switch_output_linear(kappa, s * H, g=gL, h=hL, lam=lam0, learning_rate=a)
    new_expected = p * new_H + (1.0 - p) * new_L
    return old_part + new_expected


def compute_belief_threshold(W: float, S_high: float, S_low: float) -> Optional[float]:
    """p*=(W-S_L)/(S_H-S_L), if S_H > S_L."""
    if S_high <= S_low + EPS:
        return None
    return (W - S_low) / (S_high - S_low)


def label_pain(ratio: float) -> str:
    if ratio >= 1.20:
        return "GREEN: immediate switch likely accelerates output."
    if ratio >= 0.90:
        return "NEAR-NEUTRAL: immediate switch is roughly painless."
    if ratio >= 0.60:
        return "YELLOW: immediate switch likely causes a visible short-term dip."
    return "RED: immediate switch is likely very painful."


def choose_recommendation(
    params: ModelInputs,
    path: PathValues,
    immediate_ratio_high: float,
    p_star: Optional[float],
) -> Tuple[str, str, str]:
    values = {
        "STAY_AND_MONITOR": path.stay,
        "SWITCH_NOW": path.switch_now_expected,
        "PILOT_FAST": path.pilot_first,
        "PARALLEL_TRACK": path.parallel_track,
    }
    best_key = max(values, key=values.get)
    best_value = values[best_key]

    # A small tolerance prevents unstable recommendations when two strategies are numerically tied.
    tol = 0.03 * max(abs(best_value), EPS)

    pilot_advantage = path.pilot_first - path.no_experiment_best
    parallel_close = path.parallel_track >= best_value - tol
    switch_close = path.switch_now_expected >= best_value - tol

    if best_key == "SWITCH_NOW":
        if immediate_ratio_high >= 0.90 and params.p_high >= 0.70:
            rec = "SWITCH_NOW"
            action = (
                "Commit quickly. Preserve a rollback plan, but make the new paradigm the default. "
                "Track output speed and quality for the first review interval."
            )
        elif pilot_advantage > 0:
            rec = "PILOT_FAST"
            action = (
                "Do a short, high-information pilot before full migration. The expected switch value is high, "
                "but pain or uncertainty is large enough that validation has positive option value."
            )
        elif parallel_close:
            rec = "PARALLEL_TRACK"
            action = (
                "Run the new paradigm in a protected side track while keeping the old system alive. "
                "Escalate only after measured transfer and gain improve."
            )
        else:
            rec = "SWITCH_NOW_WITH_MANAGED_DIP"
            action = (
                "Switch now, but budget for a temporary productivity dip. Do not interpret the early dip as failure "
                "unless measured gain also disappoints."
            )
    elif best_key == "PILOT_FAST":
        rec = "PILOT_FAST"
        action = (
            "Run a focused pilot. Measure: completion time, quality-adjusted output, retraining friction, "
            "reuse of old skills, and whether the niche window is decaying during validation."
        )
    elif best_key == "PARALLEL_TRACK":
        rec = "PARALLEL_TRACK"
        action = (
            "Use a dual-track strategy. Keep the old paradigm producing while allocating a controlled share "
            "of capacity to the new paradigm. Recompute after collecting pilot data."
        )
    else:
        # stay is best
        if params.g_high < 1.0 or path.switch_now_expected < 0.90 * path.stay:
            rec = "REJECT_OR_DEFER"
            action = (
                "Do not migrate now. The expected gain does not compensate for transfer loss and opportunity cost. "
                "Revisit only if credible evidence raises gain or transfer."
            )
        elif pilot_advantage > 0:
            rec = "PILOT_ONLY_LOW_BUDGET"
            action = (
                "Do not commit. A very small pilot may be justified only if it cheaply resolves uncertainty. "
                "Keep the old paradigm as the production default."
            )
        else:
            rec = "STAY_AND_MONITOR"
            action = (
                "Stay in the current paradigm and monitor evidence. Set a trigger condition for reassessment, "
                "such as higher verified gain, better transfer, or faster niche decay."
            )

    # Add belief-threshold note when useful.
    if p_star is not None and math.isfinite(p_star):
        if p_star > 1.0:
            action += " Current belief threshold is above 1, so switching is not justified without better evidence."
        elif p_star < 0.0:
            action += " Current belief threshold is below 0, so switching is structurally attractive even in the low state."
        else:
            action += f" Estimated belief threshold p* is {p_star:.3f}."

    return best_key, rec, action


def run_model(raw_inputs: ModelInputs) -> ModelResult:
    params, derived, warnings = prepare_inputs(raw_inputs)

    path = compute_path_values(params, derived)

    ratio_high = switching_ratio(params.kappa, params.g_high, params.h_high, derived.lambda_now)
    ratio_low = switching_ratio(params.kappa, derived.g_low, derived.h_low, derived.lambda_now)
    kappa_star, x_star = no_pain_threshold(params.g_high, params.h_high, derived.lambda_now)
    g_crit = critical_gain_for_no_pain(params.kappa, params.h_high, derived.lambda_now)
    h_crit = critical_transfer_for_no_pain(params.kappa, params.g_high, derived.lambda_now)
    p_star = compute_belief_threshold(path.stay, path.switch_now_high, path.switch_now_low)
    pilot_net = path.pilot_first - path.no_experiment_best

    best_path, rec, action = choose_recommendation(params, path, ratio_high, p_star)

    stay = path.stay
    relative = {
        "stay": 1.0,
        "switch_now_expected": safe_div(path.switch_now_expected, stay),
        "switch_now_high": safe_div(path.switch_now_high, stay),
        "switch_now_low": safe_div(path.switch_now_low, stay),
        "pilot_first": safe_div(path.pilot_first, stay),
        "parallel_track": safe_div(path.parallel_track, stay),
        "no_experiment_best": safe_div(path.no_experiment_best, stay),
    }

    metrics = DecisionMetrics(
        immediate_pain_index_high=ratio_high,
        immediate_pain_index_low=ratio_low,
        no_pain_kappa_threshold_high=kappa_star,
        no_pain_x_threshold_high=x_star,
        critical_gain_for_no_pain=g_crit,
        critical_transfer_for_no_pain=h_crit,
        belief_threshold=p_star,
        pilot_net_value_vs_no_experiment=pilot_net,
        best_path=best_path,
        recommendation=rec,
        next_action=action,
        pain_label=label_pain(ratio_high),
    )

    return ModelResult(
        inputs=params,
        derived=derived,
        path_values=path,
        relative_to_stay=relative,
        metrics=metrics,
        warnings=warnings,
    )


# -----------------------------------------------------------------------------
# Visual-Studio-friendly interactive input/output helpers
# -----------------------------------------------------------------------------

def parse_prob_or_percent(raw: str) -> float:
    """Parse either 0.7, 70, or 70% as probability 0.7."""
    s = raw.strip().replace("%", "")
    x = float(s)
    if x > 1.0:
        x = x / 100.0
    return x


def yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please input y or n.")


def _format_default(default: float) -> str:
    return fmt_float(default, digits=4)


def ask_numeric(
    question: str,
    default: float,
    *,
    parser=float,
    validator=None,
    explanation: Optional[str] = None,
) -> float:
    """Ask one plain numeric question with retry validation."""
    print("\n" + question)
    if explanation:
        print(explanation)
    while True:
        raw = input(f"Please input a value. Press Enter for default [{_format_default(default)}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = parser(raw)
            except Exception as exc:  # noqa: BLE001 - user-facing retry
                print(f"Invalid input: {exc}. Please try again.")
                continue
        try:
            if validator is not None:
                validator(value)
            return value
        except Exception as exc:  # noqa: BLE001 - user-facing retry
            print(f"Invalid value: {exc}. Please try again.")


def ask_preset_or_custom(
    title: str,
    options: List[Tuple[str, str, float]],
    default_index: int,
    *,
    parser=float,
    validator=None,
    explanation: Optional[str] = None,
) -> float:
    """Ask a question with numbered options plus custom numeric override.

    The user can type:
      - an option number, e.g. 3;
      - a preset key, e.g. expert;
      - a raw numeric value, e.g. 8 or 70%.
    """
    if default_index < 1 or default_index > len(options):
        raise ValueError("default_index is out of range.")

    key_to_value = {key: value for key, _, value in options}
    default_value = options[default_index - 1][2]

    print("\n" + title)
    if explanation:
        print(explanation)
    print("Options:")
    for i, (key, label, value) in enumerate(options, start=1):
        print(f"  {i}. {label:<46} -> {fmt_float(value)}")

    while True:
        raw = input(
            f"Please input option number, preset key, or custom value. "
            f"Press Enter for default option {default_index} [{fmt_float(default_value)}]: "
        ).strip()

        if not raw:
            value = default_value
        elif raw.isdigit() and 1 <= int(raw) <= len(options):
            value = options[int(raw) - 1][2]
        elif raw in key_to_value:
            value = key_to_value[raw]
        else:
            try:
                value = parser(raw)
            except Exception as exc:  # noqa: BLE001 - user-facing retry
                print(f"Invalid input: {exc}. Please try again.")
                continue

        try:
            if validator is not None:
                validator(value)
            return value
        except Exception as exc:  # noqa: BLE001 - user-facing retry
            print(f"Invalid value: {exc}. Please try again.")


def ask_pilot_cost(default_key: str = "medium") -> str:
    keys = list(PILOT_COST_PRESETS.keys())
    if default_key not in PILOT_COST_PRESETS:
        default_key = "medium"
    default_index = keys.index(default_key) + 1

    print("\nPlease input pilot cost level.")
    print("This controls pilot duration, direct cost, and retained old-paradigm production during the pilot.")
    print("Options:")
    for i, key in enumerate(keys, start=1):
        val = PILOT_COST_PRESETS[key]
        print(
            f"  {i}. {key:<10} -> duration={fmt_float(val.epsilon)} months, "
            f"direct cost={fmt_float(val.direct_cost_months)} current-output-months, "
            f"rho={fmt_float(val.rho)}"
        )

    while True:
        raw = input(
            f"Please input option number or preset key. "
            f"Press Enter for default option {default_index} [{default_key}]: "
        ).strip()
        if not raw:
            return default_key
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        if raw in PILOT_COST_PRESETS:
            return raw
        print(f"Invalid pilot cost. Choose one of: {', '.join(keys)}")


def current_mastery_options() -> List[Tuple[str, str, float]]:
    return [
        ("novice", "Novice / just able to use it", 1.0),
        ("skilled", "Skilled", 2.0),
        ("strong", "Strong", 4.0),
        ("expert", "Expert", 8.0),
        ("elite", "Elite / deep accumulated advantage", 16.0),
    ]


def gain_options() -> List[Tuple[str, str, float]]:
    return [
        ("worse", "Possibly worse than current paradigm", 0.8),
        ("slightly_better", "Slightly better", 1.2),
        ("clearly_better", "Clearly better", 1.5),
        ("major", "Major improvement", 2.0),
        ("game_changing", "Game-changing", 3.0),
        ("explosive", "Explosive paradigm", 5.0),
    ]


def transfer_options() -> List[Tuple[str, str, float]]:
    return [
        ("almost_all", "Almost all old skill transfers", 0.90),
        ("most", "Most old skill transfers", 0.75),
        ("half", "About half transfers", 0.50),
        ("little", "Only a little transfers", 0.25),
        ("almost_none", "Almost none transfers", 0.10),
    ]


def half_life_options() -> List[Tuple[str, str, float]]:
    return [
        ("very_tight", "Very tight window", 1.0),
        ("tight", "Tight window", 3.0),
        ("moderate", "Moderate window", 6.0),
        ("slow", "Slowly decaying window", 12.0),
        ("not_urgent", "Not urgent", 24.0),
    ]


def confidence_options() -> List[Tuple[str, str, float]]:
    return [
        ("hype_only", "Mostly hype / weak evidence", 0.20),
        ("possible", "Possible but uncertain", 0.40),
        ("coin_flip", "Roughly 50/50", 0.50),
        ("credible", "Credible", 0.70),
        ("high", "Highly credible", 0.85),
        ("verified", "Almost verified", 0.95),
    ]


def interactive_inputs() -> ModelInputs:
    print("\n" + "=" * 78)
    print("PARADIGM SWITCH DASHBOARD - INTERACTIVE MODE")
    print("=" * 78)
    print("This version is designed for Visual Studio / VS Code Run mode.")
    print("It will ask one question at a time. Time unit defaults to months.")
    print("You can choose an option number or type your own numeric value.")

    kappa = ask_preset_or_custom(
        "Please input current mastery.",
        current_mastery_options(),
        default_index=3,
        validator=lambda v: log_skill(v),
        explanation="Meaning: your current productivity in the old paradigm relative to a novice baseline.",
    )

    g = ask_preset_or_custom(
        "Please input expected high-state gain.",
        gain_options(),
        default_index=4,
        validator=lambda v: ensure_positive(v, "expected gain"),
        explanation="Meaning: if the new paradigm works, how many times stronger is mature output?",
    )

    h = ask_preset_or_custom(
        "Please input old-skill transfer rate.",
        transfer_options(),
        default_index=3,
        parser=parse_prob_or_percent,
        validator=lambda v: ensure_fraction_positive(v, "skill transfer"),
        explanation="Meaning: after switching, how much old log-skill survives? You may type 0.5 or 50%.",
    )

    half_life = ask_preset_or_custom(
        "Please input opportunity-window half-life.",
        half_life_options(),
        default_index=3,
        validator=lambda v: ensure_positive(v, "window half-life"),
        explanation="Meaning: after this many months of delay, the niche/window advantage is cut in half.",
    )

    p = ask_preset_or_custom(
        "Please input confidence in the high-payoff state.",
        confidence_options(),
        default_index=4,
        parser=parse_prob_or_percent,
        validator=lambda v: ensure_probability(v, "confidence"),
        explanation="Meaning: probability that the new paradigm is truly high-gain / acceptable-drift. You may type 0.7 or 70%.",
    )

    horizon = ask_numeric(
        "Please input decision horizon.",
        12.0,
        validator=lambda v: ensure_positive(v, "decision horizon"),
        explanation="Meaning: how far into the future to compare total output. Default: 12 months.",
    )

    pilot_level = ask_pilot_cost("medium")

    advanced = yes_no("Do you want advanced overrides?", default=False)
    if not advanced:
        return ModelInputs(
            kappa=kappa,
            g_high=g,
            h_high=h,
            p_high=p,
            window_half_life=half_life,
            horizon=horizon,
            pilot_cost_level=pilot_level,
        )

    default_g_low, default_h_low = derive_low_state(g, h)
    g_low = ask_numeric(
        "Please input low-state gain g_L.",
        default_g_low,
        validator=lambda v: ensure_positive(v, "low-state gain"),
        explanation="Meaning: if the new paradigm disappoints, what is its gain?",
    )
    h_low = ask_numeric(
        "Please input low-state transfer h_L.",
        default_h_low,
        parser=parse_prob_or_percent,
        validator=lambda v: ensure_fraction_positive(v, "low-state transfer"),
        explanation="Meaning: if the new paradigm disappoints, how much old log-skill survives?",
    )
    current_delay = ask_numeric(
        "Please input current delay tau.",
        0.0,
        validator=lambda v: ensure_nonnegative(v, "current delay"),
        explanation="Meaning: how long since the opportunity first appeared. Default: 0 months.",
    )
    c = ask_numeric(
        "Please input coordination efficiency c.",
        1.0,
        parser=parse_prob_or_percent,
        validator=lambda v: ensure_fraction_positive(v, "coordination efficiency"),
        explanation="Meaning: organization/team friction multiplier. Individual use usually equals 1.0.",
    )
    learning_rate = ask_numeric(
        "Please input learning rate d(kappa)/dt.",
        1.0,
        validator=lambda v: ensure_nonnegative(v, "learning rate"),
        explanation="Meaning: within-paradigm productivity growth in the linear-productivity benchmark.",
    )
    q = ask_numeric(
        "Please input pilot signal accuracy q.",
        0.75,
        parser=parse_prob_or_percent,
        validator=lambda v: (ensure_probability(v, "pilot accuracy"), fail("q must be >= 0.5") if v < 0.5 else None),
        explanation="Meaning: probability that the pilot gives the correct high/low signal. Type 0.75 or 75%.",
    )
    parallel_share = ask_numeric(
        "Please input parallel-track capacity share.",
        0.25,
        parser=parse_prob_or_percent,
        validator=lambda v: ensure_probability(v, "parallel-track capacity share"),
        explanation="Meaning: fraction of capacity allocated to the new paradigm in the dual-track strategy.",
    )

    return ModelInputs(
        kappa=kappa,
        g_high=g,
        h_high=h,
        p_high=p,
        window_half_life=half_life,
        horizon=horizon,
        pilot_cost_level=pilot_level,
        g_low=g_low,
        h_low=h_low,
        current_delay=current_delay,
        coordination_efficiency=c,
        learning_rate=learning_rate,
        pilot_accuracy=q,
        parallel_share=parallel_share,
    )


def print_report(result: ModelResult) -> None:
    p = result.inputs
    d = result.derived
    v = result.path_values
    r = result.relative_to_stay
    m = result.metrics

    print("\n" + "=" * 78)
    print("PARADIGM SWITCH DASHBOARD REPORT")
    print("=" * 78)

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    print("\n[1] Simple user inputs")
    print(f"  Current mastery kappa:              {fmt_float(p.kappa)}x novice")
    print(f"  Expected high-state gain g_H:        {fmt_float(p.g_high)}")
    print(f"  High-state skill transfer h_H:       {fmt_float(p.h_high)}")
    print(f"  Confidence p(high state):            {fmt_float(p.p_high)}")
    print(f"  Window half-life:                    {fmt_float(p.window_half_life)} months")
    print(f"  Decision horizon H:                  {fmt_float(p.horizon)} months")
    print(f"  Pilot cost level:                    {p.pilot_cost_level}")

    print("\n[2] Derived mathematical state")
    print(f"  Log-skill x = ln(kappa):             {fmt_float(d.x)}")
    print(f"  High-state drift delta_H=-ln(h_H):   {fmt_float(d.delta_high)}")
    print(f"  Low-state gain g_L:                  {fmt_float(d.g_low)}")
    print(f"  Low-state transfer h_L:              {fmt_float(d.h_low)}")
    print(f"  Low-state drift delta_L=-ln(h_L):    {fmt_float(d.delta_low)}")
    print(f"  Beta = ln(2)/half_life:              {fmt_float(d.beta)}")
    print(f"  S(tau), current niche coefficient:   {fmt_float(d.niche_now)}")
    print(f"  Lambda = c*S(tau):                   {fmt_float(d.lambda_now)}")
    print(f"  Pilot duration epsilon:              {fmt_float(d.pilot_duration)} months")
    print(f"  Pilot direct cost C_e:               {fmt_float(d.pilot_direct_cost)} output units")
    print(f"  Pilot retained production rho:       {fmt_float(d.pilot_rho)}")
    print(f"  Pilot decay factor S(tau+eps)/S(tau):{fmt_float(d.pilot_decay_factor)}")

    print("\n[3] Immediate switching metrics")
    print(f"  Pain index high state V+/V-:         {fmt_float(m.immediate_pain_index_high)}")
    print(f"  Pain index low state V+/V-:          {fmt_float(m.immediate_pain_index_low)}")
    print(f"  Pain label:                          {m.pain_label}")
    print(f"  No-pain kappa threshold:             {fmt_float(m.no_pain_kappa_threshold_high)}x novice")
    print(f"  Critical gain for no-pain:           {fmt_float(m.critical_gain_for_no_pain)}")
    print(f"  Critical transfer h for no-pain:     {fmt_float(m.critical_transfer_for_no_pain)}")
    print(f"  Belief threshold p*:                 {fmt_float(m.belief_threshold)}")

    print("\n[4] Long-horizon output by strategy")
    rows = [
        ("Stay", v.stay, r["stay"]),
        ("Switch now, expected", v.switch_now_expected, r["switch_now_expected"]),
        ("Switch now, high state", v.switch_now_high, r["switch_now_high"]),
        ("Switch now, low state", v.switch_now_low, r["switch_now_low"]),
        ("Pilot first", v.pilot_first, r["pilot_first"]),
        ("Parallel track", v.parallel_track, r["parallel_track"]),
    ]
    for name, value, rel in rows:
        print(f"  {name:<24} output={fmt_float(value):>12}   relative_to_stay={fmt_float(rel):>9}")

    print("\n[5] Decision")
    print(f"  Best raw path:                       {m.best_path}")
    print(f"  Recommendation:                      {m.recommendation}")
    print(f"  Pilot net value vs no experiment:    {fmt_float(m.pilot_net_value_vs_no_experiment)}")
    print(f"  Next action:                         {m.next_action}")
    print("=" * 78 + "\n")


def run_once() -> None:
    inputs = interactive_inputs()
    result = run_model(inputs)
    print_report(result)


def main() -> int:
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as exc:  # noqa: BLE001 - top-level interactive error reporting
            print(f"\nERROR: {exc}")

        if not yes_no("Run another scenario?", default=False):
            print("Done.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

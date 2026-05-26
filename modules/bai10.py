"""Bài 10 - Two-stage stochastic programming for AIDEOM-VN.

The implementation follows the exercise statement but avoids external LP solvers by
using the structure of the model.  A Pyomo-style model description is still shown in
Streamlit, while this module computes the optimal solution directly and reproducibly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số",
    "AI": "AI & dữ liệu",
    "H": "Nhân lực số",
}
SCENARIOS = ["s1", "s2", "s3", "s4"]
SCENARIO_NAMES = {
    "s1": "Lạc quan",
    "s2": "Cơ sở",
    "s3": "Bi quan",
    "s4": "Khủng hoảng",
}
PROBS = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
SCENARIO_META = {
    "s1": {"Tăng trưởng TG (%)": 3.5, "FDI VN (tỷ USD/năm)": 32.0, "Xuất khẩu VN tăng (%)": 12.0, "Xác suất": 0.30},
    "s2": {"Tăng trưởng TG (%)": 2.8, "FDI VN (tỷ USD/năm)": 27.0, "Xuất khẩu VN tăng (%)": 8.0, "Xác suất": 0.45},
    "s3": {"Tăng trưởng TG (%)": 1.5, "FDI VN (tỷ USD/năm)": 20.0, "Xuất khẩu VN tăng (%)": 3.0, "Xác suất": 0.20},
    "s4": {"Tăng trưởng TG (%)": 0.2, "FDI VN (tỷ USD/năm)": 12.0, "Xuất khẩu VN tăng (%)": -5.0, "Xác suất": 0.05},
}
BETA = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}


@dataclass
class StochasticSolution:
    name: str
    first_stage: Dict[str, float]
    second_stage: Dict[str, Dict[str, float]]
    scenario_values: Dict[str, float]
    expected_value: float
    h_value: float


def scenario_dataframe() -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        row = {"Kịch bản": s, "Tên": SCENARIO_NAMES[s], **SCENARIO_META[s]}
        rows.append(row)
    return pd.DataFrame(rows)


def beta_dataframe() -> pd.DataFrame:
    rows = []
    for j in ITEMS:
        row = {"Hạng mục": f"{j} - {ITEM_NAMES[j]}", "β cơ bản": BETA[j]}
        for s in SCENARIOS:
            row[f"{s} - {SCENARIO_NAMES[s]}"] = BETA_S[(s, j)]
        rows.append(row)
    return pd.DataFrame(rows)


def _best_non_ai_for_scenario(s: str) -> Tuple[str, float]:
    candidates = [(j, BETA_S[(s, j)]) for j in ITEMS if j != "AI"]
    return max(candidates, key=lambda x: x[1])


def _recourse_for_h(h: float, s: str, reserve_budget: float = 15.0) -> Tuple[Dict[str, float], float]:
    """Optimal second-stage recourse for a fixed first-stage H allocation.

    Constraint: y_AI <= 0.5 * x_H.  All coefficients are positive, so the full
    reserve is used.  The recourse budget is placed in AI if it is better than
    the best non-AI item and allowed by the H constraint; the remainder goes to
    the best non-AI item.
    """
    y = {j: 0.0 for j in ITEMS}
    best_non, best_non_beta = _best_non_ai_for_scenario(s)
    ai_beta = BETA_S[(s, "AI")]
    if ai_beta > best_non_beta:
        y_ai = min(reserve_budget, 0.5 * h)
        y["AI"] = y_ai
        y[best_non] = reserve_budget - y_ai
    else:
        y[best_non] = reserve_budget
    value = sum(BETA_S[(s, j)] * y[j] for j in ITEMS)
    return y, value


def _first_stage_for_h(h: float, first_budget: float = 65.0, first_beta: Dict[str, float] | None = None) -> Dict[str, float]:
    """Return the optimal first-stage allocation conditional on x_H=h.

    The remaining first-stage budget goes to the best non-H item under the
    supplied first-stage coefficients.  For the stochastic model these are the
    base betas; for wait-and-see deterministic scenarios they can be scenario
    betas, which gives a meaningful EVPI comparison.
    """
    if first_beta is None:
        first_beta = BETA
    h = float(np.clip(h, 0.0, first_budget))
    x = {j: 0.0 for j in ITEMS}
    x["H"] = h
    best_non_h = max([j for j in ITEMS if j != "H"], key=lambda j: first_beta[j])
    x[best_non_h] = first_budget - h
    return x


def _first_value(x: Dict[str, float], first_beta: Dict[str, float] | None = None) -> float:
    if first_beta is None:
        first_beta = BETA
    return sum(first_beta[j] * x[j] for j in ITEMS)


def _value_for_h(h: float, probs: Dict[str, float] | None = None, first_budget: float = 65.0, reserve_budget: float = 15.0, first_beta: Dict[str, float] | None = None) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, float]]:
    if probs is None:
        probs = PROBS
    if first_beta is None:
        first_beta = BETA
    x = _first_stage_for_h(h, first_budget, first_beta)
    first = _first_value(x, first_beta)
    y_by_s: Dict[str, Dict[str, float]] = {}
    value_by_s: Dict[str, float] = {}
    expected = 0.0
    for s in SCENARIOS:
        y, second = _recourse_for_h(h, s, reserve_budget)
        y_by_s[s] = y
        total_s = first + second
        value_by_s[s] = total_s
        expected += probs.get(s, 0.0) * total_s
    return expected, y_by_s, value_by_s


def optimize_stochastic(first_budget: float = 65.0, reserve_budget: float = 15.0, step: float = 0.01) -> StochasticSolution:
    grid = np.arange(0.0, first_budget + step, step)
    values = np.array([_value_for_h(float(h), PROBS, first_budget, reserve_budget, BETA)[0] for h in grid])
    idx = int(np.argmax(values))
    h = float(grid[idx])
    expected, y_by_s, scenario_values = _value_for_h(h, PROBS, first_budget, reserve_budget, BETA)
    return StochasticSolution("SP - stochastic", _first_stage_for_h(h, first_budget, BETA), y_by_s, scenario_values, expected, h)


def optimize_deterministic_scenario(s: str, first_budget: float = 65.0, reserve_budget: float = 15.0, step: float = 0.01) -> StochasticSolution:
    probs = {ss: 0.0 for ss in SCENARIOS}
    probs[s] = 1.0
    first_beta = {j: BETA_S[(s, j)] for j in ITEMS}
    grid = np.arange(0.0, first_budget + step, step)
    values = np.array([_value_for_h(float(h), probs, first_budget, reserve_budget, first_beta)[0] for h in grid])
    h = float(grid[int(np.argmax(values))])
    expected, y_by_s, scenario_values = _value_for_h(h, probs, first_budget, reserve_budget, first_beta)
    # only keep recourse for that scenario as the deterministic wait-and-see plan
    return StochasticSolution(f"Deterministic {s}", _first_stage_for_h(h, first_budget, first_beta), {s: y_by_s[s]}, {s: scenario_values[s]}, expected, h)


def _expected_beta_s() -> Dict[str, float]:
    return {j: sum(PROBS[s] * BETA_S[(s, j)] for s in SCENARIOS) for j in ITEMS}


def optimize_expected_value(first_budget: float = 65.0, reserve_budget: float = 15.0, step: float = 0.01) -> Tuple[StochasticSolution, float]:
    """Solve the deterministic expected-value model and evaluate it under scenarios.

    EV model uses one average recourse vector rather than scenario-adaptive recourse.
    EEV fixes x_EV, then allows normal second-stage recourse in each scenario.
    """
    beta_ev = _expected_beta_s()

    def ev_det_value(h: float) -> float:
        x = _first_stage_for_h(h, first_budget, beta_ev)
        first = _first_value(x, beta_ev)
        # choose one recourse plan under expected beta_s.
        y = {j: 0.0 for j in ITEMS}
        best_non = max([j for j in ITEMS if j != "AI"], key=lambda j: beta_ev[j])
        if beta_ev["AI"] > beta_ev[best_non]:
            y_ai = min(reserve_budget, 0.5 * h)
            y["AI"] = y_ai
            y[best_non] = reserve_budget - y_ai
        else:
            y[best_non] = reserve_budget
        return first + sum(beta_ev[j] * y[j] for j in ITEMS)

    grid = np.arange(0.0, first_budget + step, step)
    vals = np.array([ev_det_value(float(h)) for h in grid])
    h = float(grid[int(np.argmax(vals))])
    expected_eval, y_by_s, scenario_values = _value_for_h(h, PROBS, first_budget, reserve_budget, BETA)
    sol = StochasticSolution("EV - expected value", _first_stage_for_h(h, first_budget, beta_ev), y_by_s, scenario_values, expected_eval, h)
    return sol, float(np.max(vals))


def compute_ws_evpi_vss(first_budget: float = 65.0, reserve_budget: float = 15.0, step: float = 0.01) -> Dict[str, object]:
    sp = optimize_stochastic(first_budget, reserve_budget, step)
    ev, ev_det = optimize_expected_value(first_budget, reserve_budget, step)
    det = {s: optimize_deterministic_scenario(s, first_budget, reserve_budget, step) for s in SCENARIOS}
    ws = sum(PROBS[s] * det[s].expected_value for s in SCENARIOS)
    return {
        "sp": sp,
        "ev": ev,
        "deterministic": det,
        "WS": ws,
        "SP": sp.expected_value,
        "EEV": ev.expected_value,
        "EV_det_objective": ev_det,
        "VSS": sp.expected_value - ev.expected_value,
        "EVPI": ws - sp.expected_value,
    }


def optimize_robust_minimax_regret(first_budget: float = 65.0, reserve_budget: float = 15.0, step: float = 0.01) -> Dict[str, object]:
    best_by_s = {s: optimize_deterministic_scenario(s, first_budget, reserve_budget, step).expected_value for s in SCENARIOS}
    grid = np.arange(0.0, first_budget + step, step)
    records = []
    for h in grid:
        _, _, val_by_s_base = _value_for_h(float(h), PROBS, first_budget, reserve_budget, BETA)
        # Regret is scenario-specific: value the same here-and-now x under each realized scenario beta.
        val_by_s = {}
        for s in SCENARIOS:
            first_beta = {j: BETA_S[(s, j)] for j in ITEMS}
            _, _, tmp_val = _value_for_h(float(h), {ss: 1.0 if ss == s else 0.0 for ss in SCENARIOS}, first_budget, reserve_budget, first_beta)
            val_by_s[s] = tmp_val[s]
        regrets = {s: best_by_s[s] - val_by_s[s] for s in SCENARIOS}
        max_regret = max(regrets.values())
        exp_val = sum(PROBS[s] * val_by_s[s] for s in SCENARIOS)
        records.append((max_regret, -exp_val, float(h), regrets, val_by_s))
    records.sort(key=lambda x: (x[0], x[1]))
    max_regret, neg_exp, h, regrets, val_by_s = records[0]
    _, y_by_s, scenario_values = _value_for_h(h, PROBS, first_budget, reserve_budget, BETA)
    return {
        "solution": StochasticSolution("Robust minimax regret", _first_stage_for_h(h, first_budget, BETA), y_by_s, scenario_values, -neg_exp, h),
        "regrets": regrets,
        "max_regret": max_regret,
        "best_by_s": best_by_s,
    }


def first_stage_table(*solutions: StochasticSolution) -> pd.DataFrame:
    rows = []
    for sol in solutions:
        row = {"Mô hình": sol.name, "Expected value": sol.expected_value, "x_H để mở khóa AI": sol.h_value}
        row.update({f"x_{j} ({ITEM_NAMES[j]})": sol.first_stage[j] for j in ITEMS})
        rows.append(row)
    return pd.DataFrame(rows)


def recourse_table(sol: StochasticSolution) -> pd.DataFrame:
    rows = []
    for s, y in sol.second_stage.items():
        row = {"Kịch bản": f"{s} - {SCENARIO_NAMES[s]}", "Giá trị theo kịch bản": sol.scenario_values.get(s, np.nan)}
        row.update({f"y_{j}": y.get(j, 0.0) for j in ITEMS})
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_value_table(sol: StochasticSolution) -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        rows.append({
            "Kịch bản": f"{s} - {SCENARIO_NAMES[s]}",
            "Xác suất": PROBS[s],
            "Giá trị": sol.scenario_values[s],
            "Đóng góp kỳ vọng": PROBS[s] * sol.scenario_values[s],
        })
    return pd.DataFrame(rows)


def allocation_long_table(sol: StochasticSolution) -> pd.DataFrame:
    rows = []
    for j in ITEMS:
        rows.append({"Giai đoạn": "First-stage", "Kịch bản": "Here-and-now", "Hạng mục": j, "Ngân sách": sol.first_stage[j]})
    for s, y in sol.second_stage.items():
        for j in ITEMS:
            rows.append({"Giai đoạn": "Second-stage", "Kịch bản": s, "Hạng mục": j, "Ngân sách": y.get(j, 0.0)})
    return pd.DataFrame(rows)

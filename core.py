"""
AIDEOM-VN core calculations for 12 exercises.
All functions are deterministic and runnable without Streamlit.
Units follow the assignment: money is generally trillion VND unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

DATA_DIR = Path(__file__).resolve().parent / "data"

# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

def load_macro(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "vietnam_macro_2020_2025.csv")
    return df.sort_values("year").reset_index(drop=True)


def load_sectors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "vietnam_sectors_2024.csv")


def load_regions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "vietnam_regions_2024.csv")


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

# -----------------------------------------------------------------------------
# Bai 1
# -----------------------------------------------------------------------------

def bai1_cobb_douglas(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07) -> Dict[str, Any]:
    df = load_macro().copy()
    # The assignment provides these calibrated series.
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], dtype=float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)
    Y = df["GDP_trillion_VND"].values.astype(float)
    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_bar = A.mean()
    Y_hat = A_bar * K**alpha * L**beta * D**gamma * AI**delta * H**theta

    # Growth accounting using log differences and annual averages.
    variables = {
        "K - Vốn vật chất": K,
        "L - Lao động": L,
        "D - Số hóa": D,
        "AI - Năng lực AI": AI,
        "H - Nhân lực số": H,
        "TFP - A": A,
    }
    weights = {
        "K - Vốn vật chất": alpha,
        "L - Lao động": beta,
        "D - Số hóa": gamma,
        "AI - Năng lực AI": delta,
        "H - Nhân lực số": theta,
        "TFP - A": 1.0,
    }
    avg_dln_y = np.diff(np.log(Y)).mean()
    rows = []
    for name, arr in variables.items():
        contribution = weights[name] * np.diff(np.log(arr)).mean()
        rows.append({
            "Yếu tố": name,
            "Đóng góp log bình quân": contribution,
            "Đóng góp vào tăng trưởng (%)": contribution * 100,
            "Tỷ trọng trong tăng trưởng GDP (%)": 100 * contribution / avg_dln_y if avg_dln_y != 0 else np.nan,
        })
    growth_table = pd.DataFrame(rows)

    # 2030 simulation from 2025.
    K2030 = K[-1] * (1.06 ** 5)
    L2030 = L[-1] * (1.06 ** 5)
    D2030 = 30.0
    AI2030 = 100.0
    H2030 = 35.0
    A2030 = A[-1] * (1.012 ** 5)
    Y2030 = A2030 * K2030**alpha * L2030**beta * D2030**gamma * AI2030**delta * H2030**theta

    out = df[["year", "GDP_trillion_VND", "GDP_growth_pct", "digital_economy_share_GDP_pct", "labor_productivity_million_VND"]].copy()
    out["K"] = K; out["L"] = L; out["D"] = D; out["AI"] = AI; out["H"] = H; out["A_TFP"] = A; out["Y_hat"] = Y_hat
    return {
        "table": out,
        "A": A,
        "A_bar": A_bar,
        "MAPE": mape(Y, Y_hat),
        "growth_table": growth_table,
        "Y2030": float(Y2030),
        "scenario2030": {"K2030": K2030, "L2030": L2030, "D2030": D2030, "AI2030": AI2030, "H2030": H2030, "A2030": A2030},
    }

# -----------------------------------------------------------------------------
# Bai 2
# -----------------------------------------------------------------------------

def solve_bai2_lp(budget=100.0, min_h=20.0) -> Dict[str, Any]:
    c = np.array([-0.85, -1.20, -0.95, -1.35])
    A_ub = np.array([
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ], dtype=float)
    b_ub = np.array([budget, -25, -15, -min_h, -10, 0], dtype=float)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*4, method="highs")
    names = ["x1 Hạ tầng số", "x2 AI & dữ liệu", "x3 Nhân lực số", "x4 R&D"]
    if not res.success:
        return {"success": False, "message": res.message, "x": None, "Z": np.nan, "dual": None}
    # For the original max problem, shadow price = - marginal of minimization constraint.
    dual = None
    try:
        dual = -pd.Series(res.ineqlin.marginals, index=["Ngân sách", "x1_min", "x2_min", "x3_min", "x4_min", "AI+R&D>=35%"])
    except Exception:
        pass
    return {"success": True, "message": res.message, "x": pd.Series(res.x, index=names), "Z": float(-res.fun), "dual": dual, "raw": res}


def bai2_all() -> Dict[str, Any]:
    base = solve_bai2_lp(100, 20)
    budgets = [100, 120, 140]
    sens_rows = []
    for B in budgets:
        r = solve_bai2_lp(B, 20)
        sens_rows.append({"Ngân sách": B, "Z*": r["Z"], **({k: v for k, v in r["x"].to_dict().items()} if r["success"] else {})})
    h30 = solve_bai2_lp(100, 30)
    return {"base": base, "sensitivity": pd.DataFrame(sens_rows), "h30": h30}

# -----------------------------------------------------------------------------
# Bai 3
# -----------------------------------------------------------------------------

def minmax_good(s: pd.Series) -> pd.Series:
    d = s.max() - s.min()
    return (s - s.min()) / d if d else s * 0


def minmax_low_is_good(s: pd.Series) -> pd.Series:
    d = s.max() - s.min()
    return (s.max() - s) / d if d else s * 0


def bai3_priority(ai_weight=0.20, scheme="default") -> Dict[str, Any]:
    df = load_sectors().copy()
    cols = ["growth_rate_2024_pct", "gdp_share_2024_pct", "spillover_coef_0_1", "export_billion_USD", "labor_million", "ai_readiness_0_100", "automation_risk_pct"]
    norm = pd.DataFrame({
        "Growth": minmax_good(df["growth_rate_2024_pct"]),
        "Productivity/GDP share": minmax_good(df["gdp_share_2024_pct"]),
        "Spillover": minmax_good(df["spillover_coef_0_1"]),
        "Export": minmax_good(df["export_billion_USD"]),
        "Employment": minmax_good(df["labor_million"]),
        "AI Readiness": minmax_good(df["ai_readiness_0_100"]),
        "Risk safe": minmax_low_is_good(df["automation_risk_pct"]),
    })
    if scheme == "growth":
        w = np.array([0.25, 0.25, 0.10, 0.20, 0.05, 0.10, 0.05])
    elif scheme == "inclusive":
        w = np.array([0.10, 0.10, 0.20, 0.05, 0.25, 0.10, 0.20])
    else:
        # AI weight can be varied; other weights rescaled from default excluding AI.
        base = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])
        other_sum = base.sum() - base[5]
        w = base.copy()
        w[5] = ai_weight
        w[np.arange(len(w)) != 5] *= (1 - ai_weight) / other_sum
    score = norm.values @ w
    result = df[["sector_id", "sector_name_vi", "sector_name_en"]].copy()
    result["Priority"] = score
    result["Rank"] = result["Priority"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("Priority", ascending=False).reset_index(drop=True)
    return {"normalized": pd.concat([df[["sector_name_vi"]], norm], axis=1), "ranking": result, "weights": pd.Series(w, index=norm.columns)}


def bai3_sensitivity() -> pd.DataFrame:
    rows = []
    for w_ai in np.arange(0.05, 0.401, 0.05):
        ranking = bai3_priority(ai_weight=float(w_ai))["ranking"]
        for _, row in ranking.iterrows():
            rows.append({"w_AI": round(float(w_ai), 2), "Ngành": row["sector_name_vi"], "Rank": int(row["Rank"]), "Priority": float(row["Priority"])})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Bai 4
# -----------------------------------------------------------------------------

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}
ITEMS = ["I", "D", "AI", "H"]
BETA4 = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])
D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)


def solve_bai4_lp(fairness=True, total_budget=50000.0, region_floor=5000.0, region_cap=12000.0, soft=True) -> Dict[str, Any]:
    """Solve Bai 4 LP.

    Note: with the exact PDF parameters gamma=0.002, lambda=0.7, region cap=12,000,
    the hard fairness constraint is infeasible because the weakest region cannot catch
    up to 70% of the strongest initial digital index. Therefore the default uses a
    soft fairness slack with a large penalty; set soft=False to check hard feasibility.
    """
    n_x = 24
    if fairness and soft:
        n_var = n_x + 1 + 6  # x, M, slack per region
    else:
        n_var = n_x + (1 if fairness else 0)
    c = np.zeros(n_var)
    c[:n_x] = -BETA4.flatten()
    if fairness and soft:
        c[n_x+1:] = 3000.0  # penalty for fairness shortfall in digital-index points
    A_ub, b_ub = [], []
    row = np.zeros(n_var); row[:n_x] = 1; A_ub.append(row); b_ub.append(total_budget)
    for r in range(6):
        idx = slice(r*4, r*4+4)
        row = np.zeros(n_var); row[idx] = 1; A_ub.append(row); b_ub.append(region_cap)
        row = np.zeros(n_var); row[idx] = -1; A_ub.append(row); b_ub.append(-region_floor)
    row = np.zeros(n_var)
    for r in range(6): row[r*4+3] = -1
    A_ub.append(row); b_ub.append(-12000)
    if fairness:
        M_idx = n_x
        gamma, lam = 0.002, 0.7
        for r in range(6):
            row = np.zeros(n_var); row[r*4+1] = gamma; row[M_idx] = -1
            A_ub.append(row); b_ub.append(-D0[r])
        for r in range(6):
            row = np.zeros(n_var); row[r*4+1] = -gamma; row[M_idx] = lam
            if soft:
                row[n_x+1+r] = -1
            A_ub.append(row); b_ub.append(D0[r])
    bounds = [(0, None)] * n_var
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success and fairness and not soft:
        return {"success": False, "message": res.message, "hard_fairness_infeasible": True}
    if not res.success:
        return {"success": False, "message": res.message}
    X = res.x[:n_x].reshape(6, 4)
    alloc = pd.DataFrame(X, index=[REGION_NAMES[r] for r in REGIONS], columns=["I Hạ tầng", "D CĐS DN", "AI", "H Nhân lực"])
    true_gdp_gain = float((BETA4 * X).sum())
    slack = None
    if fairness and soft:
        slack = pd.Series(res.x[n_x+1:], index=[REGION_NAMES[r] for r in REGIONS], name="Fairness slack")
    return {"success": True, "Z": true_gdp_gain, "allocation": alloc, "M": float(res.x[n_x]) if fairness else None, "slack": slack, "raw": res}

# -----------------------------------------------------------------------------
# Bai 5
# -----------------------------------------------------------------------------

PROJECTS = pd.DataFrame([
    [1, "Trung tâm dữ liệu quốc gia Hòa Lạc", "Hạ tầng", 12000, 21500, 8500, 3500],
    [2, "Trung tâm dữ liệu quốc gia phía Nam", "Hạ tầng", 11500, 20800, 7500, 4000],
    [3, "Hệ thống 5G phủ sóng toàn quốc", "Hạ tầng", 18000, 32500, 12000, 6000],
    [4, "Hệ thống định danh điện tử VNeID 2.0", "Chính phủ số", 4500, 9200, 3500, 1000],
    [5, "Cổng dịch vụ công quốc gia v3", "Chính phủ số", 3200, 6800, 2500, 700],
    [6, "Y tế số quốc gia", "Y tế số", 5800, 11400, 4000, 1800],
    [7, "Giáo dục số K-12 toàn quốc", "Giáo dục", 6500, 12200, 4500, 2000],
    [8, "Trung tâm AI quốc gia + supercomputing", "AI", 15000, 28500, 9000, 6000],
    [9, "Sandbox tài chính số", "Tài chính số", 2500, 5800, 1800, 700],
    [10, "Logistics thông minh + cảng biển số", "Logistics", 7200, 13800, 5000, 2200],
    [11, "Nông nghiệp số ĐBSCL", "Nông nghiệp", 4800, 8500, 3500, 1300],
    [12, "Đào tạo 50.000 kỹ sư AI/bán dẫn", "Nhân lực", 8500, 16200, 5500, 3000],
    [13, "Khu CN bán dẫn Bắc Ninh - Bắc Giang", "Bán dẫn", 20000, 35000, 13000, 7000],
    [14, "An ninh mạng quốc gia (SOC)", "An ninh", 3800, 7500, 2800, 1000],
    [15, "Open Data + dữ liệu mở quốc gia", "Dữ liệu", 1500, 3800, 1200, 300],
], columns=["id", "name", "sector", "cost", "benefit", "year12", "year35"])


def solve_bai5(budget=80000, force_p1_p2=False, expected=False) -> Dict[str, Any]:
    df = PROJECTS.copy()
    benefit_col = "benefit"
    if expected:
        probs = []
        for sec in df["sector"]:
            if sec == "Hạ tầng": probs.append(0.85)
            elif sec == "Chính phủ số": probs.append(0.75)
            elif sec in ["AI", "Bán dẫn"]: probs.append(0.65)
            else: probs.append(0.80)
        df["p"] = probs
        df["expected_benefit"] = df["benefit"] * df["p"]
        benefit_col = "expected_benefit"
    best = None
    rows = df.to_dict("records")
    # 2^15 brute force is fast and avoids external MIP solvers.
    for mask in range(1 << len(rows)):
        chosen = [rows[i] for i in range(len(rows)) if (mask >> i) & 1]
        n = len(chosen)
        if n < 7 or n > 11: continue
        ids = {p["id"] for p in chosen}
        if sum(p["cost"] for p in chosen) > budget: continue
        if sum(p["year12"] for p in chosen) > 40000: continue
        if 1 in ids and 2 in ids and not force_p1_p2: continue
        if force_p1_p2 and not ({1, 2}.issubset(ids)): continue
        if 8 in ids and 12 not in ids: continue
        if 13 in ids and 12 not in ids: continue
        if not ({4, 5} & ids): continue
        if 14 not in ids: continue
        val = sum(p[benefit_col] for p in chosen)
        if best is None or val > best["Z"]:
            best = {"Z": val, "chosen_ids": sorted(ids), "chosen": pd.DataFrame(chosen), "cost": sum(p["cost"] for p in chosen), "year12": sum(p["year12"] for p in chosen)}
    if best is None:
        return {"success": False, "message": "Không tìm thấy nghiệm khả thi."}
    best["success"] = True
    best["npv_per_cost"] = best["Z"] / best["cost"]
    return best

# -----------------------------------------------------------------------------
# Bai 6
# -----------------------------------------------------------------------------

def topsis(df: pd.DataFrame, criteria: List[str], weights: np.ndarray, benefit: List[bool]) -> pd.DataFrame:
    X = df[criteria].values.astype(float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1
    R = X / denom
    V = R * weights
    benefit_arr = np.array(benefit)
    A_star = np.where(benefit_arr, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit_arr, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C = S_neg / (S_star + S_neg)
    out = df[["region_id", "region_name_vi", "region_name_en"]].copy()
    out["TOPSIS_score"] = C
    out["Rank"] = out["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)


def entropy_weights(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    # make all values positive and normalize by column
    X = X - X.min(axis=0) + 1e-9
    P = X / X.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    if d.sum() == 0:
        return np.ones(X.shape[1]) / X.shape[1]
    return d / d.sum()


def bai6_topsis(ai_weight=0.20) -> Dict[str, Any]:
    df = load_regions()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
    benefit = [True, True, True, True, True, True, True, False]
    base = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    other_sum = base.sum() - base[3]
    w = base.copy(); w[3] = ai_weight; w[np.arange(len(w)) != 3] *= (1 - ai_weight) / other_sum
    expert = topsis(df, criteria, w, benefit)
    X = df[criteria].values.astype(float).copy()
    # Turn Gini into benefit before entropy weighting to avoid cost direction issue.
    X[:, -1] = X[:, -1].max() - X[:, -1]
    ew = entropy_weights(X)
    entropy = topsis(df, criteria, ew, benefit)
    return {"expert": expert, "entropy": entropy, "weights_expert": pd.Series(w, index=criteria), "weights_entropy": pd.Series(ew, index=criteria), "criteria": criteria}


def bai6_sensitivity() -> pd.DataFrame:
    rows = []
    for w_ai in np.arange(0.10, 0.401, 0.05):
        rank = bai6_topsis(ai_weight=float(w_ai))["expert"]
        for _, r in rank.iterrows():
            rows.append({"w_AI": round(float(w_ai), 2), "Vùng": r["region_name_vi"], "Rank": int(r["Rank"]), "Score": float(r["TOPSIS_score"])})
    return pd.DataFrame(rows)


def bai6_ahp_simple() -> pd.DataFrame:
    # AHP-like direct pairwise priority approximated by normalized expert weights.
    # In a real AHP, these would come from a Saaty matrix; here we compare ranking with TOPSIS.
    res = bai6_topsis()
    return res["expert"].copy()

# -----------------------------------------------------------------------------
# Bai 7
# -----------------------------------------------------------------------------

def gini_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.abs(x - x.mean()).mean())


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    # F columns all minimization.
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]: continue
        dominated_by_any = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if dominated_by_any:
            mask[i] = False
    return mask


def generate_feasible_allocations(n=700, seed=42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    allocs = []
    # Add some anchor solutions.
    for fair in [True, False]:
        r = solve_bai4_lp(fairness=fair)
        if r.get("success"):
            allocs.append(r["allocation"].values)
    attempts = 0
    while len(allocs) < n and attempts < n * 100:
        attempts += 1
        total = rng.uniform(30000, 50000)  # C1 is <= 50,000.
        excess = max(total - 6*5000, 0)
        totals = 5000 + rng.dirichlet(np.ones(6)*1.2) * excess
        if np.any(totals > 12000):
            continue
        X = np.zeros((6, 4))
        for r in range(6):
            alpha = rng.choice([
                [1.4, 1.0, 1.0, 1.2],
                [0.6, 0.8, 0.5, 2.4],
                [1.0, 1.8, 0.7, 1.2],
                [1.2, 0.7, 1.9, 0.9],
            ])
            weights = rng.dirichlet(alpha)
            X[r] = totals[r] * weights
        if X[:, 3].sum() < 12000:
            continue
        # The exact PDF C5 is infeasible with the provided cap, so Pareto generation keeps C1-C4
        # and lets the inequality objective represent the fairness trade-off.
        allocs.append(X)
    return allocs


def bai7_pareto(n_samples=700, seed=42) -> Dict[str, Any]:
    allocs = generate_feasible_allocations(n=n_samples, seed=seed)
    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    rows = []
    for i, X in enumerate(allocs):
        f1_gain = float((BETA4 * X).sum())
        f2 = gini_mad(X.sum(axis=1))
        f3 = float((e * (X[:, 0] + X[:, 2])).sum())
        f4 = float((rho * X[:, 2]).sum() - (sig * X[:, 3]).sum())
        rows.append({"id": i, "GDP_gain": f1_gain, "Inequality_MAD": f2, "Emission": f3, "SecurityRisk": f4, "allocation": X})
    df = pd.DataFrame(rows)
    F_min = np.column_stack([-df["GDP_gain"].values, df["Inequality_MAD"].values, df["Emission"].values, df["SecurityRisk"].values])
    mask = nondominated_mask(F_min)
    pareto = df.loc[mask].reset_index(drop=True)
    # TOPSIS on Pareto set: GDP benefit; other three costs.
    P = pareto[["GDP_gain", "Inequality_MAD", "Emission", "SecurityRisk"]].values.astype(float)
    # Normalize benefit/cost to closeness using min-max benefit-coded.
    benefit_coded = np.column_stack([
        minmax_good(pd.Series(P[:, 0])).values,
        minmax_low_is_good(pd.Series(P[:, 1])).values,
        minmax_low_is_good(pd.Series(P[:, 2])).values,
        minmax_low_is_good(pd.Series(P[:, 3])).values,
    ])
    weights = np.array([0.40, 0.25, 0.20, 0.15])
    score = benefit_coded @ weights
    pareto["Compromise_score"] = score
    best = pareto.iloc[int(np.argmax(score))]
    max_growth = pareto.iloc[int(np.argmax(pareto["GDP_gain"].values))]
    return {"all": df.drop(columns=["allocation"]), "pareto": pareto.drop(columns=["allocation"]), "best": best, "best_allocation": best["allocation"], "max_growth": max_growth}

# -----------------------------------------------------------------------------
# Bai 8
# -----------------------------------------------------------------------------

def simulate_dynamic(strategy="balanced", shock=False, rho_discount=0.97, seed=42) -> pd.DataFrame:
    years = np.arange(2026, 2036)
    T = len(years)
    params = {"dK": 0.05, "dD": 0.12, "dAI": 0.15, "eta": 0.8, "mu": 0.02, "phi1": 0.003, "phi2": 0.002, "phi3": 0.004}
    K, D, AI, H, A, L = 27500.0, 20.3, 86.0, 30.0, bai1_cobb_douglas()["A"][-1] * 1.012, 53.9
    if strategy == "traditional": shares = np.tile([0.70, 0.10, 0.10, 0.10], (T, 1)); inv_rate = np.full(T, 0.24)
    elif strategy == "front_load": shares = np.tile([0.35, 0.25, 0.20, 0.20], (T, 1)); inv_rate = np.array([0.34,0.33,0.31,0.25,0.24,0.23,0.22,0.21,0.20,0.19])
    elif strategy == "even": shares = np.tile([0.40, 0.25, 0.15, 0.20], (T, 1)); inv_rate = np.full(T, 0.25)
    elif strategy == "ai_led": shares = np.tile([0.20, 0.20, 0.45, 0.15], (T, 1)); inv_rate = np.full(T, 0.27)
    elif strategy == "inclusive": shares = np.tile([0.30, 0.20, 0.10, 0.40], (T, 1)); inv_rate = np.full(T, 0.26)
    else: # balanced optimized-like
        shares = np.tile([0.36, 0.24, 0.18, 0.22], (T, 1)); inv_rate = np.linspace(0.30, 0.22, T)
    rows = []
    welfare = 0
    for t, y in enumerate(years):
        Y_plan = A * (K**0.33) * (L**0.42) * (D**0.10) * (AI**0.08) * (H**0.07)
        Y = Y_plan * (0.92 if (shock and y == 2028) else 1.0)
        I_total = inv_rate[t] * Y
        IK, ID, IAI, IH = shares[t] * I_total
        C = max(Y - I_total, 1e-6)
        welfare += (rho_discount ** t) * np.log(C)
        rows.append({"year": y, "Y": Y, "C": C, "K": K, "D": D, "AI": AI, "H": H, "A": A, "I_K": IK, "I_D": ID, "I_AI": IAI, "I_H": IH, "welfare_cum": welfare})
        K = (1-params["dK"]) * K + IK
        D = (1-params["dD"]) * D + ID / 1000.0
        AI = (1-params["dAI"]) * AI + IAI / 100.0
        H = H + params["eta"] * IH / 1000.0 - params["mu"] * H
        A = A * (1 + params["phi1"] * D/100 + params["phi2"] * AI/100 + params["phi3"] * H/100)
        L *= 1.006
    return pd.DataFrame(rows)


def bai8_compare() -> pd.DataFrame:
    rows = []
    for strategy in ["even", "front_load", "balanced", "traditional", "ai_led", "inclusive"]:
        df = simulate_dynamic(strategy=strategy)
        rows.append({"Chiến lược": strategy, "Y2035": df.iloc[-1]["Y"], "Welfare": df.iloc[-1]["welfare_cum"], "C2035": df.iloc[-1]["C"]})
    return pd.DataFrame(rows).sort_values("Welfare", ascending=False)

# -----------------------------------------------------------------------------
# Bai 9
# -----------------------------------------------------------------------------

LABOR_SECTORS = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"]
LABOR_L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
LABOR_RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
LABOR_A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
LABOR_A2 = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0])
LABOR_B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
LABOR_C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
LABOR_D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])


def solve_bai9(no_large_loss=False, budget=30000.0) -> Dict[str, Any]:
    N = 8
    # variables [x_AI_0..7, x_H_0..7]
    net_ai = LABOR_A1 - LABOR_C1 * LABOR_RISK
    net_h = LABOR_B1
    c = -np.r_[net_ai, net_h]
    A_ub, b_ub = [], []
    A_ub.append(np.ones(2*N)); b_ub.append(budget)
    # NetJob_i >= 0 -> -net_ai*xAI - net_h*xH <= 0
    for i in range(N):
        row = np.zeros(2*N); row[i] = -net_ai[i]; row[N+i] = -net_h[i]
        A_ub.append(row); b_ub.append(0)
    # Displaced <= retrain capacity -> c1*risk*xAI - d1*xH <=0
    for i in range(N):
        row = np.zeros(2*N); row[i] = LABOR_C1[i] * LABOR_RISK[i]; row[N+i] = -LABOR_D1[i]
        A_ub.append(row); b_ub.append(0)
    if no_large_loss:
        for i in range(N):
            row = np.zeros(2*N); row[i] = LABOR_C1[i] * LABOR_RISK[i]
            A_ub.append(row); b_ub.append(0.05 * LABOR_L[i] * 1_000_000)  # jobs
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)]*(2*N), method="highs")
    if not res.success:
        return {"success": False, "message": res.message}
    xAI = res.x[:N]; xH = res.x[N:]
    newjob = LABOR_A1*xAI
    displaced = LABOR_C1*LABOR_RISK*xAI
    upgrade = LABOR_B1*xH
    retrain = LABOR_D1*xH
    net = newjob + upgrade - displaced
    tbl = pd.DataFrame({"Ngành": LABOR_SECTORS, "Lao động (triệu)": LABOR_L, "Risk": LABOR_RISK, "x_AI": xAI, "x_H": xH, "NewJob": newjob, "UpgradeJob": upgrade, "DisplacedJob": displaced, "RetrainingCapacity": retrain, "NetJob": net})
    return {"success": True, "Z": float(net.sum()), "table": tbl, "raw": res}


def bai9_threshold_industry2(x_ai=30000.0) -> Dict[str, float]:
    i = 1
    threshold_net = max(0.0, (LABOR_C1[i] * LABOR_RISK[i] - LABOR_A1[i]) * x_ai / LABOR_B1[i])
    threshold_retrain = (LABOR_C1[i] * LABOR_RISK[i] * x_ai) / LABOR_D1[i]
    return {"x_AI ngành 2": x_ai, "x_H tối thiểu theo NetJob>=0": threshold_net, "x_H tối thiểu theo Displaced<=Retrain": threshold_retrain, "x_H tối thiểu cần chọn": max(threshold_net, threshold_retrain)}

# -----------------------------------------------------------------------------
# Bai 10
# -----------------------------------------------------------------------------

SCENARIOS = ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"]
PROBS = np.array([0.30, 0.45, 0.20, 0.05])
BETA_BASE = np.array([1.00, 1.10, 1.25, 0.95])
BETA_S = np.array([
    [1.25, 1.35, 1.55, 1.05],
    [1.00, 1.10, 1.25, 0.95],
    [0.75, 0.85, 0.90, 1.00],
    [0.40, 0.50, 0.55, 1.10],
])
ITEM_NAMES = ["I Hạ tầng", "D CĐS", "AI", "H Nhân lực"]


def solve_sp_model(beta_s=BETA_S, probs=PROBS, force_expected=None) -> Dict[str, Any]:
    # variables x4 + y 4*4 = 20
    n = 20
    c = np.zeros(n)
    c[:4] = -BETA_BASE
    for s in range(4):
        c[4+s*4:4+(s+1)*4] = -probs[s] * beta_s[s]
    A_ub, b_ub = [], []
    row = np.zeros(n); row[:4] = 1; A_ub.append(row); b_ub.append(65000)
    for s in range(4):
        row = np.zeros(n); row[4+s*4:4+(s+1)*4] = 1; A_ub.append(row); b_ub.append(15000)
        # y_AI_s <= 0.5*x_H -> y_AI - 0.5*xH <= 0
        row = np.zeros(n); row[4+s*4+2] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, None)] * n
    if force_expected is not None:
        # force x to supplied vector
        bounds[:4] = [(float(v), float(v)) for v in force_expected]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        return {"success": False, "message": res.message}
    x = res.x[:4]
    y = res.x[4:].reshape(4,4)
    return {"success": True, "Z": float(-res.fun), "x": pd.Series(x, index=ITEM_NAMES), "y": pd.DataFrame(y, index=SCENARIOS, columns=ITEM_NAMES), "raw": res}


def solve_deterministic_scenario(s_idx: int) -> Dict[str, Any]:
    # Solve as if scenario s is certain: beta_s for all probability mass.
    probs = np.zeros(4); probs[s_idx] = 1.0
    return solve_sp_model(BETA_S, probs=probs)


def bai10_all() -> Dict[str, Any]:
    sp = solve_sp_model()
    beta_ev = np.tile((PROBS[:, None] * BETA_S).sum(axis=0), (4,1))
    ev_solve = solve_sp_model(beta_s=beta_ev, probs=PROBS)
    ev_eval = solve_sp_model(force_expected=ev_solve["x"].values)
    dets = [solve_deterministic_scenario(i) for i in range(4)]
    ws = sum(PROBS[i] * dets[i]["Z"] for i in range(4))
    vss = sp["Z"] - ev_eval["Z"]
    evpi = ws - sp["Z"]
    return {"sp": sp, "ev_solve": ev_solve, "ev_eval": ev_eval, "deterministic": dets, "WS": ws, "VSS": vss, "EVPI": evpi}

# -----------------------------------------------------------------------------
# Bai 11
# -----------------------------------------------------------------------------

ACTIONS = {
    0: ("a0 Truyền thống", np.array([0.70, 0.10, 0.10, 0.10])),
    1: ("a1 Cân bằng", np.array([0.40, 0.25, 0.15, 0.20])),
    2: ("a2 Số hóa nhanh", np.array([0.25, 0.45, 0.15, 0.15])),
    3: ("a3 AI dẫn dắt", np.array([0.20, 0.20, 0.45, 0.15])),
    4: ("a4 Bao trùm", np.array([0.30, 0.20, 0.10, 0.40])),
}

class SimpleVietnamEnv:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.T = 10
        self.w = np.array([0.40, 0.25, 0.20, 0.15])
        self.reset()

    def reset(self, state=None):
        self.t = 0
        self.K = 27500.0; self.D = 20.3; self.AI = 86.0; self.H = 30.0
        self.prev_Y = self._Y()
        if state is None:
            self.state = np.array([1, 1, 0, 1], dtype=int)
        else:
            self.state = np.array(state, dtype=int)
        return self.state.copy()

    def _Y(self):
        return self.K**0.33 * 54.0**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07

    def _discretize(self, growth, cyber, emission):
        g = 0 if growth < 0.018 else (1 if growth < 0.045 else 2)
        d = 0 if self.D < 20 else (1 if self.D < 28 else 2)
        ai = 0 if self.AI < 95 else (1 if self.AI < 120 else 2)
        u_risk_raw = 0.08 + 0.06*(2-g) - 0.02*(self.H > 32) + 0.02*(cyber > 0.06)
        u = 0 if u_risk_raw < 0.10 else (1 if u_risk_raw < 0.17 else 2)
        return np.array([g, d, ai, u], dtype=int), u_risk_raw

    def step(self, action):
        share = ACTIONS[int(action)][1]
        budget = 1000.0
        self.K += share[0] * budget
        self.D += share[1] * budget / 100.0
        self.AI += share[2] * budget / 20.0
        self.H += share[3] * budget / 200.0
        Y = self._Y()
        growth = (Y - self.prev_Y) / max(abs(self.prev_Y), 1e-9)
        cyber = 0.02 + 0.10 * share[2] - 0.04 * share[3]
        emission = 0.05 + 0.06 * share[0] + 0.04 * share[2]
        new_state, u_risk = self._discretize(growth, cyber, emission)
        reward = 100 * (self.w[0]*growth - self.w[1]*u_risk - self.w[2]*cyber - self.w[3]*emission)
        self.prev_Y = Y
        self.state = new_state
        self.t += 1
        done = self.t >= self.T
        return new_state.copy(), float(reward), done


def train_q_learning(episodes=3000, seed=42) -> Dict[str, Any]:
    env = SimpleVietnamEnv(seed=seed)
    Q = np.zeros((3,3,3,3,5), dtype=float)
    rewards = []
    rng = np.random.default_rng(seed)
    for ep in range(int(episodes)):
        s = env.reset()
        total = 0.0
        eps = max(0.05, 1.0 - ep / max(1, episodes*0.65))
        while True:
            if rng.random() < eps:
                a = int(rng.integers(0,5))
            else:
                a = int(np.argmax(Q[tuple(s)]))
            s2, r, done = env.step(a)
            Q[tuple(s)+(a,)] += 0.1 * (r + 0.95 * Q[tuple(s2)].max() - Q[tuple(s)+(a,)])
            s = s2
            total += r
            if done:
                break
        rewards.append(total)
    return {"Q": Q, "rewards": pd.DataFrame({"episode": np.arange(1, len(rewards)+1), "reward": rewards})}


def evaluate_policy(policy, n=50, seed=123) -> float:
    vals = []
    for k in range(n):
        env = SimpleVietnamEnv(seed=seed+k)
        s = env.reset()
        total = 0.0
        while True:
            if callable(policy):
                a = policy(s)
            elif policy == "random":
                a = env.rng.integers(0,5)
            else:
                a = int(policy)
            s, r, done = env.step(a)
            total += r
            if done: break
        vals.append(total)
    return float(np.mean(vals))


def q_policy_table(Q: np.ndarray) -> pd.DataFrame:
    sample_states = {
        "VN 2026 thực tế: GDP medium, D medium, AI low, U medium": [1,1,0,1],
        "Suy giảm: GDP low, D low, AI low, U high": [0,0,0,2],
        "Số hóa tốt: GDP high, D high, AI medium, U low": [2,2,1,0],
        "AI cao nhưng thất nghiệp cao": [1,2,2,2],
        "Ổn định cao: GDP high, D high, AI high, U low": [2,2,2,0],
    }
    rows = []
    for label, state in sample_states.items():
        a = int(np.argmax(Q[tuple(state)]))
        rows.append({"Trạng thái": label, "state": str(state), "Hành động π*": ACTIONS[a][0]})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Bai 12 integration
# -----------------------------------------------------------------------------

def scenario_2030(name: str, share: np.ndarray) -> Dict[str, float]:
    # simplified integrated simulation to 2030.
    K, D, AI, H, A, L = 27500.0, 20.3, 86.0, 30.0, bai1_cobb_douglas()["A"][-1] * 1.012, 53.9
    welfare = 0.0
    total_risk = 0.0
    for t in range(5):
        Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
        I_total = 0.26 * Y
        IK, ID, IAI, IH = share * I_total
        C = Y - I_total
        welfare += (0.97**t)*np.log(C)
        total_risk += 0.02 + 0.10*share[2] + 0.05*share[0] - 0.04*share[3]
        K = 0.95*K + IK
        D = 0.88*D + ID/1000
        AI = 0.85*AI + IAI/100
        H = H + 0.8*IH/1000 - 0.02*H
        A *= (1 + 0.003*D/100 + 0.002*AI/100 + 0.004*H/100)
        L *= 1.006
    Y2030 = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
    return {"Kịch bản": name, "GDP_2030": float(Y2030), "Digital_D": float(D), "AI": float(AI), "H": float(H), "Welfare": float(welfare), "RiskIndex": float(total_risk/5)}


def bai12_scenarios() -> pd.DataFrame:
    scenarios = {
        "S1 Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
        "S2 Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
        "S3 AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
        "S4 Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
        "S5 Tối ưu cân bằng": np.array([0.36, 0.24, 0.18, 0.22]),
    }
    return pd.DataFrame([scenario_2030(k, v) for k, v in scenarios.items()]).sort_values("Welfare", ascending=False)

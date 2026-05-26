"""Bài 12 - Integrated AIDEOM-VN prototype using outputs from Bài 10 and 11."""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

SCENARIO_DESCRIPTIONS = {
    "S1": {"Tên": "Truyền thống", "Mô tả": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống", "Phân bổ": [0.70, 0.10, 0.10, 0.10]},
    "S2": {"Tên": "Số hóa nhanh", "Mô tả": "Tăng đầu tư chính phủ số, doanh nghiệp số, thành phố số", "Phân bổ": [0.25, 0.45, 0.15, 0.15]},
    "S3": {"Tên": "AI dẫn dắt", "Mô tả": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu", "Phân bổ": [0.20, 0.20, 0.45, 0.15]},
    "S4": {"Tên": "Bao trùm số", "Mô tả": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số", "Phân bổ": [0.30, 0.20, 0.10, 0.40]},
    "S5": {"Tên": "Tối ưu cân bằng", "Mô tả": "Kết quả mô phỏng cân bằng giữa tăng trưởng, việc làm và rủi ro", "Phân bổ": None},
}
COLS = ["K", "D", "AI", "H"]

MODULES = pd.DataFrame([
    ["M1", "Dự báo kinh tế", "Macro 2020-2025", "GDP, TFP, lao động 2030", "Cobb-Douglas mở rộng"],
    ["M2", "Đánh giá sẵn sàng số", "Sectors, Regions", "Digital Index + AI Readiness", "TOPSIS + entropy weight"],
    ["M3", "Tối ưu phân bổ", "Budget, β-matrix", "Phân bổ ngành-vùng-thời gian", "LP + stochastic programming"],
    ["M4", "Mô phỏng lao động", "AI, H plans", "Net job từng ngành", "Markov chain / RL"],
    ["M5", "Đánh giá rủi ro", "Risk parameters", "Cyber, environmental, dependency", "Đa mục tiêu + SP"],
    ["M6", "Dashboard ra QĐ", "Outputs M1-M5", "Trực quan kịch bản, cảnh báo, khuyến nghị", "Streamlit + Plotly"],
], columns=["Module", "Tên", "Đầu vào", "Đầu ra", "Kỹ thuật chính"])


def optimise_balanced_share() -> np.ndarray:
    """Grid-search S5 allocation on 5% increments."""
    best = None
    values = np.arange(0, 1.0001, 0.05)
    for k in values:
        for d in values:
            for ai in values:
                h = 1.0 - k - d - ai
                if h < -1e-9 or h > 1:
                    continue
                share = np.array([k, d, ai, h])
                metrics = score_allocation(share)
                # Balanced objective: strong GDP, plus jobs and readiness, minus risk.
                score = 0.45 * metrics["GDP_2030_index"] + 0.20 * metrics["Readiness"] + 0.20 * metrics["Net_jobs_index"] - 0.25 * metrics["Risk_index"]
                if best is None or score > best[0]:
                    best = (score, share)
    return best[1]


def scenario_table() -> pd.DataFrame:
    s5 = optimise_balanced_share()
    rows = []
    for code, item in SCENARIO_DESCRIPTIONS.items():
        share = s5 if item["Phân bổ"] is None else np.array(item["Phân bổ"], dtype=float)
        rows.append({
            "Kịch bản": code,
            "Tên": item["Tên"],
            "Mô tả": item["Mô tả"],
            "K": share[0], "D": share[1], "AI": share[2], "H": share[3],
            "Đặc điểm phân bổ": f"{share[0]*100:.0f}% K + {share[1]*100:.0f}% D + {share[2]*100:.0f}% AI + {share[3]*100:.0f}% H",
        })
    return pd.DataFrame(rows)


def score_allocation(share: np.ndarray) -> Dict[str, float]:
    k, d, ai, h = share
    # normalized synthetic indicators used by S5 grid and final scoring.
    gdp = 100 + 24 * k + 30 * d + 34 * ai + 18 * h + 8 * min(d, ai) + 5 * min(ai, h)
    readiness = 40 + 28 * d + 32 * ai + 22 * h + 8 * min(d, h)
    net_jobs = 50 + 12 * d - 22 * ai + 42 * h + 8 * min(ai, h)
    risk = 35 + 18 * k + 35 * ai + 12 * d - 30 * h - 7 * min(d, h)
    cyber = 25 + 45 * ai + 18 * d - 22 * h
    env = 20 + 42 * k + 28 * ai - 18 * d - 10 * h
    dep = 20 + 28 * ai + 15 * d - 15 * h
    return {
        "GDP_2030_index": float(gdp),
        "Readiness": float(np.clip(readiness, 0, 100)),
        "Net_jobs_index": float(net_jobs),
        "Risk_index": float(np.clip(risk, 0, 100)),
        "Cyber": float(np.clip(cyber, 0, 100)),
        "Environment": float(np.clip(env, 0, 100)),
        "Dependency": float(np.clip(dep, 0, 100)),
    }


def forecast_path(share: np.ndarray, start_year: int = 2025, end_year: int = 2030) -> pd.DataFrame:
    """Cobb-Douglas style scenario simulation from 2025 to 2030."""
    alpha, beta_l, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    Y0, K0, L0, D0, AI0, H0 = 12847.6, 25900.0, 53.4, 19.5, 80.1, 29.2
    A0 = Y0 / (K0**alpha * L0**beta_l * D0**gamma * AI0**delta * H0**theta)
    K, L, D, AI, H, A = K0, L0, D0, AI0, H0, A0
    rows = []
    for year in range(start_year, end_year + 1):
        Y = A * K**alpha * L**beta_l * D**gamma * AI**delta * H**theta
        rows.append({"Năm": year, "GDP": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H, "TFP": A})
        if year == end_year:
            break
        k_s, d_s, ai_s, h_s = share
        K *= 1.035 + 0.035 * k_s
        L *= 1.003 + 0.002 * h_s
        D = min(38.0, D + 0.45 + 5.2 * d_s + 0.5 * min(d_s, h_s))
        AI = min(130.0, AI + 2.0 + 32.0 * ai_s + 4.0 * min(ai_s, h_s))
        H = min(42.0, H + 0.25 + 4.6 * h_s)
        A *= 1.010 + 0.005 * d_s + 0.006 * ai_s + 0.004 * h_s
    return pd.DataFrame(rows)


def run_integrated_model() -> Dict[str, pd.DataFrame]:
    scen = scenario_table()
    summary_rows = []
    paths = []
    risk_rows = []
    for _, row in scen.iterrows():
        share = row[COLS].values.astype(float)
        path = forecast_path(share)
        path["Kịch bản"] = row["Kịch bản"]
        path["Tên"] = row["Tên"]
        paths.append(path)
        metrics = score_allocation(share)
        summary_rows.append({
            "Kịch bản": row["Kịch bản"], "Tên": row["Tên"],
            "GDP 2030": path.iloc[-1]["GDP"],
            "Tăng GDP 2025-2030 (%)": (path.iloc[-1]["GDP"] / path.iloc[0]["GDP"] - 1) * 100,
            **metrics,
        })
        for risk_name in ["Cyber", "Environment", "Dependency"]:
            risk_rows.append({"Kịch bản": row["Kịch bản"], "Tên": row["Tên"], "Loại rủi ro": risk_name, "Điểm": metrics[risk_name]})
    summary = pd.DataFrame(summary_rows).sort_values("GDP 2030", ascending=False)
    return {
        "scenario_table": scen,
        "modules": MODULES.copy(),
        "paths": pd.concat(paths, ignore_index=True),
        "summary": summary,
        "risks": pd.DataFrame(risk_rows),
    }


def recommendation(summary: pd.DataFrame) -> str:
    best_gdp = summary.sort_values("GDP 2030", ascending=False).iloc[0]
    best_jobs = summary.sort_values("Net_jobs_index", ascending=False).iloc[0]
    lowest_risk = summary.sort_values("Risk_index", ascending=True).iloc[0]
    balanced = summary.loc[summary["Kịch bản"] == "S5"].iloc[0]
    return (
        f"Kịch bản tăng GDP mạnh nhất là {best_gdp['Kịch bản']} - {best_gdp['Tên']}. "
        f"Kịch bản hỗ trợ việc làm tốt nhất là {best_jobs['Kịch bản']} - {best_jobs['Tên']}. "
        f"Kịch bản rủi ro thấp nhất là {lowest_risk['Kịch bản']} - {lowest_risk['Tên']}. "
        f"S5 có vai trò phương án thỏa hiệp: GDP 2030 khoảng {balanced['GDP 2030']:.1f} nghìn tỷ VND, "
        f"vừa giữ mức rủi ro {balanced['Risk_index']:.1f}/100 vừa duy trì chỉ số việc làm {balanced['Net_jobs_index']:.1f}."
    )

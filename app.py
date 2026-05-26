from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import core
from modules import bai10 as bai10_new, bai11 as bai11_new, bai12 as bai12_new

# -----------------------------------------------------------------------------
# Page config + global style
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AIDEOM-VN — Dashboard 12 bài",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --bg0:#070b18; --bg1:#0b1020; --bg2:#111827; --card:#111827cc;
  --line:#29364f; --muted:#94a3b8; --text:#f8fafc;
  --pink:#ec4899; --purple:#7c3aed; --cyan:#22d3ee; --green:#22c55e; --amber:#f59e0b;
}
html, body, [class*="css"] { font-family:'Inter','Segoe UI',system-ui,sans-serif !important; }
.block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1440px; }
[data-testid="stAppViewContainer"] { background:
  radial-gradient(circle at 20% 8%, rgba(236,72,153,.16), transparent 28%),
  radial-gradient(circle at 86% 18%, rgba(34,211,238,.12), transparent 24%),
  linear-gradient(135deg,#070b18 0%,#0b1020 55%,#0f172a 100%);
}
[data-testid="stSidebar"] { background: linear-gradient(180deg,#070b18 0%,#101827 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#e5e7eb; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#cbd5e1; }
h1 { font-weight:900 !important; letter-spacing:-.045em; line-height:1.04; }
h2, h3 { font-weight:800 !important; letter-spacing:-.025em; }
hr { border-color:var(--line); }

.hero {
  position:relative; overflow:hidden; border:1px solid rgba(148,163,184,.22); border-radius:28px;
  padding:28px 30px; margin-bottom:18px;
  background: linear-gradient(135deg, rgba(17,24,39,.92), rgba(15,23,42,.72));
  box-shadow:0 20px 55px rgba(0,0,0,.30);
}
.hero:after{
  content:""; position:absolute; width:260px; height:260px; right:-80px; top:-100px; border-radius:999px;
  background: radial-gradient(circle, rgba(236,72,153,.32), transparent 65%); animation: pulseGlow 5s ease-in-out infinite;
}
@keyframes pulseGlow { 0%,100%{transform:scale(1); opacity:.65} 50%{transform:scale(1.16); opacity:1} }
.badge { display:inline-flex; align-items:center; gap:6px; padding:7px 13px; border-radius:999px;
  background:linear-gradient(90deg,var(--pink),var(--purple)); color:white; font-weight:800; font-size:12px; margin:0 8px 10px 0; }
.smallbadge { display:inline-flex; padding:6px 10px; border-radius:999px; background:#182235; color:#dbeafe;
  font-weight:700; font-size:12px; border:1px solid #334155; margin:0 8px 10px 0; }
.card { background:rgba(17,24,39,.82); border:1px solid rgba(148,163,184,.18); border-radius:22px;
  padding:18px 20px; box-shadow:0 18px 42px rgba(0,0,0,.22); }
.metric-card { min-height:118px; border-radius:22px; border:1px solid rgba(236,72,153,.28);
  background: linear-gradient(135deg, rgba(236,72,153,.16), rgba(124,58,237,.10)); padding:17px 18px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 14px 35px rgba(0,0,0,.18); transition:.22s ease; }
.metric-card:hover { transform: translateY(-3px); border-color:rgba(34,211,238,.45); }
.metric-label { color:#aab3c6; font-size:13px; font-weight:700; }
.metric-value { color:#fff; font-size:26px; line-height:1.12; font-weight:900; margin-top:8px; }
.metric-help { color:#94a3b8; font-size:12px; margin-top:8px; }
.notice { border-radius:18px; border:1px solid rgba(34,211,238,.26); background:rgba(8,145,178,.10); padding:14px 16px; color:#dff9ff; }
.stButton > button { border-radius:999px; border:0; background:linear-gradient(90deg,var(--pink),var(--purple)); color:white; font-weight:800; padding:.65rem 1.1rem; box-shadow:0 10px 28px rgba(124,58,237,.25); }
.stTabs [data-baseweb="tab-list"] { gap:10px; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { border-radius:999px; background:#101827; padding:8px 16px; border:1px solid #29364f; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,rgba(236,72,153,.28),rgba(124,58,237,.22)); border-color:rgba(236,72,153,.45); }
[data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; border:1px solid rgba(148,163,184,.16); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "toast_once" not in st.session_state:
    st.toast("✅ Dashboard AIDEOM-VN đã sẵn sàng! Dùng sidebar để chỉnh tham số.", icon="🚀")
    st.session_state.toast_once = True

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def hero(title: str, level: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
          <span class="badge">✨ {level}</span>
          <span class="smallbadge">🐍 Python</span>
          <span class="smallbadge">📊 Streamlit + Plotly</span>
          <h1>{title}</h1>
          <p style="color:#cbd5e1;font-size:1.03rem;margin:0;max-width:1050px">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-help">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def df_show(df, hide_index=True):
    st.dataframe(df, use_container_width=True, hide_index=hide_index)


def fmt_df(df: pd.DataFrame, precision: int = 3):
    return df.style.format(precision=precision)


def line_fig(df, x, ys, title, labels=None, height=420):
    fig = px.line(df, x=x, y=ys, markers=True, template=PLOTLY_TEMPLATE, title=title, labels=labels or {})
    fig.update_layout(legend_title_text="", margin=dict(l=18, r=18, t=50, b=18), height=height)
    return fig


def bar_fig(df, x, y, title, color=None, orientation="v", height=430):
    fig = px.bar(df, x=x, y=y, color=color, template=PLOTLY_TEMPLATE, title=title, orientation=orientation)
    fig.update_layout(legend_title_text="", margin=dict(l=18, r=18, t=50, b=18), height=height)
    return fig


def info_box(text: str, kind="info"):
    icon = {"info":"💡", "success":"✅", "warning":"⚠️", "note":"📝"}.get(kind, "💡")
    st.markdown(f"<div class='notice'><b>{icon}</b> {text}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_bai7(n, seed):
    return core.bai7_pareto(n_samples=n, seed=seed)


@st.cache_data(show_spinner=False)
def cached_train_bai11(episodes, alpha, gamma, seed):
    return bai11_new.train_q_learning(episodes=int(episodes), alpha=float(alpha), gamma=float(gamma), seed=int(seed))

# -----------------------------------------------------------------------------
# Sidebar navigation + controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🇻🇳 AIDEOM-VN")
st.sidebar.caption("Dashboard 12 bài thực hành — mọi tham số nằm ở sidebar để màn hình chính tập trung vào kết quả.")
menu = [
    "🏠 Trang chủ",
    "📈 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🧭 Bài 4 — LP ngành-vùng",
    "🧩 Bài 5 — MIP chọn dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — Pareto đa mục tiêu",
    "⏳ Bài 8 — Tối ưu động",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🖥️ Bài 12 — Dashboard tích hợp",
]
page = st.sidebar.radio("📚 Chọn bài", menu, index=0)
page_id = menu.index(page)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu hình")

# Page-specific sidebar controls
controls = {}
if page_id == 1:
    with st.sidebar.expander("Hệ số Cobb-Douglas", expanded=True):
        controls["alpha"] = st.number_input("α — Vốn K", 0.0, 1.0, 0.33, 0.01)
        controls["beta"] = st.number_input("β — Lao động L", 0.0, 1.0, 0.42, 0.01)
        controls["gamma"] = st.number_input("γ — Số hóa D", 0.0, 1.0, 0.10, 0.01)
        controls["delta"] = st.number_input("δ — AI", 0.0, 1.0, 0.08, 0.01)
        controls["theta"] = st.number_input("θ — Nhân lực H", 0.0, 1.0, 0.07, 0.01)
elif page_id == 2:
    with st.sidebar.expander("LP ngân sách", expanded=True):
        controls["budget"] = st.slider("Ngân sách tổng B", 80, 160, 100, 10)
        controls["min_h"] = st.slider("Sàn nhân lực số x₃", 20, 50, 20, 5)
elif page_id == 3:
    with st.sidebar.expander("Priority ngành", expanded=True):
        controls["scheme"] = st.selectbox("Bộ trọng số", ["default", "growth", "inclusive"], format_func=lambda x: {"default":"Mặc định", "growth":"Tăng trưởng", "inclusive":"Bao trùm"}[x])
        controls["w_ai"] = st.slider("Trọng số AI Readiness a₆", 0.05, 0.40, 0.20, 0.05)
elif page_id == 4:
    with st.sidebar.expander("LP vùng-hạng mục", expanded=True):
        controls["fair"] = st.toggle("Bật công bằng C5", value=True)
elif page_id == 5:
    with st.sidebar.expander("MIP dự án", expanded=True):
        controls["budget"] = st.slider("Ngân sách 5 năm", 70000, 110000, 80000, 5000)
        controls["force"] = st.toggle("Bắt buộc P1 và P2", value=False)
        controls["expected"] = st.toggle("Dùng lợi ích kỳ vọng", value=False)
elif page_id == 6:
    with st.sidebar.expander("TOPSIS", expanded=True):
        controls["w_ai"] = st.slider("w_AI", 0.10, 0.40, 0.20, 0.05)
elif page_id == 7:
    with st.sidebar.expander("Pareto mô phỏng", expanded=True):
        controls["n"] = st.slider("Số phương án", 200, 1500, 700, 100)
        controls["seed"] = st.number_input("Seed", 0, 9999, 42, 1)
elif page_id == 8:
    with st.sidebar.expander("Tối ưu động", expanded=True):
        controls["strategy"] = st.selectbox("Chiến lược", ["balanced", "even", "front_load", "traditional", "ai_led", "inclusive"], format_func=lambda x: {"balanced":"Tối ưu cân bằng", "even":"Đầu tư trải đều", "front_load":"Front-load", "traditional":"Truyền thống", "ai_led":"AI dẫn dắt", "inclusive":"Bao trùm"}[x])
        controls["shock"] = st.toggle("Cú sốc 2028: Y giảm 8%", value=False)
        controls["rho"] = st.slider("Chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
elif page_id == 9:
    with st.sidebar.expander("Lao động & AI", expanded=True):
        controls["strict"] = st.toggle("Không ngành nào mất quá 5% LĐ", value=False)
        controls["xai2"] = st.slider("x_AI ngành 2 để tính ngưỡng", 0, 30000, 1000, 500)
elif page_id == 10:
    with st.sidebar.expander("Stochastic programming", expanded=True):
        controls["first_budget"] = st.number_input("First-stage budget", 10.0, 100.0, 65.0, 1.0)
        controls["reserve_budget"] = st.number_input("Second-stage reserve", 1.0, 50.0, 15.0, 1.0)
        controls["step"] = st.selectbox("Độ mịn tối ưu H", [0.10, 0.05, 0.01], index=2)
elif page_id == 11:
    with st.sidebar.expander("Q-learning", expanded=True):
        controls["episodes"] = st.slider("Số episode", 500, 10000, 3000, 500)
        controls["alpha_rl"] = st.slider("α learning rate", 0.01, 0.50, 0.10, 0.01)
        controls["gamma_rl"] = st.slider("γ discount", 0.50, 0.99, 0.95, 0.01)
        controls["seed_rl"] = st.number_input("Seed", 0, 9999, 42, 1)
elif page_id == 12:
    with st.sidebar.expander("Dashboard tích hợp", expanded=True):
        st.caption("Bài 12 tự chạy 5 kịch bản S1–S5. Không cần nhập thêm.")

st.sidebar.markdown("---")
st.sidebar.success("✅ Repo cần có: app.py, core.py, requirements.txt, data/, modules/")
st.sidebar.caption("Tip: nếu deploy lỗi, kiểm tra chữ thường `data` và `modules`.")

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page_id == 0:
    hero("AIDEOM-VN — Dashboard 12 bài thực hành", "TỔNG HỢP", "Giao diện mới dùng font hiện đại, hiệu ứng nhẹ, thông báo trực quan, sidebar cho toàn bộ tham số, expander để ẩn phần dài và bố cục columns/containers cho kết quả.")
    metric_cards([
        ("📚 Số bài", "12", "Từ Cobb-Douglas đến RL"),
        ("🧠 Module", "6", "M1–M6 tích hợp"),
        ("🎯 Kịch bản", "5", "S1–S5 đến 2030"),
        ("🚀 Deploy", "Ready", "GitHub + Streamlit Cloud"),
    ])
    info_box("Chọn bài ở sidebar. Mọi input/slider đều nằm trong sidebar; màn hình chính chỉ hiển thị kết quả, bảng, biểu đồ và nhận xét.", "success")
    with st.container(border=True):
        st.subheader("🗺️ Bản đồ 12 bài")
        df_map = pd.DataFrame({
            "Cấp độ": ["Dễ"]*3 + ["Trung bình"]*3 + ["Khá khó"]*3 + ["Khó"]*3,
            "Bài": [f"Bài {i}" for i in range(1, 13)],
            "Nội dung": [
                "Cobb-Douglas mở rộng", "LP ngân sách", "Priority ngành", "LP vùng-hạng mục", "MIP chọn dự án", "TOPSIS vùng",
                "Pareto đa mục tiêu", "Tối ưu động", "Tác động lao động", "Quy hoạch ngẫu nhiên", "Q-learning", "Dashboard tích hợp"
            ],
            "Kết quả chính": [
                "TFP, MAPE, GDP 2030", "Phân bổ x1-x4", "Top ngành ưu tiên", "Ma trận 6×4", "Tập dự án tối ưu", "Top vùng AI",
                "Tập Pareto", "Quỹ đạo 2026–2035", "NetJob & Sankey", "SP, VSS, EVPI", "π*, learning curve", "So sánh S1–S5"
            ],
        })
        df_show(df_map)
    with st.expander("📦 Hướng dẫn deploy nhanh"):
        st.markdown("""
        1. Upload đúng cấu trúc lên GitHub: `app.py`, `core.py`, `requirements.txt`, `data/`, `modules/`.  
        2. Vào Streamlit Cloud → **Deploy public app from GitHub**.  
        3. Chọn repository, branch `main`, main file path `app.py` → **Deploy**.
        """)

elif page_id == 1:
    hero("Bài 1 — Cobb-Douglas mở rộng với AI và số hóa", "DỄ", "Ước lượng TFP Aₜ, dự báo GDP, phân rã tăng trưởng và mô phỏng GDP 2030.")
    with st.expander("📘 Đề bài tóm tắt"):
        st.markdown("Mô hình **Yₜ = Aₜ·Kₜ^α·Lₜ^β·Dₜ^γ·AIₜ^δ·Hₜ^θ**. Yêu cầu tính Aₜ, Ŷₜ, MAPE, growth accounting và GDP 2030.")
    out = core.bai1_cobb_douglas(controls["alpha"], controls["beta"], controls["gamma"], controls["delta"], controls["theta"])
    metric_cards([
        ("⚙️ A trung bình", f"{out['A_bar']:.4f}", "TFP calibrated"),
        ("📉 MAPE", f"{out['MAPE']:.2f}%", "Sai số dự báo"),
        ("🚀 GDP 2030", f"{out['Y2030']:,.1f}", "nghìn tỷ VND"),
        ("🧮 Tổng hệ số", f"{sum(controls[k] for k in ['alpha','beta','gamma','delta','theta']):.2f}", "α+β+γ+δ+θ"),
    ])
    tab1, tab2, tab3 = st.tabs(["📈 TFP & dự báo", "🧩 Phân rã", "🔮 2030"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(line_fig(out["table"], "year", ["A_TFP"], "TFP Aₜ theo năm"), use_container_width=True)
        with col2: st.plotly_chart(line_fig(out["table"], "year", ["GDP_trillion_VND", "Y_hat"], "GDP thực tế và dự báo"), use_container_width=True)
        with st.expander("📋 Bảng chi tiết"):
            df_show(out["table"].round(4))
    with tab2:
        gt = out["growth_table"].copy()
        col1, col2 = st.columns([1, 1.35])
        with col1: df_show(gt.round(4))
        with col2: st.plotly_chart(bar_fig(gt, "Yếu tố", "Đóng góp vào tăng trưởng (%)", "Đóng góp tăng trưởng log bình quân"), use_container_width=True)
    with tab3:
        info_box("Kịch bản dùng D=30%, AI=100 nghìn DN số, H=35%, K và L tăng 6%/năm, TFP tăng 1,2%/năm.", "info")
        st.json({k: round(v, 3) for k, v in out["scenario2030"].items()})

elif page_id == 2:
    hero("Bài 2 — LP phân bổ ngân sách số", "DỄ", "Tối đa hóa GDP kỳ vọng với 4 hạng mục đầu tư số và phân tích độ nhạy ngân sách.")
    with st.expander("📘 Đề bài tóm tắt"):
        st.markdown("Tối đa hóa **Z = 0,85x₁ + 1,20x₂ + 0,95x₃ + 1,35x₄**, có ràng buộc ngân sách, sàn từng hạng mục và AI+R&D ≥ 35% tổng đầu tư.")
    res = core.solve_bai2_lp(controls["budget"], controls["min_h"])
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cards([
            ("🎯 Z*", f"{res['Z']:.2f}", "GDP kỳ vọng"),
            ("💸 Tổng chi", f"{res['x'].sum():.1f}", "nghìn tỷ VND"),
            ("🤖 AI+R&D", f"{res['x'].iloc[1]+res['x'].iloc[3]:.1f}", "x₂+x₄"),
            ("📌 Tỷ trọng", f"{100*(res['x'].iloc[1]+res['x'].iloc[3])/res['x'].sum():.1f}%", "chiến lược"),
        ])
        col1, col2 = st.columns([1, 1.35])
        with col1:
            df_show(res["x"].rename("Phân bổ tối ưu").reset_index().rename(columns={"index":"Hạng mục"}))
            if res["dual"] is not None:
                with st.expander("🧮 Shadow price"):
                    df_show(res["dual"].rename("Shadow price").reset_index().rename(columns={"index":"Ràng buộc"}).round(4))
        with col2:
            plot_df = res["x"].reset_index().rename(columns={"index":"Hạng mục", 0:"Giá trị"})
            st.plotly_chart(bar_fig(plot_df, "Hạng mục", "Giá trị", "Phân bổ tối ưu"), use_container_width=True)
    all2 = core.bai2_all()
    with st.container(border=True):
        st.subheader("🔍 Độ nhạy ngân sách")
        col1, col2 = st.columns([1, 1.35])
        with col1: df_show(all2["sensitivity"].round(3))
        with col2: st.plotly_chart(line_fig(all2["sensitivity"], "Ngân sách", ["Z*"], "Đường cong Z*(B)"), use_container_width=True)

elif page_id == 3:
    hero("Bài 3 — Priorityᵢ cho 10 ngành", "DỄ", "Chuẩn hóa min-max, tính chỉ số ưu tiên ngành và phân tích độ nhạy trọng số AI Readiness.")
    with st.expander("📘 Đề bài tóm tắt"):
        st.markdown("Priorityᵢ kết hợp Growth, Productivity, Spillover, Export, Employment, AI Readiness và Risk. Risk là tiêu chí bất lợi.")
    out = core.bai3_priority(ai_weight=controls["w_ai"], scheme=controls["scheme"])
    top3 = out["ranking"].head(3)
    metric_cards([(f"🏅 Top {i+1}", row["sector_name_vi"], f"Priority={row['Priority']:.3f}") for i, row in top3.iterrows()] + [("📊 Số ngành", "10", "sector data 2024")])
    tab1, tab2, tab3 = st.tabs(["🏆 Xếp hạng", "🧮 Chuẩn hóa", "🔥 Độ nhạy"])
    with tab1:
        col1, col2 = st.columns([1.1, 1.4])
        with col1: df_show(out["ranking"].round(4))
        with col2: st.plotly_chart(bar_fig(out["ranking"].sort_values("Priority"), "Priority", "sector_name_vi", "Priority theo ngành", orientation="h"), use_container_width=True)
        with st.expander("⚖️ Trọng số đang dùng"):
            df_show(out["weights"].rename("weight").reset_index().rename(columns={"index":"Tiêu chí"}).round(4))
    with tab2:
        df_show(out["normalized"].round(3))
    with tab3:
        sens = core.bai3_sensitivity()
        heat = sens.pivot(index="Ngành", columns="w_AI", values="Rank")
        st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap thứ hạng khi thay đổi trọng số AI"), use_container_width=True)
        with st.expander("📋 Bảng độ nhạy"):
            df_show(sens.sort_values(["w_AI", "Rank"]))

elif page_id == 4:
    hero("Bài 4 — LP phân bổ ngân sách theo vùng-hạng mục", "TRUNG BÌNH", "Giải ma trận 6×4 và so sánh có/không ràng buộc công bằng vùng miền.")
    res = core.solve_bai4_lp(fairness=controls["fair"])
    nofair = core.solve_bai4_lp(fairness=False)
    if not res["success"]:
        st.error(res["message"])
    else:
        alloc = res["allocation"]
        metric_cards([
            ("🎯 Z*", f"{res['Z']:,.1f}", "GDP gain"),
            ("💰 Tổng NS", f"{alloc.values.sum():,.0f}", "tỷ VND"),
            ("🗺️ Vùng max", alloc.sum(axis=1).idxmax(), f"{alloc.sum(axis=1).max():,.0f}"),
            ("⚖️ Chi phí C5", f"{(nofair['Z']-res['Z']):,.1f}", "so với bỏ C5"),
        ])
        if res.get("slack") is not None and float(res["slack"].sum()) > 1e-6:
            st.warning("⚠️ C5 với tham số gốc khá căng; app dùng soft fairness slack để vẫn hiển thị nghiệm và mức thiếu hụt.")
        col1, col2 = st.columns([1, 1.25])
        with col1:
            df_show(alloc.round(2), hide_index=False)
            with st.expander("🔎 Slack C5"):
                if res.get("slack") is not None: df_show(res["slack"].rename("Slack C5").reset_index().rename(columns={"index":"Vùng"}).round(3))
        with col2:
            st.plotly_chart(px.imshow(alloc, text_auto='.0f', aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap phân bổ tối ưu 6×4"), use_container_width=True)
        comp = pd.DataFrame({"Có C5": alloc.sum(axis=1), "Không C5": nofair["allocation"].sum(axis=1)})
        st.plotly_chart(px.bar(comp, barmode="group", template=PLOTLY_TEMPLATE, title="So sánh tổng ngân sách vùng"), use_container_width=True)

elif page_id == 5:
    hero("Bài 5 — MIP lựa chọn dự án chuyển đổi số", "TRUNG BÌNH", "Chọn dự án tối ưu với ràng buộc ngân sách, loại trừ, tiên quyết và số lượng dự án.")
    res = core.solve_bai5(budget=controls["budget"], force_p1_p2=controls["force"], expected=controls["expected"])
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cards([
            ("💎 Tổng lợi ích", f"{res['Z']:,.0f}", "tỷ VND"),
            ("💸 Tổng chi", f"{res['cost']:,.0f}", "tỷ VND"),
            ("📈 NPV/Cost", f"{res['npv_per_cost']:.2f}", "hiệu suất"),
            ("🧩 Số dự án", f"{len(res['chosen_ids'])}", ", ".join('P'+str(i) for i in res['chosen_ids'])),
        ])
        with st.container(border=True):
            st.subheader("✅ Dự án được chọn")
            df_show(res["chosen"][["id", "name", "sector", "cost", "benefit", "year12", "year35"]])
        fig = px.bar(res["chosen"], x="name", y=["cost", "benefit"], barmode="group", template=PLOTLY_TEMPLATE, title="Chi phí và lợi ích các dự án được chọn")
        fig.update_layout(xaxis_tickangle=-35, height=500)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔍 So sánh ngân sách 80.000 và 100.000"):
        comp = []
        for B in [80000, 100000]:
            r = core.solve_bai5(budget=B)
            comp.append({"Ngân sách": B, "Z*": r["Z"], "Chi phí": r["cost"], "Dự án chọn": ", ".join('P'+str(i) for i in r["chosen_ids"])})
        df_show(pd.DataFrame(comp))

elif page_id == 6:
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng", "TRUNG BÌNH", "Tính TOPSIS bằng trọng số chuyên gia, Entropy weights và phân tích độ nhạy w_AI.")
    out = core.bai6_topsis(ai_weight=controls["w_ai"])
    metric_cards([
        ("🏆 Top chuyên gia", out["expert"].iloc[0]["region_name_vi"], f"Score={out['expert'].iloc[0]['TOPSIS_score']:.3f}"),
        ("🧠 Top Entropy", out["entropy"].iloc[0]["region_name_vi"], f"Score={out['entropy'].iloc[0]['TOPSIS_score']:.3f}"),
        ("🗺️ Số vùng", "6", "KT-XH"),
        ("⚠️ Gini", "Cost", "càng thấp càng tốt"),
    ])
    tab1, tab2, tab3 = st.tabs(["🏆 TOPSIS", "🧮 Entropy", "🔥 Độ nhạy"])
    with tab1:
        col1, col2 = st.columns([1,1.3])
        with col1: df_show(out["expert"].round(4))
        with col2: st.plotly_chart(bar_fig(out["expert"].sort_values("TOPSIS_score"), "TOPSIS_score", "region_name_vi", "TOPSIS trọng số chuyên gia", orientation="h"), use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1: df_show(out["entropy"].round(4))
        with col2: df_show(out["weights_entropy"].rename("Entropy weight").reset_index().rename(columns={"index":"Tiêu chí"}).round(4))
    with tab3:
        sens = core.bai6_sensitivity()
        heat = sens.pivot(index="Vùng", columns="w_AI", values="Rank")
        st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap thứ hạng khi thay đổi w_AI"), use_container_width=True)

elif page_id == 7:
    hero("Bài 7 — Pareto đa mục tiêu", "KHÁ KHÓ", "Mô phỏng tập nghiệm không bị trội cho GDP, bao trùm, môi trường và an ninh dữ liệu.")
    with st.expander("📘 Lưu ý mô hình"):
        st.markdown("Phần này mô phỏng tập Pareto thay cho NSGA-II đầy đủ để chạy nhẹ trên Streamlit Cloud, vẫn giữ logic đa mục tiêu và chọn nghiệm thỏa hiệp.")
    out = cached_bai7(controls["n"], int(controls["seed"]))
    pareto, best = out["pareto"], out["best"]
    metric_cards([
        ("🧪 Phương án", f"{len(out['all'])}", "khả thi"),
        ("🌐 Pareto", f"{len(pareto)}", "không bị trội"),
        ("📈 GDP compromise", f"{best['GDP_gain']:,.0f}", "tỷ VND"),
        ("⭐ Score", f"{best['Compromise_score']:.3f}", "compromise"),
    ])
    fig3d = px.scatter_3d(pareto, x="GDP_gain", y="Inequality_MAD", z="Emission", color="SecurityRisk", template=PLOTLY_TEMPLATE, title="Pareto 3D: GDP - Bao trùm - Môi trường")
    fig3d.update_layout(height=620)
    st.plotly_chart(fig3d, use_container_width=True)
    col1, col2 = st.columns([1.2,1])
    with col1:
        with st.expander("📋 Top nghiệm compromise"):
            df_show(pareto.sort_values("Compromise_score", ascending=False).head(20).round(3))
    with col2:
        alloc = pd.DataFrame(out["best_allocation"], index=[core.REGION_NAMES[r] for r in core.REGIONS], columns=["I", "D", "AI", "H"])
        st.plotly_chart(px.imshow(alloc, text_auto='.0f', aspect="auto", template=PLOTLY_TEMPLATE, title="Phân bổ nghiệm thỏa hiệp"), use_container_width=True)

elif page_id == 8:
    hero("Bài 8 — Tối ưu động 2026–2035", "KHÁ KHÓ", "Mô phỏng quỹ đạo K, D, AI, H, Y, C và so sánh chiến lược đầu tư.")
    df = core.simulate_dynamic(strategy=controls["strategy"], shock=controls["shock"], rho_discount=controls["rho"])
    metric_cards([
        ("📈 Y 2035", f"{df.iloc[-1]['Y']:,.1f}", "sản lượng"),
        ("🛒 C 2035", f"{df.iloc[-1]['C']:,.1f}", "tiêu dùng"),
        ("💚 Welfare", f"{df.iloc[-1]['welfare_cum']:.2f}", "tích lũy"),
        ("🤖 AI 2035", f"{df.iloc[-1]['AI']:.1f}", "nghìn DN số"),
    ])
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(line_fig(df, "year", ["K", "D", "AI", "H"], "Trạng thái K, D, AI, H"), use_container_width=True)
    with col2: st.plotly_chart(line_fig(df, "year", ["Y", "C"], "Sản lượng Y và tiêu dùng C"), use_container_width=True)
    with st.expander("📊 So sánh chiến lược"):
        comp = core.bai8_compare()
        df_show(comp.round(3))
        st.plotly_chart(bar_fig(comp, "Chiến lược", "Welfare", "Welfare tổng theo chiến lược"), use_container_width=True)

elif page_id == 9:
    hero("Bài 9 — Lao động & AI", "KHÁ KHÓ", "Tính NetJob, ngưỡng đào tạo lại và mô phỏng luồng dịch chuyển lao động.")
    out = core.solve_bai9(no_large_loss=controls["strict"])
    if not out["success"]:
        st.error(out["message"])
    else:
        tbl = out["table"]
        metric_cards([
            ("👷 Tổng NetJob", f"{out['Z']:,.0f}", "việc làm ròng"),
            ("💸 Ngân sách", f"{(tbl['x_AI']+tbl['x_H']).sum():,.0f}", "tỷ VND"),
            ("🎓 Đào tạo max", tbl.sort_values("x_H", ascending=False).iloc[0]["Ngành"], f"x_H={tbl['x_H'].max():,.0f}"),
            ("🤖 AI max", tbl.sort_values("x_AI", ascending=False).iloc[0]["Ngành"], f"x_AI={tbl['x_AI'].max():,.0f}"),
        ])
        with st.expander("📋 Bảng NetJob chi tiết", expanded=True):
            df_show(tbl.round(2))
        st.plotly_chart(px.bar(tbl, x="Ngành", y=["NewJob", "UpgradeJob", "DisplacedJob", "NetJob"], barmode="group", template=PLOTLY_TEMPLATE, title="Việc làm tạo mới, nâng cấp, dịch chuyển và ròng"), use_container_width=True)
        with st.expander("🧮 Ngưỡng x_H ngành 2"):
            threshold = core.bai9_threshold_industry2(x_ai=float(controls["xai2"]))
            st.json({k: round(v, 2) for k, v in threshold.items()})
        st.subheader("🌊 Sankey nhóm dễ bị tổn thương")
        vuln_idx = [0,2,3]
        labels = [tbl.loc[i,"Ngành"] + " - Displaced" for i in vuln_idx] + ["Đào tạo lại", "Việc làm mới/nâng cấp", "Thất nghiệp ròng"]
        sources, targets, values = [], [], []
        retrain_node, new_node, unemp_node = len(vuln_idx), len(vuln_idx)+1, len(vuln_idx)+2
        for k,i in enumerate(vuln_idx):
            displaced = float(tbl.loc[i,"DisplacedJob"]); retrained = min(displaced, float(tbl.loc[i,"RetrainingCapacity"])); unemp = max(displaced-retrained, 0)
            sources += [k, retrain_node]; targets += [retrain_node, new_node]; values += [max(retrained,0), max(retrained + tbl.loc[i,"NewJob"] + tbl.loc[i,"UpgradeJob"],0)]
            if unemp > 1e-6: sources.append(k); targets.append(unemp_node); values.append(unemp)
        fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=sources, target=targets, value=values))])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
        st.plotly_chart(fig, use_container_width=True)

elif page_id == 10:
    hero("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn", "KHÓ", "First-stage/second-stage với 4 kịch bản, tính SP, EV, WS, VSS, EVPI và robust minimax regret.")
    st.success("✅ Đang chạy đúng Bài 10 — phiên bản mới đã tích hợp vào app chung.")
    with st.expander("📘 Đề bài tóm tắt"):
        st.markdown("First-stage phân bổ x=(I,D,AI,H) tối đa 65; second-stage yˢ dự phòng 15; liên kết **y_AIˢ ≤ 0,5x_H**; tính **SP/EV/WS/VSS/EVPI**.")
    result = bai10_new.compute_ws_evpi_vss(controls["first_budget"], controls["reserve_budget"], controls["step"])
    robust = bai10_new.optimize_robust_minimax_regret(controls["first_budget"], controls["reserve_budget"], controls["step"])
    sp, ev, rb = result["sp"], result["ev"], robust["solution"]
    metric_cards([
        ("🎲 SP", f"{result['SP']:.2f}", "stochastic solution"),
        ("📌 EEV", f"{result['EEV']:.2f}", "EV policy under uncertainty"),
        ("💎 VSS", f"{result['VSS']:.4f}", "SP - EEV"),
        ("🔮 EVPI", f"{result['EVPI']:.4f}", "WS - SP"),
    ])
    tab1, tab2, tab3, tab4 = st.tabs(["🌳 Kịch bản", "⚙️ Nghiệm", "📊 VSS/EVPI", "💻 Pyomo"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: df_show(bai10_new.scenario_dataframe())
        with col2: df_show(bai10_new.beta_dataframe())
    with tab2:
        col1, col2 = st.columns([1,1.2])
        with col1: df_show(bai10_new.first_stage_table(sp, ev, rb).round(3))
        with col2: df_show(bai10_new.recourse_table(sp).round(3))
        alloc = bai10_new.allocation_long_table(sp)
        st.plotly_chart(px.bar(alloc, x="Hạng mục", y="Ngân sách", color="Kịch bản", facet_col="Giai đoạn", barmode="group", template=PLOTLY_TEMPLATE, title="Phân bổ SP theo giai đoạn"), use_container_width=True)
    with tab3:
        det_rows = []
        for s, sol in result["deterministic"].items():
            row = {"Kịch bản": s, "Tên": bai10_new.SCENARIO_NAMES[s], "WS objective": sol.expected_value, "x_H": sol.h_value}
            row.update({f"x_{j}": sol.first_stage[j] for j in bai10_new.ITEMS})
            det_rows.append(row)
        with st.expander("📋 Nghiệm deterministic theo từng kịch bản"):
            df_show(pd.DataFrame(det_rows).round(3))
        metric_df = pd.DataFrame([
            {"Chỉ tiêu":"WS", "Giá trị":result["WS"]}, {"Chỉ tiêu":"SP", "Giá trị":result["SP"]}, {"Chỉ tiêu":"EEV", "Giá trị":result["EEV"]}, {"Chỉ tiêu":"VSS", "Giá trị":result["VSS"]}, {"Chỉ tiêu":"EVPI", "Giá trị":result["EVPI"]},
        ])
        st.plotly_chart(px.bar(metric_df, x="Chỉ tiêu", y="Giá trị", template=PLOTLY_TEMPLATE, title="So sánh WS, SP, EEV, VSS, EVPI"), use_container_width=True)
    with tab4:
        st.code("""import pyomo.environ as pyo
m = pyo.ConcreteModel()
m.J = pyo.Set(initialize=['I','D','AI','H'])
m.S = pyo.Set(initialize=['s1','s2','s3','s4'])
m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65)
m.budget2 = pyo.Constraint(m.S, rule=lambda m,s: sum(m.y[s,j] for j in m.J) <= 15)
m.ai_link = pyo.Constraint(m.S, rule=lambda m,s: m.y[s,'AI'] <= 0.5*m.x['H'])
# objective = first-stage value + expected second-stage value
""", language="python")

elif page_id == 11:
    hero("Bài 11 — Q-learning chính sách kinh tế thích nghi", "KHÓ", "MDP 81 trạng thái, 5 hành động chính sách, learning curve và so sánh với rule-based policies.")
    st.success("✅ Đang chạy đúng Bài 11 — không nhảy về Bài 1.")
    with st.expander("📘 Đề bài tóm tắt"):
        st.markdown("State = GDP growth × Digital index × AI capacity × Unemployment risk; Action = 5 gói ngân sách; Reward = tăng trưởng − thất nghiệp − cyber risk − emission.")
    with st.expander("🧭 Bảng hành động chính sách", expanded=True):
        df_show(bai11_new.action_table())
    trained = cached_train_bai11(controls["episodes"], controls["alpha_rl"], controls["gamma_rl"], controls["seed_rl"])
    Q = trained["Q"]
    rewards = pd.DataFrame({"Episode": np.arange(len(trained["rewards"])), "Reward": trained["rewards"], "Smoothed": trained["smooth_rewards"]})
    metric_cards([
        ("🏁 Reward 100 ep cuối", f"{np.mean(trained['rewards'][-100:]):.2f}", "learning curve"),
        ("🏆 Best reward", f"{np.max(trained['rewards']):.2f}", "episode tốt nhất"),
        ("🧩 Trạng thái", "81", "3⁴"),
        ("🎮 Hành động", "5", "a0–a4"),
    ])
    tab1, tab2, tab3 = st.tabs(["📈 Learning", "⚖️ So sánh", "🗺️ Policy map"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rewards["Episode"], y=rewards["Reward"], mode="lines", name="Per episode", opacity=0.35))
        fig.add_trace(go.Scatter(x=rewards["Episode"], y=rewards["Smoothed"], mode="lines", name="Smoothed"))
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Learning curve Q-learning", xaxis_title="Episode", yaxis_title="Reward")
        st.plotly_chart(fig, use_container_width=True)
        action_counts = pd.DataFrame({"Hành động": [bai11_new.ACTION_NAMES[i] for i in range(5)], "Số lần được chọn": trained["actions_count"]})
        st.plotly_chart(px.bar(action_counts, x="Hành động", y="Số lần được chọn", template=PLOTLY_TEMPLATE, title="Tần suất hành động"), use_container_width=True)
    with tab2:
        comp = bai11_new.compare_policies(Q, episodes=250, seed=int(controls["seed_rl"])+1)
        col1, col2 = st.columns([1,1.25])
        with col1: df_show(comp.round(3))
        with col2: st.plotly_chart(px.bar(comp, x="Chính sách", y="Mean cumulative reward", template=PLOTLY_TEMPLATE, title="π* vs rule-based"), use_container_width=True)
        path = bai11_new.simulate_one_episode_policy(Q=Q, seed=int(controls["seed_rl"])+5)
        st.plotly_chart(px.line(path, x="t", y=["D", "AI", "H"], markers=True, template=PLOTLY_TEMPLATE, title="Một episode theo chính sách học được"), use_container_width=True)
    with tab3:
        sl = bai11_new.policy_slice(Q, growth_state=1, unemployment_state=1)
        pivot = sl.pivot(index="Digital state", columns="AI state", values="Action")
        st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Policy heatmap: GDP medium, unemployment medium"), use_container_width=True)
        with st.expander("📋 Full policy table"):
            df_show(bai11_new.policy_table_for_states(Q))

elif page_id == 12:
    hero("Bài 12 — Dashboard tích hợp AIDEOM-VN", "KHÓ", "Nguyên mẫu tích hợp M1–M6, so sánh 5 kịch bản và đưa ra khuyến nghị chính sách.")
    st.success("✅ Đang chạy đúng Bài 12 — dashboard tổng hợp đã được làm mới.")
    outputs = bai12_new.run_integrated_model()
    scen, summary, paths, risks = outputs["scenario_table"], outputs["summary"], outputs["paths"], outputs["risks"]
    tab1, tab2, tab3, tab4 = st.tabs(["🧱 M1–M6", "📊 5 kịch bản", "⚠️ Rủi ro & việc làm", "📝 Khuyến nghị"])
    with tab1:
        col1, col2 = st.columns([1.1,1])
        with col1:
            st.subheader("Thiết kế hệ thống")
            df_show(outputs["modules"])
        with col2:
            st.subheader("Kịch bản chính sách")
            df_show(scen.round(2))
    with tab2:
        top = summary.iloc[0]
        metric_cards([
            ("🏆 GDP 2030 cao nhất", f"{top['GDP 2030']:.1f}", str(top["Kịch bản"])),
            ("🛡️ Rủi ro thấp nhất", str(summary.sort_values("Risk_index").iloc[0]["Kịch bản"]), "Risk_index min"),
            ("👷 Việc làm tốt nhất", str(summary.sort_values("Net_jobs_index", ascending=False).iloc[0]["Kịch bản"]), "NetJobs max"),
            ("🎯 Kịch bản", "5", "S1–S5"),
        ])
        with st.expander("📋 Bảng tổng hợp", expanded=True):
            df_show(summary.round(2))
        st.plotly_chart(px.line(paths, x="Năm", y="GDP", color="Kịch bản", markers=True, template=PLOTLY_TEMPLATE, title="Dự báo GDP 2025–2030 theo 5 kịch bản"), use_container_width=True)
        st.plotly_chart(px.bar(summary, x="Kịch bản", y="Tăng GDP 2025-2030 (%)", color="Tên", template=PLOTLY_TEMPLATE, title="Tăng trưởng GDP tích lũy 2025–2030"), use_container_width=True)
    with tab3:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.bar(summary, x="Kịch bản", y="Net_jobs_index", color="Tên", template=PLOTLY_TEMPLATE, title="Chỉ số việc làm ròng"), use_container_width=True)
        with col2: st.plotly_chart(px.bar(summary, x="Kịch bản", y="Risk_index", color="Tên", template=PLOTLY_TEMPLATE, title="Chỉ số rủi ro tổng hợp"), use_container_width=True)
        st.plotly_chart(px.bar(risks, x="Kịch bản", y="Điểm", color="Loại rủi ro", barmode="group", template=PLOTLY_TEMPLATE, title="Cấu phần rủi ro"), use_container_width=True)
        best = summary.iloc[0]
        cats = ["GDP_2030_index", "Readiness", "Net_jobs_index", "Risk_index"]
        fig = go.Figure(data=go.Scatterpolar(r=[best[c] for c in cats], theta=cats, fill="toself"))
        fig.update_layout(template=PLOTLY_TEMPLATE, title=f"Radar kịch bản dẫn đầu: {best['Kịch bản']}", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.markdown(f"<div class='card'>💡 {bai12_new.recommendation(summary)}</div>", unsafe_allow_html=True)
        with st.expander("📝 Gợi ý viết thảo luận chính sách"):
            st.markdown("""
            - Không nên chọn duy nhất kịch bản GDP cao nhất nếu rủi ro cyber, môi trường hoặc phụ thuộc công nghệ quá lớn.  
            - S5 phù hợp để trình bày như phương án thỏa hiệp vì kết hợp stochastic planning, Q-learning và dashboard tích hợp.  
            - Cần thêm ràng buộc an sinh xã hội, minh bạch dữ liệu và trách nhiệm giải trình khi dùng AI hỗ trợ chính sách.
            """)

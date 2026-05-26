"""Bài 11 - Tabular Q-learning for an adaptive economic-policy MDP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ACTION_NAMES = {
    0: "a0 - Truyền thống",
    1: "a1 - Cân bằng",
    2: "a2 - Số hóa nhanh",
    3: "a3 - AI dẫn dắt",
    4: "a4 - Bao trùm",
}
ACTION_SHARES = np.array([
    [0.70, 0.10, 0.10, 0.10],
    [0.40, 0.25, 0.20, 0.15],
    [0.25, 0.45, 0.15, 0.15],
    [0.20, 0.20, 0.45, 0.15],
    [0.30, 0.20, 0.10, 0.40],
], dtype=float)
ACTION_COLS = ["K", "D", "AI", "H"]
WEIGHTS = np.array([40.0, 25.0, 20.0, 15.0])  # ΔGDP, unemployment, cyber, emission


def action_table() -> pd.DataFrame:
    df = pd.DataFrame(ACTION_SHARES, columns=["K", "D", "AI", "H"])
    df.insert(0, "Hành động", [ACTION_NAMES[i] for i in range(len(ACTION_NAMES))])
    return df


@dataclass
class StepResult:
    obs: Tuple[int, int, int, int]
    reward: float
    done: bool
    info: Dict[str, float]


class VietnamEconomyEnv:
    """Small MDP with 4 discrete state dimensions and 5 policy actions.

    Observation = (GDP growth level, digital index level, AI capacity level, unemployment risk level),
    each in {0, 1, 2}. One episode has T=10 policy periods.
    """

    def __init__(self, T: int = 10, seed: int | None = None, stochastic: bool = True):
        self.T = T
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.Y = self._production()
        self.last_growth = 0.035
        self.unemp_risk = 0.45
        self.cyber_risk = 0.30
        self.emission = 0.30

    def _production(self) -> float:
        return (self.K / 25000.0) ** 0.33 * (self.D / 20.0) ** 0.10 * (self.AI / 80.0) ** 0.08 * (self.H / 30.0) ** 0.07

    @staticmethod
    def _bin_growth(g: float) -> int:
        if g < 0.025:
            return 0
        if g < 0.055:
            return 1
        return 2

    @staticmethod
    def _bin_d(D: float) -> int:
        if D < 22.0:
            return 0
        if D < 30.0:
            return 1
        return 2

    @staticmethod
    def _bin_ai(AI: float) -> int:
        if AI < 90.0:
            return 0
        if AI < 115.0:
            return 1
        return 2

    @staticmethod
    def _bin_u(u: float) -> int:
        if u < 0.35:
            return 0
        if u < 0.55:
            return 1
        return 2

    def _obs(self) -> Tuple[int, int, int, int]:
        return (self._bin_growth(self.last_growth), self._bin_d(self.D), self._bin_ai(self.AI), self._bin_u(self.unemp_risk))

    def reset(self, seed: int | None = None) -> Tuple[Tuple[int, int, int, int], Dict[str, float]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.Y = self._production()
        self.last_growth = 0.035
        self.unemp_risk = 0.45
        self.cyber_risk = 0.30
        self.emission = 0.30
        return self._obs(), self._info()

    def _info(self) -> Dict[str, float]:
        return {
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
            "growth": self.last_growth,
            "unemployment_risk": self.unemp_risk,
            "cyber_risk": self.cyber_risk,
            "emission": self.emission,
            "Y_index": self.Y,
        }

    def step(self, action: int) -> StepResult:
        shares = ACTION_SHARES[int(action)]
        k_s, d_s, ai_s, h_s = shares
        noise = self.rng.normal(0, 0.004, size=4) if self.stochastic else np.zeros(4)
        old_y = self.Y

        # State dynamics: deliberately simple, transparent and stable.
        self.K *= 1.025 + 0.020 * k_s + noise[0]
        self.D = np.clip(self.D + 0.45 + 4.2 * d_s - 0.010 * self.D + noise[1] * 10, 12.0, 42.0)
        self.AI = np.clip(self.AI + 1.2 + 24.0 * ai_s + 0.035 * self.H + noise[2] * 18, 45.0, 170.0)
        self.H = np.clip(self.H + 0.25 + 3.8 * h_s - 0.018 * max(self.AI - 100.0, 0.0) / 10 + noise[3] * 6, 18.0, 48.0)
        self.Y = self._production()
        self.last_growth = max(-0.03, self.Y / old_y - 1.0)

        # Risks: high AI without H increases unemployment and cyber risk; H mitigates shocks.
        self.unemp_risk = np.clip(0.50 + 0.016 * (self.AI - 100.0) - 0.030 * (self.H - 30.0) - 0.15 * h_s, 0.05, 0.95)
        self.cyber_risk = np.clip(0.20 + 0.55 * ai_s + 0.18 * d_s + 0.0025 * max(self.AI - 90.0, 0.0) - 0.012 * (self.H - 30.0), 0.05, 0.95)
        self.emission = np.clip(0.15 + 0.42 * k_s + 0.25 * ai_s - 0.12 * d_s - 0.08 * h_s, 0.05, 0.95)

        reward = WEIGHTS[0] * (self.last_growth * 100.0) - WEIGHTS[1] * self.unemp_risk - WEIGHTS[2] * self.cyber_risk - WEIGHTS[3] * self.emission
        self.t += 1
        done = self.t >= self.T
        return StepResult(self._obs(), float(reward), done, self._info())


def train_q_learning(
    episodes: int = 10000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 42,
    T: int = 10,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    rewards = []
    actions_count = np.zeros(5, dtype=int)
    env = VietnamEconomyEnv(T=T, seed=seed, stochastic=True)

    for ep in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        total = 0.0
        eps = max(eps_end, eps_start * (1.0 - ep / max(1, episodes)))
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(0, 5))
            else:
                a = int(np.argmax(Q[s]))
            res = env.step(a)
            ns = res.obs
            Q[s + (a,)] += alpha * (res.reward + gamma * np.max(Q[ns]) - Q[s + (a,)])
            total += res.reward
            actions_count[a] += 1
            s = ns
            done = res.done
        rewards.append(total)

    # Smooth rewards for display.
    reward_arr = np.array(rewards)
    window = min(100, max(5, episodes // 50))
    kernel = np.ones(window) / window
    smooth = np.convolve(reward_arr, kernel, mode="same")
    return {
        "Q": Q,
        "rewards": reward_arr,
        "smooth_rewards": smooth,
        "actions_count": actions_count,
        "policy": np.argmax(Q, axis=-1),
    }


def evaluate_policy(Q: np.ndarray | None = None, fixed_action: int | None = None, episodes: int = 300, seed: int = 7, T: int = 10) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    totals = []
    final_info = []
    for _ in range(episodes):
        env = VietnamEconomyEnv(T=T, seed=int(rng.integers(0, 1_000_000)), stochastic=True)
        s, _ = env.reset()
        done = False
        total = 0.0
        info = {}
        while not done:
            if fixed_action is not None:
                a = fixed_action
            elif Q is not None:
                a = int(np.argmax(Q[s]))
            else:
                a = int(rng.integers(0, 5))
            res = env.step(a)
            s = res.obs
            total += res.reward
            info = res.info
            done = res.done
        totals.append(total)
        final_info.append(info)
    totals = np.array(totals)
    info_df = pd.DataFrame(final_info)
    return {
        "Mean cumulative reward": float(totals.mean()),
        "Std reward": float(totals.std()),
        "Mean final GDP growth (%)": float(info_df["growth"].mean() * 100.0),
        "Mean final D": float(info_df["D"].mean()),
        "Mean final AI": float(info_df["AI"].mean()),
        "Mean final H": float(info_df["H"].mean()),
        "Mean unemployment risk": float(info_df["unemployment_risk"].mean()),
    }


def compare_policies(Q: np.ndarray, episodes: int = 300, seed: int = 7, T: int = 10) -> pd.DataFrame:
    rows = []
    learned = evaluate_policy(Q=Q, episodes=episodes, seed=seed, T=T)
    learned["Chính sách"] = "π* Q-learning"
    rows.append(learned)
    for a in [1, 3, 4, 0]:
        stats = evaluate_policy(fixed_action=a, episodes=episodes, seed=seed + a + 1, T=T)
        stats["Chính sách"] = ACTION_NAMES[a]
        rows.append(stats)
    random_stats = evaluate_policy(Q=None, fixed_action=None, episodes=episodes, seed=seed + 99, T=T)
    random_stats["Chính sách"] = "Random"
    rows.append(random_stats)
    cols = ["Chính sách"] + [c for c in rows[0] if c != "Chính sách"]
    return pd.DataFrame(rows)[cols].sort_values("Mean cumulative reward", ascending=False)


def policy_table_for_states(Q: np.ndarray) -> pd.DataFrame:
    rows = []
    label = {0: "low", 1: "medium", 2: "high"}
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    a = int(np.argmax(Q[g, d, ai, u]))
                    rows.append({
                        "GDP growth": label[g],
                        "Digital index": label[d],
                        "AI capacity": label[ai],
                        "Unemployment risk": label[u],
                        "Policy action": ACTION_NAMES[a],
                        "Action ID": a,
                    })
    return pd.DataFrame(rows)


def policy_slice(Q: np.ndarray, growth_state: int = 1, unemployment_state: int = 1) -> pd.DataFrame:
    rows = []
    for d in range(3):
        for ai in range(3):
            a = int(np.argmax(Q[growth_state, d, ai, unemployment_state]))
            rows.append({"Digital state": d, "AI state": ai, "Action": a, "Action name": ACTION_NAMES[a]})
    return pd.DataFrame(rows)


def simulate_one_episode_policy(Q: np.ndarray | None = None, fixed_action: int | None = None, seed: int = 123, T: int = 10) -> pd.DataFrame:
    env = VietnamEconomyEnv(T=T, seed=seed, stochastic=False)
    s, info = env.reset()
    rows = [{"t": 0, "action": "start", **info}]
    done = False
    while not done:
        if fixed_action is not None:
            a = fixed_action
        elif Q is not None:
            a = int(np.argmax(Q[s]))
        else:
            a = int(env.rng.integers(0, 5))
        res = env.step(a)
        s = res.obs
        rows.append({"t": env.t, "action": ACTION_NAMES[a], "reward": res.reward, **res.info})
        done = res.done
    return pd.DataFrame(rows)

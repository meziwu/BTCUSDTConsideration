"""
BTC/USDT 組合策略回測
比較三種資金配置方式：
  A. 純均值回歸策略 (MA5 掛單, 100% 資金)
  B. 純網格策略 (Grid, 100% 資金)
  C. 50/50 組合 (各 50% 資金，同時運行)

網格策略邏輯：
  - 以當前價為中心，向下均勻掛 n_levels 層買單，間距 grid_pct%
  - 每層買單成交後，立刻在買入價 + grid_pct% 掛賣單
  - 當價格超出網格範圍時（漲超頂部 or 跌穿底部），重置網格
"""

import numpy as np
import pandas as pd
import json
from itertools import product
from datetime import datetime, timedelta

np.random.seed(42)

# ─────────────────────────────────────────
# 1. 模擬 BTC OHLC 資料（同 backtest.py）
# ─────────────────────────────────────────

def simulate_btc_ohlc(n_bars, bar_minutes, start_price=28_000.0,
                      annual_vol=0.65, annual_drift=0.45, seed=42):
    rng  = np.random.default_rng(seed)
    dt   = bar_minutes / (365 * 24 * 60)
    z    = rng.standard_normal(n_bars)
    log_r = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * z

    cycle_bars = int(18 * 30 * 24 * 60 / bar_minutes)
    t = np.arange(n_bars)
    log_r += 0.30 * np.sin(2 * np.pi * t / cycle_bars) * dt

    prices = np.empty(n_bars + 1)
    prices[0] = start_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(log_r[i])

    close = prices[1:]
    open_ = prices[:-1]
    intra = annual_vol * np.sqrt(dt)
    hl    = open_ * intra * np.abs(rng.standard_normal(n_bars)) * 2
    high  = np.maximum(open_, close) + hl * 0.5
    low   = np.maximum(np.minimum(open_, close) - hl * 0.5, open_ * 0.001)

    start_dt = datetime(2022, 1, 1)
    times = [start_dt + timedelta(minutes=i * bar_minutes) for i in range(n_bars)]
    return pd.DataFrame({"open_time": times, "open": open_,
                         "high": high, "low": low, "close": close})


# ─────────────────────────────────────────
# 2. 策略 A：均值回歸掛單
# ─────────────────────────────────────────

def run_ma_strategy(df, N, M, capital, fee_rate=0.001):
    """
    回傳: (最終資產, 交易次數, 勝次, 最大回撤)
    """
    cash     = capital
    btc_held = 0.0
    trades   = wins = 0
    max_eq   = capital
    max_dd   = 0.0
    buy_px = sell_px = None

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    W = 5

    for i in range(W, len(df)):
        avg5  = closes[i-W:i].mean()
        close = closes[i]

        # 賣出
        if btc_held > 0 and sell_px and highs[i] >= sell_px:
            proceeds = btc_held * sell_px * (1 - fee_rate)
            profit   = proceeds - btc_held * buy_px
            cash    += proceeds
            if profit > 0: wins += 1
            trades += 1
            btc_held = 0.0
            buy_px = sell_px = None

        # 判斷是否掛買單
        if btc_held == 0 and buy_px is None and close < avg5:
            buy_px = close * (1 - N / 100)

        # 買單成交
        if btc_held == 0 and buy_px and lows[i] <= buy_px:
            btc_held = (cash / buy_px) * (1 - fee_rate)
            cash     = 0.0
            sell_px  = buy_px * (1 + M / 100)

        eq = cash + btc_held * close
        if eq > max_eq: max_eq = eq
        dd = (max_eq - eq) / max_eq
        if dd > max_dd: max_dd = dd

    # 期末平倉
    final = cash + btc_held * closes[-1] * (1 - fee_rate)
    return final, trades, wins, max_dd


# ─────────────────────────────────────────
# 3. 策略 B：網格交易
# ─────────────────────────────────────────

def run_grid_strategy(df, grid_pct, n_levels, capital, fee_rate=0.001):
    """
    grid_pct  : 每層間距百分比（如 1.5 = 1.5%）
    n_levels  : 網格層數（如 8 層）
    capital   : 配置資金
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    # 每層可用資金
    per_level = capital / n_levels

    # 初始網格以第一根收盤價為中心
    def make_grid(center_price):
        """回傳各層掛買單價格（由高到低）"""
        return [center_price * (1 - grid_pct / 100 * (i + 1))
                for i in range(n_levels)]

    grid_buys  = make_grid(closes[0])  # 各層買單目標價
    grid_top   = closes[0]             # 網格頂部（超過就重置）
    grid_bot   = grid_buys[-1] * (1 - grid_pct / 100)  # 網格底部

    # 各層狀態：None = 待買, (btc_held, sell_px) = 已持倉待賣
    slots = [None] * n_levels

    cash   = capital
    trades = wins = 0
    max_eq = capital
    max_dd = 0.0

    for i in range(1, len(df)):
        close = closes[i]

        # ── 先處理所有賣單 ──
        for idx in range(n_levels):
            if slots[idx] is not None:
                btc_held, sell_px, buy_cost = slots[idx]
                if highs[i] >= sell_px:
                    proceeds = btc_held * sell_px * (1 - fee_rate)
                    profit   = proceeds - buy_cost
                    cash    += proceeds
                    if profit > 0: wins += 1
                    trades += 1
                    slots[idx] = None

        # ── 處理各層買單 ──
        for idx in range(n_levels):
            if slots[idx] is None and cash >= per_level:
                buy_px = grid_buys[idx]
                if lows[i] <= buy_px:
                    btc  = (per_level / buy_px) * (1 - fee_rate)
                    cost = per_level
                    sell_px = buy_px * (1 + grid_pct / 100)
                    cash   -= per_level
                    slots[idx] = (btc, sell_px, cost)

        # ── 網格重置：價格超出邊界 ──
        if close > grid_top * (1 + grid_pct / 100) or close < grid_bot:
            # 強制賣出所有持倉
            for idx in range(n_levels):
                if slots[idx] is not None:
                    btc_held, _, buy_cost = slots[idx]
                    proceeds = btc_held * close * (1 - fee_rate)
                    profit   = proceeds - buy_cost
                    cash    += proceeds
                    if profit > 0: wins += 1
                    trades += 1
                    slots[idx] = None
            # 重建網格
            grid_buys = make_grid(close)
            grid_top  = close
            grid_bot  = grid_buys[-1] * (1 - grid_pct / 100)
            per_level = cash / n_levels  # 重新均分剩餘資金

        # ── 追蹤回撤 ──
        btc_total = sum(s[0] for s in slots if s is not None)
        eq = cash + btc_total * close
        if eq > max_eq: max_eq = eq
        dd = (max_eq - eq) / max_eq
        if dd > max_dd: max_dd = dd

    # 期末平倉
    btc_total = sum(s[0] for s in slots if s is not None)
    final = cash + btc_total * closes[-1] * (1 - fee_rate)
    return final, trades, wins, max_dd


# ─────────────────────────────────────────
# 4. 組合回測：A(50%) + B(50%)
# ─────────────────────────────────────────

def run_combined(df, N, M, grid_pct, n_levels,
                 initial_capital=10_000.0, fee_rate=0.001):
    half = initial_capital / 2

    ma_final,   ma_tr,   ma_wins,   ma_dd   = run_ma_strategy(
        df, N, M, half, fee_rate)
    grid_final, grid_tr, grid_wins, grid_dd = run_grid_strategy(
        df, grid_pct, n_levels, half, fee_rate)

    final   = ma_final + grid_final
    ret_pct = (final - initial_capital) / initial_capital * 100
    trades  = ma_tr + grid_tr
    wins    = ma_wins + grid_wins
    win_rate = wins / trades * 100 if trades > 0 else 0
    # 組合最大回撤近似（取兩者加權平均，不完全相關）
    comb_dd  = (ma_dd + grid_dd) / 2 * 100

    return {
        "return_pct":   round(ret_pct, 2),
        "final_equity": round(final, 2),
        "trades":       trades,
        "win_rate":     round(win_rate, 2),
        "max_drawdown": round(comb_dd, 2),
        "ma_return":    round((ma_final - half) / half * 100, 2),
        "grid_return":  round((grid_final - half) / half * 100, 2),
    }


# ─────────────────────────────────────────
# 5. 網格參數網格搜尋
# ─────────────────────────────────────────

GRID_PCT    = [0.5, 1.0, 1.5, 2.0, 3.0]
GRID_LEVELS = [5, 8, 10]

# 固定均值回歸最佳參數（日線回測結果）
MA_N = 4.0
MA_M = 5.0

INITIAL_CAPITAL = 10_000.0
FEE = 0.001


def full_comparison(df, label):
    print(f"\n{'='*65}")
    print(f"時框: {label}")
    print(f"{'='*65}")

    # ── 策略 A：純均值回歸（100%）──
    ma_f, ma_t, ma_w, ma_dd = run_ma_strategy(df, MA_N, MA_M, INITIAL_CAPITAL)
    ma_ret  = (ma_f - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    ma_wr   = ma_w / ma_t * 100 if ma_t > 0 else 0

    print(f"\n【策略 A】純均值回歸 (N={MA_N}%, M={MA_M}%, 100% 資金)")
    print(f"  總回報={ma_ret:.2f}%  交易={ma_t}次  勝率={ma_wr:.1f}%"
          f"  最大回撤={ma_dd*100:.2f}%  最終={ma_f:,.0f} USDT")

    # ── 策略 B：純網格（100%），搜尋最佳網格參數 ──
    best_grid = {"return_pct": -999}
    for gp, gl in product(GRID_PCT, GRID_LEVELS):
        gf, gt, gw, gdd = run_grid_strategy(df, gp, gl, INITIAL_CAPITAL)
        gr  = (gf - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        gwr = gw / gt * 100 if gt > 0 else 0
        if gr > best_grid["return_pct"]:
            best_grid = {"grid_pct": gp, "n_levels": gl,
                         "return_pct": gr, "trades": gt,
                         "win_rate": round(gwr, 1),
                         "max_drawdown": round(gdd * 100, 2),
                         "final": gf}

    print(f"\n【策略 B】純網格 最佳參數"
          f" (間距={best_grid['grid_pct']}%, 層數={best_grid['n_levels']}, 100% 資金)")
    print(f"  總回報={best_grid['return_pct']:.2f}%"
          f"  交易={best_grid['trades']}次"
          f"  勝率={best_grid['win_rate']}%"
          f"  最大回撤={best_grid['max_drawdown']}%"
          f"  最終={best_grid['final']:,.0f} USDT")

    # ── 策略 C：50/50 組合，搜尋最佳網格參數 ──
    best_comb = {"return_pct": -999}
    for gp, gl in product(GRID_PCT, GRID_LEVELS):
        res = run_combined(df, MA_N, MA_M, gp, gl, INITIAL_CAPITAL)
        if res["return_pct"] > best_comb["return_pct"]:
            best_comb = {"grid_pct": gp, "n_levels": gl, **res}

    print(f"\n【策略 C】50/50 組合 最佳參數"
          f" (均值回歸 N={MA_N}% M={MA_M}% +"
          f" 網格間距={best_comb['grid_pct']}% 層數={best_comb['n_levels']})")
    print(f"  總回報={best_comb['return_pct']:.2f}%"
          f"  交易={best_comb['trades']}次"
          f"  勝率={best_comb['win_rate']}%"
          f"  最大回撤={best_comb['max_drawdown']}%"
          f"  最終={best_comb['final_equity']:,.0f} USDT")
    print(f"  └ 均值回歸貢獻: {best_comb['ma_return']:.2f}%"
          f"  網格貢獻: {best_comb['grid_return']:.2f}%")

    # ── 比較摘要 ──
    print(f"\n  ┌─{'─'*30}─┬──────────┬──────────┬──────────┐")
    print(f"  │ {'指標':<30} │  策略 A  │  策略 B  │  策略 C  │")
    print(f"  ├─{'─'*30}─┼──────────┼──────────┼──────────┤")

    def row(name, a, b, c):
        print(f"  │ {name:<30} │{a:>9} │{b:>9} │{c:>9} │")

    row("總回報 (%)",
        f"{ma_ret:.1f}%", f"{best_grid['return_pct']:.1f}%", f"{best_comb['return_pct']:.1f}%")
    row("交易次數",
        str(ma_t), str(best_grid['trades']), str(best_comb['trades']))
    row("勝率 (%)",
        f"{ma_wr:.1f}%", f"{best_grid['win_rate']}%", f"{best_comb['win_rate']}%")
    row("最大回撤 (%)",
        f"{ma_dd*100:.1f}%", f"{best_grid['max_drawdown']}%", f"{best_comb['max_drawdown']}%")
    row("Calmar (回報/回撤)",
        f"{ma_ret/(ma_dd*100+1e-9):.2f}",
        f"{best_grid['return_pct']/(best_grid['max_drawdown']+1e-9):.2f}",
        f"{best_comb['return_pct']/(best_comb['max_drawdown']+1e-9):.2f}")
    row("最終資產 (USDT)",
        f"{ma_f:,.0f}", f"{best_grid['final']:,.0f}", f"{best_comb['final_equity']:,.0f}")

    print(f"  └─{'─'*30}─┴──────────┴──────────┴──────────┘")

    return {
        "A_return": ma_ret, "A_dd": ma_dd * 100,
        "B_return": best_grid["return_pct"], "B_dd": best_grid["max_drawdown"],
        "C_return": best_comb["return_pct"], "C_dd": best_comb["max_drawdown"],
        "C_grid_pct": best_comb["grid_pct"], "C_n_levels": best_comb["n_levels"],
    }


# ─────────────────────────────────────────
# 6. 主程式
# ─────────────────────────────────────────

TIMEFRAMES = {
    "1h":  {"minutes": 60,   "bars": 2500,  "label": "1小時"},
    "4h":  {"minutes": 240,  "bars": 1500,  "label": "4小時"},
    "1d":  {"minutes": 1440, "bars": 730,   "label": "日線"},
}


def main():
    print("=" * 65)
    print("BTC/USDT 三策略比較回測")
    print("  A = 均值回歸 (100%)  |  B = 網格 (100%)  |  C = 50/50 組合")
    print(f"  均值回歸固定參數: N={MA_N}%, M={MA_M}%")
    print(f"  初始資金: {INITIAL_CAPITAL:,} USDT  手續費: {FEE*100}%/筆")
    print("=" * 65)

    all_summary = []
    for tf_key, tf_info in TIMEFRAMES.items():
        df  = simulate_btc_ohlc(tf_info["bars"], tf_info["minutes"], seed=42)
        res = full_comparison(df, tf_info["label"])
        res["時框"] = tf_info["label"]
        all_summary.append(res)

    # ── 跨時框總結 ──
    print(f"\n{'='*65}")
    print("跨時框策略比較總覽")
    print(f"{'='*65}")

    hdr = f"{'時框':<8} {'策略A回報':>10} {'策略B回報':>10} {'策略C回報':>10} " \
          f"{'A回撤':>8} {'B回撤':>8} {'C回撤':>8} {'結論'}"
    print(hdr)
    print("-" * 80)
    for r in all_summary:
        # 判斷哪個策略最好（Calmar）
        calmar_a = r["A_return"] / (r["A_dd"] + 1e-9)
        calmar_b = r["B_return"] / (r["B_dd"] + 1e-9)
        calmar_c = r["C_return"] / (r["C_dd"] + 1e-9)
        best = ["A", "B", "C"][[calmar_a, calmar_b, calmar_c].index(max(calmar_a, calmar_b, calmar_c))]

        print(f"{r['時框']:<8} {r['A_return']:>9.1f}% {r['B_return']:>9.1f}% "
              f"{r['C_return']:>9.1f}% {r['A_dd']:>7.1f}% {r['B_dd']:>7.1f}% "
              f"{r['C_dd']:>7.1f}%  ★策略{best}最優")

    print(f"\n{'='*65}")
    print("★★ 策略選擇建議")
    print(f"{'='*65}")
    print("""
  【策略 C：50/50 組合】的核心優勢：

  1. 資金效率提升
     - 均值回歸策略在等買點期間資金閒置
     - 網格可讓這50%閒置資金持續產生獲利
     - 兩者幾乎不競爭同一資金

  2. 風險分散效果
     - 均值回歸：橫盤震盪時獲利
     - 網格：在設定範圍內任何雙向波動都獲利
     - 避免單一市場狀態失效時全軍覆沒

  3. 回撤控制更好
     - 均值回歸：持倉期間若暴跌損失大
     - 網格：分層持倉，跌越深買越多，反彈時全部獲利
     - 組合後單邊暴跌損失被網格多層買入攤平

  ⚠️  組合策略建議的停損規則：
     - 網格：整體資金跌破配置的 15% 時暫停重置
     - 均值回歸：買入後跌破買入價 N×1.5% 時止損
     - 兩部位各自獨立管理，不互相借用資金
""")


if __name__ == "__main__":
    main()

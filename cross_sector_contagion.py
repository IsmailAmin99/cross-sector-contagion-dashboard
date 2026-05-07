"""
Cross-Sector Impact Mapping:
Predicting Contagion Effects of Geopolitical and Macro Events for Market Alpha

Authors: Isma'il Amin & Danielson Azumah
CSCI-3412 - Machine Learning

Full pipeline: data collection > feature engineering > event-study labeling >
dynamic graph construction > GAT model > training > evaluation > visualization
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

from tqdm import tqdm

OUTPUT_DIR = Path("contagion_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SECTOR & MACRO TICKER DEFINITIONS
# ---------------------------------------------------------------------------

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLB": "Materials",
}

MACRO_TICKERS = {
    "^VIX": "VIX",
    "^TNX": "10Y_Yield",
    "DX-Y.NYB": "DXY",
    "CL=F": "WTI_Crude",
    "GC=F": "Gold",
    "^IRX": "3M_TBill",
}

BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# 2. CURATED EVENT DATABASE (2021-2025)
# ---------------------------------------------------------------------------

EVENTS = [
    # 2021
    {"date": "2021-03-23", "category": "supply_chain", "severity": 3,
     "name": "Ever Given blocks Suez Canal"},
    {"date": "2021-06-16", "category": "rate_decision", "severity": 2,
     "name": "Fed hawkish dot-plot surprise - taper talk begins"},
    {"date": "2021-09-20", "category": "recession_scare", "severity": 2,
     "name": "Evergrande default fears shake global markets"},
    {"date": "2021-11-26", "category": "recession_scare", "severity": 3,
     "name": "Omicron variant discovered - travel/growth scare"},
    # 2022
    {"date": "2022-01-26", "category": "rate_decision", "severity": 3,
     "name": "Fed signals aggressive rate hikes - hawkish pivot"},
    {"date": "2022-02-24", "category": "geopolitical", "severity": 5,
     "name": "Russia invades Ukraine"},
    {"date": "2022-03-08", "category": "energy_shock", "severity": 4,
     "name": "Oil spikes above $130/bbl on Russia sanctions"},
    {"date": "2022-03-16", "category": "rate_decision", "severity": 3,
     "name": "Fed raises rates 25bp - first hike since 2018"},
    {"date": "2022-05-04", "category": "rate_decision", "severity": 3,
     "name": "Fed raises rates 50bp"},
    {"date": "2022-06-10", "category": "inflation_surprise", "severity": 4,
     "name": "CPI hits 8.6% - inflation higher than expected"},
    {"date": "2022-06-15", "category": "rate_decision", "severity": 4,
     "name": "Fed raises rates 75bp - largest hike since 1994"},
    {"date": "2022-07-27", "category": "rate_decision", "severity": 3,
     "name": "Fed raises rates another 75bp"},
    {"date": "2022-09-13", "category": "inflation_surprise", "severity": 4,
     "name": "August CPI surprises hot - 8.3% vs 8.1% expected"},
    {"date": "2022-09-21", "category": "rate_decision", "severity": 3,
     "name": "Fed raises rates 75bp - third consecutive"},
    {"date": "2022-09-26", "category": "geopolitical", "severity": 3,
     "name": "UK gilt crisis - LDI pension blowup"},
    {"date": "2022-11-02", "category": "rate_decision", "severity": 3,
     "name": "Fed raises rates 75bp - fourth consecutive"},
    {"date": "2022-12-14", "category": "rate_decision", "severity": 2,
     "name": "Fed raises rates 50bp - step-down but hawkish guidance"},
    # 2023
    {"date": "2023-02-14", "category": "inflation_surprise", "severity": 3,
     "name": "Jan CPI hotter than expected - 6.4% vs 6.2%"},
    {"date": "2023-03-10", "category": "recession_scare", "severity": 5,
     "name": "SVB collapses - regional banking crisis begins"},
    {"date": "2023-03-13", "category": "recession_scare", "severity": 4,
     "name": "Signature Bank fails - FDIC emergency measures"},
    {"date": "2023-03-20", "category": "recession_scare", "severity": 3,
     "name": "Credit Suisse forced rescue by UBS"},
    {"date": "2023-05-01", "category": "recession_scare", "severity": 3,
     "name": "First Republic Bank seized by FDIC"},
    {"date": "2023-05-03", "category": "rate_decision", "severity": 2,
     "name": "Fed raises rates 25bp - signals possible pause"},
    {"date": "2023-07-26", "category": "rate_decision", "severity": 2,
     "name": "Fed raises rates 25bp to 5.25-5.50% - peak rate"},
    {"date": "2023-10-07", "category": "geopolitical", "severity": 4,
     "name": "Hamas attacks Israel - Middle East escalation"},
    {"date": "2023-10-19", "category": "rate_decision", "severity": 2,
     "name": "10Y Treasury hits 5% - highest since 2007"},
    {"date": "2023-11-19", "category": "supply_chain", "severity": 3,
     "name": "Houthi Red Sea attacks begin - shipping disruption"},
    # 2024
    {"date": "2024-01-11", "category": "supply_chain", "severity": 3,
     "name": "Red Sea crisis intensifies - major shipping rerouting"},
    {"date": "2024-04-10", "category": "inflation_surprise", "severity": 3,
     "name": "March CPI 3.5% - hot inflation stalls rate cut hopes"},
    {"date": "2024-04-13", "category": "geopolitical", "severity": 3,
     "name": "Iran launches drone/missile attack on Israel"},
    {"date": "2024-08-05", "category": "recession_scare", "severity": 4,
     "name": "Yen carry trade unwind - global market crash"},
    {"date": "2024-09-18", "category": "rate_decision", "severity": 3,
     "name": "Fed cuts rates 50bp - first cut since 2020"},
    {"date": "2024-11-05", "category": "geopolitical", "severity": 3,
     "name": "US election - Trump victory, tariff fears spike"},
    {"date": "2024-12-18", "category": "rate_decision", "severity": 3,
     "name": "Fed cuts 25bp but hawkish guidance tanks market"},
    # 2025
    {"date": "2025-01-20", "category": "geopolitical", "severity": 3,
     "name": "Trump inauguration - tariff policy uncertainty surges"},
    {"date": "2025-02-01", "category": "geopolitical", "severity": 4,
     "name": "US imposes 25% tariffs on Canada/Mexico, 10% on China"},
    {"date": "2025-03-04", "category": "geopolitical", "severity": 4,
     "name": "Tariffs take effect - retaliation from trading partners"},
    {"date": "2025-04-02", "category": "geopolitical", "severity": 5,
     "name": "Liberation Day reciprocal tariffs announced"},
]


# ---------------------------------------------------------------------------
# 3. DATA COLLECTION
# ---------------------------------------------------------------------------

def download_market_data(start="2021-01-01", end="2025-12-31"):
    """Download sector ETFs, benchmark, and macro proxies from yfinance."""
    all_tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK] + list(MACRO_TICKERS.keys())
    print(f"[DATA] Downloading {len(all_tickers)} tickers from {start} to {end} ...")

    raw = yf.download(all_tickers, start=start, end=end, progress=True, auto_adjust=True)

    prices = raw["Close"].copy()
    volumes = raw["Volume"].copy()

    rename_map = {**{t: t for t in list(SECTOR_ETFS.keys()) + [BENCHMARK]},
                  **MACRO_TICKERS}
    prices = prices.rename(columns=rename_map)
    volumes = volumes.rename(columns=rename_map)

    prices = prices.ffill().dropna(how="all")
    volumes = volumes.ffill().dropna(how="all")

    print(f"[DATA] Got {len(prices)} trading days, {prices.shape[1]} columns")
    return prices, volumes


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def compute_node_features(prices, volumes, sector_tickers, window_short=10, window_long=20):
    """Compute per-node (per-sector) features: returns, vol, excess return, volume z-score."""
    features = {}
    spy_ret = prices[BENCHMARK].pct_change()

    for ticker in sector_tickers:
        if ticker not in prices.columns:
            continue
        df = pd.DataFrame(index=prices.index)
        ret = prices[ticker].pct_change()

        df["ret_1d"] = ret
        df["ret_5d"] = prices[ticker].pct_change(5)
        df["ret_10d"] = prices[ticker].pct_change(10)
        df["ret_20d"] = prices[ticker].pct_change(20)
        df["rvol_10d"] = ret.rolling(window_short).std() * np.sqrt(252)
        df["rvol_20d"] = ret.rolling(window_long).std() * np.sqrt(252)
        df["excess_ret_5d"] = df["ret_5d"] - spy_ret.rolling(5).sum()

        if ticker in volumes.columns:
            vol_mean = volumes[ticker].rolling(window_long).mean()
            vol_std = volumes[ticker].rolling(window_long).std()
            df["vol_zscore"] = (volumes[ticker] - vol_mean) / (vol_std + 1e-9)
        else:
            df["vol_zscore"] = 0.0

        if "VIX" in prices.columns:
            df["vix_level"] = prices["VIX"]
            df["vix_change"] = prices["VIX"].pct_change()
        elif "^VIX" in prices.columns:
            df["vix_level"] = prices["^VIX"]
            df["vix_change"] = prices["^VIX"].pct_change()
        else:
            df["vix_level"] = 0.0
            df["vix_change"] = 0.0

        features[ticker] = df

    return features


def compute_pairwise_correlations(prices, sector_tickers, window=20):
    """Compute rolling pairwise correlations between sectors."""
    returns = prices[sector_tickers].pct_change()
    corr_series = {}
    tickers = [t for t in sector_tickers if t in returns.columns]
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            corr_series[(t1, t2)] = returns[t1].rolling(window).corr(returns[t2])
    return pd.DataFrame(corr_series, index=returns.index)


def compute_granger_edges(prices, sector_tickers, window=60, max_lag=5, pvalue_thresh=0.05):
    """
    Compute Granger-causal edges on the most recent `window` days of data.
    Returns list of (source, target, pvalue) for significant relationships.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    returns = prices[sector_tickers].pct_change().dropna().tail(window)
    edges = []
    tickers = [t for t in sector_tickers if t in returns.columns]

    for t1 in tickers:
        for t2 in tickers:
            if t1 == t2:
                continue
            try:
                data = pd.DataFrame({"y": returns[t2], "x": returns[t1]}).dropna()
                if len(data) < max_lag + 5:
                    continue
                result = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=False)
                min_p = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
                if min_p < pvalue_thresh:
                    edges.append((t1, t2, min_p))
            except Exception:
                continue
    return edges


# ---------------------------------------------------------------------------
# 5. EVENT-STUDY LABELING
# ---------------------------------------------------------------------------

def compute_beta(sector_ret, benchmark_ret, window=60):
    """Compute rolling beta of sector to benchmark."""
    cov = sector_ret.rolling(window).cov(benchmark_ret)
    var = benchmark_ret.rolling(window).var()
    return cov / (var + 1e-12)


def label_event_responses(prices, events, sector_tickers,
                          pre_window=5, post_window=10,
                          stress_threshold=1.0):
    """
    For each event, compute abnormal returns for each sector in the post-event window.
    Label sectors as: 0=unaffected, 1=immediate_stress, 2=delayed_stress
    """
    sector_list = [t for t in sector_tickers if t in prices.columns]
    returns = prices.pct_change()
    spy_ret = returns[BENCHMARK]

    records = []
    trading_dates = prices.index

    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if ev_date not in trading_dates:
            idx = trading_dates.searchsorted(ev_date)
            if idx >= len(trading_dates):
                continue
            ev_date = trading_dates[idx]

        ev_idx = trading_dates.get_loc(ev_date)
        if ev_idx < pre_window + 60 or ev_idx + post_window >= len(trading_dates):
            continue

        pre_start = ev_idx - pre_window
        pre_end = ev_idx
        post_end = ev_idx + post_window

        for sector in sector_list:
            sect_ret = returns[sector]
            beta = compute_beta(sect_ret, spy_ret).iloc[ev_idx]
            if pd.isna(beta):
                beta = 1.0

            post_sect = sect_ret.iloc[pre_end + 1: post_end + 1].values
            post_spy = spy_ret.iloc[pre_end + 1: post_end + 1].values
            abnormal = post_sect - beta * post_spy

            pre_sect = sect_ret.iloc[pre_start:pre_end].values
            pre_spy = spy_ret.iloc[pre_start:pre_end].values
            pre_abnormal = pre_sect - beta * pre_spy
            pre_std = np.std(pre_abnormal) if np.std(pre_abnormal) > 1e-6 else 0.01

            cum_abnormal = np.cumsum(abnormal)

            early_stressed = np.any(cum_abnormal[:3] < -stress_threshold * pre_std * np.sqrt(3))
            late_stressed = np.any(cum_abnormal[3:] < -stress_threshold * pre_std * np.sqrt(np.arange(4, post_window + 1)))

            if early_stressed and late_stressed:
                label = 1
            elif not early_stressed and late_stressed:
                label = 2  # delayed stress
            elif early_stressed:
                label = 1
            else:
                label = 0

            records.append({
                "event_date": ev_date,
                "event_name": ev["name"],
                "event_category": ev["category"],
                "event_severity": ev["severity"],
                "sector": sector,
                "label": label,
                "cum_abnormal_return": cum_abnormal[-1] if len(cum_abnormal) > 0 else 0.0,
                "max_drawdown": np.min(cum_abnormal) if len(cum_abnormal) > 0 else 0.0,
            })

    df = pd.DataFrame(records)
    print(f"\n[LABELS] Generated {len(df)} sector-event labels:")
    print(df["label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# 6. DYNAMIC GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def build_dynamic_graph(prices, sector_tickers, date, corr_window=20,
                        corr_threshold=0.3):
    """
    Build a market graph snapshot at a given date.
    Nodes = sectors, edges = significant rolling correlations + Granger edges.
    """
    sector_list = [t for t in sector_tickers if t in prices.columns]
    trading_dates = prices.index
    if date not in trading_dates:
        idx = trading_dates.searchsorted(date)
        if idx >= len(trading_dates):
            idx = len(trading_dates) - 1
        date = trading_dates[idx]

    date_idx = trading_dates.get_loc(date)
    if date_idx < corr_window + 5:
        return None, None

    window_prices = prices[sector_list].iloc[max(0, date_idx - corr_window):date_idx + 1]
    window_returns = window_prices.pct_change().dropna()

    if len(window_returns) < 10:
        return None, None

    corr_matrix = window_returns.corr()

    G = nx.DiGraph()
    for i, s in enumerate(sector_list):
        G.add_node(s, idx=i)

    edge_list = []
    edge_weights = []
    for i, s1 in enumerate(sector_list):
        for j, s2 in enumerate(sector_list):
            if i == j:
                continue
            c = corr_matrix.loc[s1, s2] if s1 in corr_matrix.index and s2 in corr_matrix.columns else 0
            if abs(c) > corr_threshold:
                G.add_edge(s1, s2, weight=abs(c))
                edge_list.append([i, j])
                edge_weights.append(abs(c))

    if not edge_list:
        for i, s1 in enumerate(sector_list):
            for j, s2 in enumerate(sector_list):
                if i != j:
                    c = corr_matrix.loc[s1, s2] if s1 in corr_matrix.index and s2 in corr_matrix.columns else 0.1
                    edge_list.append([i, j])
                    edge_weights.append(abs(c))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) if edge_weights else torch.zeros((0, 1))

    return G, (edge_index, edge_attr)


# ---------------------------------------------------------------------------
# 7. DATASET PREPARATION FOR GNN
# ---------------------------------------------------------------------------

EVENT_CATEGORIES = ["rate_decision", "inflation_surprise", "geopolitical",
                    "supply_chain", "energy_shock", "recession_scare"]


def prepare_gnn_dataset(labels_df, node_features, prices, sector_tickers):
    """
    Build PyG Data objects for each event.
    Each graph = snapshot of market at event time with per-node features and labels.
    """
    sector_list = [t for t in sector_tickers if t in prices.columns]
    n_sectors = len(sector_list)
    scaler = StandardScaler()

    feature_cols = ["ret_1d", "ret_5d", "ret_10d", "ret_20d",
                    "rvol_10d", "rvol_20d", "excess_ret_5d", "vol_zscore",
                    "vix_level", "vix_change"]

    event_dates = labels_df["event_date"].unique()
    dataset = []

    for ev_date in event_dates:
        ev_rows = labels_df[labels_df["event_date"] == ev_date]
        if len(ev_rows) != n_sectors:
            continue

        ev_category = ev_rows.iloc[0]["event_category"]
        ev_severity = ev_rows.iloc[0]["event_severity"]

        cat_onehot = [1.0 if c == ev_category else 0.0 for c in EVENT_CATEGORIES]
        event_vec = cat_onehot + [ev_severity / 5.0]

        x_list = []
        y_list = []
        for sector in sector_list:
            row = ev_rows[ev_rows["sector"] == sector]
            if row.empty:
                x_list.append(np.zeros(len(feature_cols) + len(event_vec)))
                y_list.append(0)
                continue

            if sector in node_features and ev_date in node_features[sector].index:
                feats = node_features[sector].loc[ev_date, feature_cols].values.astype(float)
            else:
                feats = np.zeros(len(feature_cols))

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            combined = np.concatenate([feats, event_vec])
            x_list.append(combined)

            label = row.iloc[0]["label"]
            y_list.append(1 if label >= 1 else 0)  # binary: any stress vs unaffected

        x = torch.tensor(np.array(x_list), dtype=torch.float)
        y = torch.tensor(y_list, dtype=torch.long)

        _, graph_tensors = build_dynamic_graph(prices, sector_tickers, ev_date)
        if graph_tensors is None:
            continue
        edge_index, edge_attr = graph_tensors

        if edge_index.shape[1] == 0:
            continue

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.event_date = str(ev_date.date()) if hasattr(ev_date, 'date') else str(ev_date)
        data.event_name = ev_rows.iloc[0]["event_name"]
        dataset.append(data)

    print(f"\n[DATASET] Built {len(dataset)} graph snapshots")

    total_pos = sum(d.y.sum().item() for d in dataset)
    total_neg = sum((d.y == 0).sum().item() for d in dataset)
    print(f"  Delayed-stress nodes: {total_pos}, Unaffected nodes: {total_neg}")

    return dataset


# ---------------------------------------------------------------------------
# 8. GAT MODEL
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for handling severe class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EventConditionedGAT(nn.Module):
    """
    Graph Attention Network conditioned on macro event context.
    2-layer GAT with per-node classification head.
    """
    def __init__(self, in_channels, hidden_channels=32, heads=4, dropout=0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=2, concat=False, dropout=dropout, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 2),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        out = self.classifier(x)
        return out


# ---------------------------------------------------------------------------
# 9. TRAINING PIPELINE
# ---------------------------------------------------------------------------

def train_gat(dataset, train_ratio=0.7, val_ratio=0.15, epochs=150, lr=0.003,
              weight_decay=5e-4):
    """Walk-forward train/val/test split, then train the GAT."""
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]

    print(f"\n[TRAIN] Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    if not train_set:
        print("[TRAIN] Not enough training data!")
        return None, [], [], []

    in_channels = train_set[0].x.shape[1]
    model = EventConditionedGAT(in_channels=in_channels, hidden_channels=48, heads=4, dropout=0.2)

    pos_count = sum(d.y.sum().item() for d in train_set)
    neg_count = sum((d.y == 0).sum().item() for d in train_set)
    pos_weight = max(neg_count / (pos_count + 1), 1.0)
    class_weights = torch.tensor([1.0, min(pos_weight, 15.0)])
    print(f"  Class balance: {neg_count} neg / {pos_count} pos -> weight={class_weights[1]:.1f}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    train_losses = []
    val_f1s = []
    best_val_f1 = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for data in train_set:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_set)
        train_losses.append(avg_loss)

        if val_set:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data in val_set:
                    out = model(data)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.numpy())
                    all_labels.extend(data.y.numpy())
            vf1 = f1_score(all_labels, all_preds, zero_division=0)
            val_f1s.append(vf1)
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            val_f1s.append(0)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val F1: {val_f1s[-1]:.4f}")

    if best_state is not None and best_val_f1 > 0:
        model.load_state_dict(best_state)
        print(f"  Restored best model (Val F1={best_val_f1:.4f})")

    return model, train_losses, val_f1s, test_set


# ---------------------------------------------------------------------------
# 10. BASELINE MODELS
# ---------------------------------------------------------------------------

def flatten_graphs_for_baselines(dataset):
    """Convert graph dataset to flat feature/label arrays for sklearn baselines."""
    X, y = [], []
    for data in dataset:
        for i in range(data.x.shape[0]):
            X.append(data.x[i].numpy())
            y.append(data.y[i].item())
    return np.array(X), np.array(y)


def train_baselines(train_set, test_set):
    """Train logistic regression and random forest baselines."""
    X_train, y_train = flatten_graphs_for_baselines(train_set)
    X_test, y_test = flatten_graphs_for_baselines(test_set)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))
    results["Logistic Regression"] = {
        "preds": lr_preds, "proba": lr_proba, "labels": y_test
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train_s, y_train)
    rf_preds = rf.predict(X_test_s)
    rf_proba = rf.predict_proba(X_test_s)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))
    results["Random Forest"] = {
        "preds": rf_preds, "proba": rf_proba, "labels": y_test
    }

    return results


# ---------------------------------------------------------------------------
# 11. EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, test_set, model_name="GAT"):
    """Evaluate a trained GNN model on test data."""
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for data in test_set:
            out = model(data)
            proba = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(data.y.numpy())
            all_proba.extend(proba.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_proba = np.array(all_proba)

    return {"preds": all_preds, "proba": all_proba, "labels": all_labels}


def print_evaluation_table(results_dict):
    """Print a comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - SECTOR CONTAGION / STRESS PREDICTION")
    print("=" * 80)
    print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'TNR':>7}")
    print("-" * 80)

    for name, res in results_dict.items():
        y_true, y_pred = res["labels"], res["preds"]
        y_proba = res["proba"]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"{name:<25} {acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {auc:>7.3f} {tnr:>7.3f}")

    print("=" * 80)

    for name, res in results_dict.items():
        y_true, y_pred = res["labels"], res["preds"]
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=["Unaffected", "Stressed"],
                                    zero_division=0))


# ---------------------------------------------------------------------------
# 12. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses, val_f1s):
    """Plot training loss and validation F1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, color="#2563eb", linewidth=1.5)
    ax1.set_title("GAT Training Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_f1s, color="#16a34a", linewidth=1.5)
    ax2.set_title("GAT Validation F1 Score", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved training_curves.png")


def plot_confusion_matrices(results_dict):
    """Plot confusion matrices for all models side by side."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results_dict.items()):
        y_true, y_pred = res["labels"], res["preds"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Unaff.", "Stressed"])
        ax.set_yticklabels(["Unaff.", "Stressed"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved confusion_matrices.png")


def plot_event_label_distribution(labels_df):
    """Plot how labels distribute across event categories."""
    label_map = {0: "Unaffected", 1: "Immediate Stress", 2: "Delayed Stress"}
    labels_df["label_name"] = labels_df["label"].map(label_map)

    ct = pd.crosstab(labels_df["event_category"], labels_df["label_name"])
    ct = ct.reindex(columns=["Unaffected", "Immediate Stress", "Delayed Stress"], fill_value=0)

    colors = ["#93c5fd", "#f87171", "#fbbf24"]
    ax = ct.plot(kind="barh", stacked=True, color=colors, figsize=(12, 6), edgecolor="white")
    ax.set_title("Sector Response Labels by Event Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count of Sector-Event Pairs")
    ax.legend(title="Label", loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved label_distribution.png")


def plot_market_graph_snapshot(prices, sector_tickers, date, event_name=""):
    """Visualize the market graph at a specific event date."""
    G, _ = build_dynamic_graph(prices, sector_tickers, pd.Timestamp(date))
    if G is None:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    edges = G.edges(data=True)
    weights = [d.get("weight", 0.3) for _, _, d in edges]
    max_w = max(weights) if weights else 1

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           width=[2 * w / max_w for w in weights],
                           edge_color="#94a3b8")

    node_colors = []
    sector_names = []
    for node in G.nodes():
        sector_names.append(SECTOR_ETFS.get(node, node))
        cat_color = {
            "Technology": "#3b82f6", "Energy": "#f59e0b", "Healthcare": "#10b981",
            "Financials": "#8b5cf6", "Industrials": "#6b7280", "Communication Services": "#ec4899",
            "Consumer Discretionary": "#f97316", "Consumer Staples": "#14b8a6",
            "Real Estate": "#a855f7", "Utilities": "#eab308", "Materials": "#78716c",
        }
        node_colors.append(cat_color.get(SECTOR_ETFS.get(node, ""), "#64748b"))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800,
                           node_color=node_colors, edgecolors="white", linewidths=2)
    labels = {n: SECTOR_ETFS.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    title = f"Market Graph - {date}"
    if event_name:
        title += f"\n{event_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    safe_date = str(date).replace("-", "")[:8]
    plt.savefig(OUTPUT_DIR / f"graph_{safe_date}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved graph_{safe_date}.png")


def plot_sector_heatmap(labels_df):
    """Heatmap of cumulative abnormal returns per sector per event."""
    pivot = labels_df.pivot_table(
        index="event_name", columns="sector", values="cum_abnormal_return", aggfunc="mean"
    )
    if pivot.empty:
        return

    pivot = pivot.iloc[-20:]  # last 20 events for readability

    fig, ax = plt.subplots(figsize=(16, max(8, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.08, vmax=0.08)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([SECTOR_ETFS.get(c, c) for c in pivot.columns],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    plt.colorbar(im, ax=ax, label="Cumulative Abnormal Return", shrink=0.8)
    ax.set_title("Sector Abnormal Returns After Major Events", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sector_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved sector_heatmap.png")


def plot_contagion_timeline(labels_df):
    """Show which sectors got hit (and when) across events over time."""
    delayed = labels_df[labels_df["label"] == 2].copy()
    if delayed.empty:
        delayed = labels_df[labels_df["label"] >= 1].copy()
    if delayed.empty:
        print("[VIZ] No stress events to plot timeline")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    sector_list = sorted(delayed["sector"].unique())
    sector_map = {s: i for i, s in enumerate(sector_list)}

    for _, row in delayed.iterrows():
        y = sector_map[row["sector"]]
        x = pd.Timestamp(row["event_date"])
        severity = row.get("event_severity", 3)
        color = {"geopolitical": "#ef4444", "rate_decision": "#3b82f6",
                 "inflation_surprise": "#f59e0b", "supply_chain": "#8b5cf6",
                 "energy_shock": "#f97316", "recession_scare": "#6b7280"
                 }.get(row["event_category"], "#64748b")
        ax.scatter(x, y, s=severity * 40, c=color, alpha=0.7, edgecolors="white", zorder=3)

    ax.set_yticks(range(len(sector_list)))
    ax.set_yticklabels([SECTOR_ETFS.get(s, s) for s in sector_list], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.set_title("Contagion Timeline - Stressed Sectors Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.grid(axis="x", alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=8, label='Geopolitical'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', markersize=8, label='Rate Decision'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b', markersize=8, label='Inflation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8b5cf6', markersize=8, label='Supply Chain'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f97316', markersize=8, label='Energy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6b7280', markersize=8, label='Recession'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "contagion_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved contagion_timeline.png")


# ---------------------------------------------------------------------------
# 13. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CROSS-SECTOR CONTAGION PREDICTION PIPELINE")
    print("Isma'il Amin & Danielson Azumah - CSCI-3412")
    print("=" * 80)

    # --- Step 1: Data Collection ---
    print("\n>>> STEP 1: Downloading market data ...")
    prices, volumes = download_market_data(start="2021-01-01", end="2025-12-31")
    prices.to_csv(OUTPUT_DIR / "prices.csv")
    print(f"  Saved prices.csv ({prices.shape})")

    # --- Step 2: Feature Engineering ---
    print("\n>>> STEP 2: Computing node features ...")
    sector_tickers = list(SECTOR_ETFS.keys())
    node_features = compute_node_features(prices, volumes, sector_tickers)
    print(f"  Computed features for {len(node_features)} sectors")

    # --- Step 3: Event-Study Labeling ---
    print("\n>>> STEP 3: Labeling event responses ...")
    labels_df = label_event_responses(prices, EVENTS, sector_tickers)
    labels_df.to_csv(OUTPUT_DIR / "event_labels.csv", index=False)
    print(f"  Saved event_labels.csv")

    # --- Step 4: Visualize label distribution ---
    print("\n>>> STEP 4: Generating visualizations ...")
    plot_event_label_distribution(labels_df)
    plot_sector_heatmap(labels_df)
    plot_contagion_timeline(labels_df)

    key_events = [
        ("2022-02-24", "Russia invades Ukraine"),
        ("2023-03-10", "SVB collapse"),
        ("2025-04-02", "Liberation Day tariffs"),
    ]
    for date, name in key_events:
        try:
            plot_market_graph_snapshot(prices, sector_tickers, date, name)
        except Exception as e:
            print(f"  Warning: Could not plot graph for {date}: {e}")

    # --- Step 5: Prepare GNN dataset ---
    print("\n>>> STEP 5: Building GNN dataset ...")
    dataset = prepare_gnn_dataset(labels_df, node_features, prices, sector_tickers)

    if len(dataset) < 5:
        print("[WARN] Not enough graph samples for meaningful train/test split.")
        print("  Adjusting: using all data for training demo with synthetic test.")
        train_set = dataset
        test_set = dataset[-2:] if len(dataset) >= 2 else dataset
    else:
        train_end = int(len(dataset) * 0.7)
        val_end = int(len(dataset) * 0.85)
        train_set = dataset[:train_end]
        test_set = dataset[val_end:]

    # --- Step 6: Train GAT ---
    print("\n>>> STEP 6: Training Graph Attention Network ...")
    model, train_losses, val_f1s, test_from_train = train_gat(dataset, epochs=150)

    if model is None:
        print("[ERROR] Training failed. Exiting.")
        return

    if test_from_train:
        test_set = test_from_train

    plot_training_curves(train_losses, val_f1s)

    # --- Step 7: Evaluate ---
    print("\n>>> STEP 7: Evaluating models ...")
    results = {}
    gat_results = evaluate_model(model, test_set, "GAT")
    results["GAT (Ours)"] = gat_results

    baseline_train = dataset[:int(len(dataset) * 0.7)]
    if baseline_train and test_set:
        baseline_results = train_baselines(baseline_train, test_set)
        results.update(baseline_results)

    print_evaluation_table(results)
    plot_confusion_matrices(results)

    # --- Step 8: Per-event backtest analysis ---
    print("\n>>> STEP 8: Per-event backtest analysis ...")
    model.eval()
    print(f"\n{'Event':<50} {'Date':<12} {'Predicted Stress':<35} {'Actual Stress':<35} {'Hit?'}")
    print("-" * 140)
    total_hits = 0
    total_events = 0
    for data in test_set:
        with torch.no_grad():
            out = model(data)
            preds = out.argmax(dim=1).numpy()
            proba = F.softmax(out, dim=1)[:, 1].numpy()

        predicted_sectors = [sector_tickers[i] for i in range(len(preds)) if preds[i] == 1]
        actual_sectors = [sector_tickers[i] for i in range(len(data.y)) if data.y[i].item() == 1]

        overlap = set(predicted_sectors) & set(actual_sectors)
        hit = "YES" if overlap else ("--" if not actual_sectors else "MISS")
        if overlap:
            total_hits += 1
        total_events += 1

        ev_name = data.event_name if hasattr(data, "event_name") else "Unknown"
        ev_name = ev_name[:48]
        ev_date = data.event_date if hasattr(data, "event_date") else ""
        pred_str = ", ".join(predicted_sectors) if predicted_sectors else "None"
        actual_str = ", ".join(actual_sectors) if actual_sectors else "None"
        print(f"{ev_name:<50} {str(ev_date):<12} {pred_str:<35} {actual_str:<35} {hit}")

        if predicted_sectors:
            top_proba = sorted(zip(sector_tickers[:len(proba)], proba), key=lambda x: -x[1])[:3]
            print(f"{'':>50} Top risk scores: {', '.join(f'{t}={p:.3f}' for t,p in top_proba)}")

    print(f"\n  Backtest hit rate: {total_hits}/{total_events}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"  - prices.csv           : raw price data")
    print(f"  - event_labels.csv     : event-study labels")
    print(f"  - training_curves.png  : GAT loss and F1 curves")
    print(f"  - confusion_matrices.png : model comparison")
    print(f"  - label_distribution.png : event label breakdown")
    print(f"  - sector_heatmap.png   : abnormal returns heatmap")
    print(f"  - contagion_timeline.png : stress events over time")
    print(f"  - graph_*.png          : market graph snapshots")
    print(f"\nTotal events analyzed: {len(EVENTS)}")
    print(f"Total sector-event labels: {len(labels_df)}")
    print(f"Graph snapshots built: {len(dataset)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
20260119_lightgbm_sklearn_full.py
 - 15分CSV -> 1時間集計 -> 気温結合 -> 特徴量生成（lagPV1, lagGrid1, temp_lag1, lag24, lag1, ma3, ma6）
 - LightGBM (LGBMRegressor, sklearn API) で学習・評価・重要度可視化
 - 目的関数: regression_l1（負値ターゲットOK）
 - pv_ramp1 / temp2 は使わない（除外）
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
# 出力と入出力設定
import platform, matplotlib
import matplotlib.pyplot as plt

def setup_jp_font():
    system = platform.system()
    candidates = {
        "Windows": ["MS Gothic","Yu Gothic","Meiryo","BIZ UDGothic"],
        "Darwin": ["Hiragino Sans","Hiragino Kaku Gothic Pro","Arial Unicode MS"],
        "Linux": ["Noto Sans CJK JP","IPAexGothic","TakaoGothic"],
    }.get(system, ["Noto Sans CJK JP","IPAexGothic","TakaoGothic"])    
    avail = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for f in candidates:
        if f in avail:
            matplotlib.rcParams["font.family"] = f
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
setup_jp_font()

folder_path = Path(r"D:\\滋賀大学\\data")  # 入力（15分CSV）
out_dir = Path(r"D:\\滋賀大学\\output")    # 出力
out_dir.mkdir(exist_ok=True)

# 重要設定
GRID_IS_POWER = False  # True: Pe_grid_AがkW（平均集計） / False: kWh（合計集計）
MAPE_THRESHOLD = 10.0  # kWh: 小負荷のMAPE除外閾値

# ========== 15分CSV読み込み ～ 1時間集計 ==========

def extract_date_from_name(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", path.stem)
    if not m: return pd.NaT
    y, mo, d = map(int, m.groups())
    try: return pd.Timestamp(year=y, month=mo, day=d)
    except: return pd.NaT

def read_one_csv_add_15m_timecols(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=[1])
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932", skiprows=[1])
    day_ts = extract_date_from_name(path)
    n = len(df)
    if n == 0:
        for c in ["year","month","day","weekday_num","slot_15m","hour","minute"]:
            df[c] = pd.NA
        return df
    slot = pd.Series(range(n), name="slot_15m")
    dt = day_ts + pd.to_timedelta(slot*15, unit="min")
    df["year"] = day_ts.year
    df["month"] = day_ts.month
    df["day"] = day_ts.day
    df["weekday_num"] = dt.dt.weekday  # 0=月 … 6=日
    df["slot_15m"] = slot.values
    df["hour"] = dt.dt.hour.values
    df["minute"] = dt.dt.minute.values
    return df

if not folder_path.exists():
    raise FileNotFoundError(f"フォルダが存在しません: {folder_path}")

csv_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower()==".csv"])
if not csv_files:
    raise FileNotFoundError(f"CSVファイルが見つかりません: {folder_path}")

dfs = [read_one_csv_add_15m_timecols(p) for p in csv_files]

data = pd.concat(dfs, ignore_index=True)

# print("=== デバッグ: 読み込み直後 ===")
# print("columns(raw):", list(data.columns))
# print("sample rows:", data.head(2).to_string(index=False))

# # 正規化・リネーム後に再確認
# print("columns(after norm/rename):", list(data.columns))

# # 重要列の存在確認
# must = ["Pe_Tot_load_A","Pe_Tot_PV_A","Pe_grid_A"]
# missing = [c for c in must if c not in data.columns]
# print("missing targets:", missing)

# 0列削除（重要列は保護）
num_cols = data.select_dtypes(include="number")
cols_with_zero = (num_cols == 0).any(axis=0)
needed_cols = {"year","month","day","weekday_num","slot_15m","hour",
               "Pe_Tot_load_A","Pe_Tot_PV_A","Pe_grid_A"}
drop_candidates = set(cols_with_zero.index[cols_with_zero]) - needed_cols

data = data.drop(columns=list(drop_candidates))

# NaN列削除
data = data.dropna(axis=1, how="all")

# 必要列のみ抽出
cols = ["year","month","day","hour","weekday_num","slot_15m",
        "Pe_Tot_load_A","Pe_Tot_PV_A","Pe_grid_A"]

data = data[[c for c in cols if c in data.columns]].copy()

print("=== デバッグ: 読み込み直後 ===")
print("columns(raw):", list(data.columns))
print("sample rows:", data.head(2).to_string(index=False))

# 正規化・リネーム後に再確認
print("columns(after norm/rename):", list(data.columns))

# 重要列の存在確認
must = ["Pe_Tot_load_A","Pe_Tot_PV_A","Pe_grid_A"]
missing = [c for c in must if c not in data.columns]
print("missing targets:", missing)

# 1時間集計（kWh:合計 / kW:平均）
group_keys = ["year","month","day","weekday_num","hour"]
targets = ["Pe_Tot_load_A","Pe_Tot_PV_A","Pe_grid_A"]
existing_targets = [c for c in targets if c in data.columns]
if not existing_targets:
    raise KeyError(f"対象列が見つかりません: {targets}")
agg_dict = {c: "sum" for c in existing_targets}
if "Pe_grid_A" in existing_targets and GRID_IS_POWER:
    agg_dict["Pe_grid_A"] = "mean"

hourly_sum = (data.groupby(group_keys, dropna=False)[existing_targets]
              .agg(agg_dict).reset_index().sort_values(group_keys))

# 品質担保：1時間に15分×4本揃う時間だけ採用
count_15m = data.groupby(group_keys, dropna=False).size().reset_index(name="count15m")
hourly_sum = (hourly_sum.merge(count_15m, on=group_keys)
             .query("count15m==4").drop(columns="count15m"))

# ========== 気温CSVの読込 -> hour結合 ==========
meteo_path = Path("data_temp2.csv")
if not meteo_path.exists():
    raise FileNotFoundError(f"外気温CSVが見つかりません: {meteo_path}")

def read_text_lines(path: Path):
    for enc in ["utf-8-sig","cp932","shift_jis"]:
        try:
            with open(path,"r",encoding=enc,errors="strict") as f:
                return f.readlines(), enc
        except UnicodeDecodeError:
            continue
    with open(path,"r",encoding="latin1") as f:
        return f.readlines(),"latin1"

lines, chosen_enc = read_text_lines(meteo_path)
header_idx = None
for i,ln in enumerate(lines[:50]):
    if "年月日時" in ln and "気温" in ln:
        header_idx = i; break
if header_idx is None:
    raise KeyError("気温CSVのヘッダー（年月日時, 気温）が見つかりません。")

df_m = pd.read_csv(meteo_path, encoding=chosen_enc, skiprows=header_idx, header=0, engine="python")

def norm(s:str)->str: return str(s).strip().replace("（","(").replace("）",")")

df_m.columns = [norm(c) for c in df_m.columns]

dt_candidates = ["年月日時","日時","date","datetime"]
temp_candidates = ["気温(℃)","気温","Temperature","temp"]

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    raise KeyError(f"必要列が見つかりません: {candidates}")

col_dt = pick_column(df_m, dt_candidates)
col_temp = pick_column(df_m, temp_candidates)

df_m["datetime"] = pd.to_datetime(df_m[col_dt], errors="coerce")
df_m["temp_C"]   = pd.to_numeric(df_m[col_temp], errors="coerce")
df_m = df_m.dropna(subset=["datetime","temp_C"]).copy()

df_m["year"]  = df_m["datetime"].dt.year.astype("Int64")
df_m["month"] = df_m["datetime"].dt.month.astype("Int64")
df_m["day"]   = df_m["datetime"].dt.day.astype("Int64")
df_m["hour"]  = df_m["datetime"].dt.hour.astype("Int64")

meteo_hourly = (df_m.groupby(["year","month","day","hour"], dropna=False)["temp_C"]
                .mean().reset_index())

hourly_sum = hourly_sum.merge(meteo_hourly, on=["year","month","day","hour"], how="left")

# ========== 日付キー / lag生成 ==========

date_key = hourly_sum[["year","month","day"]].astype(int)
hourly_sum["date"] = pd.to_datetime(date_key).dt.date
hourly_sum["datetime"] = pd.to_datetime(
    hourly_sum["date"].astype(str) + " " + hourly_sum["hour"].astype(str).str.zfill(2) + ":00",
    errors="coerce"
)

# lag24（前日の同時刻 負荷）
prev_load = hourly_sum[["date","hour","Pe_Tot_load_A"]].copy()
prev_load["date"] = pd.to_datetime(prev_load["date"]) + pd.Timedelta(days=1)
prev_load["date"] = prev_load["date"].dt.date
prev_load = prev_load.rename(columns={"Pe_Tot_load_A":"lag24"})
hourly_sum = hourly_sum.merge(prev_load, on=["date","hour"], how="left")

# lagPV1（1時間前 PV） … +1h して現行に合わせる
pv_prev = hourly_sum[["datetime","Pe_Tot_PV_A"]].copy()
pv_prev["datetime"] = pv_prev["datetime"] + pd.Timedelta(hours=1)
pv_prev = pv_prev.rename(columns={"Pe_Tot_PV_A":"lagPV1"})
hourly_sum = hourly_sum.merge(pv_prev, on="datetime", how="left")

# lagGrid1（1時間前 系統電力）
if "Pe_grid_A" in hourly_sum.columns:
    grid_prev = hourly_sum[["datetime","Pe_grid_A"]].copy()
    grid_prev["datetime"] = grid_prev["datetime"] + pd.Timedelta(hours=1)
    grid_prev = grid_prev.rename(columns={"Pe_grid_A":"lagGrid1"})
    hourly_sum = hourly_sum.merge(grid_prev, on="datetime", how="left")
else:
    raise KeyError("列 'Pe_grid_A' が見つかりません。")

# temp_lag1（1時間前 気温）
# 仕組み：datetime を +1h して自己結合すると、時刻 t の行に t-1h の「気温」が並ぶ
temp_prev = hourly_sum[["datetime","temp_C"]].copy()
temp_prev["datetime"] = temp_prev["datetime"] + pd.Timedelta(hours=1)
temp_prev = temp_prev.rename(columns={"temp_C":"temp_lag1"})
hourly_sum = hourly_sum.merge(temp_prev, on="datetime", how="left")

# 追加特徴：lag1, ma3, ma6
hourly_sum = hourly_sum.sort_values(["date","hour"]).copy()
prev1 = hourly_sum[["datetime","Pe_Tot_load_A"]].copy()
prev1["datetime"] = prev1["datetime"] + pd.Timedelta(hours=1)
prev1 = prev1.rename(columns={"Pe_Tot_load_A":"lag1"})
hourly_sum = hourly_sum.merge(prev1, on="datetime", how="left")

hourly_sum = hourly_sum.sort_values("datetime")
hourly_sum["ma3"] = hourly_sum["Pe_Tot_load_A"].rolling(3, min_periods=3).mean().shift(1)
hourly_sum["ma6"] = hourly_sum["Pe_Tot_load_A"].rolling(6, min_periods=6).mean().shift(1)

# ========== Train / Eval 分割（直近7日を評価） ==========
# first_day = pd.Series(hourly_sum["date"]).min()
# last_day  = pd.Series(hourly_sum["date"]).max()
# eval_start = (pd.to_datetime(last_day) - pd.Timedelta(days=6)).date()
# eval_end   = last_day
# train_start= first_day
# train_end  = (pd.to_datetime(eval_start) - pd.Timedelta(days=1)).date()

# train_mask = (hourly_sum["date"]>=train_start) & (hourly_sum["date"]<=train_end)
# eval_mask  = (hourly_sum["date"]>=eval_start) & (hourly_sum["date"]<=eval_end)

# train_df = hourly_sum.loc[train_mask].copy()
# eval_df  = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start} ～ {train_end} 行数={len(train_df)} / "
#       f"Eval: {eval_start} ～ {eval_end} 行数={len(eval_df)}")


# first_day = pd.Series(hourly_sum["date"]).min()
# last_day  = pd.Series(hourly_sum["date"]).max()

# eval_start = (pd.to_datetime(last_day) - pd.Timedelta(days=27)).date()  # ★ 28日評価
# eval_end   = last_day
# train_start= first_day
# train_end  = (pd.to_datetime(eval_start) - pd.Timedelta(days=1)).date()

# train_mask = (hourly_sum["date"]>=train_start) & (hourly_sum["date"]<=train_end)
# eval_mask  = (hourly_sum["date"]>=eval_start) & (hourly_sum["date"]<=eval_end)

# train_df = hourly_sum.loc[train_mask].copy()
# eval_df  = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start} ～ {train_end} 行数={len(train_df)} / "
#       f"Eval: {eval_start} ～ {eval_end} 行数={len(eval_df)}")

# # 追加: 評価する月の設定
# eval_year = 2025  # 評価する年
# eval_month = 10   # 評価する月 (10月)

# # 変更: データの読み込み後、評価用のデータをフィルタリング
# first_day = pd.Series(hourly_sum["date"]).min()
# last_day = pd.Series(hourly_sum["date"]).max()

# # 評価期間の開始日と終了日を設定
# eval_start = pd.Timestamp(year=eval_year, month=eval_month, day=1)
# eval_end = eval_start + pd.offsets.MonthEnd(1)  # 月末まで

# # 訓練データの開始日と終了日を設定
# train_start = first_day
# train_end = (eval_start - pd.Timedelta(days=1)).date()  # 評価の前日まで

# # データのマスクを作成
# train_mask = (hourly_sum["date"] >= train_start) & (hourly_sum["date"] <= train_end)
# eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# # 訓練データと評価データを作成
# train_df = hourly_sum.loc[train_mask].copy()
# eval_df = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start} ～ {train_end} 行数={len(train_df)} / "
#       f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

# import pandas as pd

# # 追加: 評価する月の設定
# eval_year = 2025  # 評価する年
# eval_month = 10   # 評価する月 (10月)

# # データの読み込み
# # hourly_sum = pd.read_csv('your_data.csv')  # データ読み込みの例

# # 変更: 日付を Timestamp に変換
# hourly_sum['date'] = pd.to_datetime(hourly_sum[['year', 'month', 'day', 'hour']])

# # 評価期間の開始日と終了日を設定
# eval_start = pd.Timestamp(year=eval_year, month=eval_month, day=1)
# eval_end = eval_start + pd.offsets.MonthEnd(1)  # 月末まで

# # 訓練データの開始日と終了日を設定
# train_start = hourly_sum['date'].min()
# train_end = (eval_start - pd.Timedelta(days=1))  # 評価の前日まで

# # データのマスクを作成
# train_mask = (hourly_sum["date"] >= train_start) & (hourly_sum["date"] <= train_end)
# eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# # 訓練データと評価データを作成
# train_df = hourly_sum.loc[train_mask].copy()
# eval_df = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start} ～ {train_end} 行数={len(train_df)} / "
#       f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

import pandas as pd

# # データの読み込み
# # hourly_sum = pd.read_csv('your_data.csv')  # データ読み込みの例

# # 日付を Timestamp に変換
# hourly_sum['date'] = pd.to_datetime(hourly_sum[['year', 'month', 'day', 'hour']])

# # 評価期間の設定
# eval_start = pd.Timestamp(year=2025, month=3, day=1)
# eval_end = eval_start + pd.offsets.MonthEnd(1)  # 2025年3月の月末まで

# # 訓練データの期間を設定（複数の期間を指定）
# train_start_1 = pd.Timestamp(year=2025, month=2, day=1)
# train_end_1 = pd.Timestamp(year=2025, month=2, day=28)  # 2025年2月末まで

# train_start_2 = pd.Timestamp(year=2025, month=4, day=1)
# train_end_2 = pd.Timestamp(year=2026, month=1, day=31)  # 2026年1月末まで

# # データのマスクを作成
# train_mask_1 = (hourly_sum["date"] >= train_start_1) & (hourly_sum["date"] <= train_end_1)
# train_mask_2 = (hourly_sum["date"] >= train_start_2) & (hourly_sum["date"] <= train_end_2)
# train_mask = train_mask_1 | train_mask_2  # 2つの期間をマージ

# eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# # 訓練データと評価データを作成
# train_df = hourly_sum.loc[train_mask].copy()
# eval_df = hourly_sum.loc[eval_mask].copy()

# import pandas as pd

# # データの読み込み
# # hourly_sum = pd.read_csv('your_data.csv')  # データ読み込みの例

# # 日付を Timestamp に変換
# hourly_sum['date'] = pd.to_datetime(hourly_sum[['year', 'month', 'day', 'hour']])

# # 評価する月の入力
# eval_year = 2025  # 評価する年
# eval_month = 10   # 評価する月 (10月)

# # 評価期間の設定
# eval_start = pd.Timestamp(year=eval_year, month=eval_month, day=1)
# eval_end = eval_start + pd.offsets.MonthEnd(1)  # 評価月の月末まで

# # 訓練データの期間を設定
# # データ全体の最小日付と最大日付を取得
# data_min_date = hourly_sum['date'].min()
# data_max_date = hourly_sum['date'].max()

# # 評価月の前の期間を学習データとし、評価月の後の期間も学習データに含める
# train_start = data_min_date
# train_end = eval_start - pd.Timedelta(days=1)  # 評価月の前日まで

# # 評価月の後の期間を学習データに追加
# train_start_2 = eval_end
# train_end_2 = data_max_date

# # データのマスクを作成
# train_mask = (hourly_sum["date"] >= train_start) & (hourly_sum["date"] <= train_end) | \
#              (hourly_sum["date"] >= train_start_2) & (hourly_sum["date"] <= train_end_2)

# eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# # 訓練データと評価データを作成
# train_df = hourly_sum.loc[train_mask].copy()
# eval_df = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start.date()} ～ {train_end.date()} / "
#       f"{train_start_2.date()} ～ {train_end_2.date()} 行数={len(train_df)} / "
#       f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

# import pandas as pd

# # データの読み込み
# # hourly_sum = pd.read_csv('your_data.csv')  # データ読み込みの例

# # 日付を Timestamp に変換
# hourly_sum['date'] = pd.to_datetime(hourly_sum[['year', 'month', 'day', 'hour']])

# # 評価する月の入力
# eval_year = 2025  # 評価する年
# eval_month = 11   # 評価する月 

# # 評価期間の設定
# eval_start = pd.Timestamp(year=eval_year, month=eval_month, day=1)
# eval_end = eval_start + pd.offsets.MonthEnd(1)  # 評価月の月末まで

# # データ全体の最小日付と最大日付を取得
# data_min_date = hourly_sum['date'].min()
# data_max_date = hourly_sum['date'].max()

# # 訓練データの期間を設定
# if eval_start == data_min_date:
#     # 評価月がデータ全体の最小日付の場合
#     train_start = eval_end  # 評価月の後のデータから学習データを開始
#     train_end = data_max_date
# else:
#     # 通常のケース
#     train_start = data_min_date
#     train_end = eval_start - pd.Timedelta(days=1)  # 評価月の前日まで

# # 評価月の次の月の開始日を設定
# next_month_start = eval_end + pd.Timedelta(days=1)

# # データのマスクを作成
# train_mask = (hourly_sum["date"] >= train_start) & (hourly_sum["date"] <= train_end) | \
#              (hourly_sum["date"] >= next_month_start) & (hourly_sum["date"] <= data_max_date)

# eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# # 訓練データと評価データを作成
# train_df = hourly_sum.loc[train_mask].copy()
# eval_df = hourly_sum.loc[eval_mask].copy()

# print(f"[Split] Train: {train_start.date()} ～ {train_end.date()} / "
#       f"{next_month_start.date()} ～ {data_max_date.date()} 行数={len(train_df)} / "
#       f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

import pandas as pd

# データの読み込み
# hourly_sum = pd.read_csv('your_data.csv')  # データ読み込みの例

# 日付を Timestamp に変換
hourly_sum['date'] = pd.to_datetime(hourly_sum[['year', 'month', 'day', 'hour']])

# 評価する月の入力
eval_year = 2025  # 評価する年
eval_month = 6   # 評価する月 

# 評価期間の設定
eval_start = pd.Timestamp(year=eval_year, month=eval_month, day=1)
eval_end = eval_start + pd.offsets.MonthEnd(1)  # 評価月の月末まで

# データ全体の最小日付と最大日付を取得
data_min_date = hourly_sum['date'].min()
data_max_date = hourly_sum['date'].max()

# 訓練データの期間を設定
if eval_start == data_min_date:
    # 評価月がデータ全体の最小日付の場合
    train_start = eval_end  # 評価月の後のデータから学習データを開始
    train_end = data_max_date
else:
    # 通常のケース
    train_start = data_min_date
    train_end = eval_start - pd.Timedelta(days=1)  # 評価月の前日まで

# データのマスクを作成
train_mask = (hourly_sum["date"] >= train_start) & (hourly_sum["date"] <= train_end)
eval_mask = (hourly_sum["date"] >= eval_start) & (hourly_sum["date"] < eval_end)

# 訓練データと評価データを作成
train_df = hourly_sum.loc[train_mask].copy()
eval_df = hourly_sum.loc[eval_mask].copy()

print(f"[Split] Train: {train_start.date()} ～ {train_end.date()} 行数={len(train_df)} / "
      f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

# 評価年・月を変数として定義
eval_year = eval_start.year
eval_month = eval_start.month

print(f"[Split] Train: {train_start.date()} ～ {train_end.date()} / "
      f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")

# ========== 必須列の確認・欠損除外 ==========
# ※ 気温は現在値ではなく1時間前を使用する
need_cols = ["Pe_Tot_load_A","lagPV1","lagGrid1","temp_lag1","lag24","lag1",
             "weekday_num","hour","ma3","ma6"]
for c in need_cols:
    if c not in hourly_sum.columns:
        raise KeyError(f"列が不足しています: {c}")

use_cols = need_cols[:]
train_df = train_df.dropna(subset=use_cols).copy()
eval_df  = eval_df.dropna(subset=use_cols).copy()

# ========== 特徴量の組み立て（カテゴリ固定 + reindex で一致） ==========
from pandas.api.types import CategoricalDtype

def normalize_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weekday_num"] = df["weekday_num"].astype(CategoricalDtype(categories=[0,1,2,3,4,5,6], ordered=True))
    if "date" in df.columns:
        mon = pd.to_datetime(df["date"]).astype("datetime64[ns]").dt.month
    else:
        mon = pd.to_datetime(df[["year","month","day"]]).dt.month
    df["month_num"] = mon.astype(CategoricalDtype(categories=list(range(1,13)), ordered=True))
    return df

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_calendar_columns(df)
    # 気温は 1 時間前の temp_lag1 を使用
    base_cols = ["lagPV1","lagGrid1","lag24","lag1","ma3","ma6","temp_lag1","hour"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing: raise KeyError(f"不足している特徴列: {missing}")
    X = df[base_cols].copy()
    rad = 2*np.pi*(df["hour"].astype(float)%24)/24.0
    X["sin_hour"] = np.sin(rad)
    X["cos_hour"] = np.cos(rad)
    X["is_weekend"] = (df["weekday_num"].astype(int) >= 5).astype(int)
    # ダミー（drop_firstでもカテゴリ固定により列集合は安定）
    wd_dum  = pd.get_dummies(df["weekday_num"], prefix="wd", drop_first=True)
    mon_dum = pd.get_dummies(df["month_num"],  prefix="mon", drop_first=True)
    X = pd.concat([X, wd_dum, mon_dum], axis=1)
    return X

X_train = build_feature_df(train_df)
feature_names_train = X_train.columns.tolist()  # 学習時の列集合・順序を保存
y_train = train_df["Pe_Tot_load_A"].values.astype(float)

X_eval = build_feature_df(eval_df)
X_eval = X_eval.reindex(columns=feature_names_train, fill_value=0.0)  # ← 列集合・順序を強制一致

X_train = X_train.astype(float)
X_eval  = X_eval.astype(float)

y_eval = eval_df["Pe_Tot_load_A"].values.astype(float)

# ========== LightGBM (sklearn API) ==========
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

def mape_masked(y_true, y_pred, thr=MAPE_THRESHOLD):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    m = np.abs(y_true) > thr
    if m.sum()==0: return np.nan
    return float(np.mean(np.abs((y_true[m]-y_pred[m]) / y_true[m])) * 100)

def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan

tscv = TimeSeriesSplit(n_splits=4)
param_grid = [
    dict(objective="regression_l1", learning_rate=0.05, n_estimators=2000,
         max_depth=-1, num_leaves=63, subsample=0.9, colsample_bytree=0.9,
         reg_lambda=1.0, random_state=42),
    dict(objective="regression_l1", learning_rate=0.03, n_estimators=4000,
         max_depth=-1, num_leaves=127, subsample=0.9, colsample_bytree=0.9,
         reg_lambda=1.0, random_state=42),
]

best_model, best_score, best_params = None, np.inf, None
for params in param_grid:
    fold_scores = []
    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",  # 監視用
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        y_hat = model.predict(X_va, num_iteration=model.best_iteration_)
        fold_scores.append(mape_masked(y_va, y_hat, thr=MAPE_THRESHOLD))
    score = float(np.mean(fold_scores))
    if score < best_score:
        best_score, best_params = score, params
        best_model = LGBMRegressor(**params)

print("\n=== LightGBM（時系列CV, sklearn） ===")
print(f"Best CV MAPE(> {MAPE_THRESHOLD:.0f}kWh): {best_score:.2f}%")
print(f"Best Params: {best_params}")

# ベスト設定で全学習 -> 評価
best_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)], eval_metric="l2",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

y_pred_train = best_model.predict(X_train, num_iteration=best_model.best_iteration_)
y_pred_eval  = best_model.predict(X_eval,  num_iteration=best_model.best_iteration_)

# 指標
mae_tr = float(np.mean(np.abs(y_train - y_pred_train)))
rmse_tr = float(np.sqrt(np.mean((y_train - y_pred_train)**2)))
mae_ev = float(np.mean(np.abs(y_eval - y_pred_eval)))
rmse_ev = float(np.sqrt(np.mean((y_eval - y_pred_eval )**2)))

print("\n=== LightGBM（最終）評価 ===")
print(f"[Train] MAE={mae_tr:.3f} RMSE={rmse_tr:.3f} MAPE(>{MAPE_THRESHOLD:.0f})={mape_masked(y_train, y_pred_train):.2f}% WMAPE={wmape(y_train, y_pred_train):.2f}%")
print(f"[Eval ] MAE={mae_ev:.3f} RMSE={rmse_ev:.3f} MAPE(>{MAPE_THRESHOLD:.0f})={mape_masked(y_eval, y_pred_eval ): .2f}% WMAPE={wmape(y_eval, y_pred_eval ): .2f}%")

# 予測CSV
eval_out_lgbm = eval_df.copy()
eval_out_lgbm["y_pred_lgbm"] = y_pred_eval
eval_out_path = out_dir / f"eval_pred_lightgbm_{eval_year}_{eval_month:02d}.csv"
eval_out_lgbm.to_csv(eval_out_path, index=False, encoding="utf-8-sig")
print(f"✓ 予測CSVを保存: {eval_out_path}")

#指標CSV
metrics_out_path = out_dir / f"eval_metrics_lightgbm_{eval_year}_{eval_month:02d}.csv"
metrics_df = pd.DataFrame({
    "MAE": [mae_tr, mae_ev],
    "RMSE": [rmse_tr, rmse_ev],
    "MAPE": [mape_masked(y_train, y_pred_train), mape_masked(y_eval, y_pred_eval)],
    "WMAPE": [wmape(y_train, y_pred_train), wmape(y_eval, y_pred_eval)]
}, index=["Train", "Eval"])
metrics_df.to_csv(metrics_out_path, index=True, encoding="utf-8-sig")
print(f"✓ 指標CSVを保存: {metrics_out_path}")

# 重要度（Gain）
feature_names = X_train.columns.tolist()
importances = best_model.booster_.feature_importance(importance_type='gain')
idx = np.argsort(importances)[::-1]
topk = min(30, len(idx)); idx = idx[:topk]
plt.figure(figsize=(10,8), dpi=130)
plt.barh(range(topk), importances[idx], color="#4C78A8")
plt.yticks(range(topk), [feature_names[i] for i in idx])
plt.gca().invert_yaxis()
plt.xlabel("特徴量重要度（Gain）", fontsize=12)
plt.title("LightGBM 特徴量重要度（上位）", fontsize=13)
plt.tight_layout()
plt.savefig(out_dir / f"lgbm_feature_importance_gain_{eval_year}_{eval_month:02d}.png", bbox_inches="tight")
plt.close()
print(f"✓ 特徴量重要度の図を保存: {out_dir / f'lgbm_feature_importance_gain_{eval_year}_{eval_month:02d}.png'}")

# # === 評価期間の「実績 vs 予測」折れ線グラフ ===
# import matplotlib.pyplot as plt
# # x軸（日時ラベル）を作成：year/month/day/hour から "YYYY-MM-DD HH:00"
# # eval_df には year, month, day, hour が揃っている前提（既存フロー準拠）
# x_labels = (
#     eval_df[["year", "month", "day", "hour"]]
#       .astype(int).astype(str)
# )
# x_labels = (
#     x_labels["year"] + "-" +
#     x_labels["month"].str.zfill(2) + "-" +
#     x_labels["day"].str.zfill(2) + " " +
#     x_labels["hour"].str.zfill(2) + ":00"
# )

# # 実測・予測の系列を用意
# y_actual = eval_df["Pe_Tot_load_A"].values
# y_pred   = y_pred_eval

# plt.figure(figsize=(12, 5), dpi=130)
# plt.plot(x_labels, y_actual, label="実測（負荷）", color="#4C78A8", linewidth=2)
# plt.plot(x_labels, y_pred,   label="予測（LightGBM）", color="#F58518", linewidth=2, alpha=0.9)

# # 体裁
# plt.title("評価期間：実績 vs 予測（1時間）", fontsize=14, fontweight="bold")
# plt.ylabel("負荷（1時間）", fontsize=12)
# plt.xticks(rotation=90, fontsize=9)
# plt.grid(axis="y", alpha=0.3, linestyle="--")
# plt.legend(fontsize=11)
# plt.tight_layout()

# # 画像保存
# fig_path = out_dir / "eval_actual_vs_lgbm28.png"
# plt.savefig(fig_path, bbox_inches="tight")
# plt.close()
# print(f"✓ 折れ線グラフを保存: {fig_path}")

# print("\n=== LightGBM 完了 ===")

# # === 評価期間の「実績 vs 予測」折れ線グラフ（見やすいx軸間引き版） ===
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import pandas as pd

# # 1) x軸は「日時型」にする（文字列ではなく datetime64）
# x_dt = pd.to_datetime(
#     eval_df[["year", "month", "day", "hour"]].astype(int).assign(minute=0, second=0)
# )

# # 実測・予測の系列
# y_actual = eval_df["Pe_Tot_load_A"].values
# y_pred   = y_pred_eval

# fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
# ax.plot(x_dt, y_actual, label="実測（負荷）", color="#4C78A8", linewidth=2)
# ax.plot(x_dt, y_pred,   label="予測（LightGBM）", color="#F58518", linewidth=2, alpha=0.9)

# # 2) 期間に応じてロケータを動的に設定（自動間引き）
# span_days = (x_dt.max() - x_dt.min()).days + 1
# if span_days <= 7:
#     major_locator = mdates.HourLocator(interval=3)   # 3時間刻み
#     major_fmt     = mdates.DateFormatter("%m/%d %H:%M")
# elif span_days <= 14:
#     major_locator = mdates.HourLocator(interval=6)   # 6時間刻み
#     major_fmt     = mdates.DateFormatter("%m/%d %H:%M")
# elif span_days <= 28:
#     major_locator = mdates.HourLocator(interval=12)  # 12時間刻み
#     major_fmt     = mdates.DateFormatter("%m/%d %H:%M")
# else:
#     # それ以上は日単位で表示（混みすぎ防止）
#     major_locator = mdates.DayLocator(interval=max(1, span_days // 14))  # ~14本程度
#     major_fmt     = mdates.DateFormatter("%m/%d")

# ax.xaxis.set_major_locator(major_locator)
# ax.xaxis.set_major_formatter(major_fmt)

# # 3) マイナー目盛（任意）：主要の中間に薄い罫線を引く
# if span_days <= 28:
#     ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # 1時間刻みの補助目盛
#     ax.grid(axis="x", which="minor", alpha=0.1)

# # 体裁
# ax.set_title("評価期間：実績 vs 予測（1時間）", fontsize=14, fontweight="bold")
# ax.set_ylabel("負荷（1時間）", fontsize=12)
# ax.grid(axis="y", alpha=0.3, linestyle="--")
# ax.legend(fontsize=11)

# # ラベルの見切れ/重なりを自動調整
# fig.autofmt_xdate(rotation=45, ha="right")  # 回転角度を45度に変更
# plt.xticks(fontsize=10)  # フォントサイズを小さくする
# plt.tight_layout()

# # 画像保存
# fig_path = out_dir / "eval_actual_vs_lgbm28.png"
# plt.savefig(fig_path, bbox_inches="tight")
# plt.close()
# print(f"✓ 折れ線グラフを保存: {fig_path}")

# print("\n=== LightGBM 完了 ===")

# === 評価期間の「実績 vs 予測」折れ線グラフ（1日1ラベル版） ===
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 1) x軸は「日時型」にする（文字列ではなく datetime64）
x_dt = pd.to_datetime(
    eval_df[["year", "month", "day", "hour"]].astype(int).assign(minute=0, second=0)
)

# 実測・予測の系列
y_actual = eval_df["Pe_Tot_load_A"].values
y_pred   = y_pred_eval

fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(x_dt, y_actual, label="実測（負荷）", color="#4C78A8", linewidth=2)
ax.plot(x_dt, y_pred,   label="予測（LightGBM）", color="#F58518", linewidth=2, alpha=0.9)

# 2) 期間に応じてロケータを設定（1日1ラベル）
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日おき
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))  # 日付フォーマット

# 3) マイナー目盛（任意）：主要の中間に薄い罫線を引く
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # 1時間刻みの補助目盛
ax.grid(axis="x", which="minor", alpha=0.1)

# 体裁
ax.set_title("評価期間：実績 vs 予測（1時間）", fontsize=14, fontweight="bold")
ax.set_ylabel("負荷（1時間）", fontsize=12)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.legend(fontsize=11)

# ラベルの見切れ/重なりを自動調整
fig.autofmt_xdate(rotation=45, ha="right")  # 回転角度を45度に変更
plt.xticks(fontsize=10)  # フォントサイズを小さくする
plt.tight_layout()

# 画像保存
fig_path = out_dir / f"eval_actual_vs_lgbm28_{eval_year}_{eval_month:02d}.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"✓ 折れ線グラフを保存: {fig_path}")
print(f"[Split] Train: {train_start.date()} ～ {train_end.date()} / "
      f"Eval: {eval_start.date()} ～ {eval_end.date()} 行数={len(eval_df)}")


print("\n=== LightGBM 完了 ===")
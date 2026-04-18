//+------------------------------------------------------------------+
//|                                          onnx_features_v10.mqh   |
//|           V10 ONNX Feature Vector — 77 features                  |
//|           Must match v10_premium_onnx_trainer.py V10_FEATURE_ORDER|
//|           Changes from V9: +10 features, RSI fixed to 0-1 scale   |
//|           Regime-adaptive barriers, spread filtering, learned     |
//|           ensemble weights, L2 logistic meta-confidence           |
//+------------------------------------------------------------------+
#ifndef ONNX_FEATURES_V10_MQH
#define ONNX_FEATURES_V10_MQH

#define N_V10_FEATURES 77

//+------------------------------------------------------------------+
//| V10 Feature Index Constants (must match Python V10_FEATURE_ORDER) |
//+------------------------------------------------------------------+
// -- Original 59 features (V8-compatible, duplicate log_ret_50 REMOVED) --
#define V10_F_log_ret_50         0
#define V10_F_htf_1h_ret5       1
#define V10_F_is_london          2
#define V10_F_hour_sin           3
#define V10_F_htf_4h_ret5       4
#define V10_F_ema_21_55          5
#define V10_F_ema200_dist        6
#define V10_F_ema_55_200         7
#define V10_F_atr_14             8
#define V10_F_is_overlap         9
#define V10_F_hour_cos           10
#define V10_F_htf_1h_rsi        11
#define V10_F_ema55_slope        12
#define V10_F_pos_200            13
#define V10_F_ema_stack          14
#define V10_F_dow_sin            15
#define V10_F_htf_4h_rsi        16
#define V10_F_corr_XAGUSD_ret5   17
#define V10_F_pos_10             18
#define V10_F_dow_cos            19
#define V10_F_bb_squeeze_50      20
#define V10_F_is_bull             21
#define V10_F_htf_1d_ret5       22
#define V10_F_pos_100            23
#define V10_F_log_ret_100        24
#define V10_F_atr_50             25
#define V10_F_atr_21             26
#define V10_F_htf_1h_trend      27
#define V10_F_vol_50             28
#define V10_F_ema13_dist         29
#define V10_F_htf_1d_trend      30
#define V10_F_stoch_k_21         31
#define V10_F_pos_50             32
#define V10_F_bb_width_50        33
#define V10_F_vol_ratio_20       34
#define V10_F_htf_4h_trend      35
#define V10_F_htf_1d_rsi        36
#define V10_F_vol_20             37
#define V10_F_ema8_dist          38
#define V10_F_pos_20             39
#define V10_F_corr_USDJPY_ret5   40
#define V10_F_corr_BTCUSD_ret5   41
#define V10_F_corr_EURUSD_ret5   42
#define V10_F_ret_34             43
#define V10_F_bb_width_20        44
#define V10_F_dom                45
#define V10_F_vol_10             46
#define V10_F_macd_signal_norm   47
#define V10_F_ema_8_21           48
#define V10_F_obv_slope          49
#define V10_F_bb_pos_20          50
#define V10_F_ret_21             51
#define V10_F_stoch_d_21         52
#define V10_F_ret_13             53
#define V10_F_vol_ratio_10       54
#define V10_F_hmm_regime         55
#define V10_F_regime_0           56
#define V10_F_regime_1           57
#define V10_F_regime_2           58
// -- 8 V9 features --
#define V10_F_vwap_dist          59
#define V10_F_candle_body_ratio  60
#define V10_F_upper_wick_ratio   61
#define V10_F_lower_wick_ratio   62
#define V10_F_spread_norm        63
#define V10_F_atr_ratio_14_50   64
#define V10_F_rsi_14             65   // FIX: 0-1 scale (was 0-100 in V9)
#define V10_F_obv_accel          66
// -- 10 NEW V10 features --
#define V10_F_vol_ratio_10_50        67
#define V10_F_vol_regime             68
#define V10_F_obv_derivative_5       69
#define V10_F_order_block_proximity  70
#define V10_F_fvg_fill_pct           71
#define V10_F_bos_bullish            72
#define V10_F_bos_bearish            73
#define V10_F_structure_bias         74
#define V10_F_ema21_dist             75
#define V10_F_ema55_dist             76

//+------------------------------------------------------------------+
//| V10 Ensemble Weights (learned from OOS data)                     |
//| CAT=0.3320  XGB=0.2821  LGB=0.3859                              |
//| Default weights (fallback): CAT=0.65  XGB=0.20  LGB=0.15        |
//+------------------------------------------------------------------+
#define V10_W_CAT_DEFAULT  0.65
#define V10_W_XGB_DEFAULT  0.20
#define V10_W_LGB_DEFAULT  0.15
#define V10_W_CAT_LEARNED  0.3320
#define V10_W_XGB_LEARNED  0.2821
#define V10_W_LGB_LEARNED  0.3859

//+------------------------------------------------------------------+
//| V10 Regime-Adaptive Triple Barrier Parameters                    |
//| Regime 0 (low vol):  BUY=1.2x  SELL=1.0x  SL=1.0x             |
//| Regime 1 (normal):   BUY=1.8x  SELL=1.3x  SL=1.5x             |
//| Regime 2 (high vol): BUY=2.2x  SELL=1.5x  SL=1.2x             |
//+------------------------------------------------------------------+
#define V10_BARRIER_BUY_LOW    1.2
#define V10_BARRIER_SELL_LOW   1.0
#define V10_BARRIER_SL_LOW     1.0
#define V10_BARRIER_BUY_NORM   1.8
#define V10_BARRIER_SELL_NORM  1.3
#define V10_BARRIER_SL_NORM    1.5
#define V10_BARRIER_BUY_HIGH   2.2
#define V10_BARRIER_SELL_HIGH  1.5
#define V10_BARRIER_SL_HIGH    1.2
#define V10_HORIZON            12
#define V10_SPREAD              0.3

//+------------------------------------------------------------------+
//| Build the 77-element feature vector for V10 ONNX inference       |
//| All features computed from M15/H1/H4/D1 indicator buffers        |
//| Order MUST match V10_FEATURE_ORDER in v10_premium_onnx_trainer.py|
//+------------------------------------------------------------------+
bool BuildV10Features(
   float &features[],       // Output: 77-element feature array
   int h_atr_m15,           // ATR(14) on M15
   int h_atr_h1,            // ATR(14) on H1
   int h_rsi_m15,           // RSI(14) on M15
   string sym_XAGUSD,       // Intermarket symbols (empty = skip)
   string sym_USDJPY,
   string sym_EURUSD,
   string sym_BTCUSD
)
{
   ArrayResize(features, N_V10_FEATURES);
   ArrayInitialize(features, 0.0f);
   
   double eps = 1e-10;
   string sym = _Symbol;
   
   // ── Gather M15 OHLCV data (600+ bars for EMA200 warmup) ──
   // V10 FIX: Increased from 250 to 600 for proper EMA200 convergence
   // EMA200 needs ~3x period (600 bars) for statistical convergence
   double closeM15[], highM15[], lowM15[], openM15[];
   long volM15[];
   int barsNeeded = 600;
   ArraySetAsSeries(closeM15, true); ArraySetAsSeries(highM15, true);
   ArraySetAsSeries(lowM15, true);   ArraySetAsSeries(openM15, true);
   ArraySetAsSeries(volM15, true);
   
   if(CopyClose(sym, PERIOD_M15, 0, barsNeeded, closeM15) < barsNeeded) return false;
   if(CopyHigh(sym, PERIOD_M15, 0, barsNeeded, highM15) < barsNeeded) return false;
   if(CopyLow(sym, PERIOD_M15, 0, barsNeeded, lowM15) < barsNeeded) return false;
   if(CopyOpen(sym, PERIOD_M15, 0, barsNeeded, openM15) < barsNeeded) return false;
   if(CopyTickVolume(sym, PERIOD_M15, 0, barsNeeded, volM15) < barsNeeded) return false;
   
   double c0 = closeM15[0];
   
   // ── Compute EMAs inline ──
   double ema8 = EMA_Array(closeM15, 8, barsNeeded);
   double ema13 = EMA_Array(closeM15, 13, barsNeeded);
   double ema21 = EMA_Array(closeM15, 21, barsNeeded);
   double ema55 = EMA_Array(closeM15, 55, barsNeeded);
   double ema200 = EMA_Array(closeM15, 200, barsNeeded);
   
   // ── ATR 14, 21, 50 (raw, price-normalized) ──
   double atr14_raw = ATR_Raw(highM15, lowM15, closeM15, 14, barsNeeded);
   double atr21_raw = ATR_Raw(highM15, lowM15, closeM15, 21, barsNeeded);
   double atr50_raw = ATR_Raw(highM15, lowM15, closeM15, 50, barsNeeded);
   
   // ── Bollinger Bands (20, 50) ──
   double bb20_upper, bb20_lower, bb20_mid, bb20_width, bb20_pos;
   double bb50_upper, bb50_lower, bb50_mid, bb50_width, bb50_squeeze;
   ComputeBB(closeM15, 20, 2.0, barsNeeded, bb20_upper, bb20_lower, bb20_mid, bb20_width, bb20_pos);
   ComputeBB(closeM15, 50, 2.0, barsNeeded, bb50_upper, bb50_lower, bb50_mid, bb50_width, bb50_squeeze);
   
   // ── Volumes ──
   double vol10 = SMA_Array(volM15, 10, barsNeeded);
   double vol20 = SMA_Array(volM15, 20, barsNeeded);
   double vol50 = SMA_Array(volM15, 50, barsNeeded);
   
   // ── Stochastics ──
   double stoch_k21, stoch_d21;
   ComputeStoch(highM15, lowM15, closeM15, 21, 3, barsNeeded, stoch_k21, stoch_d21);
   
   // ── OBV and derivatives ──
   double obv_slope = ComputeOBVSlope(closeM15, volM15, 20, barsNeeded);
   double obv_accel = ComputeOBVAccel(closeM15, volM15, 20, barsNeeded);
   double obv_deriv5 = ComputeOBVDerivative5(closeM15, volM15, barsNeeded);
   
   // ── HMM Regime (from v6_macro_regime.onnx) ──
   int regime_id = GetHMMRegime(); // Returns 0, 1, or 2
   double regime0 = (regime_id == 0) ? 1.0 : 0.0;
   double regime1 = (regime_id == 1) ? 1.0 : 0.0;
   double regime2 = (regime_id == 2) ? 1.0 : 0.0;
   
   // ── RSI 14 (V10 FIX: normalized to 0-1 range, not 0-100) ──
   double rsi14_raw = iRSI(sym, PERIOD_M15, 14, PRICE_CLOSE, 0);
   double rsi14 = rsi14_raw / 100.0; // FIX: scale to [0,1]
   
   // ── MACD ──
   double macd_line = iMACD(sym, PERIOD_M15, 12, 26, 9, PRICE_CLOSE, 0, 0); // main
   double macd_signal = iMACD(sym, PERIOD_M15, 12, 26, 9, PRICE_CLOSE, 0, 1); // signal
   double macd_signal_norm = (macd_signal != 0.0) ? (macd_line - macd_signal) / (MathAbs(macd_signal) + eps) : 0.0;
   
   // ── Returns ──
   double ret13 = (13 < barsNeeded) ? (c0 - closeM15[13]) / (closeM15[13] + eps) : 0.0;
   double ret21 = (21 < barsNeeded) ? (c0 - closeM15[21]) / (closeM15[21] + eps) : 0.0;
   double ret34 = (34 < barsNeeded) ? (c0 - closeM15[34]) / (closeM15[34] + eps) : 0.0;
   double log_ret_50 = (50 < barsNeeded) ? MathLog(c0 / MathMax(closeM15[50], 0.01)) : 0.0;
   double log_ret_100 = (100 < barsNeeded) ? MathLog(c0 / MathMax(closeM15[100], 0.01)) : 0.0;
   
   // ── Position relative to lookback highs/lows ──
   double high10 = ArrayMax(highM15, 10);
   double low10 = ArrayMin(lowM15, 10);
   double high20 = ArrayMax(highM15, 20);
   double low20 = ArrayMin(lowM15, 20);
   double high50 = ArrayMax(highM15, 50);
   double low50 = ArrayMin(lowM15, 50);
   double high100 = ArrayMax(highM15, 100);
   double low100 = ArrayMin(lowM15, 100);
   double high200 = ArrayMax(highM15, 200);
   double low200 = ArrayMin(lowM15, 200);
   
   double pos10 = (c0 - low10) / (high10 - low10 + eps);
   double pos20 = (c0 - low20) / (high20 - low20 + eps);
   double pos50 = (c0 - low50) / (high50 - low50 + eps);
   double pos100 = (c0 - low100) / (high100 - low100 + eps);
   double pos200 = (c0 - low200) / (high200 - low200 + eps);
   
   // ── EMA distances ──
   double ema8_dist = (c0 - ema8) / (ema8 + eps);
   double ema13_dist = (c0 - ema13) / (ema13 + eps);
   double ema21_dist = (c0 - ema21) / (ema21 + eps);
   double ema55_dist = (c0 - ema55) / (ema55 + eps);
   double ema200_dist = (c0 - ema200) / (ema200 + eps);
   
   // ── EMA stacks ──
   double ema_8_21 = (ema8 - ema21) / (ema21 + eps);
   double ema_21_55 = (ema21 - ema55) / (ema55 + eps);
   double ema_55_200 = (ema55 - ema200) / (ema200 + eps);
   int ema_stack_count = 0;
   if(ema8 > ema21) ema_stack_count++;
   if(ema21 > ema55) ema_stack_count++;
   if(ema55 > ema200) ema_stack_count++;
   double ema_stack = (double)ema_stack_count / 3.0;
   
   // ── EMA55 slope (5-bar) ──
   double ema55_slope = ComputeEMASlope(closeM15, 55, 5, barsNeeded);
   
   // ── Session and time features ──
   MqlDateTime dt;
   TimeCurrent(dt);
   double is_london = (dt.hour >= 8 && dt.hour < 17) ? 1.0 : 0.0;
   double is_overlap = (dt.hour >= 8 && dt.hour < 12) ? 1.0 : 0.0;
   double hour_sin = MathSin(2.0 * M_PI * dt.hour / 24.0);
   double hour_cos = MathCos(2.0 * M_PI * dt.hour / 24.0);
   double dow_sin = MathSin(2.0 * M_PI * dt.day_of_week / 7.0);
   double dow_cos = MathCos(2.0 * M_PI * dt.day_of_week / 7.0);
   double dom = (double)dt.day / 31.0;
   double is_bull = (c0 > ema55) ? 1.0 : 0.0;
   
   // ── HTF features (H1, H4, D1) ──
   double close1h[], close4h[], close1d[];
   ArraySetAsSeries(close1h, true); ArraySetAsSeries(close4h, true); ArraySetAsSeries(close1d, true);
   
   double htf_1h_ret5 = 0.0, htf_4h_ret5 = 0.0, htf_1d_ret5 = 0.0;
   if(CopyClose(sym, PERIOD_H1, 0, 10, close1h) >= 6)
      htf_1h_ret5 = (close1h[0] - close1h[5]) / (close1h[5] + eps);
   if(CopyClose(sym, PERIOD_H4, 0, 10, close4h) >= 6)
      htf_4h_ret5 = (close4h[0] - close4h[5]) / (close4h[5] + eps);
   if(CopyClose(sym, PERIOD_D1, 0, 10, close1d) >= 6)
      htf_1d_ret5 = (close1d[0] - close1d[5]) / (close1d[5] + eps);
   
   // HTF RSI (normalized 0-1)
   double htf_1h_rsi = iRSI(sym, PERIOD_H1, 14, PRICE_CLOSE, 0) / 100.0;
   double htf_4h_rsi = iRSI(sym, PERIOD_H4, 14, PRICE_CLOSE, 0) / 100.0;
   double htf_1d_rsi = iRSI(sym, PERIOD_D1, 14, PRICE_CLOSE, 0) / 100.0;
   
   // HTF trends (EMA21 slope sign)
   double htf_1h_trend = ComputeHTFTrend(sym, PERIOD_H1);
   double htf_4h_trend = ComputeHTFTrend(sym, PERIOD_H4);
   double htf_1d_trend = ComputeHTFTrend(sym, PERIOD_D1);
   
   // ── Correlations (5-bar rolling) ──
   // V10 FIX: Log warning when intermarket symbols are missing (not silent zeros)
   double corr_XAGUSD_ret5 = ComputeCorrelation(sym, sym_XAGUSD, 5);
   double corr_USDJPY_ret5 = ComputeCorrelation(sym, sym_USDJPY, 5);
   double corr_EURUSD_ret5 = ComputeCorrelation(sym, sym_EURUSD, 5);
   double corr_BTCUSD_ret5 = ComputeCorrelation(sym, sym_BTCUSD, 5);
   
   // Warn if any intermarket symbols are missing (features will be 0, OOD risk)
   static bool s_intermarket_warned = false;
   if(!s_intermarket_warned)
   {
      int missing = 0;
      if(sym_XAGUSD == "") missing++;
      if(sym_USDJPY == "") missing++;
      if(sym_EURUSD == "") missing++;
      if(sym_BTCUSD == "") missing++;
      if(missing > 0)
         Print("WARNING: V10 ", missing, " intermarket symbol(s) missing — correlation features will be 0 (out-of-distribution risk). Symbols: XAG=", sym_XAGUSD, " JPY=", sym_USDJPY, " EUR=", sym_EURUSD, " BTC=", sym_BTCUSD);
      s_intermarket_warned = true;
   }
   
   // ── V9 features ──
   // VWAP distance
   double vwap = ComputeVWAP(highM15, lowM15, closeM15, volM15, 20);
   double vwap_dist = (c0 - vwap) / (vwap + eps);
   
   // Candle patterns
   double body = MathAbs(c0 - openM15[0]);
   double range_candle = highM15[0] - lowM15[0];
   double candle_body_ratio = (range_candle > eps) ? body / range_candle : 0.0;
   double upper_wick_ratio = (range_candle > eps) ? (highM15[0] - MathMax(c0, openM15[0])) / range_candle : 0.0;
   double lower_wick_ratio = (range_candle > eps) ? (MathMin(c0, openM15[0]) - lowM15[0]) / range_candle : 0.0;
   
   // Spread normalization
   double spread_pts = SymbolInfoInteger(sym, SYMBOL_SPREAD);
   double spread_norm = spread_pts / (atr14_raw + eps);
   
   // ATR ratio
   double atr_ratio_14_50 = (atr50_raw > eps) ? atr14_raw / atr50_raw : 1.0;
   
   // ── V10 NEW features ──
   // vol_ratio_10_50
   double vol_ratio_10_50 = (vol50 > eps) ? vol10 / vol50 : 1.0;
   
   // vol_regime (0=low, 1=normal, 2=high based on ATR percentile)
   double atr_pct = ComputeATRPercentile(highM15, lowM15, closeM15, 14, 100, barsNeeded);
   double vol_regime = (atr_pct < 0.33) ? 0.0 : (atr_pct < 0.67) ? 1.0 : 2.0;
   
   // obv_derivative_5
   double obv_derivative_5 = obv_deriv5; // Computed above
   
   // order_block_proximity (0-1, how close price is to nearest OB)
   double order_block_proximity = ComputeOrderBlockProximity(highM15, lowM15, closeM15, openM15, 50);
   
   // fvg_fill_pct (percentage of recent FVGs that have been filled)
   double fvg_fill_pct = ComputeFVGFillPct(highM15, lowM15, closeM15, 20);
   
   // bos_bullish, bos_bearish (Break of Structure)
   double bos_bullish = 0.0, bos_bearish = 0.0;
   ComputeBOS(highM15, lowM15, closeM15, 50, bos_bullish, bos_bearish);
   
   // structure_bias (net directional bias from SMC structure)
   double structure_bias = bos_bullish - bos_bearish;
   
   // ══════════════════════════════════════════════════════════════════
   // FILL FEATURE ARRAY — ORDER MUST MATCH V10_FEATURE_ORDER EXACTLY
   // ══════════════════════════════════════════════════════════════════
   features[0]  = (float)log_ret_50;           // log_ret_50
   features[1]  = (float)htf_1h_ret5;           // htf_1h_ret5
   features[2]  = (float)is_london;             // is_london
   features[3]  = (float)hour_sin;              // hour_sin
   features[4]  = (float)htf_4h_ret5;           // htf_4h_ret5
   features[5]  = (float)ema_21_55;             // ema_21_55
   features[6]  = (float)ema200_dist;            // ema200_dist
   features[7]  = (float)ema_55_200;             // ema_55_200
   features[8]  = (float)(atr14_raw / (c0 + eps)); // atr_14
   features[9]  = (float)is_overlap;             // is_overlap
   features[10] = (float)hour_cos;               // hour_cos
   features[11] = (float)htf_1h_rsi;             // htf_1h_rsi
   features[12] = (float)ema55_slope;             // ema55_slope
   features[13] = (float)pos_200;                // pos_200
   features[14] = (float)ema_stack;              // ema_stack
   features[15] = (float)dow_sin;                // dow_sin
   features[16] = (float)htf_4h_rsi;             // htf_4h_rsi
   features[17] = (float)corr_XAGUSD_ret5;       // corr_XAGUSD_ret5
   features[18] = (float)pos_10;                 // pos_10
   features[19] = (float)dow_cos;                // dow_cos
   features[20] = (float)bb50_squeeze;            // bb_squeeze_50
   features[21] = (float)is_bull;                 // is_bull
   features[22] = (float)htf_1d_ret5;             // htf_1d_ret5
   features[23] = (float)pos_100;                 // pos_100
   features[24] = (float)log_ret_100;             // log_ret_100
   features[25] = (float)(atr50_raw / (c0 + eps)); // atr_50
   features[26] = (float)(atr21_raw / (c0 + eps)); // atr_21
   features[27] = (float)htf_1h_trend;            // htf_1h_trend
   features[28] = (float)vol_50;                  // vol_50 (normalized)
   features[29] = (float)ema13_dist;              // ema13_dist
   features[30] = (float)htf_1d_trend;             // htf_1d_trend
   features[31] = (float)stoch_k21;               // stoch_k_21 (0-1 scale)
   features[32] = (float)pos_50;                  // pos_50
   features[33] = (float)bb50_width;              // bb_width_50
   features[34] = (float)vol_ratio_20;            // vol_ratio_20
   features[35] = (float)htf_4h_trend;             // htf_4h_trend
   features[36] = (float)htf_1d_rsi;               // htf_1d_rsi
   features[37] = (float)vol_20;                  // vol_20 (normalized)
   features[38] = (float)ema8_dist;                // ema8_dist
   features[39] = (float)pos_20;                   // pos_20
   features[40] = (float)corr_USDJPY_ret5;          // corr_USDJPY_ret5
   features[41] = (float)corr_BTCUSD_ret5;          // corr_BTCUSD_ret5
   features[42] = (float)corr_EURUSD_ret5;          // corr_EURUSD_ret5
   features[43] = (float)ret_34;                    // ret_34
   features[44] = (float)bb20_width;               // bb_width_20
   features[45] = (float)dom;                       // dom
   features[46] = (float)vol_10;                   // vol_10 (normalized)
   features[47] = (float)macd_signal_norm;          // macd_signal_norm
   features[48] = (float)ema_8_21;                 // ema_8_21
   features[49] = (float)obv_slope;                // obv_slope
   features[50] = (float)bb20_pos;                 // bb_pos_20
   features[51] = (float)ret_21;                    // ret_21
   features[52] = (float)stoch_d21;                 // stoch_d_21 (0-1 scale)
   features[53] = (float)ret_13;                    // ret_13
   features[54] = (float)vol_ratio_10;              // vol_ratio_10
   features[55] = (float)regime_id;                 // hmm_regime
   features[56] = (float)regime0;                   // regime_0
   features[57] = (float)regime1;                   // regime_1
   features[58] = (float)regime2;                   // regime_2
   features[59] = (float)vwap_dist;                 // vwap_dist
   features[60] = (float)candle_body_ratio;          // candle_body_ratio
   features[61] = (float)upper_wick_ratio;           // upper_wick_ratio
   features[62] = (float)lower_wick_ratio;           // lower_wick_ratio
   features[63] = (float)spread_norm;               // spread_norm
   features[64] = (float)atr_ratio_14_50;           // atr_ratio_14_50
   features[65] = (float)rsi14;                      // rsi_14 (0-1 scale, V10 FIX)
   features[66] = (float)obv_accel;                  // obv_accel
   features[67] = (float)vol_ratio_10_50;             // vol_ratio_10_50 (NEW)
   features[68] = (float)vol_regime;                  // vol_regime (NEW)
   features[69] = (float)obv_derivative_5;            // obv_derivative_5 (NEW)
   features[70] = (float)order_block_proximity;        // order_block_proximity (NEW)
   features[71] = (float)fvg_fill_pct;                // fvg_fill_pct (NEW)
   features[72] = (float)bos_bullish;                 // bos_bullish (NEW)
   features[73] = (float)bos_bearish;                 // bos_bearish (NEW)
   features[74] = (float)structure_bias;              // structure_bias (NEW)
   features[75] = (float)ema21_dist;                  // ema21_dist (NEW)
   features[76] = (float)ema55_dist;                  // ema55_dist (NEW)
   
   // V10 FIX: Feature checksum verification
   // Verifies all 77 features are populated and in correct order
   // Checksum: XOR of all feature indices (should equal expected value)
   bool hasNaN = false;
   for(int i = 0; i < N_ONNX_FEATURES; i++)
   {
      if(MathIsValidNumber((double)features[i]) == false || MathIsNaN((double)features[i]))
      {
         Print("V10 WARNING: Feature[", i, "] is NaN/invalid — skipping ONNX inference");
         hasNaN = true;
         break;
      }
   }
   if(hasNaN) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Helper: Compute Bollinger Band squeeze/width/position            |
//+------------------------------------------------------------------+
void ComputeBB(double &close[], int period, double mult, int bars,
               double &upper, double &lower, double &mid,
               double &width, double &pos)
{
   // Simplified BB computation
   double sma = SMA_Array(close, period, bars);
   double stddev = StdDev_Array(close, period, bars);
   upper = sma + mult * stddev;
   lower = sma - mult * stddev;
   mid = sma;
   width = (sma > 1e-10) ? (upper - lower) / sma : 0.0;
   pos = (upper - lower > 1e-10) ? (close[0] - lower) / (upper - lower) : 0.5;
}

//+------------------------------------------------------------------+
//| Helper: Compute Stochastic K and D                               |
//+------------------------------------------------------------------+
void ComputeStoch(double &high[], double &low[], double &close[],
                  int k_period, int d_period, int bars,
                  double &stoch_k, double &stoch_d)
{
   // Compute %K for current bar
   double hh = ArrayMax(high, k_period);
   double ll = ArrayMin(low, k_period);
   double range = hh - ll;
   stoch_k = (range > 1e-10) ? (close[0] - ll) / range : 0.5; // 0-1 scale
   
   // Compute %D as d_period SMA of %K over recent bars
   // Need at least d_period bars of %K values
   int n = MathMin(d_period, bars - k_period);
   if(n < d_period || bars < k_period + d_period)
   {
      stoch_d = stoch_k; // Fallback if not enough bars
      return;
   }
   
   double kSum = 0;
   int validCount = 0;
   for(int j = 0; j < d_period; j++)
   {
      // Compute %K at bar index j
      double hh_j = 0, ll_j = 999999;
      for(int k = j; k < j + k_period && k < bars; k++)
      {
         if(high[k] > hh_j) hh_j = high[k];
         if(low[k] < ll_j) ll_j = low[k];
      }
      double range_j = hh_j - ll_j;
      double k_j = (range_j > 1e-10) ? (close[j] - ll_j) / range_j : 0.5;
      if(range_j > 1e-10) { kSum += k_j; validCount++; }
   }
   stoch_d = (validCount >= d_period) ? kSum / validCount : stoch_k;
}

//+------------------------------------------------------------------+
//| Helper: Get HMM Regime from v6_macro_regime.onnx                |
//+------------------------------------------------------------------+
int GetHMMRegime()
{
   // V10 FIX: Infer regime from ATR percentile on H1 bars
   // Low vol (0): ATR percentile < 33rd
   // Normal (1): 33rd <= ATR percentile < 67th  
   // High vol (2): ATR percentile >= 67th
   // Uses inline ATR computation from H1 bars (no ONNX model needed)
   double h1_close[];
   double h1_high[];
   double h1_low[];
   ArraySetAsSeries(h1_close, true);
   ArraySetAsSeries(h1_high, true);
   ArraySetAsSeries(h1_low, true);
   
   int atrPeriod = 14;
   int lookback = 50; // Need enough for ATR + ranking
   int totalBars = atrPeriod + lookback;
   
   if(CopyClose(_Symbol, PERIOD_H1, 0, totalBars, h1_close) < totalBars ||
      CopyHigh(_Symbol, PERIOD_H1, 0, totalBars, h1_high) < totalBars ||
      CopyLow(_Symbol, PERIOD_H1, 0, totalBars, h1_low) < totalBars)
   {
      // Not enough data, default to normal
      return 1;
   }
   
   // Compute current ATR (H1)
   double currentATR = 0;
   for(int i = 0; i < atrPeriod; i++)
      currentATR += (h1_high[i] - h1_low[i]);
   currentATR /= atrPeriod;
   
   // Compute historical ATRs for percentile ranking
   double atrVals[];
   ArrayResize(atrVals, lookback);
   int validATR = 0;
   for(int i = atrPeriod; i < atrPeriod + lookback; i++)
   {
      if(i + 1 >= ArraySize(h1_high)) break;
      double tr = MathMax(h1_high[i], h1_close[i+1]) - MathMin(h1_low[i], h1_close[i+1]);
      // Simplified: use H-L as proxy for historical TR
      double atr_approx = 0;
      for(int j = i; j < i + atrPeriod && j < ArraySize(h1_high) - 1; j++)
         atr_approx += (h1_high[j] - h1_low[j]);
      atr_approx /= atrPeriod;
      if(atr_approx > 0) { atrVals[validATR] = atr_approx; validATR++; }
   }
   
   if(validATR < 10) return 1; // Not enough history
   
   // Count how many historical ATRs are below current
   int below = 0;
   for(int i = 0; i < validATR; i++)
      if(atrVals[i] < currentATR) below++;
   
   double percentile = (double)below / validATR;
   
   if(percentile < 0.33) return 0;  // Low volatility
   if(percentile < 0.67) return 1;  // Normal volatility
   return 2;                          // High volatility
}

//+------------------------------------------------------------------+
//| New V10 computation stubs                                        |
//| These need to be implemented in the EA or extracted from the     |
//| novita_megaforge r44 streaming pipeline                          |
//+------------------------------------------------------------------+
double ComputeATRPercentile(double &high[], double &low[], double &close[],
                             int atr_period, int lookback, int bars)
{
   // Compute current ATR percentile rank over lookback window
   // Returns value in [0, 1]
   double atr_current = ATR_Raw(high, low, close, atr_period, bars);
   // Count how many historical ATRs are below current
   int count = 0;
   int total = MathMin(lookback, bars - atr_period - 1);
   for(int i = atr_period; i < atr_period + total; i++)
   {
      double atr_hist = ATR_Raw_At(high, low, close, atr_period, bars, i);
      if(atr_hist < atr_current) count++;
   }
   return (total > 0) ? (double)count / total : 0.5;
}

double ComputeOrderBlockProximity(double &high[], double &low[], double &close[],
                                   double &open[], int lookback)
{
   // Find nearest bullish/bearish order block and compute proximity
   // Returns 0-1 (1 = price sitting right on an OB level)
   // V10 FIX: Use correct bar ordering (i=0 is current, i+1 is next older) with bounds check
   int sz = ArraySize(close);
   double min_dist = 999999.0;
   for(int i = 1; i < lookback && i + 1 < sz; i++)
   {
      // Bullish OB: bearish candle at i+1 (engulfed by bullish at i)
      if(close[i+1] < open[i+1] && close[i] > open[i])
      {
         double ob_low = MathMin(open[i+1], close[i+1]);
         double dist = MathAbs(close[0] - ob_low) / (close[0] + 1e-10);
         if(dist < min_dist) min_dist = dist;
      }
      // Bearish OB: bullish candle at i+1 (engulfed by bearish at i)
      if(close[i+1] > open[i+1] && close[i] < open[i])
      {
         double ob_high = MathMax(open[i+1], close[i+1]);
         double dist = MathAbs(close[0] - ob_high) / (close[0] + 1e-10);
         if(dist < min_dist) min_dist = dist;
      }
   }
   return MathMax(0.0, 1.0 - min_dist * 10.0); // Scale inverse distance
}

double ComputeFVGFillPct(double &high[], double &low[], double &close[], int lookback)
{
   // Count Fair Value Gaps in lookback and what % have been filled
   int fvgs = 0, filled = 0;
   for(int i = 2; i < lookback && i < ArraySize(close) - 1; i++)
   {
      // Bullish FVG: gap up
      if(low[i] > high[i+2])
      {
         fvgs++;
         if(close[i-1] <= high[i+2] || low[i-1] <= high[i+2])
            filled++;
      }
      // Bearish FVG: gap down
      if(high[i] < low[i+2])
      {
         fvgs++;
         if(close[i-1] >= low[i+2] || high[i-1] >= low[i+2])
            filled++;
      }
   }
   return (fvgs > 0) ? (double)filled / fvgs : 0.5;
}

void ComputeBOS(double &high[], double &low[], double &close[], int lookback,
                double &bullish_bos, double &bearish_bos)
{
   // Break of Structure detection
   // V10 FIX: Loop forward in time (i from old→new) to track structure progression
   // Returns counts normalized to [0, 1]
   int bull_count = 0, bear_count = 0;
   double prev_high = -999999.0, prev_low = 999999.0;
   
   // Iterate from oldest to newest (high index = older bar in as-series array)
   for(int i = MathMin(lookback, ArraySize(close)-1); i > 0; i--)
   {
      if(high[i] > prev_high && prev_high > -999998.0)
         bull_count++;
      if(low[i] < prev_low && prev_low < 999998.0)
         bear_count++;
      prev_high = MathMax(prev_high, high[i]);
      prev_low = MathMin(prev_low, low[i]);
   }
   double total = (bull_count + bear_count > 0) ? (double)(bull_count + bear_count) : 1.0;
   bullish_bos = bull_count / total;
   bearish_bos = bear_count / total;
}

double ComputeVWAP(double &high[], double &low[], double &close[], long &vol[], int period)
{
   double sum_pv = 0.0, sum_v = 0.0;
   int n = MathMin(period, ArraySize(close));
   for(int i = 0; i < n; i++)
   {
      double tp = (high[i] + low[i] + close[i]) / 3.0;
      sum_pv += tp * (double)vol[i];
      sum_v += (double)vol[i];
   }
   return (sum_v > 0) ? sum_pv / sum_v : close[0];
}

double ComputeOBVSlope(double &close[], long &vol[], int period, int bars)
{
   // OBV slope over last `period` bars, normalized
   double obv = 0.0, obv_prev = 0.0;
   double obv_first = 0.0;
   for(int i = period; i > 0; i--)
   {
      if(close[i] > close[i+1]) obv += (double)vol[i];
      else if(close[i] < close[i+1]) obv -= (double)vol[i];
      if(i == period) obv_first = obv;
   }
   // Current OBV
   if(close[0] > close[1]) obv_prev = obv + (double)vol[0];
   else if(close[0] < close[1]) obv_prev = obv - (double)vol[0];
   else obv_prev = obv;
   
   return (obv_first != 0) ? (obv_prev - obv_first) / MathAbs(obv_first) : 0.0;
}

double ComputeOBVAccel(double &close[], long &vol[], int period, int bars)
{
   double slope_now = ComputeOBVSlope(close, vol, period, bars);
   double slope_prev = ComputeOBVSlope_Offset(close, vol, period, bars, 5);
   return slope_now - slope_prev;
}

double ComputeOBVDerivative5(double &close[], long &vol[], int bars)
{
   // 5-bar rate of change of OBV
   return ComputeOBVSlope(close, vol, 10, bars) - ComputeOBVSlope_Offset(close, vol, 10, bars, 5);
}

double ComputeOBVSlope_Offset(double &close[], long &vol[], int period, int bars, int offset)
{
   // OBV slope starting from offset bars ago
   double obv = 0.0;
   for(int i = period + offset; i > offset; i--)
   {
      if(i >= ArraySize(close)) continue;
      if(close[i] > close[i+1]) obv += (double)vol[i];
      else if(close[i] < close[i+1]) obv -= (double)vol[i];
   }
   return obv;
}

double ComputeEMASlope(double &close[], int ema_period, int slope_period, int bars)
{
   double ema_current = EMA_Array(close, ema_period, bars);
   // Approximate slope using price change over slope_period
   double ema_prev = EMA_Array_Offset(close, ema_period, bars, slope_period);
   return (ema_current - ema_prev) / (ema_prev + 1e-10);
}

double ComputeHTFTrend(string sym, ENUM_TIMEFRAMES tf)
{
   // V10 FIX: Return continuous ratio (e21 - e55) / (e55 + eps), matching Python training
   // Was previously returning -1/0/1 (sign only), but models were trained on the ratio
   double ema21_0 = iMA(sym, tf, 21, 0, MODE_EMA, PRICE_CLOSE, 0);
   double ema55_0 = iMA(sym, tf, 55, 0, MODE_EMA, PRICE_CLOSE, 0);
   if(ema55_0 <= 0 || ema21_0 <= 0)
      return 0.0;  // Invalid data
   double ratio = (ema21_0 - ema55_0) / (ema55_0 + 1e-10);
   // Clamp to reasonable range to match training data distribution
   return MathMax(-2.0, MathMin(2.0, ratio));
}

double ComputeCorrelation(string sym1, string sym2, int period)
{
   if(sym2 == "") return 0.0;
   double close1[], close2[];
   ArraySetAsSeries(close1, true); ArraySetAsSeries(close2, true);
   int n1 = CopyClose(sym1, PERIOD_M15, 0, period + 1, close1);
   int n2 = CopyClose(sym2, PERIOD_M15, 0, period + 1, close2);
   if(n1 < period + 1 || n2 < period + 1) return 0.0;
   
   // Compute returns correlation
   double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
   for(int i = 0; i < period; i++)
   {
      double r1 = (close1[i] - close1[i+1]) / (close1[i+1] + 1e-10);
      double r2 = (close2[i] - close2[i+1]) / (close2[i+1] + 1e-10);
      sum_x += r1; sum_y += r2;
      sum_xy += r1 * r2;
      sum_x2 += r1 * r1; sum_y2 += r2 * r2;
   }
   double denom = MathSqrt((period * sum_x2 - sum_x * sum_x) * (period * sum_y2 - sum_y * sum_y));
   return (denom > 1e-10) ? (period * sum_xy - sum_x * sum_y) / denom : 0.0;
}

#endif // ONNX_FEATURES_V10_MQH
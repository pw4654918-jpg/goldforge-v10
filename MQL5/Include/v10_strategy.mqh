//+------------------------------------------------------------------+
//|                                              v10_strategy.mqh    |
//|     V10 Strategy for TradingAI Predictor EA                     |
//|     INCLUDE AFTER onnx_features_v10.mqh                          |
//+------------------------------------------------------------------+
//|  Changes from V9 -> V10:                                         |
//|  1. Learned ensemble weights (CAT=31.2%, XGB=52.7%, LGB=16.1%) |
//|  2. Meta-confidence gate (L2 logistic, 14 features)             |
//|  3. Regime-adaptive triple barrier SL/TP                        |
//|  4. 77 features (was 67 in V9)                                   |
//|  5. RSI fixed to 0-1 scale (was 0-100)                          |
//|  6. All V9 improvements preserved:                              |
//|     - Structure-aware SL (snap to swing, capped at 3*ATR)      |
//|     - Chandelier trailing stop                                   |
//|     - Dynamic partial close (regime-adaptive)                   |
//|     - Confidence-scaled lot sizing                               |
//|     - Enhanced session weighting (+2 London/NY overlap)         |
//|     - Adaptive ensemble drift                                     |
//|     - Kelly Criterion sizing                                     |
//|     - DXY hard gate                                              |
//|     - Gold power zones                                           |
//|     - Session-aware spread limits                                |
//|     - FOMC/NFP hard block                                        |
//+------------------------------------------------------------------+
#ifndef V10_STRATEGY_MQH
#define V10_STRATEGY_MQH

// ── V10 Named Constants (replacing magic numbers) ──
#define V10_DRIFT_PENALTY    0.05       // Drift weight redistribution fraction (was 0.10, reduced to prevent over-shifting)
#define V10_STRUCTURE_SL_PAD 0.2        // Structure SL padding in ATR multiples
#define V10_SL_FALLBACK_MULT 2.0        // Fallback SL when no swing found (xATR)
#define V10_LOT_SCALE_MIN    0.7        // Min lot scaling multiplier
#define V10_LOT_SCALE_MAX    1.5        // Max lot scaling multiplier
#define V10_META_CONF_SCALE_MIN 0.3     // Min meta-confidence lot scale
#define V10_META_CONF_SCALE_MAX 1.2     // Max meta-confidence lot scale
#define V10_MIN_TRADES_KELLY  5          // Minimum trades before Kelly sizing activates
#define V10_MAX_KELLY_FRAC    0.15       // Max Kelly fraction cap
#define V10_KELLY_WINDOW      50         // Kelly rolling window size

// ── V10 Inputs ──
input group "═══ V10 STRATEGY ═══"
input bool     V10_UseStructureSL      = true;      // Snap SL to swing structure
input double   V10_MaxSL_ATR_Mult      = 3.0;       // Max SL = ATR * this (hard cap)
input bool     V10_UseChandelierTrail  = true;      // Use Chandelier trailing stop
input double   V10_ChandelierMult      = 2.5;       // Chandelier ATR multiplier
input int      V10_ChandelierLookback  = 15;        // Chandelier lookback bars
input bool     V10_DynamicPartialClose = true;      // Regime-adaptive partial close
input double   V10_HighVolPartialPct   = 50.0;      // Partial close % in high vol
input double   V10_LowVolPartialPct    = 25.0;      // Partial close % in low vol
input bool     V10_ConfidenceScaleLots = true;      // Scale lots by AI confidence
input double   V10_ConfBaseRef         = 0.60;       // Reference confidence for 1.0x lots
input bool     V10_AdaptiveDrift       = true;      // Enable adaptive ensemble drift
input int      V10_DriftWindow         = 20;        // Trades to evaluate drift
input double   V10_DriftThreshold       = 0.50;     // Threshold for drift adjustment
input int      V10_SignalPersistence   = 2;          // Require N consecutive M15 bars with same signal
input double   V10_ConfidenceMargin     = 0.07;       // Winner must beat runner-up by this margin
input bool     V10_UseKellySizing      = true;       // True Kelly Criterion lot sizing
input double   V10_KellyFraction        = 0.25;       // Fractional Kelly (25% = quarter-Kelly)
input bool     V10_UseDXYHardGate      = true;       // Block gold BUY when DXY is strong BUY
input bool     V10_UseGoldPowerZones   = true;       // Boost confluence during peak gold hours
input bool     V10_SessionSpread        = true;       // Adaptive spread limits per session

// ── V10 ENSEMBLE — Learned Weights ──
input group "═══ V10 ONNX MODELS ═══"
input string   V10_ONNXModelXGB        = "v10_premium_xgboost.onnx";
input string   V10_ONNXModelLGB        = "v10_premium_lightgbm.onnx";
// ── V10 ENSEMBLE — Learned Weights ──
// NOTE: These are DEFAULT values only. They're copied to g_v10_wCAT/XGB/LGB globals
// at init, and then drift-adapted at runtime. Changing these inputs overrides them.
input double   V10_WeightCAT           = 0.3125;   // CatBoost weight (default, adapted by drift)
input double   V10_WeightXGB            = 0.5266;   // XGBoost weight (default, DOMINANT, adapted by drift)
input double   V10_WeightLGB            = 0.1608;   // LightGBM weight (default, adapted by drift)

// ── V10 META-CONFIDENCE — L2 Logistic Regression ──
input group "═══ V10 META-CONFIDENCE ═══"
input bool     V10_UseMetaConfidence   = true;      // Gate trades by meta-confidence
input double   V10_MetaConfThreshold   = 0.50;      // Min meta-confidence to trade (0.5 = neutral)
input bool     V10_MetaConfLotScale    = true;      // Scale lots by meta-confidence value

// ── V10 REGIME-ADAPTIVE BARRIERS ──
input group "═══ V10 REGIME BARRIERS ═══"
input bool     V10_UseRegimeBarriers   = true;      // Use regime-adaptive TP/SL
input double   V10_LowVolBuyPT         = 1.2;       // Low vol BUY profit target (xATR)
input double   V10_LowVolSellPT        = 1.0;       // Low vol SELL profit target (xATR)
input double   V10_LowVolSL            = 1.0;       // Low vol SL (xATR)
input double   V10_NormalBuyPT         = 1.8;       // Normal BUY profit target (xATR)
input double   V10_NormalSellPT        = 1.3;       // Normal SELL profit target (xATR)
input double   V10_NormalSL            = 1.5;       // Normal SL (xATR)
input double   V10_HighVolBuyPT        = 2.2;       // High vol BUY profit target (xATR)
input double   V10_HighVolSellPT       = 1.5;       // High vol SELL profit target (xATR)
input double   V10_HighVolSL           = 1.2;       // High vol SL (xATR)

// ═════════════════════════════════════════════════════════════════════
// META-CONFIDENCE COEFFICIENTS — L2 LogisticRegression, C=1.0
// Trained on V10 validation split (31,229 bars)
// Train accuracy: 0.6728 | OOS accuracy: 0.6265
// Logistic: p = 1 / (1 + exp(-(intercept + sum(coef_i * x_i))))
// p > 0.5: ensemble likely correct -> trade
// p <= 0.5: ensemble likely wrong -> skip/reduce
// ═════════════════════════════════════════════════════════════════════
double META_INTERCEPT = -1.1873795872;
double META_COEF_CAT_BUY  = -0.4796830464;
double META_COEF_CAT_HOLD = -0.0153286839;
double META_COEF_CAT_SELL = -0.4310502119;
double META_COEF_XGB_BUY  = -1.3741882283;
double META_COEF_XGB_HOLD =  0.2873679108;
double META_COEF_XGB_SELL =  0.1607605144;
double META_COEF_LGB_BUY  = -0.1993025339;
double META_COEF_LGB_HOLD = -0.3494576429;
double META_COEF_LGB_SELL = -0.3773001162;
double META_COEF_MAX_CONF = -1.8393669094;
double META_COEF_CAT_CONF =  0.2148461211;
double META_COEF_XGB_CONF =  2.8131691929;
double META_COEF_LGB_CONF =  2.5020209236;
double META_COEF_BSS      = -0.0486328268;

// ── V10 Globals ──
double g_v10_wCAT, g_v10_wXGB, g_v10_wLGB;  // Adaptive weights (init from inputs)
int    g_v10_drift_cat_correct = 0, g_v10_drift_cat_total = 0;
int    g_v10_drift_xgb_correct = 0, g_v10_drift_xgb_total = 0;
int    g_v10_drift_lgb_correct = 0, g_v10_drift_lgb_total = 0;
double g_v10_highest_high = 0;
double g_v10_lowest_low = 999999;

// Signal stability
string g_v10_last_signal = "";
int    g_v10_signal_streak = 0;
bool   g_v10_is_tester = false;

// Kelly Criterion rolling window
double g_v10_kelly_wins[50];
double g_v10_kelly_losses[50];
int    g_v10_kelly_count = 0;
int    g_v10_kelly_wins_n = 0;
int    g_v10_kelly_losses_n = 0;
double g_v10_kelly_fraction = 0.0;

// Meta-confidence stats
int    g_v10_meta_trades = 0;
int    g_v10_meta_correct = 0;

//+------------------------------------------------------------------+
//| V10 Init — Call in OnInit() after model loading                   |
//+------------------------------------------------------------------+
void V10_Init()
{
   g_v10_wCAT = V10_WeightCAT;
   g_v10_wXGB = V10_WeightXGB;
   g_v10_wLGB = V10_WeightLGB;
   g_v10_highest_high = 0;
   g_v10_lowest_low = 999999;
   g_v10_last_signal = "";
   g_v10_signal_streak = 0;
   g_v10_is_tester = (bool)MQLInfoInteger(MQL_TESTER);
   g_v10_meta_trades = 0;
   g_v10_meta_correct = 0;
}

//+------------------------------------------------------------------+
//| V10 Weighted Ensemble Probability                                |
//| Uses LEARNED weights (not fixed input defaults)                  |
//| Weights adapt via V10_AdaptiveDrift                               |
//+------------------------------------------------------------------+
void V10_EnsembleProb(
   const float &xgb_probs[], const float &lgb_probs[], const float &cat_probs[],
   double &ens_probs[]  // Output: [BUY=idx0, HOLD=idx1, SELL=idx2]
)
{
   ArrayResize(ens_probs, 3);
   for(int i = 0; i < 3; i++)
   {
      ens_probs[i] = g_v10_wCAT * (double)cat_probs[i] +
                     g_v10_wXGB * (double)xgb_probs[i] +
                     g_v10_wLGB * (double)lgb_probs[i];
   }
   
   // Normalize to sum=1.0
   double total = ens_probs[0] + ens_probs[1] + ens_probs[2];
   if(total > 0)
   {
      for(int i = 0; i < 3; i++)
         ens_probs[i] /= total;
   }
}

//+------------------------------------------------------------------+
//| V10 Meta-Confidence Prediction                                   |
//| Native L2 Logistic Regression in MQL5                            |
//| Returns probability [0..1] that ensemble prediction is correct   |
//| 14 input features: cat_probs[3] + xgb_probs[3] + lgb_probs[3]  |
//|   + max_conf + cat_conf + xgb_conf + lgb_conf + bss             |
//+------------------------------------------------------------------+
double V10_MetaConfidence(
   const float &cat_probs[], const float &xgb_probs[], const float &lgb_probs[],
   double &ens_probs[]
)
{
   // Individual model confidences (max probability per model)
   double cat_conf = (double)MathMax(cat_probs[0], MathMax(cat_probs[1], cat_probs[2]));
   double xgb_conf = (double)MathMax(xgb_probs[0], MathMax(xgb_probs[1], xgb_probs[2]));
   double lgb_conf = (double)MathMax(lgb_probs[0], MathMax(lgb_probs[1], lgb_probs[2]));
   
   // Ensemble max confidence
   double max_conf = MathMax(ens_probs[0], MathMax(ens_probs[1], ens_probs[2]));
   
   // Buy-Sell Spread (from CatBoost, same as training)
   double bss = (double)cat_probs[0] - (double)cat_probs[2];
   
   // Logistic regression linear combination
   double z = META_INTERCEPT
            + META_COEF_CAT_BUY  * cat_probs[0] + META_COEF_CAT_HOLD * cat_probs[1] + META_COEF_CAT_SELL * cat_probs[2]
            + META_COEF_XGB_BUY  * xgb_probs[0] + META_COEF_XGB_HOLD * xgb_probs[1] + META_COEF_XGB_SELL * xgb_probs[2]
            + META_COEF_LGB_BUY  * lgb_probs[0] + META_COEF_LGB_HOLD * lgb_probs[1] + META_COEF_LGB_SELL * lgb_probs[2]
            + META_COEF_MAX_CONF * max_conf
            + META_COEF_CAT_CONF * cat_conf
            + META_COEF_XGB_CONF * xgb_conf
            + META_COEF_LGB_CONF * lgb_conf
            + META_COEF_BSS      * bss;
   
   // Sigmoid: p = 1 / (1 + exp(-z))
   // Clamp z to avoid overflow
   z = MathMax(-500.0, MathMin(500.0, z));
   double p = 1.0 / (1.0 + MathExp(-z));
   
   return p;
}

//+------------------------------------------------------------------+
//| V10 Get Regime-Adaptive Barriers                                 |
//| Returns TP and SL multipliers based on volatility regime         |
//| Low vol: tight targets, normal SL                                |
//| Normal: standard asymmetric barriers                             |
//| High vol: wide targets, tight SL (protect capital)              |
//+------------------------------------------------------------------+
void V10_GetRegimeBarriers(int volRegime, double &buyPT, double &sellPT, double &slMult)
{
   if(!V10_UseRegimeBarriers)
   {
      buyPT = 1.8; sellPT = 1.3; slMult = 1.5;  // V10 defaults
      return;
   }
   
   if(volRegime < 0)       // LOW volatility regime (regime_0)
   {
      buyPT = V10_LowVolBuyPT;     // 1.2x ATR
      sellPT = V10_LowVolSellPT;   // 1.0x ATR
      slMult = V10_LowVolSL;       // 1.0x ATR
   }
   else if(volRegime > 0)  // HIGH volatility regime (regime_2)
   {
      buyPT = V10_HighVolBuyPT;    // 2.2x ATR
      sellPT = V10_HighVolSellPT;  // 1.5x ATR
      slMult = V10_HighVolSL;      // 1.2x ATR (tight!)
   }
   else                    // NORMAL regime (regime_1)
   {
      buyPT = V10_NormalBuyPT;     // 1.8x ATR
      sellPT = V10_NormalSellPT;   // 1.3x ATR
      slMult = V10_NormalSL;       // 1.5x ATR
   }
}

//+------------------------------------------------------------------+
//| V10 Structure-Aware Stop Loss                                    |
//| Finds nearest swing level within ATR range, snaps SL there       |
//| Falls back to plain ATR-based SL if no structure found           |
//+------------------------------------------------------------------+
double V10_StructureSL(ENUM_ORDER_TYPE type, double entryPrice, double atrVal)
{
   if(!V10_UseStructureSL)
   {
      if(type == ORDER_TYPE_BUY)
         return entryPrice - atrVal * V10_SL_FALLBACK_MULT;
      else
         return entryPrice + atrVal * V10_SL_FALLBACK_MULT;
   }
   
   double highs[], lows[];
   int lookback = 25;
   ArrayResize(highs, lookback); ArrayResize(lows, lookback);
   ArraySetAsSeries(highs, true); ArraySetAsSeries(lows, true);
   
   if(CopyHigh(_Symbol, PERIOD_M15, 0, lookback, highs) < lookback ||
      CopyLow(_Symbol, PERIOD_M15, 0, lookback, lows) < lookback)
   {
      return (type == ORDER_TYPE_BUY) ?
             entryPrice - atrVal * 2.0 : entryPrice + atrVal * 2.0;
   }
   
   double maxSL = atrVal * V10_MaxSL_ATR_Mult;
   double minSL = atrVal * 1.0;
   
   if(type == ORDER_TYPE_BUY)
   {
      double bestSwing = 0;
      double bestDist = 999999.0; // V10 FIX: Track closest valid swing (not first)
      for(int i = 2; i < lookback; i++)
      {
         if(i + 1 < lookback && lows[i] < lows[i-1] && lows[i] < lows[i+1])
         {
            double dist = entryPrice - lows[i];
            if(dist > minSL && dist < maxSL && dist < bestDist)
            {
               bestSwing = lows[i];
               bestDist = dist;
            }
         }
      }
      if(bestSwing > 0)
         return bestSwing - atrVal * V10_STRUCTURE_SL_PAD;
      else
         return entryPrice - atrVal * V10_SL_FALLBACK_MULT;
   }
   else // SELL
   {
      double bestSwing = 0;
      double bestDist = 999999.0; // V10 FIX: Track closest valid swing (not first)
      for(int i = 2; i < lookback; i++)
      {
         if(i + 1 < lookback && highs[i] > highs[i-1] && highs[i] > highs[i+1])
         {
            double dist = highs[i] - entryPrice;
            if(dist > minSL && dist < maxSL && dist < bestDist)
            {
               bestSwing = highs[i];
               bestDist = dist;
            }
         }
      }
      if(bestSwing > 0)
         return bestSwing + atrVal * V10_STRUCTURE_SL_PAD;
      else
         return entryPrice + atrVal * V10_SL_FALLBACK_MULT;
   }
}

//+------------------------------------------------------------------+
//| V10 Chandelier Trailing Stop                                     |
//| Trails from highest high (BUY) or lowest low (SELL)              |
//+------------------------------------------------------------------+
double V10_ChandelierTrail(ulong ticket, ENUM_POSITION_TYPE posType, double currentSL)
{
   if(!V10_UseChandelierTrail) return currentSL;
   
   double highs[], lows[], closes[];
   int lb = V10_ChandelierLookback;
   ArrayResize(highs, lb); ArrayResize(lows, lb); ArrayResize(closes, lb);
   ArraySetAsSeries(highs, true); ArraySetAsSeries(lows, true); ArraySetAsSeries(closes, true);
   
   if(CopyHigh(_Symbol, PERIOD_M15, 0, lb + 1, highs) < lb ||
      CopyLow(_Symbol, PERIOD_M15, 0, lb + 1, lows) < lb ||
      CopyClose(_Symbol, PERIOD_M15, 0, lb + 1, closes) < lb)
      return currentSL;
   
   // V10 FIX: Use True Range instead of just High-Low for ATR calculation
   // True Range accounts for gaps, avoiding 5-15% underestimation
   double atrVal = 0;
   int atrPeriod = 14;
   int atrBars = MathMin(atrPeriod, lb - 1);
   for(int i = 0; i < atrBars; i++)
   {
      double h = highs[i];
      double l = lows[i];
      double pc = closes[i + 1]; // Previous close
      double tr = MathMax(h, pc) - MathMin(l, pc); // True Range
      atrVal += tr;
   }
   atrVal /= atrBars;
   if(atrVal <= 0) return currentSL;
   
   double newSL = currentSL;
   
   if(posType == POSITION_TYPE_BUY)
   {
      double hh = highs[0];
      for(int i = 1; i < lb; i++)
         if(highs[i] > hh) hh = highs[i];
      double chandSL = hh - atrVal * V10_ChandelierMult;
      if(chandSL > currentSL && chandSL > 0)
         newSL = chandSL;
   }
   else // SELL
   {
      double ll = lows[0];
      for(int i = 1; i < lb; i++)
         if(lows[i] < ll) ll = lows[i];
      double chandSL = ll + atrVal * V10_ChandelierMult;
      if(chandSL < currentSL || currentSL == 0)
         newSL = chandSL;
   }
   
   return newSL;
}

//+------------------------------------------------------------------+
//| V10 Dynamic Partial Close Percentage                             |
//| Returns % of position to close at TP1 based on volatility       |
//+------------------------------------------------------------------+
double V10_GetPartialClosePct(int volRegime)
{
   if(!V10_DynamicPartialClose)
      return 33.0;
   
   if(volRegime > 0)       // HIGH volatility — take more off table
      return V10_HighVolPartialPct;
   else if(volRegime < 0)  // LOW volatility — let it run
      return V10_LowVolPartialPct;
   else                    // NORMAL
      return 33.0;
}

//+------------------------------------------------------------------+
//| V10 Confidence-Scaled Lot Sizing                                  |
//| High confidence = larger lot (up to 1.5x), low = smaller (0.7x) |
//+------------------------------------------------------------------+
double V10_ScaleLotByConfidence(double baseLot, double aiConfidence, double maxRiskPct)
{
   if(!V10_ConfidenceScaleLots || aiConfidence <= 0)
      return baseLot;
   
   double scale = aiConfidence / V10_ConfBaseRef;
   scale = MathMax(0.7, MathMin(1.5, scale));
   
   double scaledLot = baseLot * scale;
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   scaledLot = MathMax(minLot, MathMin(maxLot, scaledLot));
   scaledLot = MathFloor(scaledLot / lotStep) * lotStep;
   
   return scaledLot;
}

//+------------------------------------------------------------------+
//| V10 Enhanced Session Bonus                                       |
//| London/NY overlap gets +2, London alone +1, NY alone +1         |
//+------------------------------------------------------------------+
int V10_SessionBonus(int hour)
{
   if(hour >= 13 && hour <= 17) return 2;
   if(hour >= 8 && hour < 13)   return 1;
   if(hour > 17 && hour < 21)  return 1;
   return 0;
}

//+------------------------------------------------------------------+
//| V10 Adaptive Ensemble Drift                                      |
//| Tracks per-model accuracy and adjusts weights                    |
//+------------------------------------------------------------------+
void V10_UpdateDrift(
   string modelSignal_CAT, string modelSignal_XGB, string modelSignal_LGB,
   string actualDirection, bool wasProfit)
{
   if(!V10_AdaptiveDrift) return;
   
   g_v10_drift_xgb_total++;
   if(modelSignal_XGB == actualDirection && wasProfit)
      g_v10_drift_xgb_correct++;
   
   g_v10_drift_lgb_total++;
   if(modelSignal_LGB == actualDirection && wasProfit)
      g_v10_drift_lgb_correct++;
   
   g_v10_drift_cat_total++;
   if(modelSignal_CAT == actualDirection && wasProfit)
      g_v10_drift_cat_correct++;
   
   if(g_v10_drift_xgb_total >= V10_DriftWindow && g_v10_drift_xgb_total % V10_DriftWindow == 0)
   {
      double rateXGB = (g_v10_drift_xgb_total > 0) ?
                        (double)g_v10_drift_xgb_correct / g_v10_drift_xgb_total : 0.5;
      double rateLGB = (g_v10_drift_lgb_total > 0) ?
                        (double)g_v10_drift_lgb_correct / g_v10_drift_lgb_total : 0.5;
      double rateCAT = (g_v10_drift_cat_total > 0) ?
                        (double)g_v10_drift_cat_correct / g_v10_drift_cat_total : 0.5;
      
      double minRate = MathMin(rateXGB, MathMin(rateLGB, rateCAT));
      
      if(minRate < V10_DriftThreshold)
      {
         double penalty = V10_DRIFT_PENALTY;
         
         // V10: XGB is dominant, redistribute proportionally
         if(rateXGB == minRate && g_v10_wXGB > 0.05)
         {
            double shift = g_v10_wXGB * penalty;
            g_v10_wXGB -= shift;
            // Redistribute: CAT gets 60%, LGB 40% (inverse of weights)
            g_v10_wCAT += shift * 0.6;
            g_v10_wLGB += shift * 0.4;
         }
         else if(rateCAT == minRate && g_v10_wCAT > 0.05)
         {
            double shift = g_v10_wCAT * penalty;
            g_v10_wCAT -= shift;
            g_v10_wXGB += shift * 0.7;
            g_v10_wLGB += shift * 0.3;
         }
         else if(rateLGB == minRate && g_v10_wLGB > 0.05)
         {
            double shift = g_v10_wLGB * penalty;
            g_v10_wLGB -= shift;
            g_v10_wXGB += shift * 0.5;
            g_v10_wCAT += shift * 0.5;
         }
         
         // Normalize
         double wSum = g_v10_wXGB + g_v10_wLGB + g_v10_wCAT;
         if(wSum > 0)
         {
            g_v10_wXGB /= wSum;
            g_v10_wLGB /= wSum;
            g_v10_wCAT /= wSum;
         }
      }
      
      // Reset drift counters
      g_v10_drift_xgb_correct = 0; g_v10_drift_xgb_total = 0;
      g_v10_drift_lgb_correct = 0; g_v10_drift_lgb_total = 0;
      g_v10_drift_cat_correct = 0; g_v10_drift_cat_total = 0;
   }
}

//+------------------------------------------------------------------+
//| V10 Get Regime-Adaptive TP Multiplier                            |
//| In trending markets, extend TP. In ranging, tighten it.          |
//+------------------------------------------------------------------+
double V10_GetTPMultiplier(int volRegime, string regime)
{
   double base = 3.0;
   
   // V10 FIX: Case-insensitive string comparison for regime names
   string regimeUpper = regime;
   StringToUpper(regimeUpper);
   
   if(regimeUpper == "TRENDING" || regimeUpper == "STRONG_TREND")
      base = 4.0;
   else if(regimeUpper == "RANGING")
      base = 2.5;
   
   if(volRegime > 0)   base += 0.5;
   else if(volRegime < 0) base -= 0.5;
   
   return MathMax(2.0, MathMin(5.0, base));
}

//+------------------------------------------------------------------+
//| V10 Signal Stability Filter                                      |
//| Requires N consecutive M15 bars with same signal before trading   |
//+------------------------------------------------------------------+
bool V10_IsSignalStable(string currentSignal)
{
   if(V10_SignalPersistence <= 1) return true;
   
   if(currentSignal == g_v10_last_signal)
      g_v10_signal_streak++;
   else
   {
      g_v10_last_signal = currentSignal;
      g_v10_signal_streak = 1;
   }
   
   return (g_v10_signal_streak >= V10_SignalPersistence);
}

//+------------------------------------------------------------------+
//| V10 Confidence Margin Filter                                     |
//| Winner class must beat runner-up by minimum margin               |
//+------------------------------------------------------------------+
bool V10_HasConfidenceMargin(double pBuy, double pHold, double pSell)
{
   if(V10_ConfidenceMargin <= 0) return true;
   
   double probs[3];
   probs[0] = pBuy; probs[1] = pHold; probs[2] = pSell;
   
   double maxP = 0, secondP = 0;
   for(int i = 0; i < 3; i++)
   {
      if(probs[i] > maxP) { secondP = maxP; maxP = probs[i]; }
      else if(probs[i] > secondP) secondP = probs[i];
   }
   
   return (maxP - secondP >= V10_ConfidenceMargin);
}

//+------------------------------------------------------------------+
//| V10 Meta-Confidence Gate                                        |
//| Returns true if trade is allowed, false if blocked              |
//| Also returns meta-confidence value for lot scaling              |
//+------------------------------------------------------------------+
bool V10_MetaConfidenceGate(
   const float &cat_probs[], const float &xgb_probs[], const float &lgb_probs[],
   double &ens_probs[], double &metaConfValue)
{
   metaConfValue = V10_MetaConfidence(cat_probs, xgb_probs, lgb_probs, ens_probs);
   
   if(!V10_UseMetaConfidence) return true;
   
   return (metaConfValue >= V10_MetaConfThreshold);
}

//+------------------------------------------------------------------+
//| V10 Meta-Confidence Lot Scaling                                  |
//| Reduces lot size when meta-confidence is below threshold        |
//+------------------------------------------------------------------+
double V10_ScaleLotByMetaConfidence(double baseLot, double metaConf)
{
   if(!V10_MetaConfLotScale || !V10_UseMetaConfidence)
      return baseLot;
   
   // Scale: metaConf / threshold, clamped to [0.3, 1.0]
   // Above threshold: full lot or slight boost
   // Below threshold: reduce (but allow with small size if close to 0.5)
   double scale = MathMax(0.3, MathMin(1.2, metaConf / V10_MetaConfThreshold));
   
   double scaledLot = baseLot * scale;
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   scaledLot = MathMax(minLot, MathMin(maxLot, scaledLot));
   scaledLot = MathFloor(scaledLot / lotStep) * lotStep;
   
   return scaledLot;
}

//+------------------------------------------------------------------+
//| V10 Record Meta-Confidence Outcome                              |
//| Called after trade closes to track accuracy                      |
//+------------------------------------------------------------------+
void V10_RecordMetaOutcome(bool wasCorrect)
{
   g_v10_meta_trades++;
   if(wasCorrect) g_v10_meta_correct++;
}

//+------------------------------------------------------------------+
//| V10 FOMC/NFP Hard Block                                         |
//| Blocks trading on FOMC decision days and NFP Fridays            |
//+------------------------------------------------------------------+
bool V10_IsFOMCOrNFPBlock()
{
   MqlDateTime utc;
   TimeGMT(utc);
   int month = utc.mon;
   int day = utc.day;
   int hour = utc.hour;
   
   // NFP: First Friday of every month, block 12:00-15:00 UTC
   if(utc.day_of_week == 5 && day <= 7 && hour >= 12 && hour <= 15)
      return true;
   
   // 2026 FOMC dates (block 12:00-21:00 UTC)
   if(hour >= 12 && hour <= 21)
   {
      if((month==1 && (day==28||day==29)) ||
         (month==3 && (day==18||day==19)) ||
         (month==5 && (day==6||day==7)) ||
         (month==6 && (day==17||day==18)) ||
         (month==7 && (day==29||day==30)) ||
         (month==9 && (day==16||day==17)) ||
         (month==10 && (day==28||day==29)) ||
         (month==12 && (day==9||day==10)))
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| V10 Kelly Criterion Lot Sizing for XAUUSD                        |
//| f* = (WR * avgW/avgL - (1-WR)) / (avgW/avgL)                   |
//| Applied as fractional Kelly (25%) for drawdown control           |
//+------------------------------------------------------------------+
void V10_RecordTradeResult(double profitDollars)
{
   if(!V10_UseKellySizing) return;
   
   // V10 FIX: Clear the slot before writing to avoid stale data on wrap-around
   int idx = g_v10_kelly_count % 50;
   
   // Clear both arrays at this slot (stale data from previous cycle)
   g_v10_kelly_wins[idx] = 0;
   g_v10_kelly_losses[idx] = 0;
   
   if(profitDollars >= 0)
   {
      g_v10_kelly_wins[idx] = profitDollars;
      g_v10_kelly_wins_n++;
   }
   else
   {
      g_v10_kelly_losses[idx] = MathAbs(profitDollars);
      g_v10_kelly_losses_n++;
   }
   g_v10_kelly_count++;
   
   if(g_v10_kelly_count >= 5)
   {
      int n = MathMin(g_v10_kelly_count, 50);
      double totalWin = 0, totalLoss = 0;
      int wins = 0, losses = 0;
      for(int i = 0; i < n; i++)
      {
         if(g_v10_kelly_wins[i] > 0) { totalWin += g_v10_kelly_wins[i]; wins++; }
         if(g_v10_kelly_losses[i] > 0) { totalLoss += g_v10_kelly_losses[i]; losses++; }
      }
      if(wins == 0 || losses == 0 || totalLoss == 0)
      {
         g_v10_kelly_fraction = 0.01;
         return;
      }
      double wr = (double)wins / n;
      double avgWin = totalWin / wins;
      double avgLoss = totalLoss / losses;
      double wlRatio = avgWin / avgLoss;
      
      double fullKelly = (wr * wlRatio - (1.0 - wr)) / wlRatio;
      fullKelly = MathMax(0.001, MathMin(0.15, fullKelly));
      g_v10_kelly_fraction = fullKelly * V10_KellyFraction;
   }
}

double V10_KellyLotSize(double equity, double slDistDollars, double tickVal, double tickSize)
{
   if(!V10_UseKellySizing || g_v10_kelly_fraction <= 0 || g_v10_kelly_count < 5)
      return 0;
   
   double riskAmount = equity * g_v10_kelly_fraction;
   double lots = 0;
   if(tickVal > 0 && tickSize > 0)
      lots = riskAmount / (slDistDollars * tickVal / tickSize);
   else
      lots = riskAmount / (slDistDollars * 100.0);
   
   return lots;
}

string V10_KellyStatusString()
{
   if(g_v10_kelly_count < 5)
      return StringFormat("Kelly: WARMING (%d/5)", g_v10_kelly_count);
   return StringFormat("Kelly: %.2f%% (frac=%.0f%% n=%d)",
                       g_v10_kelly_fraction * 100, V10_KellyFraction * 100, g_v10_kelly_count);
}

//+------------------------------------------------------------------+
//| V10 Gold Power Zones                                             |
//| Peak institutional gold activity hours (UTC)                    |
//+------------------------------------------------------------------+
int V10_GoldPowerZoneBonus()
{
   if(!V10_UseGoldPowerZones) return 0;
   
   MqlDateTime utc;
   TimeGMT(utc);
   int h = utc.hour;
   int m = utc.min;
   
   if(h == 10) return 2;                          // London AM Fix
   if(h == 13 || (h == 14 && m < 30)) return 3;  // NY open power hour
   if(h == 15 && m < 30) return 2;                // London PM Fix
   if(h == 17 && m < 30) return 1;                // Comex close
   
   return 0;
}

//+------------------------------------------------------------------+
//| V10 DXY Inverse Hard Gate for Gold                                |
//| Strong USD blocks gold BUY, weak USD blocks gold SELL           |
//+------------------------------------------------------------------+
bool V10_DXYHardGate(string goldDirection, string dxySignal, int dxyScore)
{
   if(!V10_UseDXYHardGate) return true;
   
   if(goldDirection == "BUY" && dxySignal == "BUY" && dxyScore >= 2)
      return false;
   if(goldDirection == "SELL" && dxySignal == "SELL" && dxyScore <= -2)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| V10 Session-Aware Spread Limits for XAUUSD                       |
//+------------------------------------------------------------------+
double V10_SessionMaxSpread()
{
   if(!V10_SessionSpread) return 0;
   
   MqlDateTime utc;
   TimeGMT(utc);
   int h = utc.hour;
   
   if(h >= 0 && h < 7) return 0.80;   // Asia
   if(h >= 7 && h < 8) return 0.50;   // London open
   if(h >= 8 && h < 12) return 0.35;  // London prime
   if(h >= 12 && h < 16) return 0.30; // London/NY overlap
   if(h >= 16 && h < 21) return 0.50; // NY afternoon
   return 0.70;                        // Late NY / pre-Asia
}

//+------------------------------------------------------------------+
//| V10 Summary String for HUD                                      |
//+------------------------------------------------------------------+
string V10_StatusString()
{
   string metaStr = "OFF";
   if(V10_UseMetaConfidence)
   {
      double acc = (g_v10_meta_trades > 0) ? (double)g_v10_meta_correct / g_v10_meta_trades : 0;
      metaStr = StringFormat("ON (%.0f%% n=%d)", acc * 100, g_v10_meta_trades);
   }
   
   return StringFormat(
      "V10 | W: CAT=%.0f%% XGB=%.0f%% LGB=%.0f%% | Trail=%s | StructSL=%s | Persist=%d | Meta=%s",
      g_v10_wCAT * 100, g_v10_wXGB * 100, g_v10_wLGB * 100,
      (V10_UseChandelierTrail ? "CHAND" : "LINEAR"),
      (V10_UseStructureSL ? "SWING" : "ATR"),
      V10_SignalPersistence,
      metaStr
   );
}

//+------------------------------------------------------------------+
//| V10 IsTester helper                                              |
//+------------------------------------------------------------------+
bool V10_IsTester() { return g_v10_is_tester; }

#endif // V10_STRATEGY_MQH
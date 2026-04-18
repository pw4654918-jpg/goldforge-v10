//+------------------------------------------------------------------+
//|                                 TradingAI_Predictor_EA_v10.mq5   |
//|                     Copyright 2026, AI Trading Systems           |
//|   V10.0 — Optuna-50 77-Feature Ensemble + Meta-Confidence EA    |
//+------------------------------------------------------------------+
//| Architecture V10:                                                  |
//|   3-Model ONNX Ensemble (CAT 31.2% + XGB 52.7% + LGB 16.1%)    |
//|   77 Features: V9 base + 10 SMC/microstructure                   |
//|   Meta-Confidence Gate: L2 Logistic, 14 features, 62.6% OOS   |
//|   Regime-Adaptive Triple Barriers: Low/Normal/High vol          |
//|   Structure-aware SL, Chandelier trailing, dynamic partials       |
//|   Adaptive ensemble drift, confidence-scaled lot sizing           |
//|   Kelly Criterion lot sizing, DXY hard gate, gold power zones    |
//|   ICT/SMC: BOS, CHoCH, FVG, OB, Liquidity Sweeps                |
//|   Session-adaptive: Asia/London/NY with enhanced overlap bonus    |
//|   OOS: PF=9.79 WR=80.9% 737 trades (14 days, Optuna-50 tuned)  |
//+------------------------------------------------------------------+
#property copyright "AI Trading Systems"
#property link      "https://github.com/ZiadFrancis"
#property version   "10.00"
#property description "V10 XAUUSD EA - Optuna-50 Meta-Confidence AI + ICT/SMC"
#property description "V10 AI: PF 9.79 | WR 80.9% | 77 features | Regime-Adaptive Barriers"
#property description "Meta-Confidence Gate | Chandelier Trail | Adaptive Drift | Kelly"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Canvas\Canvas.mqh>
#include <WebhookSender.mqh>
#include <onnx_features_v10.mqh>
#include <v10_strategy.mqh>
// WebhookSender: InsForge POST .../functions/webhook-mt5 + Bearer WebhookSecret.
// LP() -> Experts + buffer -> expert_logs batch (heartbeat + LogSendIntervalSec).
// Paths: WebhookTradeOpen/Close/Modify/PartialClose, WebhookHeartbeat, WebhookAlert, WebhookSendLogs.

//+------------------------------------------------------------------+
//| XAUUSD-SPECIFIC CONSTANTS                                        |
//| Gold = 1 pip is $0.01 | 1 lot = 100 troy oz                     |
//| Typical spread: 20-40 points | ATR H1: 5-15 | ATR D1: 20-50     |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "â•â•â• GENERAL â•â•â•"
input int      MagicNumber           = 202603;    // Magic Number
input double   LotSize               = 0.10;      // Fixed Lot (FundYourFX $10K challenge)
input bool     UseDynamicLots        = true;      // Dynamic lot sizing

input group "â•â•â• RISK MANAGEMENT â•â•â•"
input double   RiskPerTradePct           = 0.70;      // Challenge-safe default risk per trade
input double   MaxDailyLossPct       = 2.8;       // Hard daily stop under 4% prop limit
input double   MaxDailyDrawdownPct        = 5.5;       // Hard equity stop under 6% prop limit
input int      MaxConcurrentTrades   = 2;         // Max simultaneous positions
input double   PartialClosePercent   = 33.0;      // % volume to close at TP1 (33% keeps 67% running)
input double   BreakevenATR          = 1.0;       // Move SL to breakeven after ATR * this
input int      MaxConsecLosses       = 2;         // Pause trading after N consecutive losses
input bool     UseConfidenceScaling  = true;      // Scale R:R and lots by AI confidence

input group "â•â•â• SL/TP (ATR-Based) â•â•â•"
input double   SL_ATR_Mult           = 2.0;       // SL distance = ATR * this (wider for crash protection)
input double   TP1_ATR_Mult          = 3.0;       // TP1 = ATR * this (1.5:1 RR with SL=2.0)
input double   TP2_ATR_Mult          = 3.5;       // TP2 = ATR * this (after partial)
input double   TrailStart_ATR        = 1.5;       // Start trailing after profit > ATR * this
input double   TrailStep_ATR         = 0.5;        // Trail step = ATR * this
input int      ATR_Period            = 14;        // ATR period

input group "â•â•â• SESSIONS (Server Time Hours) â•â•â•"
input bool     UseSessionFilter      = true;      // true = Killzone filter (London/NY/Asia only)
input bool     TradeAsia             = true;      // Trade Asian session
input int      AsiaStart             = 0;         // Asia start hour
input int      AsiaEnd               = 7;         // Asia end hour
input bool     TradeLondon           = true;      // Trade London session
input int      LondonStart           = 7;         // London start hour
input int      LondonEnd             = 12;        // London end hour
input bool     TradeNY               = true;      // Trade NY session
input int      NYStart               = 12;        // NY start hour
input int      NYEnd                 = 21;        // NY end hour
input bool     UseWeekendGap         = true;      // Sunday open gap trading
input double   GapMinDollars         = 5.0;       // Min gap size in $ for gap trade
input bool     FridayCloseAll        = true;      // Close positions Friday 16:00
input int      AsiaBonus             = 1;         // Confluence bonus during strong Asia momentum

input group "â•â•â• CONFLUENCE â•â•â•"
input int      MinScore              = 8;         // Challenge mode: favor quality over frequency
input int      RSI_Period            = 21;        // RSI period (21 optimal for gold)
input int      EMA_Fast             = 21;        // Fast EMA
input int      EMA_Slow             = 55;        // Slow EMA
input int      EMA_Trend            = 200;       // Trend EMA (H4)
input int      ADX_Period           = 14;        // ADX period
input int      ADX_MinTrend         = 25;        // ADX trending threshold (25 for gold)


input group "â•â•â• SAFETY â•â•â•"
input double   MaxSpreadDollars      = 0.60;      // Tighter spread gate for prop protection
input int      Slippage              = 20;        // Max slippage points
input bool     Debug                 = false;     // Print debug info (false for fast backtests)
input int      MaxDailyTrades        = 4;         // Lower churn for challenge consistency
input double   DailyProfitTargetPct  = 2.5;       // Bank days sooner to protect equity curve

input group "â•â•â• PROP CHALLENGE GUARDRAILS â•â•â•"
input bool     EnablePropChallengeMode = true;    // Enable strict challenge protections
input double   PropDailyLossSoftStopPct = 2.2;    // Soft stop before hard daily breaker
input double   PropOverallDDSoftStopPct = 4.8;    // Soft stop before hard drawdown breaker
input int      PropMaxTradesPerDay      = 3;      // Extra cap in challenge mode
input double   PropMinSignalConfidence  = 0.50;   // V10: Meta-confidence gate handles filtering
input double   PropMaxRiskPerTradePct   = 0.80;   // Hard cap for dynamic sizing
input int      PropPauseAfterLosses     = 2;      // Cool-off trigger before daily meltdown
input bool     PropUseSessionLock       = true;   // Only trade during challenge prime hours
input int      PropSessionStartUTC      = 8;      // Session lock start hour UTC
input int      PropSessionEndUTC        = 21;     // Session lock end hour UTC

input group "â•â•â• NEWS FILTER â•â•â•"
input bool     UseNewsFilter         = true;       // Pause trading near high-impact news
input int      NewsMinutesBefore     = 15;         // Minutes before event to stop
input int      NewsMinutesAfter      = 15;         // Minutes after event to resume

//+------------------------------------------------------------------+
//| GLOBALS                                                           |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  posInfo;
CSymbolInfo    symInfo;
CAccountInfo   accInfo;

// Indicator handles
int h_rsi, h_adx, h_atr, h_atr_h4;
int h_ema_fast, h_ema_slow, h_ema_trend;
int h_bb;

// V6.1: VWAP + RSI Divergence features
double g_daily_vwap = 0;            // Current daily VWAP level
int    g_rsi_divergence = 0;        // 1=bullish, -1=bearish, 0=none
bool   g_vwap_bullish = false;      // Price above VWAP

// V6.1b: Fibonacci EMAs + Stochastic + Waddah Attar Explosion (from directory audit)
int h_fib_ema8 = INVALID_HANDLE, h_fib_ema13 = INVALID_HANDLE;
int h_fib_ema34 = INVALID_HANDLE, h_fib_ema55 = INVALID_HANDLE;
int h_fib_ema89 = INVALID_HANDLE, h_fib_ema144 = INVALID_HANDLE;
int h_fib_ema233 = INVALID_HANDLE;
int h_stoch = INVALID_HANDLE;         // Stochastic oscillator
int h_macd_wae = INVALID_HANDLE;      // MACD for WAE calculation
int h_atr_wae = INVALID_HANDLE;       // ATR for WAE dead zone
int h_macro_rsi_xau = INVALID_HANDLE;
int h_macro_bb_xau = INVALID_HANDLE;
int h_macro_rsi_eur = INVALID_HANDLE;
int h_macro_rsi_jpy = INVALID_HANDLE;
int h_macro_rsi_btc = INVALID_HANDLE;

// V6.1b State tracking
bool   g_wae_explosion = false;       // Waddah Attar explosion detected
double g_wae_trend_strength = 0;      // WAE trend power
int    g_fib_ema_alignment = 0;       // How many Fib EMAs are stacked (0-7)
bool   g_stoch_oversold = false;      // Stochastic < 20
bool   g_stoch_overbought = false;    // Stochastic > 80
double g_equity_target_pct = 3.0;     // Daily equity target % 
double g_equity_stop_pct = 5.0;       // Daily equity stop %

// ONNX AI Model â€” V5 3-Model Ensemble (XGB+LGB+CatBoost)
long h_xgb = INVALID_HANDLE;
long h_lgb = INVALID_HANDLE;
long h_cat = INVALID_HANDLE;
long h_macro_dxy = INVALID_HANDLE;
long h_macro_regime = INVALID_HANDLE;
long h_macro_orb = INVALID_HANDLE;
int g_macro_loaded = 0;
#define N_ONNX_FEATURES 77

// Resolved intermarket symbol names (set in OnInit)
string sym_XAGUSD = "";
string sym_USDJPY = "";
string sym_EURUSD = "";
string sym_BTCUSD = "";
// V10 ONNX model filenames are defined in v10_strategy.mqh (V10_ONNXModelXGB/LGB/CAT)
// Note: V10 ensemble weights are in v10_strategy.mqh (V10_WeightXGB/LGB/CAT inputs)
input bool     UseONNX               = true;                  // V10 Ensemble — primary brain
input bool     UseONNXEnsemble       = true;                  // Use all 3 models (false=XGB only)

// State tracking
datetime       lastBarTime  = 0;
double         dayStartEquity = 0;

// â”€â”€â”€ Trade Gate Diagnostics â”€â”€â”€
bool   g_gate_spread     = true;   // true = pass
bool   g_gate_session    = true;
bool   g_gate_news       = true;
bool   g_gate_friday     = true;
bool   g_gate_maxpos     = true;
bool   g_gate_dailycap   = true;
bool   g_gate_losspause  = true;
bool   g_gate_ai         = true;   // AI not HOLD
bool   g_gate_confluence = true;   // Score >= MinScore
bool   g_gate_prop       = true;   // Prop challenge guardrail pass
string g_gate_primary    = "READY"; // Primary blocking reason
string g_gate_spread_detail = "";
string g_gate_dailycap_detail = "";  // V7.2: Shows actual count e.g. "3/8"
string g_gate_ai_detail     = "";
string g_gate_score_detail  = "";
string g_gate_prop_detail   = "";
double         peakEquity = 0;
bool           dailyLimitHit = false;
bool           emergencyStop = false;
int            todayDay = 0;
int            totalTradesOpened = 0;
int            totalTradesClosed = 0;
double         totalPnL = 0;
double         todayClosedPnL = 0;  // Closed-deal P&L today
int            todayTradeCount = 0; // Trades opened today
bool           profitTargetHit = false; // Daily profit target reached
int            lastBuyScore = 0;   // Persist for HUD between bars
int            lastSellScore = 0;
int            tickCount = 0;       // Live tick counter
datetime       lastHUDUpdate = 0;   // For "last updated" display
int            lastDealCount = 0;   // Track deal history for OnTradeTransaction
datetime       lastM5BarTime = 0;   // V4.2: Track M5 bars for ONNX inference
int            consecLosses = 0;    // Phase A: Consecutive loss counter
bool           lossPauseActive = false; // Phase A: Pause after N consecutive losses
string         tradingStatus = "Starting..."; // Why EA is/isn't trading â€” shown on HUD

// V6: Real-time brain + M15 entry
double         lastInferencePrice = 0;   // Price when ONNX last ran
double         cachedATR14 = 0;          // Cached H1 ATR for throttle
datetime       lastATRUpdate = 0;        // When ATR was last refreshed
datetime       lastM15BarTime_V6 = 0;    // Track fresh M15 bars for entries
int            onnxRunCount = 0;         // ONNX runs this session (debug)

// V5 AI Signal (expanded with DXY, ORB, regime)
struct AISignal {
   string signal;
   double confidence;
   datetime timestamp;
   bool isValid;
   // V5 fields
   string dxy;           // DXY gold signal: BUY/SELL/HOLD
   int    dxy_score;     // DXY score: -2 to +2
   string regime;        // Market regime: TRENDING/RANGING/STRONG_TREND
   string orb_dir;       // ORB breakout direction: BUY/SELL or ""
   double orb_tp;        // ORB take profit level
   double orb_sl;        // ORB stop loss level
   string version;       // Signal version: "5.0-V5"
};
AISignal aiSig;

// ZipMapProb: Still needed for V6 macro regime model (uses ZipMap output)
struct ZipMapProb
{
   long   key[];
   float  value[];
};

// Trade tracking for partial close
struct TradeRecord {
   ulong  ticket;
   bool   partialDone;
   double entryPrice;
   double originalSL;
};

//+------------------------------------------------------------------+
//| INITIALIZATION                                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   eaStartTime = TimeCurrent();
   // Validate symbol
   if(!symInfo.Name(_Symbol))
   {
      LP("FATAL: Cannot initialize symbol: ", _Symbol);
      return INIT_FAILED;
   }
   symInfo.Refresh();
   
   // XAUUSD validation â€” warn if not gold
   string sym = _Symbol;
   StringToUpper(sym);
   if(StringFind(sym, "XAU") < 0 && StringFind(sym, "GOLD") < 0)
      LP("WARNING: This EA is optimized for XAUUSD/Gold. Current: ", _Symbol);
   
   // Trade object setup
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);
   trade.SetAsyncMode(false);
   
   // Determine fill type â€” broker dependent
   ENUM_ORDER_TYPE_FILLING fillType = ORDER_FILLING_FOK;
   long fillPolicy = 0;
   if(SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE, fillPolicy))
   {
      if((fillPolicy & SYMBOL_FILLING_IOC) != 0)
         fillType = ORDER_FILLING_IOC;
      else if((fillPolicy & SYMBOL_FILLING_FOK) != 0)
         fillType = ORDER_FILLING_FOK;
      else
         fillType = ORDER_FILLING_RETURN;
   }
   trade.SetTypeFilling(fillType);
   
   // Create indicators â€” all with explicit timeframes
   h_rsi       = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   h_adx       = iADX(_Symbol, PERIOD_H1, ADX_Period);
   h_atr       = iATR(_Symbol, PERIOD_H1, ATR_Period);
   h_ema_fast  = iMA(_Symbol, PERIOD_M15, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_slow  = iMA(_Symbol, PERIOD_M15, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_trend = iMA(_Symbol, PERIOD_H4, EMA_Trend, 0, MODE_EMA, PRICE_CLOSE);
   h_bb        = iBands(_Symbol, PERIOD_M15, 20, 0, 2.0, PRICE_CLOSE);
   h_atr_h4    = iATR(_Symbol, PERIOD_H4, ATR_Period);
   
    // V6.1: VWAP_CALC â€” compute daily VWAP inline (no custom indicator needed)
    // VWAP resets at 00:00 broker time, computed from M5 bars
    
    // V6.1b: Fibonacci EMAs (institutional secret sauce: 8/13/34/55/89/144/233)
    h_fib_ema8   = iMA(_Symbol, PERIOD_M15, 8,   0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema13  = iMA(_Symbol, PERIOD_M15, 13,  0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema34  = iMA(_Symbol, PERIOD_M15, 34,  0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema55  = iMA(_Symbol, PERIOD_M15, 55,  0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema89  = iMA(_Symbol, PERIOD_M15, 89,  0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema144 = iMA(_Symbol, PERIOD_M15, 144, 0, MODE_EMA, PRICE_CLOSE);
    h_fib_ema233 = iMA(_Symbol, PERIOD_M15, 233, 0, MODE_EMA, PRICE_CLOSE);
    
    // V6.1b: Stochastic oscillator (5,3,3 â€” optimized for M15 gold scalping)
    h_stoch = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
    
    // V6.1b: MACD + ATR for Waddah Attar Explosion calculation
    h_macd_wae = iMACD(_Symbol, PERIOD_M15, 5, 13, 5, PRICE_CLOSE);
    h_atr_wae  = iATR(_Symbol, PERIOD_M15, 100);  // 100-bar ATR for WAE dead zone
    
    h_macro_rsi_xau = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
    h_macro_bb_xau  = iBands(_Symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
    if(sym_EURUSD != "") h_macro_rsi_eur = iRSI(sym_EURUSD, PERIOD_H1, 14, PRICE_CLOSE);
    if(sym_USDJPY != "") h_macro_rsi_jpy = iRSI(sym_USDJPY, PERIOD_H1, 14, PRICE_CLOSE);
    if(sym_BTCUSD != "") h_macro_rsi_btc = iRSI(sym_BTCUSD, PERIOD_H1, 14, PRICE_CLOSE);
    
    if(h_fib_ema8 == INVALID_HANDLE || h_stoch == INVALID_HANDLE || h_macd_wae == INVALID_HANDLE)
       LP("WARNING: Some V6.1b indicators failed to initialize â€” features will be skipped");
    
    // Set equity target from input parameter
    g_equity_target_pct = DailyProfitTargetPct;
    
   
   if(h_rsi == INVALID_HANDLE || h_adx == INVALID_HANDLE ||
      h_atr == INVALID_HANDLE || h_ema_fast == INVALID_HANDLE ||
      h_ema_slow == INVALID_HANDLE || h_ema_trend == INVALID_HANDLE ||
      h_bb == INVALID_HANDLE || h_atr_h4 == INVALID_HANDLE)
   {
      LP("FATAL: Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   // Initialize tracking
    dayStartEquity = accInfo.Balance();    // Use balance, not equity (excludes floating P&L)
    peakEquity = MathMax(dayStartEquity, accInfo.Equity());
   MqlDateTime initDt;
   TimeCurrent(initDt);
   todayDay = 0;
   dailyLimitHit = false;
   emergencyStop = false;
   
   // V5: Load 3-model ONNX ensemble for native inference
   if(UseONNX)
   {
      ulong inputShape[] = {1, N_ONNX_FEATURES};
      ulong outShape1[] = {1};
      ulong outShape2[] = {1, 3};
      
      // V10: Load ONNX with fallback to Common folder (for Strategy Tester)
      h_xgb = OnnxCreate(V10_ONNXModelXGB, ONNX_DEFAULT);
      if(h_xgb == INVALID_HANDLE) h_xgb = OnnxCreate(V10_ONNXModelXGB, ONNX_COMMON_FOLDER);
      if(h_xgb != INVALID_HANDLE)
      {
         OnnxSetInputShape(h_xgb, 0, inputShape);
         OnnxSetOutputShape(h_xgb, 0, outShape1);
         OnnxSetOutputShape(h_xgb, 1, outShape2);
         LP("âœ V10 XGBoost loaded: ", V10_ONNXModelXGB);
      }
      else LP("âš ï¸ XGBoost ONNX not found: ", V10_ONNXModelXGB);
      
      if(UseONNXEnsemble)
      {
         h_lgb = OnnxCreate(V10_ONNXModelLGB, ONNX_DEFAULT);
         if(h_lgb == INVALID_HANDLE) h_lgb = OnnxCreate(V10_ONNXModelLGB, ONNX_COMMON_FOLDER);
         if(h_lgb != INVALID_HANDLE)
         {
            OnnxSetInputShape(h_lgb, 0, inputShape);
            OnnxSetOutputShape(h_lgb, 0, outShape1);
            OnnxSetOutputShape(h_lgb, 1, outShape2);
            LP("âœ V10 LightGBM loaded: ", V10_ONNXModelLGB);
         }
         else LP("âš ï¸ LightGBM ONNX not found: ", V10_ONNXModelLGB);
         
         h_cat = OnnxCreate(V10_ONNXModelCAT, ONNX_DEFAULT);
         if(h_cat == INVALID_HANDLE) h_cat = OnnxCreate(V10_ONNXModelCAT, ONNX_COMMON_FOLDER);
         if(h_cat != INVALID_HANDLE)
         {
            OnnxSetInputShape(h_cat, 0, inputShape);
            OnnxSetOutputShape(h_cat, 0, outShape1);
            OnnxSetOutputShape(h_cat, 1, outShape2);
            LP("âœ V10 CatBoost loaded: ", V10_ONNXModelCAT);
         }
         else LP("âš ï¸ CatBoost ONNX not found: ", V10_ONNXModelCAT);
      }
      
      int loaded = (h_xgb != INVALID_HANDLE ? 1 : 0) + 
                   (h_lgb != INVALID_HANDLE ? 1 : 0) + 
                   (h_cat != INVALID_HANDLE ? 1 : 0);
      LP("🧠 V10 Ensemble: ", loaded, "/3 models loaded");
   
   // V6: Load Macro Brain ONNX models (DXY/Regime/ORB)
   {
             ulong macroIn[] = {1, 32};
       ulong macroOut1[] = {1};
       ulong macroOut2[] = {1, 3};
       
       h_macro_dxy = OnnxCreate("v6_macro_dxy_dir.onnx", ONNX_DEFAULT);
       if(h_macro_dxy == INVALID_HANDLE) h_macro_dxy = OnnxCreate("v6_macro_dxy_dir.onnx", ONNX_COMMON_FOLDER);
       if(h_macro_dxy != INVALID_HANDLE)
       {
          OnnxSetInputShape(h_macro_dxy, 0, macroIn);
          OnnxSetOutputShape(h_macro_dxy, 0, macroOut1);
          OnnxSetOutputShape(h_macro_dxy, 1, macroOut2);
          LP("✅ V10 Macro DXY loaded");
       }
       
       h_macro_regime = OnnxCreate("v6_macro_regime.onnx", ONNX_DEFAULT);
       if(h_macro_regime == INVALID_HANDLE) h_macro_regime = OnnxCreate("v6_macro_regime.onnx", ONNX_COMMON_FOLDER);
       if(h_macro_regime != INVALID_HANDLE)
       {
          OnnxSetInputShape(h_macro_regime, 0, macroIn);
          OnnxSetOutputShape(h_macro_regime, 0, macroOut1);
           // ZipMap Model Shape
          LP("✅ V10 Macro Regime loaded");
       }
       
       h_macro_orb = OnnxCreate("v6_macro_orb_dir.onnx", ONNX_DEFAULT);
       if(h_macro_orb == INVALID_HANDLE) h_macro_orb = OnnxCreate("v6_macro_orb_dir.onnx", ONNX_COMMON_FOLDER);
       if(h_macro_orb != INVALID_HANDLE)
       {
          OnnxSetInputShape(h_macro_orb, 0, macroIn);
          OnnxSetOutputShape(h_macro_orb, 0, macroOut1);
          OnnxSetOutputShape(h_macro_orb, 1, macroOut2);
          LP("✅ V10 Macro ORB loaded");
       }
      
      g_macro_loaded = (h_macro_dxy != INVALID_HANDLE ? 1 : 0) +
                       (h_macro_regime != INVALID_HANDLE ? 1 : 0) +
                       (h_macro_orb != INVALID_HANDLE ? 1 : 0);
      LP("🧠 V10 Macro Brain: ", g_macro_loaded, "/3 models loaded");
   }
   }
   
   // Resolve intermarket symbols (broker may use different naming)
   sym_XAGUSD = ResolveSymbol("XAGUSD");
   sym_USDJPY = ResolveSymbol("USDJPY");
   sym_EURUSD = ResolveSymbol("EURUSD");
   sym_BTCUSD = ResolveSymbol("BTCUSD");
   LP("ðŸ“Š Intermarket: XAG=", (sym_XAGUSD != "" ? sym_XAGUSD : "N/A"),
         " JPY=", (sym_USDJPY != "" ? sym_USDJPY : "N/A"),
         " EUR=", (sym_EURUSD != "" ? sym_EURUSD : "N/A"),
         " BTC=", (sym_BTCUSD != "" ? sym_BTCUSD : "N/A"));
    todayDay = initDt.day;
    aiSig.isValid = false;
    
    // Print startup info
   LP("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
   LP("  TradingAI Predictor EA v10.0 - Meta-Confidence Ensemble");
   LP("  Symbol: ", _Symbol, " | Digits: ", symInfo.Digits());
   LP("  Point: ", DoubleToString(symInfo.Point(), 6));
   LP("  TickValue: ", DoubleToString(symInfo.TickValue(), 4),
         " | TickSize: ", DoubleToString(symInfo.TickSize(), 6));
   LP("  MinLot: ", symInfo.LotsMin(), " | MaxLot: ", symInfo.LotsMax(),
         " | Step: ", symInfo.LotsStep());
   LP("  Risk: ", RiskPerTradePct, "% | MaxDD: ", MaxDailyDrawdownPct, "%");
   LP("  RSI: ", RSI_Period, " | ADX: ", ADX_MinTrend, " | EMA: ", EMA_Fast, "/", EMA_Slow, "/", EMA_Trend);
   LP("  Sessions: Asia=", TradeAsia, " London=", TradeLondon, " NY=", TradeNY);
   LP("  Confluence threshold: ", MinScore, "/30 (V10 Optuna-50)");
   LP("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
   
   // Timer for always-on HUD (refreshes even when market is closed)
   EventSetTimer(1);
   
   // Reconstruct trade stats from deal history (survives restarts)
   ReconstructTradeHistory();
   
   // V10: Initialize adaptive ensemble weights and strategy params
   V10_Init();
   LP("V10 Strategy: ", V10_StatusString());
   
   // Draw initial HUD immediately
   UpdateHUD(0, 0, 0, 0);
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| DEINITIALIZATION                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(h_rsi != INVALID_HANDLE)       IndicatorRelease(h_rsi);
   if(h_adx != INVALID_HANDLE)       IndicatorRelease(h_adx);
   if(h_atr != INVALID_HANDLE)       IndicatorRelease(h_atr);
   if(h_ema_fast != INVALID_HANDLE)  IndicatorRelease(h_ema_fast);
   if(h_ema_slow != INVALID_HANDLE)  IndicatorRelease(h_ema_slow);
   if(h_ema_trend != INVALID_HANDLE) IndicatorRelease(h_ema_trend);
   if(h_bb != INVALID_HANDLE)        IndicatorRelease(h_bb);
   if(h_atr_h4 != INVALID_HANDLE)    IndicatorRelease(h_atr_h4);
   if(h_fib_ema8 != INVALID_HANDLE)  IndicatorRelease(h_fib_ema8);
   if(h_fib_ema13 != INVALID_HANDLE) IndicatorRelease(h_fib_ema13);
   if(h_fib_ema34 != INVALID_HANDLE) IndicatorRelease(h_fib_ema34);
   if(h_fib_ema55 != INVALID_HANDLE) IndicatorRelease(h_fib_ema55);
   if(h_fib_ema89 != INVALID_HANDLE) IndicatorRelease(h_fib_ema89);
   if(h_fib_ema144 != INVALID_HANDLE) IndicatorRelease(h_fib_ema144);
   if(h_fib_ema233 != INVALID_HANDLE) IndicatorRelease(h_fib_ema233);
   if(h_stoch != INVALID_HANDLE)     IndicatorRelease(h_stoch);
   if(h_macd_wae != INVALID_HANDLE)  IndicatorRelease(h_macd_wae);
    if(h_atr_wae != INVALID_HANDLE)   IndicatorRelease(h_atr_wae);
    if(h_macro_rsi_xau != INVALID_HANDLE) IndicatorRelease(h_macro_rsi_xau);
    if(h_macro_bb_xau != INVALID_HANDLE)  IndicatorRelease(h_macro_bb_xau);
    if(h_macro_rsi_eur != INVALID_HANDLE) IndicatorRelease(h_macro_rsi_eur);
    if(h_macro_rsi_jpy != INVALID_HANDLE) IndicatorRelease(h_macro_rsi_jpy);
    if(h_macro_rsi_btc != INVALID_HANDLE) IndicatorRelease(h_macro_rsi_btc);
   // V5: Release ONNX ensemble handles
   if(h_xgb != INVALID_HANDLE) OnnxRelease(h_xgb);
   if(h_lgb != INVALID_HANDLE) OnnxRelease(h_lgb);
   if(h_cat != INVALID_HANDLE) OnnxRelease(h_cat);
   if(h_macro_dxy != INVALID_HANDLE) OnnxRelease(h_macro_dxy);
   if(h_macro_regime != INVALID_HANDLE) OnnxRelease(h_macro_regime);
   if(h_macro_orb != INVALID_HANDLE) OnnxRelease(h_macro_orb);
   // V6: Clean chart overlays
   CleanChartOverlays("V10_FVG_");
   CleanChartOverlays("V10_OB_");
   CleanChartOverlays("V10_BOS_");
   CleanChartOverlays("V10_SLTP_");
   DestroyPanel();
   Comment("");
   double winRate = (winsToday + lossesToday) > 0 ? (double)winsToday / (winsToday + lossesToday) * 100.0 : 0;
   LP("EA stopped. Trades opened: ", totalTradesOpened,
         " | Closed: ", totalTradesClosed, " | Net P&L: $", DoubleToString(totalPnL, 2),
         " | WR: ", DoubleToString(winRate, 1), "%");
}

//+------------------------------------------------------------------+
//| RECONSTRUCT â€” Rebuild trade stats from deal history on restart   |
//+------------------------------------------------------------------+
void ReconstructTradeHistory()
{
   // Use TimeCurrent() for "today" â€” deal timestamps are in SERVER time, must match
   MqlDateTime nowDt;
   TimeCurrent(nowDt);
   datetime todayStart = StringToTime(StringFormat("%04d.%02d.%02d 00:00", nowDt.year, nowDt.mon, nowDt.day));
   
   // Select ALL history (use far future to ensure nothing is excluded)
   if(!HistorySelect(0, D'2099.12.31'))
   {
      LP("âš ï¸ RECONSTRUCT: Failed to select deal history");
      return;
   }
   
   // Reset counters before reconstruction
   totalTradesOpened = 0;
   totalTradesClosed = 0;
   totalPnL = 0;
   winsToday = 0;
   lossesToday = 0;
   todayClosedPnL = 0;
   todayTradeCount = 0;
   consecLosses = 0;
   g_pnl_count = 0;
   profitTargetHit = false;
   ArrayInitialize(g_pnl_history, 0);
   
   // Use balance for dayStartEquity fallback (not equity - excludes floating P&L)
   dayStartEquity = accInfo.Balance();
   bool foundDayStart = false;
   
   int totalDeals = HistoryDealsTotal();
   int magicMatched = 0;
   int todayMatched = 0;
   datetime lastDealTime = 0;
   datetime firstDealTime = D'2099.12.31';
   
   for(int i = 0; i < totalDeals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      
      // Only our EA's deals
      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic != MagicNumber) continue;
      
      magicMatched++;
      datetime dealTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      if(dealTime > lastDealTime)  lastDealTime = dealTime;
      if(dealTime < firstDealTime) firstDealTime = dealTime;
      
      // Only closing deals (DEAL_ENTRY_OUT or DEAL_ENTRY_INOUT)
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) 
      {
         // Count entries for totalTradesOpened
         if(entry == DEAL_ENTRY_IN)
         {
            totalTradesOpened++;
            if(dealTime >= todayStart)
               todayTradeCount++;
         }
         continue;
      }
      
      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      
      // Lifetime stats
      totalTradesClosed++;
      totalPnL += profit;
      
      // P&L history for sparkline (last 50)
      if(g_pnl_count < 50)
     {
         g_pnl_history[g_pnl_count++] = profit;
           }
      else
     {
         for(int k = 0; k < 49; k++) g_pnl_history[k] = g_pnl_history[k+1];
         g_pnl_history[49] = profit;
      }
      
      // Today's stats
      if(dealTime >= todayStart)
      {
         todayMatched++;
         todayClosedPnL += profit;
         if(profit >= 0)
         {
            winsToday++;
            consecLosses = 0;
         }
         else
         {
            lossesToday++;
            consecLosses++;
         }
         
         // Reconstruct dayStartEquity from balance before today's first close
         if(!foundDayStart)
         {
            dayStartEquity = accInfo.Balance() - todayClosedPnL;
            foundDayStart = true;
         }
      }
   }
   
   // If we found today's deals, recalculate dayStartEquity properly
   // (subtract ALL today's closed P&L from current balance)
   if(foundDayStart)
   {
      dayStartEquity = accInfo.Balance() - todayClosedPnL;
   }
   
   // Reconstruct peakEquity
   peakEquity = MathMax(dayStartEquity, accInfo.Equity());
   
   // Reconstruct profitTargetHit from closed P&L only
   double closedPnLPct = (dayStartEquity > 0) ? (todayClosedPnL / dayStartEquity * 100.0) : 0;
   if(DailyProfitTargetPct > 0 && closedPnLPct >= DailyProfitTargetPct)
      profitTargetHit = true;
   
   // Diagnostic logging
   double winRate = (winsToday + lossesToday) > 0 ? (double)winsToday / (winsToday + lossesToday) * 100.0 : 0;
   LP("ðŸ”„ RECONSTRUCT: ", totalDeals, " total deals | ", magicMatched, " magic-matched (magic=", MagicNumber, ")");
   LP("   TodayStart: ", TimeToString(todayStart), " | LocalNow: ", TimeToString(TimeLocal()),
         " | ServerNow: ", TimeToString(TimeCurrent()));
   if(magicMatched > 0)
      LP("   DealRange: ", TimeToString(firstDealTime), " â†’ ", TimeToString(lastDealTime),
            " | TodayDeals: ", todayMatched);
   LP("ðŸ”„ LIFETIME: ", totalTradesClosed, " closed | P&L: $", DoubleToString(totalPnL, 2));
   LP("ðŸ“… TODAY: ", winsToday, "W ", lossesToday, "L (",
         DoubleToString(winRate, 1), "%) | Closed P&L: $", DoubleToString(todayClosedPnL, 2),
         " | DayStartEq: $", DoubleToString(dayStartEquity, 2),
         " | Target: ", (profitTargetHit ? "HIT âœ…" : "NO"));
}

//+------------------------------------------------------------------+
//| TIMER â€” Keep HUD alive even when market is closed                |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Check if market is open by comparing last tick time
   MqlTick lastTick;
   bool marketOpen = false;
   if(SymbolInfoTick(_Symbol, lastTick))
   {
      datetime tickAge = TimeCurrent() - lastTick.time;
      marketOpen = (tickAge < 60);  // If last tick was within 60s, market is open
   }
   
   if(!marketOpen)
   {
      tradingStatus = "ðŸ”’ MARKET CLOSED â€” waiting for session open";
      g_gate_session = false;
      g_gate_primary = "MARKET CLOSED";
   }
   
   // Recalculate dailyPnL for display
   double equity = accInfo.Equity();
   double timerDailyPnL = 0;
   if(dayStartEquity > 0)
      timerDailyPnL = (equity - dayStartEquity) / dayStartEquity * 100.0;
   double timerDrawdown = 0;
   if(peakEquity > 0)
      timerDrawdown = (peakEquity - equity) / peakEquity * 100.0;
   
   UpdateHUD(lastBuyScore, lastSellScore, timerDailyPnL, timerDrawdown);
   
   // Flush buffered expert logs (respects LogSendIntervalSec in WebhookSender)
   WebhookSendLogs();
}

//+------------------------------------------------------------------+
//| TRADE TRANSACTION â€” Track wins/losses/PnL from closed deals      |
//| Best practice: OnTradeTransaction > OnTrade for granular tracking |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   // Only process deal additions (trade closures show up as new deals)
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   
   // Get deal details from history
   ulong dealTicket = trans.deal;
   if(dealTicket == 0) return;
   
   // Must select the deal in history first
   if(!HistoryDealSelect(dealTicket)) return;
   
   // Filter: only our EA's deals
   long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
   if(dealMagic != MagicNumber) return;
   
   // Filter: only closing deals (DEAL_ENTRY_OUT)
   long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
   if(dealEntry != DEAL_ENTRY_OUT && dealEntry != DEAL_ENTRY_OUT_BY) return;
   
   // Get P&L including swap and commission
   double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT)
                  + HistoryDealGetDouble(dealTicket, DEAL_SWAP)
                  + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
   
   // Accumulate stats
   totalPnL += profit;
    // V6: Record P&L for sparkline

    if(g_pnl_count < 50)
     {

       g_pnl_history[g_pnl_count++] = profit;
     }
    else
     {


       for(int k = 0; k < 49; k++) g_pnl_history[k] = g_pnl_history[k+1];

       g_pnl_history[49] = profit;

    }

   todayClosedPnL += profit;
   totalTradesClosed++;
   
   // Get trade details for webhook
   double dealPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
   double dealVolume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
   long dealType = HistoryDealGetInteger(dealTicket, DEAL_TYPE);
   string dealDir = (dealType == DEAL_TYPE_BUY) ? "SELL" : "BUY";  // Close deal is opposite
   long posId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
   
   if(profit >= 0)
   {
      winsToday++;
      consecLosses = 0;        // Reset consecutive loss counter on win
      lossPauseActive = false; // Resume trading
   }
   else
   {
      lossesToday++;
      consecLosses++;
      if(consecLosses >= MaxConsecLosses)
      {
         lossPauseActive = true;
         LP("âš ï¸  LOSS PAUSE: ", consecLosses, " consecutive losses â€” pausing new trades");
         // ðŸ“¡ WEBHOOK: Alert on loss pause
         WebhookAlert("LOSS_PAUSE", StringFormat("%d consecutive losses â€” trading paused", consecLosses), "WARNING");
      }
   }
   
   // ðŸ“¡ WEBHOOK: Notify backend of trade close
    // ðŸ“¡ V6: Find entry price for accurate P&L tracking
    double entryPrice = 0;
    if(HistorySelectByPosition(posId))
    {
       for(int d = HistoryDealsTotal()-1; d >= 0; d--)
       {
          ulong dt = HistoryDealGetTicket(d);
          if(HistoryDealGetInteger(dt, DEAL_ENTRY) == DEAL_ENTRY_IN)
          {
             entryPrice = HistoryDealGetDouble(dt, DEAL_PRICE);
             break;
          }
       }
    }
    double profitPips = (dealDir == "BUY") ? (dealPrice - entryPrice) * 100.0 : (entryPrice - dealPrice) * 100.0;
    WebhookTradeClose((ulong)posId, dealDir, entryPrice, dealPrice, profit, profitPips, dealVolume, 0);
    

    // V6 Phase 4: Drift Detection â€” did AI predict correctly?

    g_drift_total++;

    bool predictedCorrect = false;

    // If AI said BUY and trade was profitable BUY, or AI said SELL and trade was profitable SELL

    if((aiSig.signal == "BUY" && dealDir == "BUY" && profit > 0) ||

       (aiSig.signal == "SELL" && dealDir == "SELL" && profit > 0))

       predictedCorrect = true;

    if(predictedCorrect) g_drift_correct++;
    
    // V10: Record trade result for Kelly sizing
    V10_RecordTradeResult(profit);
    
   // V10: Adaptive ensemble drift tracking (per-model accuracy)
   // V10 FIX: Parameter order corrected to match V10_UpdateDrift signature (CAT, XGB, LGB)
   string xgbSig = (g_xgb_probs[0] > g_xgb_probs[2]) ? "BUY" : "SELL";
   string lgbSig = (g_lgb_probs[0] > g_lgb_probs[2]) ? "BUY" : "SELL";
   string catSig = (g_cat_probs[0] > g_cat_probs[2]) ? "BUY" : "SELL";
   V10_UpdateDrift(catSig, xgbSig, lgbSig, dealDir, profit > 0);
    
    // V10: Record meta-confidence outcome for accuracy tracking
    V10_RecordMetaOutcome(predictedCorrect);

    double driftRate = g_drift_total > 0 ? (double)g_drift_correct / g_drift_total * 100.0 : 0;

    if(Debug) LP("ðŸŽ¯ DRIFT: ", (predictedCorrect ? "âœ… CORRECT" : "âŒ MISS"),

                    " | Rate: ", DoubleToString(driftRate, 1), "% (", g_drift_correct, "/", g_drift_total, ")");

    

    // Alert if drift rate drops below 45% (worse than random)

    if(g_drift_total >= 10 && driftRate < 45.0)
     {


       WebhookAlert("DRIFT_WARNING",

          StringFormat("Prediction accuracy %.1f%% (%d/%d) â€” below 45%% threshold",

                       driftRate, g_drift_correct, g_drift_total), "WARNING");

    }

   
   if(Debug)
   {
      double wr = (winsToday + lossesToday) > 0 ? (double)winsToday / (winsToday + lossesToday) * 100.0 : 0;
      LP("ðŸ“Š DEAL CLOSED: $", DoubleToString(profit, 2),
            " | Total: $", DoubleToString(totalPnL, 2),
            " | Today: W:", winsToday, " L:", lossesToday,
            " | WR: ", DoubleToString(wr, 1), "%");
   }
}

//+------------------------------------------------------------------+
//| MAIN TICK FUNCTION                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   // Refresh market data
   symInfo.Refresh();
   symInfo.RefreshRates();
   
   // â”€â”€â”€ Daily Reset â”€â”€â”€
   MqlDateTime nowDt;
   TimeCurrent(nowDt);
   if(nowDt.day != todayDay)
   {
      todayDay = nowDt.day;
      dayStartEquity = accInfo.Equity();
   dailyLimitHit = false;
   emergencyStop = false;
   profitTargetHit = false;
   lossPauseActive = false;
      if(Debug) LP("ðŸ“… New day â€” equity reset to $", DoubleToString(dayStartEquity, 2));
   }
   
   // â”€â”€â”€ Peak equity tracking â”€â”€â”€
   double equity = accInfo.Equity();
   if(equity > peakEquity) peakEquity = equity;
   
   // â”€â”€â”€ CIRCUIT BREAKERS â”€â”€â”€
   double dailyPnL = 0;
   if(dayStartEquity > 0)
      dailyPnL = (equity - dayStartEquity) / dayStartEquity * 100.0;
   
   double drawdown = 0;
   if(peakEquity > 0)
      drawdown = (peakEquity - equity) / peakEquity * 100.0;
   
   // Calculate floating PnL for heartbeat
   floatingPnL = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong position_ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(position_ticket))
      {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            floatingPnL += PositionGetDouble(POSITION_PROFIT);
      }
   }
   
   // â“€â“€â“€ Heartbeat (every HeartbeatIntervalSec) â“€â“€â“€
   if(!V10_IsTester())  // V10: Skip webhooks in Strategy Tester
      WebhookHeartbeat(accInfo.Balance(), accInfo.Equity(), floatingPnL,
                       CountMyPositions(), dailyPnL, drawdown, winsToday, lossesToday,
                       profitTargetHit, onnxRunCount, aiSig.signal, aiSig.confidence,
                       g_adaptive_regime, todayClosedPnL, todayTradeCount,
                       symInfo.Ask()-symInfo.Bid(), tradingStatus);
   
   // â”€â”€â”€ ALWAYS manage existing positions (BE/trailing/partials) even when halted â”€â”€â”€
   ManagePositions();
   
   if(dailyPnL < -MaxDailyLossPct && !dailyLimitHit)
   {
      LP("CIRCUIT BREAKER: Daily loss ", DoubleToString(dailyPnL, 2),
            "% exceeds limit of -", DoubleToString(MaxDailyLossPct, 2), "%");
      CloseAllPositions();
      dailyLimitHit = true;
      // ðŸ“¡ WEBHOOK: Circuit breaker alert
      WebhookAlert("DAILY_LOSS_LIMIT", StringFormat("Daily loss %.2f%% exceeded limit -%.2f%%", dailyPnL, MaxDailyLossPct), "CRITICAL");
   }
   // Daily profit target â€” ONLY on REALIZED (closed) P&L, NOT floating equity
   double closedPnLPct = (dayStartEquity > 0) ? (todayClosedPnL / dayStartEquity * 100.0) : 0;
   if(DailyProfitTargetPct > 0 && closedPnLPct >= DailyProfitTargetPct && !profitTargetHit)
   {
      LP("ðŸŽ¯ PROFIT TARGET: Closed P&L $", DoubleToString(todayClosedPnL, 2),
            " (", DoubleToString(closedPnLPct, 2), "%) >= +",
            DoubleToString(DailyProfitTargetPct, 2), "%");
      profitTargetHit = true;
   }
   if(dailyLimitHit || profitTargetHit)
   {
      tradingStatus = dailyLimitHit ? "â›” DAILY LOSS LIMIT â€” trading halted" : "ðŸŽ¯ PROFIT TARGET HIT â€” done for today";
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown); return;
   }
   

   
   if(drawdown > MaxDailyDrawdownPct && !emergencyStop)
   {
      LP("EMERGENCY STOP: Drawdown ", DoubleToString(drawdown, 2),
            "% exceeds limit of ", DoubleToString(MaxDailyDrawdownPct, 2), "%");
      CloseAllPositions();
      emergencyStop = true;
      // ðŸ“¡ WEBHOOK: Emergency stop alert
      WebhookAlert("EMERGENCY_STOP", StringFormat("Drawdown %.2f%% exceeded limit %.2f%%", drawdown, MaxDailyDrawdownPct), "CRITICAL");
   }
   if(emergencyStop) { tradingStatus = "ðŸš¨ EMERGENCY STOP â€” max drawdown exceeded"; UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown); return; }

   // â”€â”€â”€ Prop challenge soft guardrails (stop before hard breach) â”€â”€â”€
   g_gate_prop = true;
   g_gate_prop_detail = "OK";
   if(EnablePropChallengeMode)
   {
      bool sessionLockFail = false;
      if(PropUseSessionLock)
      {
         // V10: Use actual UTC time, not server time
         MqlDateTime utcDt;
         TimeGMT(utcDt);
         int hUtc = utcDt.hour;
         sessionLockFail = (hUtc < PropSessionStartUTC || hUtc >= PropSessionEndUTC);
      }
      if(dailyPnL <= -PropDailyLossSoftStopPct)
      {
         g_gate_prop = false;
         g_gate_prop_detail = StringFormat("SOFT DAILY %.2f/%.2f", MathAbs(dailyPnL), PropDailyLossSoftStopPct);
      }
      else if(drawdown >= PropOverallDDSoftStopPct)
      {
         g_gate_prop = false;
         g_gate_prop_detail = StringFormat("SOFT DD %.2f/%.2f", drawdown, PropOverallDDSoftStopPct);
      }
      else if(todayTradeCount >= PropMaxTradesPerDay)
      {
         g_gate_prop = false;
         g_gate_prop_detail = StringFormat("PROP CAP %d/%d", todayTradeCount, PropMaxTradesPerDay);
      }
      else if(consecLosses >= PropPauseAfterLosses)
      {
         g_gate_prop = false;
         g_gate_prop_detail = StringFormat("PROP LOSS PAUSE %d", consecLosses);
      }
      else if(sessionLockFail)
      {
         g_gate_prop = false;
         g_gate_prop_detail = StringFormat("UTC %02d OUTSIDE [%02d,%02d)", nowDt.hour, PropSessionStartUTC, PropSessionEndUTC);
      }

      if(!g_gate_prop)
      {
         tradingStatus = "ðŸ›¡ï¸ PROP GUARDRAIL â€” " + g_gate_prop_detail;
         g_gate_primary = "PROP GUARDRAIL";
         UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
         return;
      }
   }
   
   // â”€â”€â”€ AI Signal Source (runs FIRST â€” brain must always stay fresh) â”€â”€â”€
   // V6: ONNX runs on tick when price moves >0.3*ATR (responsive + CPU-efficient)
   if(UseONNX && (h_xgb != INVALID_HANDLE || h_lgb != INVALID_HANDLE || h_cat != INVALID_HANDLE))
   {
      
      // V6: ATR-throttled inference â€” runs when price moves significantly OR on new M5 bar OR timeout
      // Refresh ATR every 60 seconds to avoid CopyBuffer overhead
      if(TimeLocal() - lastATRUpdate > 60 || cachedATR14 == 0)
      {
         double atrBuf[];
         ArrayResize(atrBuf, 1);
         ArraySetAsSeries(atrBuf, true);
         if(CopyBuffer(h_atr, 0, 0, 1, atrBuf) > 0 && atrBuf[0] > 0)
            cachedATR14 = atrBuf[0];
         else
            cachedATR14 = 10.0; // Fallback ATR for XAUUSD
         lastATRUpdate = TimeLocal();
      }
      
      double curBid = symInfo.Bid();
      double priceDelta = MathAbs(curBid - lastInferencePrice);
      double atrThreshold = cachedATR14 * 0.3;
      datetime curM5Bar = iTime(_Symbol, PERIOD_M5, 0);
      bool newM5Bar = (curM5Bar != lastM5BarTime);
      
      // V6: Time-based fallback â€” ONNX must run at least every 60s to keep brain fresh
      long secsSinceOnnx = (long)(TimeGMT() - aiSig.timestamp);
      bool timeoutTrigger = (secsSinceOnnx >= 60);
      
      // Run ONNX if: price moved >0.3*ATR OR new M5 bar OR timeout OR first run
      string trigReason = "";
      if(lastInferencePrice == 0)          trigReason = "FIRST_RUN";
      else if(newM5Bar)                    trigReason = "M5_BAR";
      else if(priceDelta > atrThreshold)   trigReason = "ATR_DELTA";
      else if(timeoutTrigger)              trigReason = "TIMEOUT_60s";
      
      if(trigReason != "")
      {
         if(newM5Bar) lastM5BarTime = curM5Bar;
         lastInferencePrice = curBid;
         onnxRunCount++;
         RunONNXInference();
   DeriveMacroSignals();  // Populate DXY/Regime/ORB from intermarket data
          

          // V6 Phase 4: Fetch adaptive params from InsForge

          FetchAdaptiveParams();

          

          // V6 Phase 4: Signal change detection

          if(g_last_onnx_signal != "" && aiSig.signal != g_last_onnx_signal)
          {
             if(Debug) LP("ðŸ”„ SIGNAL FLIP: ", g_last_onnx_signal, " â†’ ", aiSig.signal,

                             " | Conf: ", DoubleToString(g_last_onnx_conf*100,0), "% â†’ ",

                             DoubleToString(aiSig.confidence*100,0), "%");

             // Notify backend of signal change

             WebhookAlert("SIGNAL_CHANGE", 

                StringFormat("%sâ†’%s conf:%d%%â†’%d%%", 

                   g_last_onnx_signal, aiSig.signal, 

                   (int)(g_last_onnx_conf*100), (int)(aiSig.confidence*100)),

                "INFO");

          }

          g_last_onnx_signal = aiSig.signal;

          g_last_onnx_conf = aiSig.confidence;

         
         if(Debug)
            LP("ðŸ§  ONNX run #", onnxRunCount, 
                  " | trigger=", (newM5Bar ? "M5_BAR" : "ATR_DELTA"),
                  " | delta=$", DoubleToString(priceDelta, 2),
                  " | threshold=$", DoubleToString(atrThreshold, 2));
      }
   }
   
   // â€”â€”â€” V7.2: Keep DAILY CAP gate LIVE on every tick â€”â€”â€”
   // Previously only evaluated on M15 bars â€” caused stale HIT display after daily reset
   g_gate_dailycap = !(MaxDailyTrades > 0 && todayTradeCount >= MaxDailyTrades);
   g_gate_dailycap_detail = StringFormat("%d/%d", todayTradeCount, MaxDailyTrades);
   
   // â“€â“€â“€ Spread Check (V10: session-aware limits for XAUUSD) â“€â“€â“€
   double spreadDollars = symInfo.Ask() - symInfo.Bid();
   double sessionSpreadLimit = V10_SessionMaxSpread();
   double effectiveSpreadLimit = (sessionSpreadLimit > 0) ? sessionSpreadLimit : MaxSpreadDollars;
   g_gate_spread = !(effectiveSpreadLimit > 0 && spreadDollars > effectiveSpreadLimit);
   g_gate_spread_detail = StringFormat("$%.2f / $%.2f", spreadDollars, MaxSpreadDollars);
   if(!g_gate_spread)
   {
      tradingStatus = StringFormat("â ¸ SPREAD HIGH â€” $%.2f > $%.2f limit", spreadDollars, MaxSpreadDollars);
      g_gate_primary = StringFormat("SPREAD $%.2f > $%.2f", spreadDollars, MaxSpreadDollars);
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // (ManagePositions already called at L826 â€” removed duplicate here)
   
   // â”€â”€â”€ V6: Fresh M15 Candle Gate â”€â”€â”€
   // Trade decisions ONLY on first tick of new M15 bar
   datetime currentM15 = iTime(_Symbol, PERIOD_M15, 0);
   if(currentM15 == lastM15BarTime_V6)
   {
      // Not a new M15 bar â€” just update HUD and return
      // Sync gate AI with live ensemble signal (so gate panel matches main HUD)
      bool propConfOkLive = !(EnablePropChallengeMode && aiSig.isValid && aiSig.confidence < PropMinSignalConfidence);
      g_gate_ai = !(aiSig.isValid && aiSig.signal == "HOLD") && propConfOkLive;
      g_gate_ai_detail = aiSig.isValid ? StringFormat("%s %.0f%%", aiSig.signal, aiSig.confidence*100) : "WAITING";
      if(aiSig.isValid && !propConfOkLive)
         g_gate_ai_detail = StringFormat("LOW CONF %.0f%% < %.0f%%", aiSig.confidence*100, PropMinSignalConfidence*100);
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â•â•â• FRESH M15 BAR â€” EVALUATE SETUP â•â•â•
   lastM15BarTime_V6 = currentM15;
   lastBarTime = currentM15;  // Keep legacy tracking in sync
   
   if(Debug)
   {
      MqlDateTime m15Dt;
      TimeToStruct(currentM15, m15Dt);
      LP("â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”");
      LP("â° NEW M15 BAR: ", TimeToString(currentM15, TIME_DATE|TIME_MINUTES));
      LP("â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”");
   }
   
   // â”€â”€â”€ Friday Close â”€â”€â”€
   g_gate_friday = !(FridayCloseAll && nowDt.day_of_week == 5 && nowDt.hour >= 16);
   if(!g_gate_friday)
   {
      tradingStatus = "ðŸ“… FRIDAY CLOSE â€” positions closed for weekend";
      g_gate_primary = "FRIDAY CLOSE";
      CloseAllPositions();
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â”€â”€â”€ Session Filter â”€â”€â”€
   string session = GetSession(nowDt.hour);
   g_gate_session = !(UseSessionFilter && session == "OFF");
   if(!g_gate_session)
   {
      tradingStatus = "ðŸŒ™ OFF SESSION â€” waiting for session open";
      g_gate_primary = "SESSION OFF";
      if(Debug) LP("ðŸ“  Session: OFF | Hour: ", nowDt.hour, " UTC");
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â”€â”€â”€ News Filter â”€â”€â”€
   g_gate_news = !IsHighImpactNews() && !V10_IsFOMCOrNFPBlock();
   if(!g_gate_news)
   {
      tradingStatus = "ðŸ“° NEWS FILTER â€” high-impact event nearby, paused";
      g_gate_primary = "NEWS FILTER";
      if(Debug) LP("ðŸ“ News filter active â€” skipping trade evaluation");
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â”€â”€â”€ Weekend Gap â”€â”€â”€
   if(UseWeekendGap && (nowDt.day_of_week == 0 || (nowDt.day_of_week == 1 && nowDt.hour <= 3)))
   {
      int gap = CheckWeekendGap();
      if(gap != 0 && CountMyPositions() == 0)
      {
         LP("WEEKEND GAP: ", (gap > 0 ? "BULLISH" : "BEARISH"), " â€” entering trade");
         ExecuteTrade(gap > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
      }
   }
   
   // â”€â”€â”€ Max positions check â”€â”€â”€
   g_gate_maxpos = !(CountMyPositions() >= MaxConcurrentTrades);
   if(!g_gate_maxpos)
   {
      tradingStatus = StringFormat("ðŸ“Š MAX POSITIONS â€” %d/%d slots full", CountMyPositions(), MaxConcurrentTrades);
      g_gate_primary = StringFormat("MAX POS %d/%d", CountMyPositions(), MaxConcurrentTrades);
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â”€â”€â”€ Daily trade cap â”€â”€â”€
   g_gate_dailycap = !(MaxDailyTrades > 0 && todayTradeCount >= MaxDailyTrades);
   if(!g_gate_dailycap)
   {
      tradingStatus = StringFormat("ðŸ”¢ DAILY CAP â€” %d/%d trades used", todayTradeCount, MaxDailyTrades);
      g_gate_primary = StringFormat("DAILY CAP %d/%d", todayTradeCount, MaxDailyTrades);
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // â”€â”€â”€ Loss Pause Check â”€â”€â”€
   g_gate_losspause = !lossPauseActive;
   if(!g_gate_losspause)
   {
      tradingStatus = StringFormat("â¸ LOSS PAUSE â€” %d consec losses, cooling off", consecLosses);
      g_gate_primary = StringFormat("LOSS PAUSE %d losses", consecLosses);
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // All pre-gates passed, reset those to pass
   g_gate_spread = true;
   g_gate_session = true;
   g_gate_news = true;
   g_gate_friday = true;
   g_gate_maxpos = true;
   // g_gate_dailycap NOT reset here â evaluated on every tick at L1004
   g_gate_losspause = true;
   g_gate_prop = true;
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   // CONFLUENCE SCORING + V6 DECISION ENGINE
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   tradingStatus = "âœ… ACTIVE â€” scanning for setups";
   int buyScore  = ScoreConfluence(1);
   int sellScore = ScoreConfluence(-1);
   
   // V10: Enhanced session bonus (London/NY overlap +2, Asia +0)
   int bonus = V10_SessionBonus(nowDt.hour);
   
   // V10: Gold power zone bonus (London fix, NY open, PM fix)
   int goldZoneBonus = V10_GoldPowerZoneBonus();
   buyScore += goldZoneBonus;
   sellScore += goldZoneBonus;
   if(bonus > 0)
   {
      double c0 = iClose(_Symbol, PERIOD_M15, 0);
      double c3 = iClose(_Symbol, PERIOD_M15, 3);
      if(c0 > c3) buyScore += bonus;    // Asia momentum is bullish
      else if(c0 < c3) sellScore += bonus; // Asia momentum is bearish
   }
   
   // Save for HUD persistence
   lastBuyScore = buyScore;
   lastSellScore = sellScore;
   
   // V7.1 BUG FIX: Use adaptive MinScore from InsForge backend (was dead code!)
   // g_adaptive_minScore is fetched from backend adaptive_params — previously ignored
   int baseMinScore = (g_adaptive_minScore >= 3) ? (int)g_adaptive_minScore : MinScore;
   int effectiveMinScore = baseMinScore;
   int volRegimeForScore = GetVolatilityRegime();
   
   // Volatility-adaptive threshold (now uses adaptive base, not hardcoded input)
   if(volRegimeForScore < 0)        effectiveMinScore = MathMax(3, baseMinScore - 1);  // LOW vol
    else if(volRegimeForScore > 0)   effectiveMinScore = baseMinScore + 1;              // HIGH vol - tighten

   
   // â”€â”€â”€ Update AI and Confluence gate states â”€â”€â”€
   bool propConfOk = !(EnablePropChallengeMode && aiSig.isValid && aiSig.confidence < PropMinSignalConfidence);
   g_gate_ai = !(aiSig.isValid && aiSig.signal == "HOLD") && propConfOk;
   g_gate_ai_detail = aiSig.isValid ? StringFormat("%s %.0f%%", aiSig.signal, aiSig.confidence*100) : "WAITING";
   if(aiSig.isValid && !propConfOk)
      g_gate_ai_detail = StringFormat("LOW CONF %.0f%% < %.0f%%", aiSig.confidence*100, PropMinSignalConfidence*100);
   
   int bestScore = MathMax(buyScore, sellScore);
   g_gate_confluence = (bestScore >= effectiveMinScore);
   g_gate_score_detail = StringFormat("B:%d S:%d Min:%d", buyScore, sellScore, effectiveMinScore);
   
   // Determine primary block
   if(!g_gate_ai && !g_gate_confluence) g_gate_primary = "AI HOLD + LOW SCORE";
   else if(!g_gate_ai)                  g_gate_primary = StringFormat("AI HOLD %s", g_gate_ai_detail);
   else if(!g_gate_confluence)           g_gate_primary = StringFormat("LOW SCORE %s", g_gate_score_detail);
   else                                  g_gate_primary = "READY";
   
   
   // â”€â”€â”€ V6 COMPREHENSIVE DEBUG LOG â”€â”€â”€
   if(Debug)
   {
      // AI Brain status
      LP("ðŸ§  BRAIN XGB: B=", DoubleToString(g_xgb_probs[0]*100,0), 
            "% N=", DoubleToString(g_xgb_probs[1]*100,0),
            "% S=", DoubleToString(g_xgb_probs[2]*100,0), "%",
            " | LGB: B=", DoubleToString(g_lgb_probs[0]*100,0),
            "% N=", DoubleToString(g_lgb_probs[1]*100,0),
            "% S=", DoubleToString(g_lgb_probs[2]*100,0), "%",
            " | CAT: B=", DoubleToString(g_cat_probs[0]*100,0),
            "% N=", DoubleToString(g_cat_probs[1]*100,0),
            "% S=", DoubleToString(g_cat_probs[2]*100,0), "%");
      
      LP("ðŸ§  ENSEMBLE: ", aiSig.signal, " ", DoubleToString(aiSig.confidence*100,0),
            "% | Threshold: ", effectiveMinScore, "/40",
            " | ONNX runs: ", onnxRunCount);
      
      // Confluence + volatility detail
      string volLabel = (volRegimeForScore < 0 ? "LOW" : (volRegimeForScore > 0 ? "HIGH" : "NORM"));
      LP("ðŸ“ CONFLUENCE: BUY=", buyScore, "/40 SELL=", sellScore, "/40",
            " | Min=", effectiveMinScore, "(base:", MinScore, " vol:", volLabel, ")",
            " | Session=", session, "+", bonus,
            " | Spread=$", DoubleToString(spreadDollars, 2));
      
      // Current position status
      bool hasBuy = HasPosition(POSITION_TYPE_BUY);
      bool hasSell = HasPosition(POSITION_TYPE_SELL);
      LP("ðŸ“Š POSITIONS: ", (hasBuy ? "BUY active" : ""), (hasSell ? "SELL active" : ""),
            (!hasBuy && !hasSell ? "FLAT" : ""),
            " | Today: ", todayTradeCount, " trades | Consec losses: ", consecLosses);
   }
   
   // === V7: HOLD SIGNAL HARD GATE ===
   // BUG FIX: When AI says HOLD, block ALL new trades
   // Previously HOLD was shown on HUD but trades still executed!
   if(!g_gate_ai)
   {
      tradingStatus = "AI HOLD - brain says wait | " + g_gate_ai_detail;
      g_gate_primary = "AI HOLD " + g_gate_ai_detail;
      if(Debug) LP("HOLD GATE: AI=", aiSig.signal, " conf=",
                      DoubleToString(aiSig.confidence*100,0), "% - BLOCKING new trades");
      UpdateHUD(lastBuyScore, lastSellScore, dailyPnL, drawdown);
      return;
   }
   
   // === V7 TRADE DECISION WITH ANTI-HEDGING ===
   bool hasBuyPos = HasPosition(POSITION_TYPE_BUY);
   bool hasSellPos = HasPosition(POSITION_TYPE_SELL);
   
   
   // V10: Require AI signal to AGREE with confluence direction
   bool aiAgreesBuy  = (aiSig.signal == "BUY");
   bool aiAgreesSell = (aiSig.signal == "SELL");
   
   // V10: DXY inverse hard gate (strong USD blocks gold BUY, weak USD blocks gold SELL)
   if(aiAgreesBuy && !V10_DXYHardGate("BUY", aiSig.dxy, aiSig.dxy_score))
   {
      if(Debug) LP("V10 DXY BLOCK: gold BUY blocked by strong USD (DXY=", aiSig.dxy, " score=", aiSig.dxy_score, ")");
      aiAgreesBuy = false;
   }
   if(aiAgreesSell && !V10_DXYHardGate("SELL", aiSig.dxy, aiSig.dxy_score))
   {
      if(Debug) LP("V10 DXY BLOCK: gold SELL blocked by weak USD (DXY=", aiSig.dxy, " score=", aiSig.dxy_score, ")");
      aiAgreesSell = false;
   }
   
   // V10: Meta-confidence was already computed in RunONNXInference
   // and stored in g_v10_meta_conf. If onnxSignal was "HOLD" due to
   // meta-gate, aiSig already reflects that. No need to re-gate here.
   
   if(buyScore >= effectiveMinScore && buyScore > sellScore && aiAgreesBuy)
   {
      if(hasSellPos)
      {
         ClosePositionsByType(POSITION_TYPE_SELL);
         if(Debug) LP("ðŸ“„ FLIP: Closed SELL to open BUY (brain+confluence aligned)");
      }
      
      if(!HasPosition(POSITION_TYPE_BUY))
      {
         if(Debug) LP("âš¡ DECISION: BUY | Score=", buyScore, "/40 | AI=", aiSig.signal,
                         " ", DoubleToString(aiSig.confidence*100,0), "%");
         ExecuteTrade(ORDER_TYPE_BUY);
      }
      else
      {
         if(Debug) LP("âš¡ BUY signal but position already active");
         tradingStatus = "BUY HOLD — position active, signal agrees";
      }
   }
   else if(sellScore >= effectiveMinScore && sellScore > buyScore && aiAgreesSell)
   {
      if(hasBuyPos)
      {
         ClosePositionsByType(POSITION_TYPE_BUY);
         if(Debug) LP("ðŸ“„ FLIP: Closed BUY to open SELL (brain+confluence aligned)");
      }
      
      if(!HasPosition(POSITION_TYPE_SELL))
      {
         if(Debug) LP("âš¡ DECISION: SELL | Score=", sellScore, "/40 | AI=", aiSig.signal,
                         " ", DoubleToString(aiSig.confidence*100,0), "%");
         ExecuteTrade(ORDER_TYPE_SELL);
      }
      else
      {
         if(Debug) LP("âš¡ SELL signal but position already active");
         tradingStatus = "SELL HOLD — position active, signal agrees";
      }
   }
   else
   {
      // No signal met threshold
      if(Debug) LP("âš¡ DECISION: NO_TRADE | BUY=", buyScore, " SELL=", sellScore,
                      " < Min=", effectiveMinScore,
                      " | AI=", aiSig.signal, " ", DoubleToString(aiSig.confidence*100,0), "%");
      tradingStatus = StringFormat("âœ… SCANNING â€” BUY:%d SELL:%d < Min:%d", buyScore, sellScore, effectiveMinScore);
   }
   
   UpdateHUD(buyScore, sellScore, dailyPnL, drawdown);
}

//+------------------------------------------------------------------+
//| CONFLUENCE SCORING â€” 36 possible points (V6.1b Full Arsenal)           |
//| V5: +DXY inverse, +ORB breakout, +regime, upgraded AI tiers       |
//+------------------------------------------------------------------+
int ScoreConfluence(int dir)
{
   int score = 0;
   
   // === 1. H4 EMA 200 Trend (2 pts â€” strong bonus, NOT a hard gate) ===
   // Professional EAs (APEX Confluence) use additive scoring, not kill switches
   double ema200[];
   ArrayResize(ema200, 2);
   ArraySetAsSeries(ema200, true);
   if(CopyBuffer(h_ema_trend, 0, 0, 2, ema200) > 0)
   {
      double price = iClose(_Symbol, PERIOD_H4, 0);
      if(dir == 1 && price > ema200[0])       score += 2;  // H4 trend aligned for BUY
      else if(dir == -1 && price < ema200[0]) score += 2;  // H4 trend aligned for SELL
      // No hard gate â€” AI signal and other factors still count even against H4 trend
   }
   
   // === 2. EMA 21/55 Alignment (1 pt) + Fresh Cross (1 pt) + EMA Stacking (1 pt) ===
   double ef[], es[];
   ArrayResize(ef, 3); ArrayResize(es, 3);
   ArraySetAsSeries(ef, true);
   ArraySetAsSeries(es, true);
   if(CopyBuffer(h_ema_fast, 0, 0, 3, ef) > 0 && CopyBuffer(h_ema_slow, 0, 0, 3, es) > 0)
   {
      if(dir == 1 && ef[0] > es[0])
      {
         score += 1;
         if(ef[1] <= es[1]) score += 1;  // Fresh cross!
      }
      else if(dir == -1 && ef[0] < es[0])
      {
         score += 1;
         if(ef[1] >= es[1]) score += 1;
      }
   }
   
   // === 3. RSI Zone + Divergence (up to 2 pts) â€” V4: RSI 21, zones 20/80 ===
   double rsi[];
   ArrayResize(rsi, 3);
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(h_rsi, 0, 0, 3, rsi) > 0)
   {
      // V4: Wider zones (80/20) â€” gold has stronger momentum
      if(dir == 1 && rsi[0] < 20) score += 1;  // Deeply oversold
      if(dir == -1 && rsi[0] > 80) score += 1; // Deeply overbought
      
      // Divergence
      double c0 = iClose(_Symbol, PERIOD_M15, 0);
      double c1 = iClose(_Symbol, PERIOD_M15, 1);
      if(dir == 1 && rsi[0] > rsi[1] && c0 < c1) score += 1;
      if(dir == -1 && rsi[0] < rsi[1] && c0 > c1) score += 1;
   }
   
   // === 4. ADX Trending + DI Alignment (1-2 pts) â€” V4: threshold 25, strength bonus ===
   double adxMain[], adxPlus[], adxMinus[];
   ArrayResize(adxMain, 2); ArrayResize(adxPlus, 2); ArrayResize(adxMinus, 2);
   ArraySetAsSeries(adxMain, true);
   ArraySetAsSeries(adxPlus, true);
   ArraySetAsSeries(adxMinus, true);
   if(CopyBuffer(h_adx, 0, 0, 2, adxMain) > 0 &&
      CopyBuffer(h_adx, 1, 0, 2, adxPlus) > 0 &&
      CopyBuffer(h_adx, 2, 0, 2, adxMinus) > 0)
   {
      if(adxMain[0] > ADX_MinTrend)
      {
         if(dir == 1 && adxPlus[0] > adxMinus[0]) score += 1;
         if(dir == -1 && adxMinus[0] > adxPlus[0]) score += 1;
         // V4: Strong trend bonus (ADX > 35)
         if(adxMain[0] > 35) score += 1;
      }
   }
   
   // === 5. Fair Value Gap (1 pt) ===
   if(DetectFVG(dir)) score += 1;
   
   // === 6. Order Block (1 pt) ===
   if(DetectOB(dir)) score += 1;
   
   // === 7. Liquidity Sweep (1 pt) ===
   if(DetectSweep(dir)) score += 1;
   
   
   // === 7b. Break of Structure (1-2 pts) - V7 FIX: now scored ===
   {
      int bosDir = DetectBOS();
      if(bosDir == dir) score += 2;
      else if(bosDir == -dir) score -= 1;
   }
   
   // === 7c. Change of Character (1-2 pts) - V7 FIX: now scored ===
   {
      int chochDir = DetectCHoCH();
      if(chochDir == dir) score += 2;
      else if(chochDir == -dir) score -= 1;
   }
   // === 8. AI Signal (3-8 pts â€” AI is PRIMARY decision maker) ===
   // PF=3.09 WF-Optuna validated model with 80 features
   if(UseONNX && aiSig.isValid)
   {
      if(dir == 1 && aiSig.signal == "BUY")
      {
         if(aiSig.confidence > 0.70) score += 8;      // Very high confidence = 8 pts (dominant)
         else if(aiSig.confidence > 0.55) score += 6;  // Strong conviction = 6 pts
         else if(aiSig.confidence > 0.40) score += 4;  // Moderate conviction = 4 pts
         else if(aiSig.confidence > 0.25) score += 3;  // Low conviction = 3 pts
      }
      else if(dir == -1 && aiSig.signal == "SELL")
      {
         if(aiSig.confidence > 0.70) score += 8;      // Symmetric with BUY
         else if(aiSig.confidence > 0.55) score += 6;
         else if(aiSig.confidence > 0.40) score += 4;
         else if(aiSig.confidence > 0.25) score += 3;
      }
   }
   
   // === 10. DXY Inverse Correlation (1-2 pts â€” V5 NEW) ===
   {
      // DXY inverse: weak USD (dxy=SELL) = gold bullish, strong USD (dxy=BUY) = gold bearish
      if(dir == 1 && aiSig.dxy == "SELL")    // Weak USD â†’ gold BUY
      {
         score += 1;
         if(aiSig.dxy_score <= -2) score += 1;  // Strong USD weakness = very gold bullish
      }
      else if(dir == -1 && aiSig.dxy == "BUY")  // Strong USD â†’ gold SELL
      {
         score += 1;
         if(aiSig.dxy_score >= 2) score += 1;
      }
   }
   
   
   // === 12. Regime Bonus (1 pt â€” V5 NEW) ===
   if(UseONNX && aiSig.isValid)
   {
      if(aiSig.regime == "TRENDING") score += 1;       // AI sees strong directional move
      else if(aiSig.regime == "VOLATILE") score += 1;  // High volatility = potential opportunity
   }
   
   // === 9. Bollinger Squeeze Breakout (1 pt) â€” V4 NEW ===
   double bbUpper[], bbLower[], bbMid[];
   ArrayResize(bbUpper, 20); ArrayResize(bbLower, 20); ArrayResize(bbMid, 20);
   ArraySetAsSeries(bbUpper, true); ArraySetAsSeries(bbLower, true); ArraySetAsSeries(bbMid, true);
   if(CopyBuffer(h_bb, 1, 0, 20, bbUpper) > 0 &&
      CopyBuffer(h_bb, 2, 0, 20, bbLower) > 0 &&
      CopyBuffer(h_bb, 0, 0, 20, bbMid) > 0)
   {
      double curWidth = bbUpper[0] - bbLower[0];
      // Check if current width is narrowest in last 20 bars (squeeze)
      bool isSqueeze = true;
      for(int i = 1; i < 20; i++)
      {
         if((bbUpper[i] - bbLower[i]) < curWidth) { isSqueeze = false; break; }
      }
      if(isSqueeze)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(dir == 1 && bid > bbUpper[0]) score += 1;  // Squeeze breakout UP
         if(dir == -1 && bid < bbLower[0]) score += 1;  // Squeeze breakout DOWN
      }
   }
   
   // === 10. London-NY Overlap Bonus (1 pt) â€” V4 NEW ===
   MqlDateTime scoreDt;
   TimeCurrent(scoreDt);
   if(scoreDt.hour >= 15 && scoreDt.hour <= 19)  // London-NY overlap (13-17 UTC = 15-19 EET server)
      score += 1;
   
   // === 13. RSI Divergence (2 pts â€” V6.1 institutional-grade) ===
    g_rsi_divergence = DetectRSIDivergence(10);
    if(dir == 1 && g_rsi_divergence == 1)  score += 2;  // Bullish divergence confirms BUY
    if(dir == -1 && g_rsi_divergence == -1) score += 2;  // Bearish divergence confirms SELL
    
    // === 14. VWAP Bias (2 pts â€” V6.1 institutional VWAP) ===
    g_daily_vwap = ComputeDailyVWAP();
    if(g_daily_vwap > 0)
    {
       double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
       g_vwap_bullish = (bid > g_daily_vwap);
       if(dir == 1 && bid > g_daily_vwap)   score += 2;  // Above VWAP = bullish bias confirmed
       if(dir == -1 && bid < g_daily_vwap)  score += 2;  // Below VWAP = bearish bias confirmed
    }
    
    // === 15. Waddah Attar Explosion (2 pts â€” V6.1b dead market filter) ===
    int waeDir = 0;
    bool waeActive = ComputeWaddahAttar(waeDir);
    if(waeActive && g_wae_explosion)
    {
       if(dir == 1 && waeDir == 1)    score += 2;  // Bullish explosion
       if(dir == -1 && waeDir == -1)  score += 2;  // Bearish explosion
    }
    
    // === 16. Fibonacci EMA Stack (2 pts â€” V6.1b institutional EMAs) ===
    int fibStack = CountFibEMAStack(dir);
    if(fibStack >= 4) score += 2;       // 4+ Fib EMAs aligned = strong stack
    else if(fibStack >= 2) score += 1;  // 2-3 = partial alignment
    
    // === 17. Stochastic Confirmation (2-3 pts â€” V6.1b overbought/oversold) ===
    if(h_stoch != INVALID_HANDLE)
    {
       double stochK[], stochD[];
       ArraySetAsSeries(stochK, true);
       ArraySetAsSeries(stochD, true);
       if(CopyBuffer(h_stoch, 0, 1, 2, stochK) > 0 && CopyBuffer(h_stoch, 1, 1, 2, stochD) > 0)
       {
          g_stoch_oversold = (stochK[0] < 20);
          g_stoch_overbought = (stochK[0] > 80);
          
          // Bullish: stoch in oversold zone + K crossing above D
          if(dir == 1 && stochK[0] < 30 && stochK[0] > stochD[0]) score += 2;
          // Bearish: stoch in overbought zone + K crossing below D 
          if(dir == -1 && stochK[0] > 70 && stochK[0] < stochD[0]) score += 2;
          // Bonus: extreme oversold/overbought
          if(dir == 1 && stochK[0] < 10) score += 1;
           if(dir == -1 && stochK[0] > 90) score += 1;
        }
     }
   // â”€â”€â”€ V7: XAG/Gold Correlation Monitor â”€â”€â”€
   if(sym_XAGUSD != "")
   {
      double xagClose[], xauClose[];
      if(CopyClose(sym_XAGUSD, PERIOD_H1, 0, 20, xagClose) == 20 &&
         CopyClose(_Symbol, PERIOD_H1, 0, 20, xauClose) == 20)
      {
         // Compute ratio and check divergence from mean
         double ratioSum = 0, ratioMean = 0;
         double ratios[];
         ArrayResize(ratios, 20);
         for(int r=0; r<20; r++) { ratios[r] = (xagClose[r] > 0) ? xauClose[r] / xagClose[r] : 0; ratioSum += ratios[r]; }
         ratioMean = ratioSum / 20.0;
         
         // Standard deviation
         double ratioVar = 0;
         for(int r=0; r<20; r++) ratioVar += MathPow(ratios[r] - ratioMean, 2);
         double ratioStd = MathSqrt(ratioVar / 20.0);
         double currentRatio = (xagClose[19] > 0) ? xauClose[19] / xagClose[19] : 0;
         double zScore = (ratioStd > 0) ? (currentRatio - ratioMean) / ratioStd : 0;
         
          if(zScore > 1.5)       { if(dir == 1) score -= 2; else score += 1; }
          else if(zScore < -1.5) { if(dir == -1) score -= 2; else score += 1; }
          else if(MathAbs(zScore) < 0.5) score += 1;
          
          if(Debug && tickCount % 60 == 0) LP("🥈 XAG CORR: ratio=", DoubleToString(currentRatio,2),
             " mean=", DoubleToString(ratioMean,2), " z=", DoubleToString(zScore,2),
             " dir=", dir, " adj=", (MathAbs(zScore) > 1.5 ? (zScore*dir > 0 ? "-2" : "+1") : (MathAbs(zScore) < 0.5 ? "+1" : "0")));
      }
   }


    
   // === DETAILED BREAKDOWN LOG ===
   if(Debug && score >= 2)
   {
       LP("SCORE V4 [", (dir==1?"BUY":"SELL"), "]=", score, "/35:",
            " EMA200=PASS",
            " | RSI(", RSI_Period, ")=", (CopyBuffer(h_rsi, 0, 0, 1, rsi) > 0 ? DoubleToString(rsi[0], 1) : "?"),
            " | ADX>", ADX_MinTrend,
            " | AI=", (aiSig.isValid ? aiSig.signal + " " + DoubleToString(aiSig.confidence*100, 0) + "%" : "N/A"));
   }
   
   return score;
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Detect RSI Divergence (institutional 10-bar swing analysis)      |
//| Returns: 1=bullish divergence, -1=bearish, 0=none               |
//| Bullish: Price lower low but RSI higher low (reversal signal)    |
//| Bearish: Price higher high but RSI lower high (reversal signal)  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Waddah Attar Explosion â€” Momentum filter (from Supreme_Alpha)    |
//| Returns: true if market has explosive momentum (good to trade)   |
//| false if market is dead/choppy (avoid entries)                    |
//+------------------------------------------------------------------+
bool ComputeWaddahAttar(int &waeDir)
{
   waeDir = 0;
   if(h_macd_wae == INVALID_HANDLE || h_atr_wae == INVALID_HANDLE) return true; // Failsafe: allow trading
   
   double macdMain[], macdSig[], atrVal[];
   ArraySetAsSeries(macdMain, true);
   ArraySetAsSeries(macdSig, true);
   ArraySetAsSeries(atrVal, true);
   
   if(CopyBuffer(h_macd_wae, 0, 0, 3, macdMain) <= 0) return true;
   if(CopyBuffer(h_macd_wae, 1, 0, 3, macdSig) <= 0) return true;
   if(CopyBuffer(h_atr_wae, 0, 0, 1, atrVal) <= 0) return true;
   
   // WAE trend strength = difference between current and previous MACD histogram
   double trend1 = macdMain[0] - macdSig[0]; // Current histogram
   double trend2 = macdMain[1] - macdSig[1]; // Previous histogram
   double trendPower = MathAbs(trend1 - trend2);  // Explosion = acceleration
   
   // Dead zone = BB-based multiplier of ATR (Bollinger Band width proxy)
   double deadZone = atrVal[0] * 1.0;  // ATR as dead zone threshold
   
   g_wae_trend_strength = trendPower;
   
   if(trendPower > deadZone)
   {
      g_wae_explosion = true;
      waeDir = (trend1 > 0) ? 1 : -1;  // Positive = bullish explosion, negative = bearish
      return true;
   }
   
   g_wae_explosion = false;
   return false;  // Dead market â€” don't trade
}

//+------------------------------------------------------------------+
//| Count Fibonacci EMA Alignment (7 EMAs: 8/13/34/55/89/144/233)   |
//| Returns: count of Fib EMAs stacked in trade direction             |
//| Bullish stack: 8 > 13 > 34 > 55 > 89 > 144 > 233               |
//+------------------------------------------------------------------+
int CountFibEMAStack(int dir)
{
   double e8[], e13[], e34[], e55[], e89[], e144[], e233[];
   ArraySetAsSeries(e8, true); ArraySetAsSeries(e13, true);
   ArraySetAsSeries(e34, true); ArraySetAsSeries(e55, true);
   ArraySetAsSeries(e89, true); ArraySetAsSeries(e144, true);
   ArraySetAsSeries(e233, true);
   
   if(CopyBuffer(h_fib_ema8,   0, 0, 1, e8)   <= 0) return 0;
   if(CopyBuffer(h_fib_ema13,  0, 0, 1, e13)  <= 0) return 0;
   if(CopyBuffer(h_fib_ema34,  0, 0, 1, e34)  <= 0) return 0;
   if(CopyBuffer(h_fib_ema55,  0, 0, 1, e55)  <= 0) return 0;
   if(CopyBuffer(h_fib_ema89,  0, 0, 1, e89)  <= 0) return 0;
   if(CopyBuffer(h_fib_ema144, 0, 0, 1, e144) <= 0) return 0;
   if(CopyBuffer(h_fib_ema233, 0, 0, 1, e233) <= 0) return 0;
   
   // Count how many consecutive Fib EMAs are stacked correctly
   double vals[7];
   vals[0] = e8[0]; vals[1] = e13[0]; vals[2] = e34[0]; vals[3] = e55[0];
   vals[4] = e89[0]; vals[5] = e144[0]; vals[6] = e233[0];
   
   int aligned = 0;
   for(int i = 0; i < 6; i++)
   {
      if(dir == 1 && vals[i] > vals[i+1])  aligned++;  // Bullish: fast > slow
      else if(dir == -1 && vals[i] < vals[i+1]) aligned++;  // Bearish: fast < slow
      else break;  // Chain broken
   }
   
   g_fib_ema_alignment = aligned;
   return aligned;
}

//+------------------------------------------------------------------+
//| Equity Target Close â€” auto-close all if daily target/stop hit    |
//| Called every tick in OnTick() before signal analysis              |
//+------------------------------------------------------------------+
void CheckEquityTargetClose()
{
   if(dayStartEquity <= 0) return;
   
   double currentEquity = accInfo.Equity();
   double dailyPnLPct = (currentEquity - dayStartEquity) / dayStartEquity * 100.0;
   
   // Daily PROFIT target hit â€” close all and celebrate
   if(g_equity_target_pct > 0 && dailyPnLPct >= g_equity_target_pct)
   {
      if(!profitTargetHit)
      {
         LP("ðŸŽ¯ EQUITY TARGET HIT: +", DoubleToString(dailyPnLPct, 2), "% â€” Closing all positions");
         ClosePositionsByType(POSITION_TYPE_BUY);
         ClosePositionsByType(POSITION_TYPE_SELL);
         profitTargetHit = true;
         tradingStatus = StringFormat("ðŸŽ¯ DAILY TARGET +%.1f%% â€” Paused until tomorrow", dailyPnLPct);
      }
   }
   
   // Daily LOSS limit hit â€” emergency close
   if(g_equity_stop_pct > 0 && dailyPnLPct <= -g_equity_stop_pct)
   {
      if(!dailyLimitHit)
      {
         LP("ðŸ›‘ EQUITY STOP HIT: ", DoubleToString(dailyPnLPct, 2), "% â€” Emergency close all");
         ClosePositionsByType(POSITION_TYPE_BUY);
         ClosePositionsByType(POSITION_TYPE_SELL);
         dailyLimitHit = true;
         tradingStatus = StringFormat("ðŸ›‘ DAILY STOP %.1f%% â€” Paused until tomorrow", dailyPnLPct);
      }
   }
}

int DetectRSIDivergence(int lookback = 10)
{
   double rsiArr[], closeArr[];
   ArrayResize(rsiArr, lookback + 2);
   ArrayResize(closeArr, lookback + 2);
   ArraySetAsSeries(rsiArr, true);
   ArraySetAsSeries(closeArr, true);
   
   if(CopyBuffer(h_rsi, 0, 1, lookback + 2, rsiArr) <= 0) return 0;
   if(CopyClose(_Symbol, PERIOD_M15, 1, lookback + 2, closeArr) <= 0) return 0;
   
   // Find two lowest price points in lookback
   double pLow1 = closeArr[0], pLow2 = closeArr[0];
   double rLow1 = rsiArr[0], rLow2 = rsiArr[0];
   int low1Idx = 0, low2Idx = -1;
   
   for(int i = 1; i < lookback && i < ArraySize(closeArr); i++)
   {
      if(closeArr[i] < pLow1)
      {
         pLow2 = pLow1; rLow2 = rLow1; low2Idx = low1Idx;
         pLow1 = closeArr[i]; rLow1 = rsiArr[i]; low1Idx = i;
      }
      else if(low2Idx < 0 || closeArr[i] < pLow2)
      {
         pLow2 = closeArr[i]; rLow2 = rsiArr[i]; low2Idx = i;
      }
   }
   
   // Bullish divergence: price made lower low, RSI made higher low
   if(low2Idx > 0 && low1Idx != low2Idx)
   {
      if(pLow1 < pLow2 && rLow1 > rLow2 && rLow1 < 40)
         return 1;  // Bullish divergence
   }
   
   // Find two highest price points in lookback
   double pHigh1 = closeArr[0], pHigh2 = closeArr[0];
   double rHigh1 = rsiArr[0], rHigh2 = rsiArr[0];
   int high1Idx = 0, high2Idx = -1;
   
   for(int i = 1; i < lookback && i < ArraySize(closeArr); i++)
   {
      if(closeArr[i] > pHigh1)
      {
         pHigh2 = pHigh1; rHigh2 = rHigh1; high2Idx = high1Idx;
         pHigh1 = closeArr[i]; rHigh1 = rsiArr[i]; high1Idx = i;
      }
      else if(high2Idx < 0 || closeArr[i] > pHigh2)
      {
         pHigh2 = closeArr[i]; rHigh2 = rsiArr[i]; high2Idx = i;
      }
   }
   
   // Bearish divergence: price made higher high, RSI made lower high
   if(high2Idx > 0 && high1Idx != high2Idx)
   {
      if(pHigh1 > pHigh2 && rHigh1 < rHigh2 && rHigh1 > 60)
         return -1;  // Bearish divergence
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Compute Daily VWAP (Volume-Weighted Average Price)               |
//| Resets at 00:00 broker time. Uses M5 bar data.                   |
//+------------------------------------------------------------------+
double ComputeDailyVWAP()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   // Find how many M5 bars since day start
   datetime dayStart = StringToTime(StringFormat("%04d.%02d.%02d 00:00", dt.year, dt.mon, dt.day));
   int barsToday = Bars(_Symbol, PERIOD_M5, dayStart, TimeCurrent());
   if(barsToday < 2) return 0;
   if(barsToday > 300) barsToday = 300;  // Safety cap
   
   double high[], low[], close[];
   long vol[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(vol, true);
   
   if(CopyHigh(_Symbol, PERIOD_M5, 0, barsToday, high) <= 0) return 0;
   if(CopyLow(_Symbol, PERIOD_M5, 0, barsToday, low) <= 0) return 0;
   if(CopyClose(_Symbol, PERIOD_M5, 0, barsToday, close) <= 0) return 0;
   if(CopyTickVolume(_Symbol, PERIOD_M5, 0, barsToday, vol) <= 0) return 0;
   
   double cumPV = 0, cumVol = 0;
   for(int i = barsToday - 1; i >= 0; i--)
   {
      double typicalPrice = (high[i] + low[i] + close[i]) / 3.0;
      cumPV += typicalPrice * (double)vol[i];
      cumVol += (double)vol[i];
   }
   
   if(cumVol == 0) return 0;
   return cumPV / cumVol;
}

//| Detect Fair Value Gap (3-candle imbalance)                        |
//| XAUUSD-tuned: gap threshold based on $ not points                |
//+------------------------------------------------------------------+
bool DetectFVG(int dir)
{
   double highs[], lows[];
   ArrayResize(highs, 10); ArrayResize(lows, 10);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   if(CopyHigh(_Symbol, PERIOD_M15, 0, 10, highs) < 0 ||
      CopyLow(_Symbol, PERIOD_M15, 0, 10, lows) < 0)
      return false;
   
   double minGap = 0.50;  // $0.50 minimum FVG for XAUUSD at $5,000+ level
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   for(int i = 1; i < 7; i++)
   {
      if(dir == 1)  // Bullish FVG: gap up
      {
         double gap = lows[i-1] - highs[i+1];
         if(gap > minGap)
         {
            if(bid >= highs[i+1] && bid <= lows[i-1])
               return true;
         }
      }
      else  // Bearish FVG: gap down
      {
         double gap = lows[i+1] - highs[i-1];
         if(gap > minGap)
         {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            if(ask >= highs[i-1] && ask <= lows[i+1])
               return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Detect Order Block                                                |
//| Last opposing candle before strong impulse                        |
//+------------------------------------------------------------------+
bool DetectOB(int dir)
{
   double opens[], closes[], highs[], lows[];
   ArrayResize(opens, 25); ArrayResize(closes, 25); ArrayResize(highs, 25); ArrayResize(lows, 25);
   ArraySetAsSeries(opens, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   
   if(CopyOpen(_Symbol, PERIOD_M15, 0, 25, opens) < 0 ||
      CopyClose(_Symbol, PERIOD_M15, 0, 25, closes) < 0 ||
      CopyHigh(_Symbol, PERIOD_M15, 0, 25, highs) < 0 ||
      CopyLow(_Symbol, PERIOD_M15, 0, 25, lows) < 0)
      return false;
   
   double price = SymbolInfoDouble(_Symbol, dir == 1 ? SYMBOL_BID : SYMBOL_ASK);
   
   for(int i = 3; i < 20; i++)
   {
      double body_i = MathAbs(closes[i] - opens[i]);
      double body_next = MathAbs(closes[i-1] - opens[i-1]);
      double range_i = highs[i] - lows[i];
      
      if(range_i <= 0 || body_next <= 0) continue;
      
      if(dir == 1)  // Bullish OB: bearish candle â†’ strong bullish
      {
         if(closes[i] < opens[i] && closes[i-1] > opens[i-1] && body_next > range_i * 1.5)
         {
            if(price >= lows[i] && price <= highs[i])
               return true;
         }
      }
      else  // Bearish OB: bullish candle â†’ strong bearish
      {
         if(closes[i] > opens[i] && closes[i-1] < opens[i-1] && body_next > range_i * 1.5)
         {
            if(price >= lows[i] && price <= highs[i])
               return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Detect Liquidity Sweep                                            |
//| Wick beyond range, close inside â€” institutional trap              |
//+------------------------------------------------------------------+
bool DetectSweep(int dir)
{
   double highs[], lows[], closes[];
   ArrayResize(highs, 35); ArrayResize(lows, 35); ArrayResize(closes, 35);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   
   if(CopyHigh(_Symbol, PERIOD_M15, 0, 35, highs) < 0 ||
      CopyLow(_Symbol, PERIOD_M15, 0, 35, lows) < 0 ||
      CopyClose(_Symbol, PERIOD_M15, 0, 35, closes) < 0)
      return false;
   
   // Find range extremes (bars 3-30)
   double hi = 0, lo = 999999;
   for(int i = 3; i < 30; i++)
   {
      if(highs[i] > hi) hi = highs[i];
      if(lows[i] < lo) lo = lows[i];
   }
   
   // Recent bar swept and closed back
   for(int i = 0; i < 3; i++)
   {
      if(dir == 1 && lows[i] < lo && closes[i] > lo)
         return true;
      if(dir == -1 && highs[i] > hi && closes[i] < hi)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Detect Break of Structure (BOS) - V7 FIX: now scored             |
//| BOS: Price breaks above recent swing HH (bullish)                |
//|      or below recent swing LL (bearish)                          |
//| Returns: 1=bullish BOS, -1=bearish BOS, 0=none                  |
//+------------------------------------------------------------------+
int DetectBOS()
{
   double highs[], lows[], closes[];
   ArrayResize(highs, 20); ArrayResize(lows, 20); ArrayResize(closes, 20);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);

   if(CopyHigh(_Symbol, PERIOD_M15, 0, 20, highs) < 0 ||
      CopyLow(_Symbol, PERIOD_M15, 0, 20, lows) < 0 ||
      CopyClose(_Symbol, PERIOD_M15, 0, 20, closes) < 0)
      return 0;

   double recentHH = 0, recentLL = 999999;
   int hhBar = 0, llBar = 0;
   for(int i = 2; i < 15; i++)
   {
      if(highs[i] > recentHH) { recentHH = highs[i]; hhBar = i; }
      if(lows[i] < recentLL) { recentLL = lows[i]; llBar = i; }
   }

   if(closes[0] > recentHH && hhBar > 1) return 1;
   if(closes[0] < recentLL && llBar > 1) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| Detect Change of Character (CHoCH) - V7 FIX: now scored         |
//| CHoCH: Reversal pattern - prior bearish trend breaks above       |
//|        recent swing high, or prior bullish breaks below LL       |
//| Returns: 1=bullish CHoCH, -1=bearish CHoCH, 0=none              |
//+------------------------------------------------------------------+
int DetectCHoCH()
{
   double highs[], lows[], closes[];
   ArrayResize(highs, 30); ArrayResize(lows, 30); ArrayResize(closes, 30);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);

   if(CopyHigh(_Symbol, PERIOD_M15, 0, 30, highs) < 0 ||
      CopyLow(_Symbol, PERIOD_M15, 0, 30, lows) < 0 ||
      CopyClose(_Symbol, PERIOD_M15, 0, 30, closes) < 0)
      return 0;

   double priorHH = 0, priorLL = 999999;
   for(int i = 5; i < 25; i++)
   {
      if(highs[i] > priorHH) priorHH = highs[i];
      if(lows[i] < priorLL) priorLL = lows[i];
   }

   int bearishBars = 0, bullishBars = 0;
   for(int i = 1; i <= 5; i++)
   {
      if(closes[i] < closes[i-1]) bearishBars++;
      if(closes[i] > closes[i-1]) bullishBars++;
   }

   if(bearishBars >= 3 && closes[0] > priorHH) return 1;
   if(bullishBars >= 3 && closes[0] < priorLL) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| V6: Chart Overlays â€” FVG, OB, BOS/CHoCH + SL/TP                  |
//| Called on new M15 bars to draw SMC structures on chart            |
//+------------------------------------------------------------------+
void CleanChartOverlays(string prefix, int maxAge = 25)
{
   // Delete overlay objects older than maxAge bars
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, prefix) == 0)
         ObjectDelete(0, name);
   }
}

void DrawChartOverlays()
{
   // V6 Phase 4: Interactive Position Level Overlays

   for(int pi = PositionsTotal() - 1; pi >= 0; pi--)
   {


      if(!posInfo.SelectByIndex(pi)) continue;

      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;

      

      double entry = posInfo.PriceOpen();

      double slLvl = posInfo.StopLoss();

      double tpLvl = posInfo.TakeProfit();

      bool isBuy = posInfo.PositionType() == POSITION_TYPE_BUY;

      double posProfit = posInfo.Profit() + posInfo.Swap() + posInfo.Commission();

      

      // Entry line (cyan dashed)

      string entryName = "V10_ENTRY_" + IntegerToString(posInfo.Ticket());

      ObjectCreate(0, entryName, OBJ_HLINE, 0, 0, entry);

      ObjectSetInteger(0, entryName, OBJPROP_COLOR, C'0,229,204');

      ObjectSetInteger(0, entryName, OBJPROP_STYLE, STYLE_DASH);

      ObjectSetInteger(0, entryName, OBJPROP_WIDTH, 1);

      ObjectSetInteger(0, entryName, OBJPROP_BACK, true);

      ObjectSetString(0, entryName, OBJPROP_TEXT, 

         StringFormat("â–¸ ENTRY $%.2f  %s %.2f lots  P&L: $%+.2f",

            entry, isBuy ? "BUY" : "SELL", posInfo.Volume(), posProfit));

      

      // SL line (red solid)

      if(slLvl > 0)
      {


         string slName = "V10_SL_" + IntegerToString(posInfo.Ticket());

         ObjectCreate(0, slName, OBJ_HLINE, 0, 0, slLvl);

         ObjectSetInteger(0, slName, OBJPROP_COLOR, C'255,75,75');

         ObjectSetInteger(0, slName, OBJPROP_STYLE, STYLE_DOT);

         ObjectSetInteger(0, slName, OBJPROP_WIDTH, 1);

         ObjectSetInteger(0, slName, OBJPROP_BACK, true);

         double slDist = MathAbs(entry - slLvl);

         ObjectSetString(0, slName, OBJPROP_TEXT,

            StringFormat("âœ• SL $%.2f  (-$%.2f)", slLvl, slDist));

      }

      

      // TP line (green solid)

      if(tpLvl > 0)
      {


         string tpName = "V10_TP_" + IntegerToString(posInfo.Ticket());

         ObjectCreate(0, tpName, OBJ_HLINE, 0, 0, tpLvl);

         ObjectSetInteger(0, tpName, OBJPROP_COLOR, C'0,255,135');

         ObjectSetInteger(0, tpName, OBJPROP_STYLE, STYLE_DOT);

         ObjectSetInteger(0, tpName, OBJPROP_WIDTH, 1);

         ObjectSetInteger(0, tpName, OBJPROP_BACK, true);

         double tpDist = MathAbs(tpLvl - entry);

         double rr = (slLvl > 0 && MathAbs(entry - slLvl) > 0) ? tpDist / MathAbs(entry - slLvl) : 0;

         ObjectSetString(0, tpName, OBJPROP_TEXT,

            StringFormat("âœ“ TP $%.2f  (+$%.2f  R:R %.1f:1)", tpLvl, tpDist, rr));

      }

      break;  // Only draw for first (latest) position

   }

   

   // Clean old entry/SL/TP lines when no position

   if(PositionsTotal() == 0)
   {


      for(int oi = ObjectsTotal(0) - 1; oi >= 0; oi--)
      {


         string oName = ObjectName(0, oi);

         if(StringFind(oName, "V10_ENTRY_") >= 0 || StringFind(oName, "V10_SL_") >= 0 || StringFind(oName, "V10_TP_") >= 0)
         {

            ObjectDelete(0, oName);

      }
   }
   }

   

   // Clean old overlays first
   CleanChartOverlays("V10_FVG_");
   CleanChartOverlays("V10_OB_");
   CleanChartOverlays("V10_BOS_");
   CleanChartOverlays("V10_SLTP_");
   
   double opens[], closes[], highs[], lows[];
   datetime times[];
   int bars = 25;
   ArrayResize(opens, bars); ArrayResize(closes, bars);
   ArrayResize(highs, bars); ArrayResize(lows, bars);
   ArrayResize(times, bars);
   ArraySetAsSeries(opens, true); ArraySetAsSeries(closes, true);
   ArraySetAsSeries(highs, true); ArraySetAsSeries(lows, true);
   ArraySetAsSeries(times, true);
   
   if(CopyOpen(_Symbol, PERIOD_M15, 0, bars, opens) < bars ||
      CopyClose(_Symbol, PERIOD_M15, 0, bars, closes) < bars ||
      CopyHigh(_Symbol, PERIOD_M15, 0, bars, highs) < bars ||
      CopyLow(_Symbol, PERIOD_M15, 0, bars, lows) < bars ||
      CopyTime(_Symbol, PERIOD_M15, 0, bars, times) < bars)
      return;
   
   double atrBuf[];
   ArrayResize(atrBuf, 1);
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(h_atr, 0, 0, 1, atrBuf) < 0) return;
   double atrNow = atrBuf[0];
   double minGap = atrNow * 0.3;
   
   int fvgCount = 0, obCount = 0;
   
   // â”€â”€ Draw FVG Zones â”€â”€
   for(int i = 1; i < bars - 1 && fvgCount < 5; i++)
   {
      // Bullish FVG: gap between bar[i+1] high and bar[i-1] low
      double bullGap = lows[i-1] - highs[i+1];
      if(bullGap > minGap)
      {
         string name = "V10_FVG_B" + IntegerToString(i);
         datetime t1 = times[i+1];
         datetime t2 = times[MathMax(0, i-3)]; // Extend forward
         ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, highs[i+1], t2, lows[i-1]);
         ObjectSetInteger(0, name, OBJPROP_COLOR, C'0,255,135');
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, name, OBJPROP_FILL, true);
         ObjectSetInteger(0, name, OBJPROP_BACK, true);
         ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
         fvgCount++;
      }
      // Bearish FVG
      double bearGap = lows[i+1] - highs[i-1];
      if(bearGap > minGap)
      {
         string name = "V10_FVG_S" + IntegerToString(i);
         datetime t1 = times[i+1];
         datetime t2 = times[MathMax(0, i-3)];
         ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, highs[i-1], t2, lows[i+1]);
         ObjectSetInteger(0, name, OBJPROP_COLOR, C'255,75,75');
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, name, OBJPROP_FILL, true);
         ObjectSetInteger(0, name, OBJPROP_BACK, true);
         ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
         fvgCount++;
      }
   }
   
   // â”€â”€ Draw Order Block Zones â”€â”€
   for(int i = 3; i < 20 && obCount < 4; i++)
   {
      double body_i = MathAbs(closes[i] - opens[i]);
      double body_next = MathAbs(closes[i-1] - opens[i-1]);
      double range_i = highs[i] - lows[i];
      if(range_i <= 0 || body_next <= 0) continue;
      
      // Bullish OB: bearish candle before strong bullish impulse
      if(closes[i] < opens[i] && closes[i-1] > opens[i-1] && body_next > range_i * 1.5)
      {
         double price_now = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(price_now >= lows[i] && price_now <= highs[i])
         {
            string name = "V10_OB_B" + IntegerToString(i);
            datetime t1 = times[i];
            datetime t2 = times[MathMax(0, i-4)];
            ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, lows[i], t2, highs[i]);
            ObjectSetInteger(0, name, OBJPROP_COLOR, C'0,180,135');
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, name, OBJPROP_FILL, true);
            ObjectSetInteger(0, name, OBJPROP_BACK, true);
            ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
            obCount++;
         }
      }
      // Bearish OB: bullish candle before strong bearish impulse
      if(closes[i] > opens[i] && closes[i-1] < opens[i-1] && body_next > range_i * 1.5)
      {
         double price_now = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(price_now >= lows[i] && price_now <= highs[i])
         {
            string name = "V10_OB_S" + IntegerToString(i);
            datetime t1 = times[i];
            datetime t2 = times[MathMax(0, i-4)];
            ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, highs[i], t2, lows[i]);
            ObjectSetInteger(0, name, OBJPROP_COLOR, C'180,50,50');
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, name, OBJPROP_FILL, true);
            ObjectSetInteger(0, name, OBJPROP_BACK, true);
            ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
            obCount++;
         }
      }
   }
   
   // â”€â”€ Draw BOS/CHoCH Labels â”€â”€
   // Find recent structure breaks (higher high / lower low violations)
   double recentHH = 0, recentLL = 999999;
   int hhBar = 0, llBar = 0;
   for(int i = 1; i < 15; i++)
   {
      if(highs[i] > recentHH) { recentHH = highs[i]; hhBar = i; }
      if(lows[i] < recentLL) { recentLL = lows[i]; llBar = i; }
   }
   
   // BOS: price breaks above recent HH (bullish) or below LL (bearish)
   if(closes[0] > recentHH && hhBar > 1)
   {
      string name = "V10_BOS_B";
      ObjectCreate(0, name, OBJ_TREND, 0, times[hhBar], recentHH, times[0], recentHH);
      ObjectSetInteger(0, name, OBJPROP_COLOR, C'0,255,135');
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
      // Label
      string lblName = "V10_BOS_LBL_B";
      ObjectCreate(0, lblName, OBJ_TEXT, 0, times[0], recentHH + atrNow * 0.1);
      ObjectSetString(0, lblName, OBJPROP_TEXT, "BOS â–²");
      ObjectSetString(0, lblName, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, lblName, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, lblName, OBJPROP_COLOR, C'0,255,135');
      ObjectSetInteger(0, lblName, OBJPROP_SELECTABLE, false);
   }
   if(closes[0] < recentLL && llBar > 1)
   {
      string name = "V10_BOS_S";
      ObjectCreate(0, name, OBJ_TREND, 0, times[llBar], recentLL, times[0], recentLL);
      ObjectSetInteger(0, name, OBJPROP_COLOR, C'255,75,75');
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
      // Label
      string lblName = "V10_BOS_LBL_S";
      ObjectCreate(0, lblName, OBJ_TEXT, 0, times[0], recentLL - atrNow * 0.1);
      ObjectSetString(0, lblName, OBJPROP_TEXT, "BOS â–¼");
      ObjectSetString(0, lblName, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, lblName, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, lblName, OBJPROP_COLOR, C'255,75,75');
      ObjectSetInteger(0, lblName, OBJPROP_SELECTABLE, false);
   }
   
   // â”€â”€ Draw Active Trade SL/TP Lines â”€â”€
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      if(!posInfo.SelectByIndex(p)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;
      
      double slLevel = posInfo.StopLoss();
      double tpLevel = posInfo.TakeProfit();
      double entryLevel = posInfo.PriceOpen();
      
      if(slLevel > 0)
      {
         string slName = "V10_SLTP_SL";
         ObjectDelete(0, slName);
         ObjectCreate(0, slName, OBJ_HLINE, 0, 0, slLevel);
         ObjectSetInteger(0, slName, OBJPROP_COLOR, C'255,75,75');
         ObjectSetInteger(0, slName, OBJPROP_STYLE, STYLE_DASHDOTDOT);
         ObjectSetInteger(0, slName, OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, slName, OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, slName, OBJPROP_BACK, true);
      }
      if(tpLevel > 0)
      {
         string tpName = "V10_SLTP_TP";
         ObjectDelete(0, tpName);
         ObjectCreate(0, tpName, OBJ_HLINE, 0, 0, tpLevel);
         ObjectSetInteger(0, tpName, OBJPROP_COLOR, C'0,255,135');
         ObjectSetInteger(0, tpName, OBJPROP_STYLE, STYLE_DASHDOTDOT);
         ObjectSetInteger(0, tpName, OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, tpName, OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, tpName, OBJPROP_BACK, true);
      }
      break; // Only draw for first matching position
   }
}

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| V6: Volatility Regime â€” LOW / NORMAL / HIGH                       |
//| Compares current ATR to rolling ATR average for dynamic thresholds|
//+------------------------------------------------------------------+
int GetVolatilityRegime()
{
   double atr_curr[], atr_long[];
   ArrayResize(atr_curr, 1);
   ArrayResize(atr_long, 50);
   ArraySetAsSeries(atr_curr, true);
   ArraySetAsSeries(atr_long, true);
   
   if(CopyBuffer(h_atr, 0, 0, 1, atr_curr) < 0 ||
      CopyBuffer(h_atr, 0, 0, 50, atr_long) < 50)
      return 0; // NORMAL fallback
   
   double avgATR = 0;
   for(int i = 0; i < 50; i++) avgATR += atr_long[i];
   avgATR /= 50.0;
   
   if(avgATR <= 0) return 0;
   
   double ratio = atr_curr[0] / avgATR;
   
   if(ratio < 0.7) return -1;  // LOW volatility (ranging)
   if(ratio > 1.3) return  1;  // HIGH volatility (trending/spiking)
   return 0;                    // NORMAL
}

//+------------------------------------------------------------------+
//| V6: Find Swing Level (structure-based SL/TP)                      |
//| Scans recent M15 candles for swing high/low for SL placement     |
//+------------------------------------------------------------------+
double FindSwingLevel(int dir, int lookback = 20)
{
   double highs[], lows[];
   ArrayResize(highs, lookback);
   ArrayResize(lows, lookback);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   
   if(CopyHigh(_Symbol, PERIOD_M15, 0, lookback, highs) < lookback ||
      CopyLow(_Symbol, PERIOD_M15, 0, lookback, lows) < lookback)
      return 0;
   
   if(dir == 1) // BUY â€” find recent swing low for SL
   {
      double swingLow = lows[0];
      for(int i = 1; i < lookback; i++)
         if(lows[i] < swingLow) swingLow = lows[i];
      return swingLow;
   }
   else // SELL â€” find recent swing high for SL
   {
      double swingHigh = highs[0];
      for(int i = 1; i < lookback; i++)
         if(highs[i] > swingHigh) swingHigh = highs[i];
      return swingHigh;
   }
}

//+------------------------------------------------------------------+
//| Execute Trade â€” V6: VaR-Adjusted + Structure SL/TP               |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type)
{
   // Get ATR for SL/TP
   double atr[];
   ArrayResize(atr, 2);
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(h_atr, 0, 0, 2, atr) < 0 || atr[0] <= 0)
   {
      LP("ERROR: Cannot get ATR");
      return;
   }
   
   double atrVal = atr[0];
   double price, sl, tp;
   int digits = (int)symInfo.Digits();
   
   // â”€â”€â”€ Phase A: Consecutive Loss Pause â”€â”€â”€
   if(lossPauseActive && MaxConsecLosses > 0)
   {
      if(Debug) LP("â¸ï¸ PAUSED: ", consecLosses, " consecutive losses â€” skipping trade");
      return;
   }
   
   // â”€â”€â”€ Phase A: Confidence-Based R:R â”€â”€â”€
   double dynamicTP = TP1_ATR_Mult;  // Default 3.0
   if(UseConfidenceScaling && aiSig.isValid)
   {
      if(aiSig.confidence > 0.70)
         dynamicTP = 4.0;    // High confidence â†’ 2:1 R:R (wider TP, let winners run)
      else if(aiSig.confidence > 0.55)
         dynamicTP = 3.0;    // Medium â†’ 1.5:1 R:R (standard)
      else if(aiSig.confidence > 0.40)
         dynamicTP = 2.5;    // Low â†’ 1.25:1 R:R (quicker exit)
      else
         dynamicTP = 2.0;    // Very low â†’ 1:1 R:R (scalp mode)
      
      if(Debug && dynamicTP != TP1_ATR_Mult)
         LP("ðŸ“Š Confidence R:R: conf=", DoubleToString(aiSig.confidence*100,0),
               "% â†’ TP=", DoubleToString(dynamicTP,1), "Ã—ATR (R:R=", 
               DoubleToString(dynamicTP/SL_ATR_Mult, 1), ":1)");
   }
   
   symInfo.RefreshRates();
   // V10: Structure-Aware SL + Regime-Adaptive TP
   int tradeDir = (type == ORDER_TYPE_BUY) ? 1 : -1;
   string slSource = "V10_STRUCT";
   
   // V10: Regime-adaptive triple barriers (buyPT, sellPT, slMult from V10_GetRegimeBarriers)
   int volRegTP = GetVolatilityRegime();
   double v10_buyPT, v10_sellPT, v10_slMult;
   V10_GetRegimeBarriers(volRegTP, v10_buyPT, v10_sellPT, v10_slMult);
   double v10TP = (type == ORDER_TYPE_BUY) ? v10_buyPT : v10_sellPT;
   if(UseConfidenceScaling && aiSig.isValid)
      v10TP = MathMax(v10TP, dynamicTP);  // Use higher of V10 regime TP or confidence TP
   
   if(type == ORDER_TYPE_BUY)
   {
      price = symInfo.Ask();
      // V10: Structure-aware SL (snaps to swing lows, capped at volRegime-adjusted ATR)
      sl = NormalizeDouble(V10_StructureSL(type, price, atrVal * v10_slMult), digits);
      tp = NormalizeDouble(price + atrVal * v10TP, digits);
   }
   else
   {
      price = symInfo.Bid();
      sl = NormalizeDouble(V10_StructureSL(type, price, atrVal * v10_slMult), digits);
      tp = NormalizeDouble(price - atrVal * v10TP, digits);
   }
   
   // â“€â“€â“€ XAUUSD Lot Size Calculation
   // Gold: 1 standard lot = 100 oz
   // If price = $5000, 1 lot = $500,000 notional
   // SL distance in $ * 100 (oz per lot) = risk per lot
   double lots = LotSize;
   if(UseDynamicLots)
   {
      double slDist = MathAbs(price - sl);  // In dollars
      if(slDist <= 0) { LP("ERROR: SL distance is 0"); return; }
      
      // â”€â”€â”€ V6: VaR-Adjusted + Kelly Risk Scaling â”€â”€â”€
      double kellyRisk = RiskPerTradePct;
      
      // V10: Kelly Criterion sizing (replaces old V6 confidence formula)
      // Uses rolling 50-trade window to compute optimal bet fraction
      double kellyLots = V10_KellyLotSize(accInfo.Equity(), slDist, symInfo.TickValue(), symInfo.TickSize());
      if(kellyLots > 0)
      {
         kellyRisk = -1;  // Flag: use Kelly lots directly instead of risk%
         lots = kellyLots;
         if(Debug) LP("V10 KELLY: f*=", DoubleToString(g_v10_kelly_fraction*100,2),
                         "% lots=", DoubleToString(kellyLots,3),
                         " (", V10_KellyStatusString(), ")");
      }
      
      // V6: VaR adjustment â€” scale risk based on volatility regime
      int volRegime = GetVolatilityRegime();
      double varScale = 1.0;
      string varLabel = "NORMAL";
      if(volRegime > 0) { varScale = 0.7; varLabel = "HIGH-VOL(-30%)"; }       // High vol: reduce risk 30%
      else if(volRegime < 0) { varScale = 1.1; varLabel = "LOW-VOL(+10%)"; }   // Low vol: slight increase
      kellyRisk *= varScale;
       
       // V6: Session-aware lot scaling â€” trade bigger in liquid sessions
       MqlDateTime sessLotDt;
       TimeToStruct(TimeGMT(), sessLotDt);
       int sessHour = sessLotDt.hour;
       double sessionScale = 1.0;
       string sessionLabel = "NORMAL";
       if(sessHour >= 0 && sessHour < 7)         { sessionScale = 0.6; sessionLabel = "ASIA(-40%)"; }
       else if(sessHour >= 13 && sessHour < 16)   { sessionScale = 1.2; sessionLabel = "LON-NY(+20%)"; }
       else if(sessHour >= 7 && sessHour < 10)    { sessionScale = 1.0; sessionLabel = "LONDON"; }
       else if(sessHour >= 10 && sessHour < 13)   { sessionScale = 1.0; sessionLabel = "NY"; }
       kellyRisk *= sessionScale;
       if(Debug) LP("ðŸ“Š SESSION LOT: ", sessionLabel, " | scale=", DoubleToString(sessionScale,1), "x");

      // Prop challenge hard cap for per-trade risk.
      if(EnablePropChallengeMode)
         kellyRisk = MathMin(kellyRisk, PropMaxRiskPerTradePct);
      
      if(Debug) LP("ðŸ’° Risk: base=", DoubleToString(RiskPerTradePct,1),
                      "% | Kelly=", DoubleToString(kellyRisk/varScale,2),
                      "% | VaR=", varLabel,
                      " | Final=", DoubleToString(kellyRisk,2), "%",
                      " | SL=", slSource, " $", DoubleToString(slDist,2));
      double riskAmount = accInfo.Equity() * kellyRisk / 100.0;
      
      // Use TickValue for precise calculation
      double tickVal = symInfo.TickValue();
      double tickSize = symInfo.TickSize();
      
      if(tickVal > 0 && tickSize > 0)
      {
         double valuePerLot = slDist * tickVal / tickSize;
         lots = riskAmount / valuePerLot;
      }
      else
      {
         // Fallback: 1 lot = 100 oz, so $1 move = $100 per lot
         lots = riskAmount / (slDist * 100.0);
      }
      
      // Clamp to broker limits
      double minLot = symInfo.LotsMin();
      double maxLot = symInfo.LotsMax();
      double lotStep = symInfo.LotsStep();
      
      if(lotStep > 0)
         lots = MathFloor(lots / lotStep) * lotStep;
      lots = MathMax(minLot, MathMin(maxLot, lots));
      lots = NormalizeDouble(lots, 2);
   }
   
   // V10: Confidence-scaled lot sizing (scale by AI confidence relative to 0.60 base)
   if(aiSig.isValid)
      lots = V10_ScaleLotByConfidence(lots, aiSig.confidence, PropMaxRiskPerTradePct);
   
   // V10: Meta-confidence lot scaling (reduce when meta-conf is low, boost when high)
   if(V10_UseMetaConfidence && g_v10_meta_conf > 0)
      lots = V10_ScaleLotByMetaConfidence(lots, g_v10_meta_conf);
   
   if(lots <= 0) { LP("ERROR: Calculated lot size is 0"); return; }
   
   // Execute
   int vr = GetVolatilityRegime();
   string comment = StringFormat("AIPred_v10|ATR=%.2f|SL=%s|Vol=%s|Meta=%.0f%%", atrVal, slSource,
                     (vr < 0 ? "LOW" : (vr > 0 ? "HIGH" : "NORM")),
                     g_v10_meta_conf * 100);
   bool ok = false;
   
   if(type == ORDER_TYPE_BUY)
      ok = trade.Buy(lots, _Symbol, price, sl, tp, comment);
   else
      ok = trade.Sell(lots, _Symbol, price, sl, tp, comment);
   
   if(ok && trade.ResultRetcode() == TRADE_RETCODE_DONE)
   {
      totalTradesOpened++;
      todayTradeCount++;
      
      string tradeDir = (type == ORDER_TYPE_BUY ? "BUY" : "SELL");
      string msg = StringFormat("%s %s %.2f lots @ $%.2f | SL: $%.2f | TP: $%.2f | ATR: $%.2f | Session: %s",
                                "ðŸš€ TRADE OPENED:", tradeDir, lots, price, sl, tp, atrVal, GetSession(nowDt_hour()));
      LP(msg);
      
      // ðŸ”” POPUP ALERT
      Alert("TradingAI v4.0 â€” ", tradeDir, " SIGNAL EXECUTED!",
            "\nLots: ", DoubleToString(lots, 2),
            "\nEntry: $", DoubleToString(price, (int)symInfo.Digits()),
            "\nSL: $", DoubleToString(sl, (int)symInfo.Digits()),
            "\nTP: $", DoubleToString(tp, (int)symInfo.Digits()),
            "\nATR: $", DoubleToString(atrVal, 2),
            "\nSession: ", GetSession(nowDt_hour()));
      
      // Play sound
      PlaySound("alert2.wav");
      
      // Record for partial close tracking
      TradeRecord rec;
      rec.ticket = trade.ResultOrder();
      rec.partialDone = false;
      rec.entryPrice = price;
       rec.originalSL = sl;
       
       // V6 WEBHOOK: Send trade open with full AI metrics
       string sess = GetSession(nowDt_hour());
       string regLabel = (vr < 0 ? "LOW_VOL" : (vr > 0 ? "HIGH_VOL" : "NORMAL"));
       string dxyStr = (aiSig.isValid ? aiSig.dxy : "");
       WebhookTradeOpen(rec.ticket, tradeDir, price, sl, tp, lots, atrVal,
                        aiSig.signal, aiSig.confidence, 
                        (tradeDir == "BUY" ? lastBuyScore : lastSellScore),
                        sess, regLabel, dxyStr, "V10");
   }
   else
   {
      LP("TRADE FAILED: ", trade.ResultRetcodeDescription(),
            " | Code: ", trade.ResultRetcode(),
            " | Price: ", DoubleToString(price, digits));
      Alert("âš ï¸ Trade FAILED: ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| V4.2: ONNX XGBoost Brain â€” PF 3.62, 55 features, sub-1ms        |
//| Primary decision maker: BUY=0, NO_TRADE=1, SELL=2                |
//| Features MUST match v4_features.json order EXACTLY                |
//+------------------------------------------------------------------+

// Helper: compute EMA from close array (span bars, newest first)
double ComputeEMA(double &arr[], int span, int maxBars)
{
   double alpha = 2.0 / (span + 1.0);
   int start = MathMin(span * 3, maxBars - 1);
   double ema = arr[start];
   for(int i = start - 1; i >= 0; i--)
      ema = alpha * arr[i] + (1 - alpha) * ema;
   return ema;
}

// Helper: compute EMA at bar shift (for slope)
double ComputeEMAShifted(double &arr[], int span, int shift, int maxBars)
{
   double alpha = 2.0 / (span + 1.0);
   int start = MathMin(span * 3 + shift, maxBars - 1);
   double ema = arr[start];
   for(int i = start - 1; i >= shift; i--)
      ema = alpha * arr[i] + (1 - alpha) * ema;
   return ema;
}

// Helper: compute RSI from close array
double ComputeRSI(double &arr[], int period, int maxBars)
{
   double avgGain = 0, avgLoss = 0;
   int start = MathMin(period + 1, maxBars - 1);
   for(int i = 1; i <= start; i++)
   {
      double d = arr[i-1] - arr[i];
      if(d > 0) avgGain += d; else avgLoss += MathAbs(d);
   }
   avgGain /= period;
   avgLoss /= period;
   if(avgLoss < 1e-10) return 1.0;
   return (100.0 - 100.0 / (1.0 + avgGain / avgLoss)) / 100.0;
}

//+------------------------------------------------------------------+
//| HMM Vol-Regime Approximation (matches training 3-state GaussianHMM)|
//+------------------------------------------------------------------+
int ClassifyHMMRegime(double &closes[], int maxBars)
{
   if(maxBars < 25) return 1;
   
   double returns[];
   ArrayResize(returns, 21);
   for(int i = 0; i < 21; i++)
      returns[i] = (closes[i] - closes[i+1]) / (closes[i+1] + 1e-10);
   
   double sum = 0, sumsq = 0;
   for(int i = 0; i < 20; i++) { sum += returns[i]; sumsq += returns[i]*returns[i]; }
   double mean = sum / 20.0;
   double currentVol = MathSqrt(MathMax(sumsq/20.0 - mean*mean, 0));
   
   double volHistory[];
   int histBars = MathMin(100, maxBars - 25);
   ArrayResize(volHistory, histBars);
   
   for(int h = 0; h < histBars; h++)
   {
      double s = 0, sq = 0;
      for(int i = 0; i < 20; i++)
      {
         double r = (closes[h+i] - closes[h+i+1]) / (closes[h+i+1] + 1e-10);
         s += r; sq += r*r;
      }
      double m = s / 20.0;
      volHistory[h] = MathSqrt(MathMax(sq/20.0 - m*m, 0));
   }
   
   double sorted[];
   ArrayCopy(sorted, volHistory);
   ArraySort(sorted);
   
   int pn = ArraySize(sorted);
   double p39 = sorted[(int)(pn * 0.39)];
   double p73 = sorted[(int)(pn * 0.73)];
   
   if(currentVol <= p39) return 0;
   else if(currentVol <= p73) return 1;
   else return 2;
}

//+------------------------------------------------------------------+
//| Resolve broker symbol â€” auto-detects suffix from chart symbol     |
//+------------------------------------------------------------------+
string ResolveSymbol(string base)
{
   string chartSym = _Symbol;
   string brokerSuffix = "";
   
   for(int len = 6; len <= 7; len++)
   {
      if(StringLen(chartSym) > len)
      {
         string potential = StringSubstr(chartSym, len);
         if(StringLen(potential) > 0)
         {
            brokerSuffix = potential;
            break;
         }
      }
   }
   
   string variants[];
   int vn = 0;
   ArrayResize(variants, 12);
   
   if(brokerSuffix != "")
      variants[vn++] = base + brokerSuffix;
   
   variants[vn++] = base;
   variants[vn++] = base + ".sml";
   variants[vn++] = base + "m";
   variants[vn++] = base + ".";
   variants[vn++] = base + "_";
   variants[vn++] = base + ".a";
   variants[vn++] = base + "pro";
   variants[vn++] = base + "-";
   variants[vn++] = base + "c";
   
   for(int i = 0; i < vn; i++)
   {
      SymbolSelect(variants[i], true);
      Sleep(50);
      double bid = 0;
      if(SymbolInfoDouble(variants[i], SYMBOL_BID, bid) && bid > 0)
         return variants[i];
      else
         SymbolSelect(variants[i], false);
   }
   return "";
}

//+------------------------------------------------------------------+
//| V6 Phase 4: Fetch Adaptive Parameters from InsForge               |
//| Reads adaptive_params table via REST API every 5 minutes          |
//+------------------------------------------------------------------+
void FetchAdaptiveParams()
{
   if(TimeLocal() - g_last_adaptive_fetch < 300) return;
   g_last_adaptive_fetch = TimeLocal();

    string baseUrl = InsForgeApiURL;
    int funcPos = StringFind(baseUrl, "/functions");
    if(funcPos > 0) baseUrl = StringSubstr(baseUrl, 0, funcPos);
    int apiPos = StringFind(baseUrl, "/api");
    if(apiPos > 0) baseUrl = StringSubstr(baseUrl, 0, apiPos);
    string url = baseUrl + "/api/database/records/adaptive_params?select=param_name,param_value";

   string anonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3OC0xMjM0LTU2NzgtOTBhYi1jZGVmMTIzNDU2NzgiLCJlbWFpbCI6ImFub25AaW5zZm9yZ2UuY29tIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM3NTE3OTd9.vhgw16dysf63KjR8MtQUB00pgUjSrfDqp5Fr4SW1zS4";
   string headers = "Content-Type: application/json\r\n";
   headers += "Authorization: Bearer " + anonKey + "\r\n";

   char   post[];
   char   result[];
   string resHeaders;
   int timeout = 3000;

   int res = WebRequest("GET", url, headers, timeout, post, result, resHeaders);

   

   if(res != 200)
   {
      if(Debug) LP("âš¡ Adaptive fetch failed: HTTP ", res);
      return;
   }

   

   string json = CharArrayToString(result);

   

   // Parse simple JSON array response

   // Expected format: [{"param_name":"min_score","param_value":8},...]

   double minScore = ParseParamValue(json, "min_score");

   double kelly = ParseParamValue(json, "kelly_fraction");

   double slMult = ParseParamValue(json, "sl_mult");

   double tpMult = ParseParamValue(json, "tp_mult");

   double regime = ParseParamValue(json, "regime");

   double dxyBias = ParseParamValue(json, "dxy_bias_active");

   

   // Apply fetched values

   if(minScore >= 3 && minScore <= 18) g_adaptive_minScore = minScore;

   if(kelly > 0 && kelly <= 0.10) g_adaptive_kelly = kelly;

   if(regime >= 0 && regime <= 2) g_adaptive_regime = (regime == 0 ? "NORMAL" : (regime == 1 ? "LOW_VOL" : "HIGH_VOL"));

   

   // V10 Phase 4: Regime-based model weight switching
   // V10 uses LEARNED Optuna weights: XGB=52.66% (DOMINANT), CAT=31.25%, LGB=16.08%
   // HIGH_VOL: Slightly more weight to CAT/LGB for diversity
   // LOW_VOL: Extend XGB dominance (trend-following)
   // NORMAL: V10 learned defaults
   if(g_adaptive_regime == "HIGH_VOL") {
      g_regime_w_xgb = 0.40; g_regime_w_lgb = 0.25; g_regime_w_cat = 0.35;
   } else if(g_adaptive_regime == "LOW_VOL") {
      g_regime_w_xgb = 0.60; g_regime_w_lgb = 0.12; g_regime_w_cat = 0.28;
   } else {
      g_regime_w_xgb = V10_WeightXGB; g_regime_w_lgb = V10_WeightLGB; g_regime_w_cat = V10_WeightCAT;
   }
   // V10: Sync regime weights into V10 adaptive drift tracker
   g_v10_wXGB = g_regime_w_xgb;
   g_v10_wLGB = g_regime_w_lgb;
   g_v10_wCAT = g_regime_w_cat;

   

   if(Debug) LP("ðŸ”„ ADAPTIVE FETCH: MinScore=", (int)g_adaptive_minScore,

                    " | Kelly=", DoubleToString(g_adaptive_kelly*100,2), "%",

                    " | Regime=", g_adaptive_regime,

                    " | Weights: XGB=", DoubleToString(g_regime_w_xgb,2),

                    " LGB=", DoubleToString(g_regime_w_lgb,2),

                    " CAT=", DoubleToString(g_regime_w_cat,2));

}

// Simple JSON param value parser for adaptive_params response

double ParseParamValue(string json, string paramName)
{

   string search = "\"param_name\":\"" + paramName + "\"";

   int pos = StringFind(json, search);

   if(pos < 0) return -1;

   

   // Find param_value after this match

   string valKey = "\"param_value\":";

   int valPos = StringFind(json, valKey, pos);

   if(valPos < 0) return -1;

   valPos += StringLen(valKey);

   

   // Extract number until comma, }, or ]

   string numStr = "";

   for(int k = valPos; k < StringLen(json) && k < valPos + 20; k++)
   {


      ushort ch = StringGetCharacter(json, k);

      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-') numStr += ShortToString(ch);

      else if(ch == ',' || ch == 125 || ch == ']') break;

   }

   return StringToDouble(numStr);

}

void RunONNXInference()
{
   if((h_xgb == INVALID_HANDLE && h_lgb == INVALID_HANDLE && h_cat == INVALID_HANDLE) || !UseONNX) return;
   
   // â•â•â• Gather price data on PRIMARY TF (M5 to match training) â•â•â•
   int bars = 600;  // V10 FIX: Need 600+ bars for EMA200 convergence (was 250)
   double c_m5[], h_m5[], l_m5[], o_m5[];
   long   v_m5_raw[];
   ArrayResize(c_m5, bars); ArrayResize(h_m5, bars); ArrayResize(l_m5, bars);
   ArrayResize(o_m5, bars); ArrayResize(v_m5_raw, bars);
   ArraySetAsSeries(c_m5, true); ArraySetAsSeries(h_m5, true);
   ArraySetAsSeries(l_m5, true); ArraySetAsSeries(o_m5, true); ArraySetAsSeries(v_m5_raw, true);
   
   if(CopyClose(_Symbol, PERIOD_M5, 0, bars, c_m5) < bars) return;
   if(CopyHigh(_Symbol, PERIOD_M5, 0, bars, h_m5) < bars) return;
   if(CopyLow(_Symbol, PERIOD_M5, 0, bars, l_m5) < bars) return;
   if(CopyOpen(_Symbol, PERIOD_M5, 0, bars, o_m5) < bars) return;
   if(CopyTickVolume(_Symbol, PERIOD_M5, 0, bars, v_m5_raw) < 0)
      ArrayInitialize(v_m5_raw, 0);
   // Convert volume to double for calculations
   double v_m5[];
   ArrayResize(v_m5, bars);
   for(int i = 0; i < bars; i++) v_m5[i] = (double)v_m5_raw[i];
   
   double close0 = c_m5[0];
   double eps = 1e-10;
   
   // â•â•â• Compute True Range array â•â•â•
   double tr[];
   ArrayResize(tr, bars);
   tr[bars-1] = h_m5[bars-1] - l_m5[bars-1];
   for(int i = bars-2; i >= 0; i--)
   {
      double hl = h_m5[i] - l_m5[i];
      double hc = MathAbs(h_m5[i] - c_m5[i+1]);
      double lc = MathAbs(l_m5[i] - c_m5[i+1]);
      tr[i] = MathMax(hl, MathMax(hc, lc));
   }
   
   // â•â•â• Pre-compute EMAs â•â•â•
   double ema5 = ComputeEMA(c_m5, 5, bars);
   double ema8 = ComputeEMA(c_m5, 8, bars);
   double ema21 = ComputeEMA(c_m5, 21, bars);
   double ema34 = ComputeEMA(c_m5, 34, bars);
   double ema55 = ComputeEMA(c_m5, 55, bars);
   double ema89 = ComputeEMA(c_m5, 89, bars);
   double ema200 = ComputeEMA(c_m5, 200, bars);
   
   // EMA slopes (5-bar lagged)
   double ema55_5ago = ComputeEMAShifted(c_m5, 55, 5, bars);
   
   // â•â•â• Pre-compute ATRs â•â•â•
   double atr7 = 0, atr14 = 0, atr21 = 0, atr50 = 0;
   for(int i = 0; i < 7; i++) atr7 += tr[i]; atr7 /= 7.0;
   for(int i = 0; i < 14; i++) atr14 += tr[i]; atr14 /= 14.0;
   for(int i = 0; i < 21; i++) atr21 += tr[i]; atr21 /= 21.0;
   for(int i = 0; i < 50; i++) atr50 += tr[i]; atr50 /= 50.0;
   
   
   // â•â•â• Pre-compute volatilities â•â•â•
   double vol5 = 0, vol10 = 0, vol20 = 0, vol50 = 0;
   {
      // vol_N = std of returns over N bars
      double s, sq;
      s = 0; sq = 0;
      for(int k = 0; k < 5 && k < bars-1; k++) { double r = (c_m5[k]-c_m5[k+1])/(c_m5[k+1]+eps); s+=r; sq+=r*r; }
      vol5 = MathSqrt(MathMax(sq/5.0-(s/5.0)*(s/5.0), 0));
      
      s = 0; sq = 0;
      for(int k = 0; k < 10 && k < bars-1; k++) { double r = (c_m5[k]-c_m5[k+1])/(c_m5[k+1]+eps); s+=r; sq+=r*r; }
      vol10 = MathSqrt(MathMax(sq/10.0-(s/10.0)*(s/10.0), 0));
      
      s = 0; sq = 0;
      for(int k = 0; k < 20 && k < bars-1; k++) { double r = (c_m5[k]-c_m5[k+1])/(c_m5[k+1]+eps); s+=r; sq+=r*r; }
      vol20 = MathSqrt(MathMax(sq/20.0-(s/20.0)*(s/20.0), 0));
      
      s = 0; sq = 0;
      for(int k = 0; k < 50 && k < bars-1; k++) { double r = (c_m5[k]-c_m5[k+1])/(c_m5[k+1]+eps); s+=r; sq+=r*r; }
      vol50 = MathSqrt(MathMax(sq/50.0-(s/50.0)*(s/50.0), 0));
   }
   
   // â•â•â• MACD â•â•â•
   double ema12 = ComputeEMA(c_m5, 12, bars);
   double ema26 = ComputeEMA(c_m5, 26, bars);
   double macd_line = ema12 - ema26;
   // MACD signal = EMA9 of MACD line (approximate)
   // Need MACD history for signal
   double macd_hist_arr[];
   ArrayResize(macd_hist_arr, 20);
   for(int i = 0; i < 20; i++)
   {
      double e12 = ComputeEMAShifted(c_m5, 12, i, bars);
      double e26 = ComputeEMAShifted(c_m5, 26, i, bars);
      macd_hist_arr[i] = e12 - e26;
   }
   double macd_signal = macd_hist_arr[0];
   {
      double alpha9 = 2.0/10.0;
      macd_signal = macd_hist_arr[MathMin(19, 9*3)];
      for(int i = MathMin(18, 9*3-1); i >= 0; i--)
         macd_signal = alpha9 * macd_hist_arr[i] + (1-alpha9) * macd_signal;
   }
   double macd_histogram = macd_line - macd_signal;
   
   // MACD hist slope (3-bar diff)
   double macd_hist_3ago = 0;
   {
      double e12_3 = ComputeEMAShifted(c_m5, 12, 3, bars);
      double e26_3 = ComputeEMAShifted(c_m5, 26, 3, bars);
      double ml3 = e12_3 - e26_3;
      // Signal 3 bars ago (approximate)
      double ms3 = macd_hist_arr[MathMin(19, 9*3)];
      double a9 = 2.0/10.0;
      for(int i = MathMin(18, 9*3-1); i >= 3; i--)
         ms3 = a9 * macd_hist_arr[i] + (1-a9) * ms3;
      macd_hist_3ago = ml3 - ms3;
   }
   
   // â•â•â• Stochastic â•â•â•
   double stoch_k_21 = 0, stoch_d_14 = 0;
   {
      double lo21 = l_m5[0], hi21 = h_m5[0];
      for(int k = 0; k < 21; k++) { if(l_m5[k]<lo21) lo21=l_m5[k]; if(h_m5[k]>hi21) hi21=h_m5[k]; }
      stoch_k_21 = (close0 - lo21) / (hi21 - lo21 + eps);
      
      double lo14 = l_m5[0], hi14 = h_m5[0];
      for(int k = 0; k < 14; k++) { if(l_m5[k]<lo14) lo14=l_m5[k]; if(h_m5[k]>hi14) hi14=h_m5[k]; }
      double sk14 = (close0 - lo14) / (hi14 - lo14 + eps);
      // D = 3-bar SMA of K
      double sk14_1 = 0, sk14_2 = 0;
      {
         double lo14_1 = l_m5[1], hi14_1 = h_m5[1];
         for(int k = 1; k < 15; k++) { if(l_m5[k]<lo14_1) lo14_1=l_m5[k]; if(h_m5[k]>hi14_1) hi14_1=h_m5[k]; }
         sk14_1 = (c_m5[1] - lo14_1) / (hi14_1 - lo14_1 + eps);
         double lo14_2 = l_m5[2], hi14_2 = h_m5[2];
         for(int k = 2; k < 16; k++) { if(l_m5[k]<lo14_2) lo14_2=l_m5[k]; if(h_m5[k]>hi14_2) hi14_2=h_m5[k]; }
         sk14_2 = (c_m5[2] - lo14_2) / (hi14_2 - lo14_2 + eps);
      }
      stoch_d_14 = (sk14 + sk14_1 + sk14_2) / 3.0;
   }
   
   // â•â•â• Position features â•â•â•
   double pos10 = 0, pos100 = 0, pos200 = 0;
   {
      double lo_p, hi_p;
      lo_p = l_m5[0]; hi_p = h_m5[0];
      for(int k = 0; k < 10; k++) { if(l_m5[k]<lo_p) lo_p=l_m5[k]; if(h_m5[k]>hi_p) hi_p=h_m5[k]; }
      pos10 = (close0 - lo_p) / (hi_p - lo_p + eps);
      
      lo_p = l_m5[0]; hi_p = h_m5[0];
      for(int k = 0; k < MathMin(100, bars); k++) { if(l_m5[k]<lo_p) lo_p=l_m5[k]; if(h_m5[k]>hi_p) hi_p=h_m5[k]; }
      pos100 = (close0 - lo_p) / (hi_p - lo_p + eps);
      
      lo_p = l_m5[0]; hi_p = h_m5[0];
      for(int k = 0; k < MathMin(200, bars); k++) { if(l_m5[k]<lo_p) lo_p=l_m5[k]; if(h_m5[k]>hi_p) hi_p=h_m5[k]; }
      pos200 = (close0 - lo_p) / (hi_p - lo_p + eps);
   }
   
   // â•â•â• Bollinger â•â•â•
   double bb_pos_50 = 0, bb_width_20 = 0;
   {
      // BB(50)
      double sum_50 = 0, sumsq_50 = 0;
      for(int k = 0; k < 50 && k < bars; k++) { sum_50 += c_m5[k]; sumsq_50 += c_m5[k]*c_m5[k]; }
      double mean_50 = sum_50/50.0;
      double std_50 = MathSqrt(MathMax(sumsq_50/50.0 - mean_50*mean_50, 0));
      bb_pos_50 = (close0 - (mean_50 - 2*std_50)) / (4*std_50 + eps);
      
      // BB(20) width
      double sum_20 = 0, sumsq_20 = 0;
      for(int k = 0; k < 20 && k < bars; k++) { sum_20 += c_m5[k]; sumsq_20 += c_m5[k]*c_m5[k]; }
      double mean_20 = sum_20/20.0;
      double std_20 = MathSqrt(MathMax(sumsq_20/20.0 - mean_20*mean_20, 0));
      bb_width_20 = (4*std_20) / (mean_20 + eps);
   }
   
   // â•â•â• Returns â•â•â•
   double ret_13 = (close0 - c_m5[13]) / (c_m5[13] + eps);
   double ret_34 = (close0 - c_m5[34]) / (c_m5[34] + eps);
   double log_ret_20 = MathLog(close0 / (c_m5[20] + eps));
   
   // â•â•â• Volume ratios â•â•â•
   double vol_avg_10 = 0, vol_avg_50 = 0;
   for(int k = 0; k < 10; k++) vol_avg_10 += v_m5[k]; vol_avg_10 /= 10.0;
   for(int k = 0; k < 50; k++) vol_avg_50 += v_m5[k]; vol_avg_50 /= 50.0;
   double vol_ratio_10 = v_m5[0] / (vol_avg_10 + eps) - 1;
   double vol_ratio_50 = v_m5[0] / (vol_avg_50 + eps) - 1;
   
   // OBV slope
   double obv = 0;
   for(int k = bars-2; k >= 0; k--)
   {
      double sign = (c_m5[k] > c_m5[k+1]) ? 1.0 : -1.0;
      obv += v_m5[k] * sign;
   }
   double obv_10ago = 0;
   for(int k = bars-2; k >= 10; k--)
   {
      double sign = (c_m5[k] > c_m5[k+1]) ? 1.0 : -1.0;
      obv_10ago += v_m5[k] * sign;
   }
   double obv_slope = (obv_10ago > 0) ? (obv - obv_10ago) / (MathAbs(obv_10ago) + eps) : 0;
   
   // â•â•â• is_bull (current candle) â•â•â•
   double is_bull = (c_m5[0] > o_m5[0]) ? 1.0 : 0.0;
   
   // â•â•â• Time features â•â•â•
   MqlDateTime dt;
   TimeCurrent(dt);
   double hour_sin = MathSin(2.0 * M_PI * dt.hour / 24.0);
   double hour_cos = MathCos(2.0 * M_PI * dt.hour / 24.0);
   double dow_sin = MathSin(2.0 * M_PI * dt.day_of_week / 5.0);
   double dow_cos = MathCos(2.0 * M_PI * dt.day_of_week / 5.0);
   double month_sin = MathSin(2.0 * M_PI * dt.mon / 12.0);
   double dom = dt.day / 31.0;
   double is_london = (dt.hour >= 7 && dt.hour < 15) ? 1.0 : 0.0;
   double is_ny = (dt.hour >= 12 && dt.hour < 21) ? 1.0 : 0.0;
   double is_overlap = (dt.hour >= 12 && dt.hour < 15) ? 1.0 : 0.0;
   
   // â•â•â• Higher TF context â•â•â•
   double htf_1h_ret5 = 0, htf_1h_rsi = 0, htf_1h_trend = 0;
   double htf_4h_ret5 = 0, htf_4h_rsi = 0, htf_4h_trend = 0;
   double htf_1d_ret5 = 0, htf_1d_rsi = 0, htf_1d_trend = 0;
   {
      double c_h1[]; ArrayResize(c_h1, 60); ArraySetAsSeries(c_h1, true);
      if(CopyClose(_Symbol, PERIOD_H1, 0, 60, c_h1) >= 60)
      {
         htf_1h_ret5 = (c_h1[0] - c_h1[5]) / (c_h1[5] + eps);
         htf_1h_rsi = ComputeRSI(c_h1, 14, 60);
         double e21_h1 = ComputeEMA(c_h1, 21, 60);
         double e55_h1 = ComputeEMA(c_h1, 55, 60);
         htf_1h_trend = (e21_h1 - e55_h1) / (e55_h1 + eps);
      }
      
      double c_h4[]; ArrayResize(c_h4, 60); ArraySetAsSeries(c_h4, true);
      if(CopyClose(_Symbol, PERIOD_H4, 0, 60, c_h4) >= 60)
      {
         htf_4h_ret5 = (c_h4[0] - c_h4[5]) / (c_h4[5] + eps);
         htf_4h_rsi = ComputeRSI(c_h4, 14, 60);
         double e21_h4 = ComputeEMA(c_h4, 21, 60);
         double e55_h4 = ComputeEMA(c_h4, 55, 60);
         htf_4h_trend = (e21_h4 - e55_h4) / (e55_h4 + eps);
      }
      
      double c_d1[]; ArrayResize(c_d1, 60); ArraySetAsSeries(c_d1, true);
      if(CopyClose(_Symbol, PERIOD_D1, 0, 60, c_d1) >= 60)
      {
         htf_1d_ret5 = (c_d1[0] - c_d1[5]) / (c_d1[5] + eps);
         htf_1d_rsi = ComputeRSI(c_d1, 14, 60);
         double e21_d1 = ComputeEMA(c_d1, 21, 60);
         double e55_d1 = ComputeEMA(c_d1, 55, 60);
         htf_1d_trend = (e21_d1 - e55_d1) / (e55_d1 + eps);
      }
   }
   
   // â•â•â• Intermarket correlations â•â•â•
   // Uses resolved symbols from OnInit (broker-agnostic)
   double corr_xag_ret5 = 0, corr_xag_20 = 0;
   double corr_usdjpy_ret5 = 0, corr_eurusd_ret5 = 0, corr_btcusd_ret5 = 0;
   {
      // Silver
      if(sym_XAGUSD != "")
      {
         double c_xag[]; ArrayResize(c_xag, 25); ArraySetAsSeries(c_xag, true);
         if(CopyClose(sym_XAGUSD, PERIOD_H1, 0, 25, c_xag) >= 6)
            corr_xag_ret5 = (c_xag[0] - c_xag[5]) / (c_xag[5] + eps);
      }
      
      // USD/JPY
      if(sym_USDJPY != "")
      {
         double c_jpy[]; ArrayResize(c_jpy, 10); ArraySetAsSeries(c_jpy, true);
         if(CopyClose(sym_USDJPY, PERIOD_H1, 0, 10, c_jpy) >= 6)
            corr_usdjpy_ret5 = (c_jpy[0] - c_jpy[5]) / (c_jpy[5] + eps);
      }
      
      // EUR/USD
      if(sym_EURUSD != "")
      {
         double c_eur[]; ArrayResize(c_eur, 10); ArraySetAsSeries(c_eur, true);
         if(CopyClose(sym_EURUSD, PERIOD_H1, 0, 10, c_eur) >= 6)
            corr_eurusd_ret5 = (c_eur[0] - c_eur[5]) / (c_eur[5] + eps);
      }
      
      // BTC/USD
      if(sym_BTCUSD != "")
      {
         double c_btc[]; ArrayResize(c_btc, 10); ArraySetAsSeries(c_btc, true);
         if(CopyClose(sym_BTCUSD, PERIOD_H1, 0, 10, c_btc) >= 6)
            corr_btcusd_ret5 = (c_btc[0] - c_btc[5]) / (c_btc[5] + eps);
      }
   }
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  BUILD 59 FEATURES â€” EXACT ORDER from V5 v5_features.json
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   
   // â”€â”€ Additional V5 computations â”€â”€
   double log_ret_50 = MathLog(close0 / (c_m5[50] + eps));
   double log_ret_100 = MathLog(close0 / (c_m5[MathMin(100, bars-1)] + eps));
   double ema13 = ComputeEMA(c_m5, 13, bars);
   double ret_21 = (close0 - c_m5[21]) / (c_m5[21] + eps);
   
   // EMA stack: 1 if ema5>ema8>ema21>ema55, -1 if reverse, 0 otherwise
   double ema_stack = 0;
   if(ema5 > ema8 && ema8 > ema21 && ema21 > ema55) ema_stack = 1.0;
   else if(ema5 < ema8 && ema8 < ema21 && ema21 < ema55) ema_stack = -1.0;
   
   // BB squeeze: bb_width < threshold (low volatility)
   double bb_squeeze_50 = 0;
   {
      double sum50b = 0, sumsq50b = 0;
      for(int k = 0; k < 50 && k < bars; k++) { sum50b += c_m5[k]; sumsq50b += c_m5[k]*c_m5[k]; }
      double mean50b = sum50b/50.0;
      double std50b = MathSqrt(MathMax(sumsq50b/50.0 - mean50b*mean50b, 0));
      double bw50 = (4*std50b) / (mean50b + eps);
      bb_squeeze_50 = (bw50 < 0.01) ? 1.0 : 0.0;  // Squeeze when width < 1%
   }
   
   // Position in 20-bar and 50-bar range
   double pos20 = 0, pos50 = 0;
   {
      double lo_p, hi_p;
      lo_p = l_m5[0]; hi_p = h_m5[0];
      for(int k = 0; k < 20 && k < bars; k++) { if(l_m5[k]<lo_p) lo_p=l_m5[k]; if(h_m5[k]>hi_p) hi_p=h_m5[k]; }
      pos20 = (close0 - lo_p) / (hi_p - lo_p + eps);
      lo_p = l_m5[0]; hi_p = h_m5[0];
      for(int k = 0; k < 50 && k < bars; k++) { if(l_m5[k]<lo_p) lo_p=l_m5[k]; if(h_m5[k]>hi_p) hi_p=h_m5[k]; }
      pos50 = (close0 - lo_p) / (hi_p - lo_p + eps);
   }
   
   // BB width 50 and BB pos 20
   double bb_width_50 = 0, bb_pos_20 = 0;
   {
      double sum50c = 0, sumsq50c = 0;
      for(int k = 0; k < 50 && k < bars; k++) { sum50c += c_m5[k]; sumsq50c += c_m5[k]*c_m5[k]; }
      double m50 = sum50c/50.0;
      double s50 = MathSqrt(MathMax(sumsq50c/50.0 - m50*m50, 0));
      bb_width_50 = (4*s50) / (m50 + eps);
      
      double sum20b = 0, sumsq20b = 0;
      for(int k = 0; k < 20 && k < bars; k++) { sum20b += c_m5[k]; sumsq20b += c_m5[k]*c_m5[k]; }
      double m20 = sum20b/20.0;
      double s20 = MathSqrt(MathMax(sumsq20b/20.0 - m20*m20, 0));
      bb_pos_20 = (close0 - (m20 - 2*s20)) / (4*s20 + eps);
   }
   
   // Volume ratio 20
   double vol_ratio_20 = 0;
   {
      double va20 = 0;
      for(int k = 0; k < 20; k++) va20 += v_m5[k]; va20 /= 20.0;
      vol_ratio_20 = v_m5[0] / (va20 + eps) - 1;
   }
   
   // Stochastic D 21 (3-bar SMA of K 21)
   double stoch_d_21 = 0;
   {
      double sk21_1 = 0, sk21_2 = 0;
      double lo21_1 = l_m5[1], hi21_1 = h_m5[1];
      for(int k = 1; k < 22 && k < bars; k++) { if(l_m5[k]<lo21_1) lo21_1=l_m5[k]; if(h_m5[k]>hi21_1) hi21_1=h_m5[k]; }
      sk21_1 = (c_m5[1] - lo21_1) / (hi21_1 - lo21_1 + eps);
      double lo21_2 = l_m5[2], hi21_2 = h_m5[2];
      for(int k = 2; k < 23 && k < bars; k++) { if(l_m5[k]<lo21_2) lo21_2=l_m5[k]; if(h_m5[k]>hi21_2) hi21_2=h_m5[k]; }
      sk21_2 = (c_m5[2] - lo21_2) / (hi21_2 - lo21_2 + eps);
      stoch_d_21 = (stoch_k_21 + sk21_1 + sk21_2) / 3.0;
   }
   
   float features[];
   ArrayResize(features, N_ONNX_FEATURES);
   ArrayInitialize(features, 0);
   
   // V5 feature order (59 features) from v5_features.json
   features[0]  = (float)log_ret_50;                              // log_ret_50
   features[1]  = (float)htf_1h_ret5;                            // htf_1h_ret5
   features[2]  = (float)is_london;                               // is_london
   features[3]  = (float)hour_sin;                                // hour_sin
   features[4]  = (float)htf_4h_ret5;                            // htf_4h_ret5
   features[5]  = (float)((ema21 - ema55) / (ema55 + eps));      // ema_21_55
   features[6]  = (float)((close0 - ema200) / (close0 + eps));   // ema200_dist
   features[7]  = (float)((ema55 - ema200) / (ema200 + eps));    // ema_55_200
   features[8]  = (float)(atr14 / (close0 + eps));               // atr_14
   features[9]  = (float)is_overlap;                              // is_overlap
   features[10] = (float)hour_cos;                                // hour_cos
   features[11] = (float)htf_1h_rsi;                             // htf_1h_rsi
   features[12] = (float)((ema55 - ema55_5ago) / (ema55_5ago + eps)); // ema55_slope
   features[13] = (float)pos200;                                  // pos_200
   features[14] = (float)ema_stack;                               // ema_stack
   features[15] = (float)dow_sin;                                 // dow_sin
   features[16] = (float)htf_4h_rsi;                             // htf_4h_rsi
   features[17] = (float)corr_xag_ret5;                          // corr_XAGUSD_ret5
   features[18] = (float)pos10;                                   // pos_10
   features[19] = (float)dow_cos;                                 // dow_cos
   features[20] = (float)bb_squeeze_50;                           // bb_squeeze_50
   features[21] = (float)is_bull;                                 // is_bull
   features[22] = (float)htf_1d_ret5;                            // htf_1d_ret5
   features[23] = (float)pos100;                                  // pos_100
   features[24] = (float)log_ret_100;                             // log_ret_100
   features[25] = (float)(atr50 / (close0 + eps));               // atr_50
   features[26] = (float)(atr21 / (close0 + eps));               // atr_21
   features[27] = (float)htf_1h_trend;                           // htf_1h_trend
   features[28] = (float)vol50;                                   // vol_50
   features[29] = (float)((close0 - ema13) / (close0 + eps));   // ema13_dist
   features[30] = (float)htf_1d_trend;                           // htf_1d_trend
   features[31] = (float)stoch_k_21;                              // stoch_k_21
   features[32] = (float)pos50;                                   // pos_50
   features[33] = (float)bb_width_50;                             // bb_width_50
   features[34] = (float)vol_ratio_20;                            // vol_ratio_20
   features[35] = (float)htf_4h_trend;                           // htf_4h_trend
   features[36] = (float)htf_1d_rsi;                             // htf_1d_rsi
   features[37] = (float)vol20;                                   // vol_20
   features[38] = (float)((close0 - ema8) / (close0 + eps));    // ema8_dist
   features[39] = (float)pos20;                                   // pos_20
   features[40] = (float)corr_usdjpy_ret5;                       // corr_USDJPY_ret5
   features[41] = (float)corr_btcusd_ret5;                       // corr_BTCUSD_ret5
   features[42] = (float)corr_eurusd_ret5;                       // corr_EURUSD_ret5
   features[43] = (float)ret_34;                                  // ret_34
   features[44] = (float)bb_width_20;                             // bb_width_20
   features[45] = (float)dom;                                     // dom
   features[46] = (float)vol10;                                   // vol_10
   features[47] = (float)(macd_signal / (close0 + eps));         // macd_signal
   features[48] = (float)((ema8 - ema21) / (ema21 + eps));       // ema_8_21
   features[49] = (float)obv_slope;                               // obv_slope
   features[50] = (float)bb_pos_20;                               // bb_pos_20
   features[51] = (float)ret_21;                                  // ret_21
   features[52] = (float)stoch_d_21;                              // stoch_d_21
   features[53] = (float)ret_13;                                  // ret_13
   features[54] = (float)vol_ratio_10;                            // vol_ratio_10
   
   // HMM Vol-Regime: compute live from M5 closes (matches training GaussianHMM)
   int hmmRegime = ClassifyHMMRegime(c_m5, bars);
   features[55] = (float)hmmRegime;                               // hmm_regime (0, 1, or 2)
   features[56] = (float)(hmmRegime == 0 ? 1.0 : 0.0);            // regime_0 (one-hot)
   features[57] = (float)(hmmRegime == 1 ? 1.0 : 0.0);            // regime_1 (one-hot)
   features[58] = (float)(hmmRegime == 2 ? 1.0 : 0.0);            // regime_2 (one-hot)
   
   // ═══ Features 59-66 (from V9) + 67-76 (V10 NEW) ═══
   // 59: vwap_dist — daily VWAP distance
   {
      MqlDateTime vwapDt; TimeCurrent(vwapDt);
      datetime dayStart = StringToTime(StringFormat("%04d.%02d.%02d 00:00", vwapDt.year, vwapDt.mon, vwapDt.day));
      int barsToday = Bars(_Symbol, PERIOD_M5, dayStart, TimeCurrent());
      int nv = MathMin(barsToday, MathMin(300, bars));
      double cumTPV = 0, cumVol = 0;
      for(int k = nv - 1; k >= 0; k--)
      {
         double tp = (h_m5[k] + l_m5[k] + c_m5[k]) / 3.0;
         cumTPV += tp * v_m5[k];
         cumVol += v_m5[k];
      }
      double vwap = (cumVol > 0) ? cumTPV / cumVol : close0;
      features[59] = (float)((close0 - vwap) / (vwap + eps));
   }
   // 60: candle_body_ratio
   {
      double body = MathAbs(c_m5[0] - o_m5[0]);
      double range = h_m5[0] - l_m5[0];
      features[60] = (float)(body / (range + eps));
   }
   // 61: upper_wick_ratio
   {
      double upper = h_m5[0] - MathMax(c_m5[0], o_m5[0]);
      double range = h_m5[0] - l_m5[0];
      features[61] = (float)(upper / (range + eps));
   }
   // 62: lower_wick_ratio
   {
      double lower = MathMin(c_m5[0], o_m5[0]) - l_m5[0];
      double range = h_m5[0] - l_m5[0];
      features[62] = (float)(lower / (range + eps));
   }
   // 63: spread_norm — candle range / ATR14
   {
      double range = h_m5[0] - l_m5[0];
      features[63] = (float)(range / (atr14 + eps));
   }
   // 64: atr_ratio_14_50
   features[64] = (float)(atr14 / (atr50 + eps));
   // 65: rsi_14 (0-1 scale)
   features[65] = (float)ComputeRSI(c_m5, 14, bars);
   // 66: obv_accel — OBV MA5 vs MA20
   {
      double obv_arr[25];
      ArrayInitialize(obv_arr, 0);
      obv_arr[24] = 0;
      for(int k = 23; k >= 0; k--)
      {
         double sign = (c_m5[k] > c_m5[k+1]) ? 1.0 : (c_m5[k] < c_m5[k+1] ? -1.0 : 0.0);
         obv_arr[k] = obv_arr[k+1] + sign * v_m5[k];
      }
      double maFast = 0, maSlow = 0;
      for(int k = 0; k < 5; k++) maFast += obv_arr[k]; maFast /= 5.0;
      for(int k = 0; k < 20; k++) maSlow += obv_arr[k]; maSlow /= 20.0;
      features[66] = (float)((maFast - maSlow) / (MathAbs(maSlow) + eps));
   }
   
   // ═══ V10 NEW FEATURES (indices 67-76) ═══
   // 67: vol_ratio_10_50 — short vs long volume ratio
   {
      double vol10s = 0, vol50s = 0;
      for(int k = 0; k < 10 && k < bars; k++) vol10s += v_m5[k];
      for(int k = 0; k < 50 && k < bars; k++) vol50s += v_m5[k];
      double avg10 = vol10s / 10.0;
      double avg50 = vol50s / 50.0;
      features[67] = (float)((avg50 > eps) ? (avg10 / avg50) : 1.0);
   }
   // 68: vol_regime — ATR percentile-based (0=low, 1=normal, 2=high)
   {
      double atr_current = atr14;
      int belowCount = 0;
      int lookN = MathMin(100, bars - 15);
      for(int k = 14; k < 14 + lookN; k++)
      {
         double atr_k = 0;
         for(int j = k; j < k + 14 && j < bars - 1; j++) atr_k += tr[j];
         atr_k /= 14.0;
         if(atr_k < atr_current) belowCount++;
      }
      double atr_pct = (lookN > 0) ? (double)belowCount / lookN : 0.5;
      double vol_regime_val = (atr_pct < 0.33) ? 0.0 : (atr_pct < 0.67) ? 1.0 : 2.0;
      features[68] = (float)vol_regime_val;
   }
   // 69: obv_derivative_5 — 5-bar rate of change of OBV MA
   {
      double obv_now = 0, obv_5ago = 0;
      for(int k = 0; k < 5 && k < bars; k++)
      {
         double sign_n = (c_m5[k] > c_m5[k+1]) ? 1.0 : (c_m5[k] < c_m5[k+1] ? -1.0 : 0.0);
         obv_now += sign_n * v_m5[k];
      }
      for(int k = 5; k < 10 && k < bars; k++)
      {
         double sign_n = (c_m5[k] > c_m5[k+1]) ? 1.0 : (c_m5[k] < c_m5[k+1] ? -1.0 : 0.0);
         obv_5ago += sign_n * v_m5[k];
      }
      features[69] = (float)((MathAbs(obv_5ago) > eps) ? (obv_now - obv_5ago) / (MathAbs(obv_5ago) + eps) : 0.0);
   }
   // 70: order_block_proximity — nearest bullish/bearish OB (0-1)
   {
      double min_ob_dist = 999999.0;
      for(int k = 1; k < 50 && k + 1 < bars; k++)
      {
         // Bullish OB: prior bearish candle, current breaks above
         if(o_m5[k+1] > c_m5[k+1] && c_m5[k] > o_m5[k])
         {
            double ob_top = MathMax(o_m5[k+1], c_m5[k+1]);
            double dist = MathAbs(close0 - ob_top) / (atr14 + eps);
            if(dist < min_ob_dist) min_ob_dist = dist;
         }
         // Bearish OB: prior bullish candle, current breaks below
         if(c_m5[k+1] > o_m5[k+1] && o_m5[k] > c_m5[k])
         {
            double ob_bot = MathMin(o_m5[k+1], c_m5[k+1]);
            double dist = MathAbs(close0 - ob_bot) / (atr14 + eps);
            if(dist < min_ob_dist) min_ob_dist = dist;
         }
      }
      // Convert distance to proximity (0=far, 1=on top of OB)
      double obp = (min_ob_dist < 5.0) ? (1.0 - min_ob_dist / 5.0) : 0.0;
      features[70] = (float)MathMax(0.0, MathMin(1.0, obp));
   }
   // 71: fvg_fill_pct — fraction of recent FVGs filled
   {
      int fvg_count = 0;
      int fvg_filled = 0;
      for(int k = 1; k < 20 && k + 2 < bars; k++)
      {
         // Bullish FVG: gap between bar[k+2] high and bar[k] low
         double fvg_top = l_m5[k];
         double fvg_bot = h_m5[k + 2];
         if(fvg_top > fvg_bot + atr14 * 0.1)
         {
            fvg_count++;
            // Check if filled: price went below fvg_top
            for(int j = k - 1; j >= 0 && j < bars; j++)
            {
               if(l_m5[j] <= fvg_top) { fvg_filled++; break; }
            }
         }
         // Bearish FVG: gap between bar[k+2] low and bar[k] high
         double fvg_top_b = l_m5[k + 2];
         double fvg_bot_b = h_m5[k];
         if(fvg_top_b > fvg_bot_b + atr14 * 0.1)
         {
            fvg_count++;
            for(int j = k - 1; j >= 0 && j < bars; j++)
            {
               if(h_m5[j] >= fvg_top_b) { fvg_filled++; break; }
            }
         }
      }
      features[71] = (float)((fvg_count > 0) ? (double)fvg_filled / fvg_count : 0.5);
   }
   // 72 & 73: bos_bullish, bos_bearish — Break of Structure
   {
      double bos_bull = 0.0, bos_bear = 0.0;
      // Find swing highs/lows in lookback, check if current breaks them
      double swing_hi = 0, swing_lo = 999999;
      for(int k = 2; k < 50 && k + 1 < bars; k++)
      {
         if(h_m5[k] > h_m5[k-1] && h_m5[k] > h_m5[k+1] && h_m5[k] > swing_hi)
            swing_hi = h_m5[k];
         if(l_m5[k] < l_m5[k-1] && l_m5[k] < l_m5[k+1] && l_m5[k] < swing_lo)
            swing_lo = l_m5[k];
      }
      if(close0 > swing_hi && swing_hi > 0) bos_bull = 1.0;
      if(close0 < swing_lo && swing_lo < 999999) bos_bear = 1.0;
      features[72] = (float)bos_bull;
      features[73] = (float)bos_bear;
   }
   // 74: structure_bias = bos_bullish - bos_bearish
   features[74] = (float)(features[72] - features[73]);
   // 75: ema21_dist = (close - ema21) / ema21
   features[75] = (float)((ema21 > eps) ? (close0 - ema21) / ema21 : 0.0);
   // 76: ema55_dist = (close - ema55) / ema55
   features[76] = (float)((ema55 > eps) ? (close0 - ema55) / ema55 : 0.0);
   // ═══ End V10 New Features ═══
   
   // Debug: log HMM + intermarket diagnostics every 10th tick
   if(Debug && tickCount % 10 == 1)
      LP("ðŸ“Š Features: HMM=", hmmRegime, 
            " [", features[56], "/", features[57], "/", features[58], "]",
            " XAG=", DoubleToString(corr_xag_ret5, 5),
            " JPY=", DoubleToString(corr_usdjpy_ret5, 5),
            " EUR=", DoubleToString(corr_eurusd_ret5, 5),
            " BTC=", DoubleToString(corr_btcusd_ret5, 5));
   
   // Clip to [-100, 100] matching training
   for(int i = 0; i < N_ONNX_FEATURES; i++)
      features[i] = (float)MathMax(-100, MathMin(100, features[i]));
   
   // ═══ Run V10 3-Model Ensemble ONNX Inference ═══
   // V10: Per-model probs stored as float for V10_EnsembleProb + V10_MetaConfidence
   float f_cat_probs[3], f_xgb_probs[3], f_lgb_probs[3];
   ArrayInitialize(f_cat_probs, 0.0f);
   ArrayInitialize(f_xgb_probs, 0.0f);
   ArrayInitialize(f_lgb_probs, 0.0f);
   int modelsRun = 0;
   uint startTick = GetTickCount();
   
   // XGBoost (V10: DOMINANT weight 52.7%)
   if(h_xgb != INVALID_HANDLE)
   {
      long pred_xgb[]; float prob_xgb[];
      ArrayResize(pred_xgb, 1); ArrayResize(prob_xgb, 3);
      if(OnnxRun(h_xgb, ONNX_DEFAULT, features, pred_xgb, prob_xgb))
      {
         modelsRun++;
         f_xgb_probs[0] = prob_xgb[0]; f_xgb_probs[1] = prob_xgb[1]; f_xgb_probs[2] = prob_xgb[2];
         g_xgb_probs[0] = (double)prob_xgb[0]; g_xgb_probs[1] = (double)prob_xgb[1]; g_xgb_probs[2] = (double)prob_xgb[2];
      }
      else if(Debug) LP("⚠️ XGBoost inference failed");
   }
   
   // LightGBM (V10: ZipMap removed during export)
   if(h_lgb != INVALID_HANDLE && UseONNXEnsemble)
   {
      long pred_lgb[]; float prob_lgb[];
      ArrayResize(pred_lgb, 1); ArrayResize(prob_lgb, 3);
      if(OnnxRun(h_lgb, ONNX_DEFAULT, features, pred_lgb, prob_lgb))
      {
         modelsRun++;
         f_lgb_probs[0] = prob_lgb[0]; f_lgb_probs[1] = prob_lgb[1]; f_lgb_probs[2] = prob_lgb[2];
         g_lgb_probs[0] = (double)prob_lgb[0]; g_lgb_probs[1] = (double)prob_lgb[1]; g_lgb_probs[2] = (double)prob_lgb[2];
      }
      else if(Debug) LP("⚠ LightGBM inference failed: ", GetLastError());
   }
   
   // CatBoost (V10: ZipMap removed during export)
   if(h_cat != INVALID_HANDLE && UseONNXEnsemble)
   {
      long pred_cat[]; float prob_cat[];
      ArrayResize(pred_cat, 1); ArrayResize(prob_cat, 3);
      if(OnnxRun(h_cat, ONNX_DEFAULT, features, pred_cat, prob_cat))
      {
         modelsRun++;
         f_cat_probs[0] = prob_cat[0]; f_cat_probs[1] = prob_cat[1]; f_cat_probs[2] = prob_cat[2];
         g_cat_probs[0] = (double)prob_cat[0]; g_cat_probs[1] = (double)prob_cat[1]; g_cat_probs[2] = (double)prob_cat[2];
      }
      else if(Debug) LP("⚠ CatBoost inference failed: ", GetLastError());
   }
   
   uint elapsed = GetTickCount() - startTick;
   
   if(modelsRun == 0) { if(Debug) LP("⌛ All ONNX models failed"); return; }
   
   // V10: Compute ensemble probs using LEARNED weights via V10_EnsembleProb
   double ens_probs[3];
   V10_EnsembleProb(f_xgb_probs, f_lgb_probs, f_cat_probs, ens_probs);
   
   double pBuy  = ens_probs[0];
   double pHold = ens_probs[1];
   double pSell = ens_probs[2];
   
   // Determine signal from V10 weighted ensemble
   string onnxSignal = "HOLD";
   double onnxConf = 0;
   
   if(pBuy > pHold && pBuy > pSell)       { onnxSignal = "BUY";  onnxConf = pBuy; }
   else if(pSell > pHold && pSell > pBuy) { onnxSignal = "SELL"; onnxConf = pSell; }
   else                                   { onnxSignal = "HOLD"; onnxConf = pHold; }
   
   // V10: Meta-Confidence Gate — L2 logistic regression filter
   double v10_meta_conf_val = 0;
   bool v10_meta_gate_pass = V10_MetaConfidenceGate(f_cat_probs, f_xgb_probs, f_lgb_probs, ens_probs, v10_meta_conf_val);
   g_v10_meta_conf = v10_meta_conf_val;
   
   if(onnxSignal != "HOLD" && !v10_meta_gate_pass)
   {
      if(Debug && tickCount % 30 == 0)
         LP("V10 META-GATE REJECT: ", onnxSignal, " metaConf=", DoubleToString(v10_meta_conf_val*100,1),
               "% < ", DoubleToString(V10_MetaConfThreshold*100,0), "%");
      onnxSignal = "HOLD";
      onnxConf = pHold;
   }
   
   // V10: Confidence margin filter — reject if winner barely beats runner-up
   if(onnxSignal != "HOLD" && !V10_HasConfidenceMargin(pBuy, pHold, pSell))
   {
      if(Debug && tickCount % 30 == 0)
         LP("V10 MARGIN REJECT: ", onnxSignal, " B:", DoubleToString(pBuy*100,0),
               " N:", DoubleToString(pHold*100,0), " S:", DoubleToString(pSell*100,0),
               " margin<", DoubleToString(V10_ConfidenceMargin*100,0), "%");
      onnxSignal = "HOLD";
      onnxConf = pHold;
   }
   
   g_ens_probs[0] = pBuy; g_ens_probs[1] = pHold; g_ens_probs[2] = pSell;
   g_models_active = modelsRun;
   
   if(Debug && !V10_IsTester() && tickCount % 10 == 0)
      LP("V10 Ensemble (", modelsRun, "x): ", onnxSignal, " conf=", DoubleToString(onnxConf*100, 1), 
            "% meta=", DoubleToString(g_v10_meta_conf*100,0), "%",
            " [B:", DoubleToString(pBuy*100,0), 
            " N:", DoubleToString(pHold*100,0), 
            " S:", DoubleToString(pSell*100,0), 
            "] ", elapsed, "ms");
   
   // V10: Signal stability filter — require persistence before acting
   if(onnxSignal != "HOLD" && !V10_IsSignalStable(onnxSignal))
   {
      if(Debug && !V10_IsTester() && tickCount % 30 == 0)
         LP("V10 PERSIST WAIT: ", onnxSignal, " streak=", g_v10_signal_streak,
               "/", V10_SignalPersistence);
      onnxSignal = "HOLD";
      onnxConf = pHold;
   }
   
   // V10 ENSEMBLE IS PRIMARY BRAIN
   aiSig.signal = onnxSignal;
   aiSig.confidence = onnxConf;
   aiSig.isValid = true;
   aiSig.timestamp = (datetime)TimeGMT();
   aiSig.version = "V10-Ensemble";
}

//+------------------------------------------------------------------+
//| Derive DXY/Regime/ORB from intermarket + indicators               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| V10 Macro Brain: ONNX inference + indicator fallback              |
//+------------------------------------------------------------------+
void DeriveMacroSignals()
{
   // If macro ONNX models loaded, use them
   if(g_macro_loaded > 0)
   {
      // Build 32-feature vector matching v6_macro_features.txt
      float macroFeat[32];
      ArrayInitialize(macroFeat, 0.0f);
      
      // Gold price data
      double goldClose[];
      if(CopyClose(_Symbol, PERIOD_H1, 0, 97, goldClose) >= 97)
      {
         int n = ArraySize(goldClose);
         macroFeat[0]  = (float)((goldClose[n-1] - goldClose[n-2]) / goldClose[n-2]);  // gold_ret_1
         macroFeat[1]  = (float)((goldClose[n-1] - goldClose[n-5]) / goldClose[n-5]);  // gold_ret_4
         macroFeat[2]  = (float)((goldClose[n-1] - goldClose[n-17]) / goldClose[n-17]); // gold_ret_16
         macroFeat[3]  = (float)((goldClose[n-1] - goldClose[n-97]) / goldClose[n-97]); // gold_ret_96
      }
      
      // Gold ATR
      double atrBuf[];
      if(CopyBuffer(h_atr, 0, 0, 15, atrBuf) >= 15)
      {
         macroFeat[4] = (float)atrBuf[14];  // gold_atr_14
         double atrMean = 0; for(int a=0; a<14; a++) atrMean += atrBuf[a]; atrMean /= 14.0;
         macroFeat[5] = (atrMean > 0) ? (float)(atrBuf[14] / atrMean) : 1.0f; // gold_atr_ratio
      }
      
       // Gold RSI
       double rsiBuf[];
       if(h_macro_rsi_xau != INVALID_HANDLE && CopyBuffer(h_macro_rsi_xau, 0, 0, 1, rsiBuf) == 1)
          macroFeat[6] = (float)(rsiBuf[0] / 100.0);
       
       // Gold BB width
       double bbUpper[], bbLower[], bbMid[];
       if(h_macro_bb_xau != INVALID_HANDLE)
       {
          if(CopyBuffer(h_macro_bb_xau, 1, 0, 1, bbUpper) == 1 && CopyBuffer(h_macro_bb_xau, 2, 0, 1, bbLower) == 1 &&
             CopyBuffer(h_macro_bb_xau, 0, 0, 1, bbMid) == 1 && bbMid[0] > 0)
             macroFeat[7] = (float)((bbUpper[0] - bbLower[0]) / bbMid[0]);
       }
      
      // EUR returns + RSI
      if(sym_EURUSD != "")
      {
         double eurClose[];
         if(CopyClose(sym_EURUSD, PERIOD_H1, 0, 17, eurClose) >= 17)
         {
            int ne = ArraySize(eurClose);
            macroFeat[8]  = (float)((eurClose[ne-1] - eurClose[ne-2]) / eurClose[ne-2]); // eur_ret_1
            macroFeat[9]  = (float)((eurClose[ne-1] - eurClose[ne-5]) / eurClose[ne-5]); // eur_ret_4
            macroFeat[10] = (float)((eurClose[ne-1] - eurClose[ne-17]) / eurClose[ne-17]); // eur_ret_16
         }
          double eurRsi[];
          if(h_macro_rsi_eur != INVALID_HANDLE && CopyBuffer(h_macro_rsi_eur, 0, 0, 1, eurRsi) == 1)
             macroFeat[11] = (float)(eurRsi[0] / 100.0);
      }
      
      // JPY returns + RSI
      if(sym_USDJPY != "")
      {
         double jpyClose[];
         if(CopyClose(sym_USDJPY, PERIOD_H1, 0, 17, jpyClose) >= 17)
         {
            int nj = ArraySize(jpyClose);
            macroFeat[12] = (float)((jpyClose[nj-1] - jpyClose[nj-2]) / jpyClose[nj-2]); // jpy_ret_1
            macroFeat[13] = (float)((jpyClose[nj-1] - jpyClose[nj-5]) / jpyClose[nj-5]); // jpy_ret_4
            macroFeat[14] = (float)((jpyClose[nj-1] - jpyClose[nj-17]) / jpyClose[nj-17]); // jpy_ret_16
         }
          double jpyRsi[];
          if(h_macro_rsi_jpy != INVALID_HANDLE && CopyBuffer(h_macro_rsi_jpy, 0, 0, 1, jpyRsi) == 1)
             macroFeat[15] = (float)(jpyRsi[0] / 100.0);
       }
      
      // DXY proxy (from USDJPY+EURUSD)
      if(sym_USDJPY != "" && sym_EURUSD != "")
      {
         double jpyH1[], eurH1[], jpyH4[], eurH4[];
         if(CopyClose(sym_USDJPY, PERIOD_H1, 0, 2, jpyH1) == 2 &&
            CopyClose(sym_EURUSD, PERIOD_H1, 0, 2, eurH1) == 2)
            macroFeat[16] = (float)((jpyH1[1]-jpyH1[0])/jpyH1[0] - (eurH1[1]-eurH1[0])/eurH1[0]); // dxy_proxy_1h
         if(CopyClose(sym_USDJPY, PERIOD_H4, 0, 2, jpyH4) == 2 &&
            CopyClose(sym_EURUSD, PERIOD_H4, 0, 2, eurH4) == 2)
            macroFeat[17] = (float)((jpyH4[1]-jpyH4[0])/jpyH4[0] - (eurH4[1]-eurH4[0])/eurH4[0]); // dxy_proxy_4h
      }
      
      // XAG features
      if(sym_XAGUSD != "")
      {
         double xagClose[], xauClose[];
         if(CopyClose(sym_XAGUSD, PERIOD_H1, 0, 21, xagClose) >= 5 &&
            CopyClose(_Symbol, PERIOD_H1, 0, 21, xauClose) >= 5)
         {
            int nx = ArraySize(xagClose);
            macroFeat[18] = (float)((xagClose[nx-1] - xagClose[nx-5]) / xagClose[nx-5]); // xag_ret_4
            double ratio = (xagClose[nx-1] > 0) ? xauClose[nx-1] / xagClose[nx-1] : 0;
            macroFeat[19] = (float)ratio; // xag_gold_ratio
            // Z-score of ratio
            double rSum=0, rMean=0;
            for(int r=0; r<MathMin(nx, 20); r++) { double rv = (xagClose[r]>0) ? xauClose[r]/xagClose[r] : 0; rSum += rv; }
            rMean = rSum / MathMin(nx, 20);
            double rVar = 0;
            for(int r=0; r<MathMin(nx, 20); r++) { double rv = (xagClose[r]>0) ? xauClose[r]/xagClose[r] : 0; rVar += MathPow(rv-rMean,2); }
            double rStd = MathSqrt(rVar / MathMin(nx, 20));
            macroFeat[20] = (rStd > 0) ? (float)((ratio - rMean) / rStd) : 0.0f; // xag_gold_ratio_z
            macroFeat[21] = macroFeat[18] - macroFeat[1]; // xag_gold_divergence (xag_ret - gold_ret)
         }
      }
      
      // BTC features
      if(sym_BTCUSD != "")
      {
         double btcClose[];
         if(CopyClose(sym_BTCUSD, PERIOD_H1, 0, 17, btcClose) >= 17)
         {
            int nb = ArraySize(btcClose);
            macroFeat[22] = (float)((btcClose[nb-1] - btcClose[nb-5]) / btcClose[nb-5]); // btc_ret_4
            macroFeat[23] = (float)((btcClose[nb-1] - btcClose[nb-17]) / btcClose[nb-17]); // btc_ret_16
         }
          double btcRsi[];
          if(h_macro_rsi_btc != INVALID_HANDLE && CopyBuffer(h_macro_rsi_btc, 0, 0, 1, btcRsi) == 1)
             macroFeat[24] = (float)(btcRsi[0] / 100.0);
       }
      
      // Volatility features
      // Volatility features (re-use h_atr handle)
      {
         double longAtr[];
         if(CopyBuffer(h_atr, 0, 0, 100, longAtr) >= 100)
         {
            macroFeat[25] = (longAtr[99] > 0) ? (float)(longAtr[ArraySize(longAtr)-1] / longAtr[99]) : 1.0f; // vol_ratio
            // vol_percentile
            int below = 0;
            for(int v=0; v<100; v++) if(longAtr[v] < longAtr[ArraySize(longAtr)-1]) below++;
            macroFeat[26] = (float)(below / 100.0); // vol_percentile
         }
      }
      macroFeat[27] = (macroFeat[5] > 1.5f) ? 1.0f : 0.0f; // vol_expansion
      macroFeat[28] = (macroFeat[5] > 1.0f && macroFeat[5] < 1.5f) ? 1.0f : 0.0f; // vol_cluster
      
      // Session flags
      MqlDateTime mdt;
      TimeToStruct(TimeGMT(), mdt);
      macroFeat[29] = (mdt.hour >= 0 && mdt.hour < 7) ? 1.0f : 0.0f;   // is_asia
      macroFeat[30] = (mdt.hour >= 7 && mdt.hour < 16) ? 1.0f : 0.0f;  // is_london
      macroFeat[31] = (mdt.hour >= 13 && mdt.hour < 22) ? 1.0f : 0.0f; // is_ny
      
      // === Run ONNX inference ===
             long classOut[1];
        float dxyProb[3], orbProb[3];
       string labels[] = {"BUY", "HOLD", "SELL"};
       string regLabels[] = {"RANGING", "STRONG_TREND", "TRENDING"};
       
       // DXY direction
       if(h_macro_dxy != INVALID_HANDLE)
       {
          if(OnnxRun(h_macro_dxy, ONNX_NO_CONVERSION, macroFeat, classOut, dxyProb))
          {
             int bestIdx = 0; float bestP = 0;
             for(int d=0; d<3; d++)
                if(dxyProb[d] > bestP) { bestP = dxyProb[d]; bestIdx = d; }
             if(bestIdx < 3) aiSig.dxy = labels[bestIdx];
             aiSig.dxy_score = (aiSig.dxy == "BUY") ? 1 : (aiSig.dxy == "SELL" ? -1 : 0);
             if(bestP > 0.70) aiSig.dxy_score *= 2; // High confidence doubles score
          }
       }
       
       // Regime (ZipMap signature mapping)
       if(h_macro_regime != INVALID_HANDLE)
       {
          ZipMapProb regProbMap[];
          if(OnnxRun(h_macro_regime, ONNX_DEFAULT, macroFeat, classOut, regProbMap) && ArraySize(regProbMap) > 0)
          {
             int bestIdx = 0; float bestP = 0;
             for(int d=0; d<ArraySize(regProbMap[0].value); d++)
                if(regProbMap[0].value[d] > bestP) { bestP = regProbMap[0].value[d]; bestIdx = d; }
             if(bestIdx < 3) aiSig.regime = regLabels[bestIdx];
          }
       }
       
       // ORB direction
       if(h_macro_orb != INVALID_HANDLE)
       {
          if(OnnxRun(h_macro_orb, ONNX_NO_CONVERSION, macroFeat, classOut, orbProb))
          {
             int bestIdx = 0; float bestP = 0;
             for(int d=0; d<3; d++)
                if(orbProb[d] > bestP) { bestP = orbProb[d]; bestIdx = d; }
             if(bestIdx < 3) aiSig.orb_dir = labels[bestIdx];
          }
       }
      
      if(Debug && tickCount % 30 == 0)
         LP("ðŸ§  MACRO: DXY=", aiSig.dxy, "(", aiSig.dxy_score, ") REG=", aiSig.regime, " ORB=", aiSig.orb_dir);
      return;
   }
   
   // === FALLBACK: indicator-based when ONNX not loaded ===
   if(sym_USDJPY != "")
   {
      double jpyClose[];
      if(CopyClose(sym_USDJPY, PERIOD_H1, 0, 21, jpyClose) == 21)
      {
         double jpyEma = 0;
         for(int j=0; j<21; j++) jpyEma += jpyClose[j];
         jpyEma /= 21.0;
         double jpyNow = jpyClose[20];
         double jpyDelta = (jpyNow - jpyEma) / jpyEma * 100.0;
         if(jpyDelta > 0.15)      { aiSig.dxy = "BUY";  aiSig.dxy_score = 1; }
         else if(jpyDelta < -0.15) { aiSig.dxy = "SELL"; aiSig.dxy_score = -1; }
         else                       { aiSig.dxy = "HOLD"; aiSig.dxy_score = 0; }
         if(jpyDelta > 0.30) aiSig.dxy_score = 2;
         if(jpyDelta < -0.30) aiSig.dxy_score = -2;
      }
   }
   double adxVal[];
   int hAdx = iADX(_Symbol, PERIOD_H1, 14);
   if(hAdx != INVALID_HANDLE && CopyBuffer(hAdx, 0, 0, 1, adxVal) == 1)
   {
      if(adxVal[0] > 35)       aiSig.regime = "STRONG_TREND";
      else if(adxVal[0] > 25)  aiSig.regime = "TRENDING";
      else if(adxVal[0] > 18)  aiSig.regime = "RANGING";
      else                      aiSig.regime = "VOLATILE";
   }
   if(hAdx != INVALID_HANDLE) IndicatorRelease(hAdx);
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   if(dt.hour >= 8 && dt.hour <= 16)
   {
      double orbH[], orbL[];
      datetime orbStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " 08:00");
      int orbBarsNeeded = (int)((TimeCurrent() - orbStart) / PeriodSeconds(PERIOD_M5));
      if(orbBarsNeeded > 12) orbBarsNeeded = 12;
      if(orbBarsNeeded >= 1)
      {
         int shift = iBarShift(_Symbol, PERIOD_M5, orbStart);
         if(shift >= 0 && CopyHigh(_Symbol, PERIOD_M5, shift, MathMin(orbBarsNeeded, 12), orbH) > 0
            && CopyLow(_Symbol, PERIOD_M5, shift, MathMin(orbBarsNeeded, 12), orbL) > 0)
         {
            double oHigh = orbH[ArrayMaximum(orbH)];
            double oLow  = orbL[ArrayMinimum(orbL)];
            double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            if(bid > oHigh)      aiSig.orb_dir = "BUY";
            else if(bid < oLow)  aiSig.orb_dir = "SELL";
            else                  aiSig.orb_dir = "";
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Helper: get current hour for session detection                    |
//+------------------------------------------------------------------+
int nowDt_hour()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   return dt.hour;
}

//+------------------------------------------------------------------+
//| Manage Open Positions â€” V5: Chandelier Exit + 3-Tier TP          |
//+------------------------------------------------------------------+
// V6.2: Track partial close per ticket (survives EA restart)
static ulong g_partial_done_tickets[50];
static int   g_partial_done_count = 0;

bool IsPartialDone(ulong ticket)
{
   for(int i = 0; i < g_partial_done_count; i++)
      if(g_partial_done_tickets[i] == ticket) return true;
   return false;
}

void MarkPartialDone(ulong ticket)
{
   if(g_partial_done_count < 50)
      g_partial_done_tickets[g_partial_done_count++] = ticket;
}

void PurgeClosedPartials()
{
   // Remove tickets no longer in open positions
   int write = 0;
   for(int i = 0; i < g_partial_done_count; i++)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         if(PositionGetTicket(p) == g_partial_done_tickets[i]) { found = true; break; }
      }
      if(found)
         g_partial_done_tickets[write++] = g_partial_done_tickets[i];
   }
   g_partial_done_count = write;
}

void ManagePositions()
{
   PurgeClosedPartials();  // Bug 10: prevent array overflow at 50
   double atr[];
   ArrayResize(atr, 2);
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(h_atr, 0, 0, 2, atr) < 0) return;
   double atrVal = atr[0];
   if(atrVal <= 0) return;
   
   int digits = (int)symInfo.Digits();
   
   // Get highest high / lowest low for Chandelier Exit (22-bar lookback)
   double highs22[], lows22[];
   int chLookback = 22;
   ArrayResize(highs22, chLookback);
   ArrayResize(lows22, chLookback);
   ArraySetAsSeries(highs22, true);
   ArraySetAsSeries(lows22, true);
   CopyHigh(_Symbol, PERIOD_M15, 0, chLookback, highs22);
   CopyLow(_Symbol, PERIOD_M15, 0, chLookback, lows22);
   
   double highestHigh = 0, lowestLow = 999999;
   for(int k = 0; k < chLookback; k++)
   {
      if(highs22[k] > highestHigh) highestHigh = highs22[k];
      if(lows22[k] < lowestLow) lowestLow = lows22[k];
   }
   
   // Diagnostic: log once per minute
   static datetime lastManageLog = 0;
   bool logThisTick = (TimeLocal() - lastManageLog >= 60);
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() != MagicNumber) continue;
      if(posInfo.Symbol() != _Symbol) continue;
      
      double entry   = posInfo.PriceOpen();
      double current = posInfo.PriceCurrent();
      double curSL   = posInfo.StopLoss();
      double curTP   = posInfo.TakeProfit();
      ulong  ticket  = posInfo.Ticket();
      bool   isBuy   = (posInfo.PositionType() == POSITION_TYPE_BUY);
      double profit  = isBuy ? (current - entry) : (entry - current);
      
// â”€â”€â”€ Diagnostic Log (once per minute) â”€â”€â”€
       if(logThisTick && Debug)
       {
          LP("🔧 MANAGE: T#", ticket, (isBuy ? " BUY" : " SELL"),
                " | entry=", DoubleToString(entry, digits),
                " cur=", DoubleToString(current, digits),
                " SL=", DoubleToString(curSL, digits),
                " TP=", DoubleToString(curTP, digits),
                " | profit=$", DoubleToString(profit, 2),
                " ATR=$", DoubleToString(atrVal, 2));
          LP("🔧 BE TRIGGER: level=$", DoubleToString(atrVal * BreakevenATR, 2),
                " triggered=", (profit >= atrVal * BreakevenATR ? "YES" : "NO"),
                " | LOCK1(1.2x)=$", DoubleToString(atrVal * 1.2, 2),
                " LOCK2(1.3x)=$", DoubleToString(atrVal * 1.3, 2),
                " LOCK3(1.4x)=$", DoubleToString(atrVal * 1.4, 2),
                " | TRAIL: level=$", DoubleToString(atrVal * TrailStart_ATR, 2),
                " | LL=", DoubleToString(lowestLow, digits),
                " HH=", DoubleToString(highestHigh, digits));
          lastManageLog = TimeLocal();
       }
      
      // â”€â”€â”€ Breakeven â”€â”€â”€
      double beLevel = atrVal * BreakevenATR;
      if(profit >= beLevel)
      {
         double beSL;
         if(isBuy)
         {
            beSL = NormalizeDouble(entry + symInfo.Spread() * symInfo.Point(), digits);
            if(curSL < beSL)
            {
int retryCount = 0;
                 bool beModified = false;
                 
                 while(retryCount < 3 && !beModified)
                 {
                    beModified = trade.PositionModify(ticket, beSL, curTP);
                    if(!beModified)
                    {
                       retryCount++;
                       LP("BE RETRY ", retryCount, " (BUY): ", trade.ResultRetcodeDescription(),
                             " newSL=", DoubleToString(beSL, digits),
                             " curSL=", DoubleToString(curSL, digits),
                             " entry=", DoubleToString(entry, digits));
                       Sleep(100);
                    }
                 }
                 
                 if(beModified)
                 {
                    LP("✅ BE: Ticket ", ticket, " SL -> ", DoubleToString(beSL, digits), " (was ", DoubleToString(curSL, digits), ")");
                    WebhookTradeModify(ticket, "BUY", posInfo.Volume(), beSL, curTP, "BREAKEVEN");
                    
                    // Force immediate position refresh for MatchTrader
                    symInfo.RefreshRates();
                    if(posInfo.SelectByTicket(ticket))
                       LP("🔁 POSITION REFRESHED: curSL=", DoubleToString(posInfo.StopLoss(), digits));
                 }
                 else
                 {
                    LP("❌ BE FAIL FINAL (BUY): ", trade.ResultRetcodeDescription(),
                          " entry=", DoubleToString(entry, digits),
                          " cur=", DoubleToString(current, digits),
                          " profit=", DoubleToString(profit, 2),
                          " beLevel=", DoubleToString(beLevel, 2));
                    Alert("❌ BREAKEVEN FAILED - Check MT5 connection");
                 }
            }
         }
         else
         {
            beSL = NormalizeDouble(entry - symInfo.Spread() * symInfo.Point(), digits);
            if(curSL > beSL || curSL == 0)
            {
int retryCount = 0;
                 bool beModified = false;
                 
                 while(retryCount < 3 && !beModified)
                 {
                    beModified = trade.PositionModify(ticket, beSL, curTP);
                    if(!beModified)
                    {
                       retryCount++;
                       LP("BE RETRY ", retryCount, " (SELL): ", trade.ResultRetcodeDescription(),
                             " newSL=", DoubleToString(beSL, digits),
                             " curSL=", DoubleToString(curSL, digits),
                             " entry=", DoubleToString(entry, digits));
                       Sleep(100);
                    }
                 }
                 
                 if(beModified)
                 {
                    LP("✅ BE: Ticket ", ticket, " SL -> ", DoubleToString(beSL, digits), " (was ", DoubleToString(curSL, digits), ")");
                    WebhookTradeModify(ticket, "SELL", posInfo.Volume(), beSL, curTP, "BREAKEVEN");
                    
                    // Force immediate position refresh for MatchTrader
                    symInfo.RefreshRates();
                    if(posInfo.SelectByTicket(ticket))
                       LP("🔁 POSITION REFRESHED: curSL=", DoubleToString(posInfo.StopLoss(), digits));
                 }
                 else
                 {
                    LP("❌ BE FAIL FINAL (SELL): ", trade.ResultRetcodeDescription(),
                          " entry=", DoubleToString(entry, digits),
                          " cur=", DoubleToString(current, digits),
                          " profit=", DoubleToString(profit, 2),
                          " beLevel=", DoubleToString(beLevel, 2));
                    Alert("❌ BREAKEVEN FAILED - Check MT5 connection");
                 }
            }
         }
      }
      
       // 6.1: Stepped Profit Lock (between BE and Trailing) ═══
       // Ratchets SL progressively to lock partial profit before trail activates
       if(profit >= beLevel)  // Only after BE threshold is met
       {
          double trigMult[3] = {1.2, 1.3, 1.4};   // trigger at N×ATR profit
          double lockMult[3] = {0.3, 0.5, 0.7};   // lock N×ATR above entry
          
          for(int s = 2; s >= 0; s--)  // Check highest step first
          {
             if(profit >= atrVal * trigMult[s])
             {
                double lockSL;
                if(isBuy)
                 {
                    lockSL = NormalizeDouble(entry + atrVal * lockMult[s], digits);
                    if(lockSL > curSL + symInfo.Point() && lockSL > entry)
                    {
                       if(trade.PositionModify(ticket, lockSL, curTP))
                       {
                          LP("🔒 LOCK: T#", ticket, " SL → ", DoubleToString(lockSL, digits));
                          WebhookTradeModify(ticket, "BUY", posInfo.Volume(), lockSL, curTP, "PROFIT_LOCK");
                       }
                       curSL = lockSL;
                    }
                 }
                 else
                 {
                    lockSL = NormalizeDouble(entry - atrVal * lockMult[s], digits);
                    if((lockSL < curSL - symInfo.Point() || curSL == 0) && lockSL < entry)
                    {
                       if(trade.PositionModify(ticket, lockSL, curTP))
                       {
                          LP("🔒 LOCK: T#", ticket, " SL → ", DoubleToString(lockSL, digits));
                          WebhookTradeModify(ticket, "SELL", posInfo.Volume(), lockSL, curTP, "PROFIT_LOCK");
                       }
                       curSL = lockSL;
                    }
                 }
                break;  // Only apply the highest matching step
             }
          }
       }
       
       // V10: Chandelier Exit Trailing (uses V10_ChandelierMult input)
      double trailActivation = atrVal * TrailStart_ATR;
      
      if(profit > trailActivation)
      {
         ENUM_POSITION_TYPE posType = isBuy ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
         double newSL = V10_ChandelierTrail(ticket, posType, curSL);
         
         if(isBuy)
         {
            if(newSL > curSL + symInfo.Point() && newSL > entry)
            {
                newSL = NormalizeDouble(newSL, digits);
                if(!trade.PositionModify(ticket, newSL, curTP))
                   LP("âšï¸ TRAIL FAIL (BUY): ", trade.ResultRetcodeDescription());
                else
                {
                   LP("ðŸ“ V10 TRAIL: T#", ticket, " SL â†’ $", DoubleToString(newSL, digits));
                   WebhookTradeModify(ticket, "BUY", posInfo.Volume(), newSL, curTP, "V10_CHANDELIER");
                }
             }
          }
          else
          {
             if((newSL < curSL - symInfo.Point() || curSL == 0) && newSL < entry)
             {
                newSL = NormalizeDouble(newSL, digits);
                if(!trade.PositionModify(ticket, newSL, curTP))
                   LP("âšï¸ TRAIL FAIL (SELL): ", trade.ResultRetcodeDescription());
                else
                {
                   LP("ðŸ“ V10 TRAIL: T#", ticket, " SL â†’ $", DoubleToString(newSL, digits));
                   WebhookTradeModify(ticket, "SELL", posInfo.Volume(), newSL, curTP, "V10_CHANDELIER");
                }
            }
         }
      }
      
      // â”€â”€â”€ V6.2: Partial Close at 50% of TP distance (no activeRecords dependency) â”€â”€â”€
      if(!IsPartialDone(ticket) && curTP != 0)
      {
         double tpDist = isBuy ? (curTP - entry) : (entry - curTP);
         double tp1Threshold = tpDist * 0.5;
         
         if(tpDist > 0 && profit >= tp1Threshold)
         {
            double vol = posInfo.Volume();
            double minL = symInfo.LotsMin();
            double step = symInfo.LotsStep();
            // V10: Dynamic partial close % based on volatility regime
            int volRegPC = GetVolatilityRegime();
            double v10PartialPct = V10_GetPartialClosePct(volRegPC);
            double closeVol = MathFloor(vol * (v10PartialPct / 100.0) / step) * step;
            
            if(closeVol >= minL && (vol - closeVol) >= minL)
            {
                if(trade.PositionClosePartial(ticket, closeVol))
                {
                   MarkPartialDone(ticket);
                   LP("✅ PARTIAL: Closed ", DoubleToString(closeVol, 3), 
                         " lots at 50% TP | profit=$", DoubleToString(profit, 2),
                         " threshold=$", DoubleToString(tp1Threshold, 2));
                   WebhookPartialClose(ticket, (isBuy ? "BUY" : "SELL"), closeVol, vol - closeVol, profit);
                }
               else
                  LP("âš ï¸ PARTIAL FAIL: ", trade.ResultRetcodeDescription());
            }
            else if(logThisTick && Debug)
            {
               LP("ðŸ”§ PARTIAL SKIP: vol=", DoubleToString(vol, 3),
                     " closeVol=", DoubleToString(closeVol, 3),
                     " minLot=", DoubleToString(minL, 3),
                     " â€” lot too small");
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Session Detection â€” includes overlaps                             |
//+------------------------------------------------------------------+
string GetSession(int hour)
{
   bool inAsia = (hour >= AsiaStart && hour < AsiaEnd);
   bool inLondon = (hour >= LondonStart && hour < LondonEnd);
   bool inNY = (hour >= NYStart && hour < NYEnd);
   
   // Overlap detection (sessions share boundary hours)
   // Asia-London overlap: last 2h of Asia + first 2h of London
   if(hour >= (AsiaEnd - 2) && hour < (LondonStart + 2) && TradeAsia && TradeLondon)
      return "ASIA-LON";
   // London-NY overlap: prime gold hours
   if(hour >= NYStart && hour < LondonEnd && TradeLondon && TradeNY)
      return "LON-NY";
   // For default session definitions (Asia 0-7, London 7-12, NY 12-21):
   // Asia-London crossover happens at hours 5-9
   // London-NY crossover: since London ends at 12 and NY starts at 12, no overlap
   // But in reality with EET server, we check 15-19 for the real overlap
   if(hour >= 15 && hour <= 19)
      return "LON-NY";
   
   if(inAsia && TradeAsia) return "ASIA";
   if(inLondon && TradeLondon) return "LONDON";
   if(inNY && TradeNY) return "NY";
   return "OFF";
}

//+------------------------------------------------------------------+
//| Session Momentum Bonus                                            |
//+------------------------------------------------------------------+
int SessionBonus(int hour)
{
   if(GetSession(hour) != "ASIA") return 0;
   
   double c[];
   ArrayResize(c, 4);
   ArraySetAsSeries(c, true);
   if(CopyClose(_Symbol, PERIOD_M15, 0, 4, c) < 0) return 0;
   
   if((c[0] > c[1] && c[1] > c[2] && c[2] > c[3]) ||
      (c[0] < c[1] && c[1] < c[2] && c[2] < c[3]))
      return AsiaBonus;
   
   return 0;
}

//+------------------------------------------------------------------+
//| Weekend Gap Detection                                             |
//+------------------------------------------------------------------+
int CheckWeekendGap()
{
   double closes[];
   ArrayResize(closes, 3);
   ArraySetAsSeries(closes, true);
   if(CopyClose(_Symbol, PERIOD_D1, 0, 3, closes) < 0) return 0;
   
   double fridayClose = closes[1];
   double current = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double gap = current - fridayClose;
   
   if(MathAbs(gap) < GapMinDollars) return 0;
   
   LP("Weekend Gap: ", (gap > 0 ? "+" : ""), DoubleToString(gap, 2), " (min: $", DoubleToString(GapMinDollars, 2), ")");
   return (gap > 0) ? 1 : -1;
}

//+------------------------------------------------------------------+
//| Close All Positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() != MagicNumber) continue;
      if(posInfo.Symbol() != _Symbol) continue;
      if(trade.PositionClose(posInfo.Ticket()))
         LP("✅ Closed ticket ", posInfo.Ticket(), " in CloseAllPositions");
      else
         LP("⚠️ CLOSE FAIL: ticket ", posInfo.Ticket(), " — ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Close positions by type (BUY or SELL only) — anti-hedging         |
//+------------------------------------------------------------------+
void ClosePositionsByType(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() != MagicNumber) continue;
      if(posInfo.Symbol() != _Symbol) continue;
      if(posInfo.PositionType() != type) continue;
      if(trade.PositionClose(posInfo.Ticket()))
         LP("✅ Closed ticket ", posInfo.Ticket(), " in ClosePositionsByType");
      else
         LP("⚠️ CLOSE FAIL: ticket ", posInfo.Ticket(), " — ", trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Count positions                                                   |
//+------------------------------------------------------------------+
int CountMyPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() == MagicNumber && posInfo.Symbol() == _Symbol)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Check if has specific position type                               |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() != MagicNumber || posInfo.Symbol() != _Symbol) continue;
      if(posInfo.PositionType() == type) return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| JSON Helpers                                                      |
//+------------------------------------------------------------------+
string ParseJSONStr(string &json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos = StringFind(json, ":", pos);
   if(pos < 0) return "";
   int q1 = StringFind(json, "\"", pos + 1);
   if(q1 < 0) return "";
   int q2 = StringFind(json, "\"", q1 + 1);
   if(q2 < 0) return "";
   return StringSubstr(json, q1 + 1, q2 - q1 - 1);
}

double ParseJSONDbl(string &json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos = StringFind(json, ":", pos);
   if(pos < 0) return 0;
   string num = "";
   for(int i = pos + 1; i < StringLen(json); i++)
   {
      ushort ch = StringGetCharacter(json, i);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-')
         num += ShortToString(ch);
      else if(StringLen(num) > 0)
         break;
   }
   return StringToDouble(num);
}

//+------------------------------------------------------------------+
//| NEWS FILTER â€” High-Impact Event Avoidance                        |
//+------------------------------------------------------------------+
bool IsHighImpactNews()
{
   if(!UseNewsFilter) return false;
   
   // MQL5 Economic Calendar check
   datetime now = TimeCurrent();
   datetime from = now - NewsMinutesBefore * 60;
   datetime to   = now + NewsMinutesBefore * 60;
   
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, from, to);
   
   if(count <= 0) return false;
   
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      
      MqlCalendarCountry country;
      if(!CalendarCountryById(event.country_id, country)) continue;
      
      // Filter: only USD-related high impact events affect gold
      if(country.currency != "USD" && country.currency != "CNY" && 
         country.currency != "EUR" && country.currency != "GBP")
         continue;
      
      // High impact only
      if(event.importance == CALENDAR_IMPORTANCE_HIGH)
      {
         if(Debug)
            LP("NEWS FILTER: High-impact event '", event.name, 
                  "' (", country.currency, ") within window â€” pausing");
         return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| ANTIGRAVITY V6 â€” PREMIUM CCANVAS CONTROL ROOM HUD                 |
//| Pixel-level bitmap rendering with gradients, rounded corners,     |
//| glowing indicators, and anti-aliased text                          |
//+------------------------------------------------------------------+
#define PNL_X       10
#define PNL_Y       25
#define PNL_W       460
#define PNL_H       720

// V10: Brain panel (right side, top)
#define BRAIN_X     480
#define BRAIN_Y     260
#define BRAIN_W     260
#define BRAIN_H     310
CCanvas brainCanvas;
bool    brainCanvasReady = false;

// V10: P&L chart panel (right side, bottom)
#define PNL_CHART_X  480
#define PNL_CHART_Y  550
#define PNL_CHART_W  260
#define PNL_CHART_H  180
CCanvas pnlCanvas;
bool    pnlCanvasReady = false;
#define FNT         "Consolas"
#define FNT_TITLE   "Segoe UI Semibold"

// Session P&L and win/loss tracking
datetime eaStartTime = 0;
int      winsToday = 0;
int      lossesToday = 0;
double   floatingPnL = 0;
int      lastScoreComponents[12];

// V10 Ensemble per-model probabilities
double g_xgb_probs[3];
double g_lgb_probs[3];
double g_cat_probs[3];
double g_ens_probs[3];
int    g_models_active = 0;
// V6: Equity sparkline + adaptive memory globals
double g_pnl_history[50];  // Last 50 trade P&Ls for sparkline
int    g_pnl_count = 0;    // How many P&L values recorded
int    g_memory_trades = 0; // Trades in InsForge memory
double g_adaptive_minScore = 8; // From InsForge adaptive_params
double g_adaptive_kelly = 0.02;
string g_adaptive_regime = "NORMAL";
bool   g_dxy_blocking = false;  // DXY anti-BUY active


// V6 Phase 4: Adaptive Intelligence Globals

datetime g_last_adaptive_fetch = 0;     // Throttle WebRequest to 5min

// V10: These feed into ONNX inference AND get updated by FetchAdaptiveParams + V10_UpdateDrift
double   g_regime_w_xgb = 0.5266;   // V10 default: XGB dominant
double   g_regime_w_lgb = 0.1608;
double   g_regime_w_cat = 0.3125;

// V10: Meta-confidence from last gate evaluation (used by ExecuteTrade for lot scaling)
double   g_v10_meta_conf = 0;

string   g_last_onnx_signal = "";         // For signal-change detection

double   g_last_onnx_conf = 0;

int      g_drift_correct = 0;             // Drift tracking: correct predictions

int      g_drift_total = 0;                // Drift tracking: total predictions

CCanvas canvas;
bool    canvasReady = false;

// â”€â”€â”€ ARGB Color macro for CCanvas (resolved at compile time) â”€â”€â”€
#define ARGB(a,r,g,b) (((uint)(a)<<24)|((uint)(r)<<16)|((uint)(g)<<8)|(uint)(b))

#define C_BG        ARGB(245, 10, 12, 18)
#define C_BG2       ARGB(230, 16, 19, 28)
#define C_CARD      ARGB(220, 20, 24, 35)
#define C_CARD_GLOW ARGB(60,  0, 180, 160)
#define C_LINE      ARGB(180, 35, 40, 55)
#define C_ACCENT    ARGB(255, 0, 229, 204)
#define C_GOLD      ARGB(255, 218, 175, 70)
#define C_GOLD_DIM  ARGB(180, 140, 112, 45)
#define C_TEXT      ARGB(255, 160, 168, 190)
#define C_BRIGHT    ARGB(255, 225, 230, 245)
#define C_DIM       ARGB(200, 85, 92, 115)
#define C_GREEN     ARGB(255, 0, 255, 135)
#define C_GREEN_DK  ARGB(200, 0, 120, 65)
#define C_RED       ARGB(255, 255, 75, 75)
#define C_RED_DK    ARGB(200, 120, 35, 35)
#define C_YELLOW    ARGB(255, 255, 200, 65)
#define C_BLUE      ARGB(255, 65, 165, 255)
#define C_CYAN      ARGB(255, 80, 220, 235)
#define C_PURPLE    ARGB(255, 175, 130, 255)
#define C_BAR_BG    ARGB(220, 28, 32, 45)

// â”€â”€â”€ Trade Gate Mini-Panel (right side) â”€â”€â”€
#define GATE_W  260
#define GATE_H  240
CCanvas gateCanvas;
bool    gateCanvasReady = false;

void CreateGatePanel()
{
   if(!gateCanvasReady)
   {
      if(gateCanvas.CreateBitmapLabel("V10_GATES", 480, 25, GATE_W, GATE_H, COLOR_FORMAT_ARGB_NORMALIZE))
      {
         gateCanvasReady = true;
         ObjectSetInteger(0, "V10_GATES", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "V10_GATES", OBJPROP_HIDDEN, true);
         ObjectSetInteger(0, "V10_GATES", OBJPROP_ZORDER, 100);
         ChartRedraw(0);
      }
   }
}

void DestroyGatePanel()
{
   if(gateCanvasReady) { gateCanvas.Destroy(); gateCanvasReady = false; }
   ObjectDelete(0, "V10_GATES");
}

void UpdateGateHUD()
{
   static bool gatePanelBuilt = false;
   if(!gatePanelBuilt) { CreateGatePanel(); gatePanelBuilt = true; }
   if(!gateCanvasReady) return;
   
   gateCanvas.Erase(ARGB(240, 10, 12, 18));
   
   int gx = 10;
   int gy = 0;
   int gW = GATE_W - 20;
   
   // Header bar
   gateCanvas.FillRectangle(0, 0, GATE_W, 24, ARGB(200, 15, 18, 28));
   gateCanvas.FillRectangle(0, 0, 3, 24, C_ACCENT);
   
   // Count gates
   int gPass = 0; int gFail = 0;
   if(g_gate_spread) gPass++; else gFail++;
   if(g_gate_session) gPass++; else gFail++;
   if(g_gate_news) gPass++; else gFail++;
   if(g_gate_friday) gPass++; else gFail++;
   if(g_gate_maxpos) gPass++; else gFail++;
   if(g_gate_dailycap) gPass++; else gFail++;
   if(g_gate_losspause) gPass++; else gFail++;
   if(g_gate_ai) gPass++; else gFail++;
   if(g_gate_confluence) gPass++; else gFail++;
   if(g_gate_prop) gPass++; else gFail++;
   
   uint headerClr = gFail == 0 ? C_GREEN : (gFail <= 2 ? C_YELLOW : C_RED);
   gateCanvas.FontSet(FNT, -90);
   gateCanvas.TextOut(gx, 4, "TRADE GATES", headerClr);
   gateCanvas.FontSet(FNT, -70);
   gateCanvas.TextOut(gx+160, 6, StringFormat("%d/10", gPass), headerClr);
   
   // Separator
   gy = 24;
   gateCanvas.FillRectangle(0, gy, GATE_W, gy+1, ARGB(80, 0, 180, 160));
   gy += 4;
   
   // Gate rows - dot indicators
   gateCanvas.FontSet(FNT, -75);
   
   string gLabels[11];  bool gStates[11];  string gDetails[11]; uint gFailClrs[11];
   gLabels[0]="SPREAD";    gStates[0]=g_gate_spread;    gDetails[0]=g_gate_spread_detail; gFailClrs[0]=C_RED;
   gLabels[1]="SESSION";   gStates[1]=g_gate_session;   gDetails[1]=g_gate_session?"ACTIVE":"OFF"; gFailClrs[1]=C_RED;
   gLabels[2]="NEWS";      gStates[2]=g_gate_news;      gDetails[2]=g_gate_news?"CLEAR":"BLOCKED"; gFailClrs[2]=C_RED;
   gLabels[3]="FRIDAY";    gStates[3]=g_gate_friday;    gDetails[3]=g_gate_friday?"OK":"CLOSING"; gFailClrs[3]=C_RED;
   gLabels[4]="POSITIONS"; gStates[4]=g_gate_maxpos;    gDetails[4]=g_gate_maxpos?"OK":"FULL"; gFailClrs[4]=C_RED;
   gLabels[5]="DAILY CAP"; gStates[5]=g_gate_dailycap;  gDetails[5]=StringFormat("%d/%d", todayTradeCount, MaxDailyTrades); gFailClrs[5]=C_RED;
   gLabels[6]="LOSS COOL"; gStates[6]=g_gate_losspause; gDetails[6]=g_gate_losspause?"OK":"PAUSED"; gFailClrs[6]=C_YELLOW;
   gLabels[7]="AI SIGNAL"; gStates[7]=g_gate_ai;        gDetails[7]=g_gate_ai_detail; gFailClrs[7]=C_YELLOW;
   gLabels[8]="CONFLUENCE";gStates[8]=g_gate_confluence; gDetails[8]=g_gate_score_detail; gFailClrs[8]=C_RED;
   gLabels[9]="PROP";      gStates[9]=g_gate_prop;       gDetails[9]=g_gate_prop_detail; gFailClrs[9]=C_RED;
   // V10: Meta-Confidence Gate status
   bool g_gate_metaconf = (g_v10_meta_conf >= V10_MetaConfThreshold) || (!V10_UseMetaConfidence);
   string g_gate_meta_detail = V10_UseMetaConfidence ? StringFormat("%.0f%%/%.0f%%", g_v10_meta_conf*100, V10_MetaConfThreshold*100)
                                                       : "OFF";
   gLabels[10]="META-CONF"; gStates[10]=g_gate_metaconf; gDetails[10]=g_gate_meta_detail; gFailClrs[10]=C_PURPLE;
   
   for(int gi = 0; gi < 11; gi++)
   {
      uint dotClr = gStates[gi] ? C_GREEN : gFailClrs[gi];
      uint txtClr = gStates[gi] ? C_DIM : gFailClrs[gi];
      
      // Glow dot
      gateCanvas.FillCircle(gx+5, gy+6, 4, ARGB(40, (int)((dotClr>>16)&0xFF), (int)((dotClr>>8)&0xFF), (int)(dotClr&0xFF)));
      gateCanvas.FillCircle(gx+5, gy+6, 3, dotClr);
      
      // Label
      gateCanvas.TextOut(gx+14, gy, gLabels[gi], gStates[gi] ? C_TEXT : gFailClrs[gi]);
      
      // Status (right-aligned)
      string statusTxt = gStates[gi] ? "PASS" : gDetails[gi];
      gateCanvas.TextOut(gx+140, gy, statusTxt, txtClr);
      
      gy += 16;
   }
   
   // Primary block reason â€” prominent
   gy += 4;
   gateCanvas.FillRectangle(0, gy, GATE_W, gy+1, C_LINE);
   gy += 6;
   
   gateCanvas.FontSet(FNT, -80);
   if(g_gate_primary == "READY")
   {
      gateCanvas.FillCircle(gx+5, gy+5, 4, C_GREEN);
      gateCanvas.TextOut(gx+14, gy, "READY TO TRADE", C_GREEN);
   }
   else
   {
      uint blockClr = gFail > 1 ? C_RED : C_YELLOW;
      gateCanvas.FillCircle(gx+5, gy+5, 4, blockClr);
      gateCanvas.TextOut(gx+14, gy, g_gate_primary, blockClr);
   }
   
   gateCanvas.Update();
}


// (Color macros moved above gate panel section)

//+------------------------------------------------------------------+
//| Canvas Drawing Helpers                                             |
//+------------------------------------------------------------------+
void FillGradientH(int x1, int y1, int w, int h, uint clrLeft, uint clrRight)
{
   // Horizontal linear gradient fill
   if(w <= 0 || h <= 0) return;
   for(int ix = 0; ix < w; ix++)
   {
      double t = (double)ix / (double)MathMax(1, w-1);
      int aL=(int)((clrLeft>>24)&0xFF), rL=(int)((clrLeft>>16)&0xFF), gL=(int)((clrLeft>>8)&0xFF), bL=(int)(clrLeft&0xFF);
      int aR=(int)((clrRight>>24)&0xFF), rR=(int)((clrRight>>16)&0xFF), gR=(int)((clrRight>>8)&0xFF), bR=(int)(clrRight&0xFF);
      uint mixed = ARGB((int)(aL+(aR-aL)*t), (int)(rL+(rR-rL)*t), (int)(gL+(gR-gL)*t), (int)(bL+(bR-bL)*t));
      canvas.LineVertical(x1+ix, y1, y1+h-1, mixed);
   }
}

void DrawRoundedCard(int x, int y, int w, int h, uint bgClr, uint borderClr, int radius=6)
{
   // Main body
   canvas.FillRectangle(x+radius, y, x+w-radius, y+h, bgClr);
   canvas.FillRectangle(x, y+radius, x+w, y+h-radius, bgClr);
   // Rounded corners (fill circles at corners)
   canvas.FillCircle(x+radius, y+radius, radius, bgClr);
   canvas.FillCircle(x+w-radius, y+radius, radius, bgClr);
   canvas.FillCircle(x+radius, y+h-radius, radius, bgClr);
   canvas.FillCircle(x+w-radius, y+h-radius, radius, bgClr);
   // Border lines
   canvas.Line(x+radius, y, x+w-radius, y, borderClr);
   canvas.Line(x+radius, y+h, x+w-radius, y+h, borderClr);
   canvas.Line(x, y+radius, x, y+h-radius, borderClr);
   canvas.Line(x+w, y+radius, x+w, y+h-radius, borderClr);
}

void DrawGlowDot(int cx, int cy, int r, uint clr)
{
   // Outer glow
   uint glowClr = ARGB(40, (int)((clr>>16)&0xFF), (int)((clr>>8)&0xFF), (int)(clr&0xFF));
   canvas.FillCircle(cx, cy, r+3, glowClr);
   uint midClr = ARGB(120, (int)((clr>>16)&0xFF), (int)((clr>>8)&0xFF), (int)(clr&0xFF));
   canvas.FillCircle(cx, cy, r+1, midClr);
   // Core
   canvas.FillCircle(cx, cy, r, clr);
}

void DrawProgressBar(int x, int y, int w, int h, double pct, uint fgClr, uint bgClr=0)
{
   if(bgClr == 0) bgClr = C_BAR_BG;
   // Background
   canvas.FillRectangle(x, y, x+w, y+h, bgClr);
   // Fill with gradient
   int fillW = (int)(MathMax(0, MathMin(1.0, pct)) * w);
   if(fillW > 1)
   {
      uint darkFg = ARGB(255, (int)(((fgClr>>16)&0xFF)*0.4), (int)(((fgClr>>8)&0xFF)*0.4), (int)((fgClr&0xFF)*0.4));
      FillGradientH(x, y, fillW, h, darkFg, fgClr);
   }
   // Rounded ends
   canvas.FillCircle(x+1, y+h/2, h/2, bgClr);
   canvas.FillCircle(x+w-1, y+h/2, h/2, bgClr);
   if(fillW > h)
      canvas.FillCircle(x+fillW-1, y+h/2, h/2-1, fgClr);
}

void DrawPillBadge(int x, int y, int w, int h, string text, uint bgClr, uint textClr)
{
   int r = h/2;
   canvas.FillRectangle(x+r, y, x+w-r, y+h, bgClr);
   canvas.FillCircle(x+r, y+r, r, bgClr);
   canvas.FillCircle(x+w-r, y+r, r, bgClr);
   canvas.FontSet(FNT, -80);
   canvas.TextOut(x+r+2, y+2, text, textClr);
}

void DrawSectionHeader(int x, int y, int w, string title, uint accentClr)
{
   // Header bar with accent stripe
   canvas.FillRectangle(x, y, x+w, y+22, ARGB(200, 15, 18, 28));
   canvas.FillRectangle(x, y, x+3, y+22, accentClr);
   canvas.FontSet(FNT, -90);
   canvas.TextOut(x+10, y+4, title, accentClr);
}

string ReadFullSignalField(string fieldName)
{
   int h = FileOpen("signal_full.json", FILE_READ | FILE_TXT);
   if(h == INVALID_HANDLE) return "";
   string content = "";
   while(!FileIsEnding(h)) content += FileReadString(h) + "\n";
   FileClose(h);
   return ParseJSONStr(content, fieldName);
}

void CreatePanel()
{
   // Create the CCanvas bitmap surface
   if(!canvasReady)
   {
      if(canvas.CreateBitmapLabel("V10_HUD", PNL_X, PNL_Y, PNL_W, PNL_H, COLOR_FORMAT_ARGB_NORMALIZE))
      {
         canvasReady = true;
         ObjectSetInteger(0, "V10_HUD", OBJPROP_CORNER, CORNER_LEFT_UPPER);
         ObjectSetInteger(0, "V10_HUD", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "V10_HUD", OBJPROP_HIDDEN, true);
      }
   }
}

void DestroyPanel()
{
   if(canvasReady) { canvas.Destroy(); canvasReady = false; }
   if(brainCanvasReady) { brainCanvas.Destroy(); brainCanvasReady = false; }
   if(pnlCanvasReady) { pnlCanvas.Destroy(); pnlCanvasReady = false; }
   ObjectDelete(0, "V10_HUD");
   ObjectDelete(0, "V10_BRAIN");
   ObjectDelete(0, "V10_PNL");
   DestroyGatePanel();
}

//+------------------------------------------------------------------+
//| V10: Brain ONNX Panel (separate canvas, right side)              |
//+------------------------------------------------------------------+
void UpdateBrainPanel()
{
   if(!brainCanvasReady)
   {
      if(brainCanvas.CreateBitmapLabel("V10_BRAIN", BRAIN_X, BRAIN_Y, BRAIN_W, BRAIN_H, COLOR_FORMAT_ARGB_NORMALIZE))
      {
         brainCanvasReady = true;
         ObjectSetInteger(0, "V10_BRAIN", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "V10_BRAIN", OBJPROP_HIDDEN, true);
      }
      else return;
   }
   brainCanvas.Erase(C_BG);
   int bx = 8, by = 0, bW = BRAIN_W - 16;
   
   // Header
   brainCanvas.FillRectangle(0, 0, BRAIN_W, 22, ARGB(200, 15, 18, 28));
   brainCanvas.FillRectangle(0, 0, 3, 22, C_PURPLE);
   brainCanvas.FontSet(FNT_TITLE, -90);
   brainCanvas.TextOut(bx, 4, "AI BRAIN", C_PURPLE);
   brainCanvas.FontSet(FNT, -70);
   brainCanvas.TextOut(bx+80, 5, "V10", C_ACCENT);
   brainCanvas.FontSet(FNT, -70);
   brainCanvas.TextOut(bx+120, 6, StringFormat("%d/3 active", g_models_active), g_models_active==3 ? C_GREEN : C_YELLOW);
   by = 28;
   
   // Per-model probs (color-coded bars)
   string modelNames[3] = {"XGB", "LGB", "CAT"};
   double modelW[3];
   modelW[0] = g_v10_wXGB; modelW[1] = g_v10_wLGB; modelW[2] = g_v10_wCAT;
   for(int m = 0; m < 3; m++)
   {
      double pb = 0, pn = 0, ps = 0;
      if(m==0) { pb=g_xgb_probs[0]; pn=g_xgb_probs[1]; ps=g_xgb_probs[2]; }
      if(m==1) { pb=g_lgb_probs[0]; pn=g_lgb_probs[1]; ps=g_lgb_probs[2]; }
      if(m==2) { pb=g_cat_probs[0]; pn=g_cat_probs[1]; ps=g_cat_probs[2]; }
      
      brainCanvas.FontSet(FNT, -75);
      brainCanvas.TextOut(bx, by, modelNames[m], C_DIM);
      brainCanvas.TextOut(bx+30, by, StringFormat("%.0f%%", modelW[m]*100), C_CYAN);
      
      // Stacked probability bar
      int barX = bx + 62, barW = bW - 62, barH = 10;
      int buyW = (int)(pb * barW);
      int holdW = (int)(pn * barW);
      int sellW = barW - buyW - holdW;
      brainCanvas.FillRectangle(barX, by+2, barX+buyW, by+2+barH, C_GREEN_DK);
      brainCanvas.FillRectangle(barX+buyW, by+2, barX+buyW+holdW, by+2+barH, ARGB(200, 60, 60, 80));
      brainCanvas.FillRectangle(barX+buyW+holdW, by+2, barX+barW, by+2+barH, C_RED_DK);
      
      // Labels on bar
      brainCanvas.FontSet(FNT, -60);
      if(buyW > 20) brainCanvas.TextOut(barX+2, by+2, StringFormat("%.0f", pb*100), C_GREEN);
      if(sellW > 20) brainCanvas.TextOut(barX+barW-18, by+2, StringFormat("%.0f", ps*100), C_RED);
      by += 18;
   }
   
   // Ensemble result
   by += 4;
   brainCanvas.FillRectangle(0, by, BRAIN_W, by+1, C_LINE);
   by += 6;
   brainCanvas.FontSet(FNT, -100);
   uint sigClr = (aiSig.signal=="BUY") ? C_GREEN : (aiSig.signal=="SELL" ? C_RED : C_DIM);
   brainCanvas.TextOut(bx, by, aiSig.signal, sigClr);
   brainCanvas.TextOut(bx+130, by+3, aiSig.version, C_DIM);

   // V10: Meta-Confidence in Brain Panel
   by += 22;
   if(V10_UseMetaConfidence && g_v10_meta_conf > 0) {
      brainCanvas.FillRectangle(0, by, BRAIN_W, by+24, ARGB(40, 175, 130, 255));
      brainCanvas.FontSet(FNT, -70);
      brainCanvas.TextOut(bx, by+2, "META", C_PURPLE);
      uint mcClr = (g_v10_meta_conf >= 0.60) ? C_GREEN : (g_v10_meta_conf >= 0.40 ? C_YELLOW : C_RED);
      brainCanvas.TextOut(bx+40, by+2, StringFormat("%.0f%%", g_v10_meta_conf*100), mcClr);
      // Mini bar
      int barW = 80;
      int barX = bx+80;
      brainCanvas.FillRectangle(barX, by+6, barX+barW, by+14, C_BAR_BG);
      brainCanvas.FillRectangle(barX, by+6, barX+(int)(g_v10_meta_conf*barW), by+14, mcClr);
      // Threshold line
      if(V10_MetaConfThreshold > 0) {
         int tX = barX + (int)(V10_MetaConfThreshold * barW);
         brainCanvas.Line(tX, by+4, tX, by+16, C_YELLOW);
      }
      brainCanvas.FontSet(FNT, -55);
      brainCanvas.TextOut(bx+170, by+4, g_v10_meta_conf >= V10_MetaConfThreshold ? "PASS" : "GATE", 
                          g_v10_meta_conf >= V10_MetaConfThreshold ? C_GREEN : C_RED);
   } else {
      brainCanvas.FontSet(FNT, -70);
      brainCanvas.TextOut(bx, by+2, "META-OFF", C_DIM);
   }
   brainCanvas.TextOut(bx+65, by+3, StringFormat("%.1f%%", aiSig.confidence*100), sigClr);
   by += 22;
   
   // Signal quality
   brainCanvas.FontSet(FNT, -70);
   brainCanvas.TextOut(bx, by, StringFormat("Streak: %d/%d", g_v10_signal_streak, V10_SignalPersistence), C_TEXT);
   brainCanvas.TextOut(bx+110, by, StringFormat("Margin: %.0f%%", V10_ConfidenceMargin*100), C_TEXT);
   by += 16;
   
   // Kelly
   brainCanvas.TextOut(bx, by, V10_KellyStatusString(), (g_v10_kelly_count>=5 ? C_CYAN : C_DIM));
   by += 16;
   
   // Gold zone + DXY
   int pzB = V10_GoldPowerZoneBonus();
   if(pzB > 0)
      brainCanvas.TextOut(bx, by, StringFormat("GOLD ZONE +%d", pzB), C_GOLD);
   else
      brainCanvas.TextOut(bx, by, "Gold Zone: --", C_DIM);
   
   string dxyTxt = (aiSig.dxy != "") ? StringFormat("DXY:%s(%d)", aiSig.dxy, aiSig.dxy_score) : "DXY:N/A";
   uint dxyC2 = (aiSig.dxy=="BUY") ? C_RED : (aiSig.dxy=="SELL" ? C_GREEN : C_DIM);
   brainCanvas.TextOut(bx+130, by, dxyTxt, dxyC2);
   by += 16;
   
   // Drift
   double driftP = g_drift_total > 0 ? (double)g_drift_correct / g_drift_total * 100.0 : 0;
   uint drClr = driftP >= 60 ? C_GREEN : (driftP >= 45 ? C_YELLOW : C_RED);
   brainCanvas.TextOut(bx, by, StringFormat("Drift: %.0f%% (%d/%d)", driftP, g_drift_correct, g_drift_total), drClr);
   brainCanvas.TextOut(bx+150, by, StringFormat("W: X%.0f L%.0f C%.0f", g_regime_w_xgb*100, g_regime_w_lgb*100, g_regime_w_cat*100), C_CYAN);
   
   brainCanvas.Update();
}

//+------------------------------------------------------------------+
//| V10: P&L Chart Panel (separate canvas, right side)               |
//+------------------------------------------------------------------+
void UpdatePnLPanel()
{
   if(!pnlCanvasReady)
   {
      if(pnlCanvas.CreateBitmapLabel("V10_PNL", PNL_CHART_X, PNL_CHART_Y, PNL_CHART_W, PNL_CHART_H, COLOR_FORMAT_ARGB_NORMALIZE))
      {
         pnlCanvasReady = true;
         ObjectSetInteger(0, "V10_PNL", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "V10_PNL", OBJPROP_HIDDEN, true);
      }
      else return;
   }
   pnlCanvas.Erase(C_BG);
   int px = 8, py = 0, pW = PNL_CHART_W - 16;
   
   // Header
   pnlCanvas.FillRectangle(0, 0, PNL_CHART_W, 22, ARGB(200, 15, 18, 28));
   pnlCanvas.FillRectangle(0, 0, 3, 22, C_GOLD);
   pnlCanvas.FontSet(FNT, -90);
   pnlCanvas.TextOut(px, 4, "P&L CURVE", C_GOLD);
   pnlCanvas.FontSet(FNT, -70);
   pnlCanvas.TextOut(px+130, 6, StringFormat("$%+.2f", totalPnL), totalPnL >= 0 ? C_GREEN : C_RED);
   py = 28;
   
   // Sparkline chart
   int sparkH = PNL_CHART_H - 50;
   int sparkW = pW;
   
   if(g_pnl_count > 1)
   {
      double pMin = 0, pMax = 0, cumPnl2 = 0;
      double cumArr2[];
      ArrayResize(cumArr2, g_pnl_count);
      for(int ci = 0; ci < g_pnl_count; ci++)
      {
         cumPnl2 += g_pnl_history[ci];
         cumArr2[ci] = cumPnl2;
         if(cumPnl2 < pMin) pMin = cumPnl2;
         if(cumPnl2 > pMax) pMax = cumPnl2;
      }
      double range2 = MathMax(pMax - pMin, 1.0);
      
      // Zero line
      int zeroY = py + (int)((pMax / range2) * sparkH);
      if(zeroY >= py && zeroY <= py + sparkH)
         pnlCanvas.Line(px, zeroY, px + sparkW, zeroY, ARGB(80, 100, 100, 120));
      
      // Draw curve
      for(int ci = 1; ci < g_pnl_count; ci++)
      {
         int x1 = px + (int)((double)(ci-1) / (g_pnl_count-1) * sparkW);
         int x2 = px + (int)((double)ci / (g_pnl_count-1) * sparkW);
         int y1 = py + (int)((pMax - cumArr2[ci-1]) / range2 * sparkH);
         int y2 = py + (int)((pMax - cumArr2[ci]) / range2 * sparkH);
         uint lineClr = cumArr2[ci] >= 0 ? C_GREEN : C_RED;
         pnlCanvas.Line(x1, y1, x2, y2, lineClr);
      }
      
      // Current value label
      pnlCanvas.FontSet(FNT, -70);
      pnlCanvas.TextOut(px, py + sparkH + 4, StringFormat("Trades: %d | W:%d L:%d", g_pnl_count, winsToday, lossesToday), C_DIM);
   }
   else
   {
      pnlCanvas.FontSet(FNT, -80);
      pnlCanvas.TextOut(px, py + sparkH/2, "Waiting for trades...", C_DIM);
   }
   
   pnlCanvas.Update();
}

void BuildDashboard() { /* Not needed â€” CCanvas draws everything in UpdateHUD */ }

//+------------------------------------------------------------------+
//| UpdateHUD â€” Full CCanvas Bitmap Render                            |
//+------------------------------------------------------------------+
void UpdateHUD(int buyScore, int sellScore, double dailyPnL, double drawdown)
{
   symInfo.RefreshRates();
   
   // Create canvas once
   static bool panelBuilt = false;
   if(!panelBuilt)
   {
      CreatePanel();
      panelBuilt = true;
   }
   if(!canvasReady) return;
   
   tickCount++;
   lastHUDUpdate = TimeLocal();
   
   // â•â•â• Clear canvas â•â•â•
   canvas.Erase(C_BG);
   
   int x = 14;
   int y = 0;
   int cW = PNL_W - 28;  // Card width
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  HEADER â€” Antigravity Brand Bar
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   FillGradientH(0, 0, PNL_W, 54, ARGB(255,15,18,28), ARGB(255,22,28,42));
   canvas.FillRectangle(0, 0, 3, 54, C_ACCENT);
   
   canvas.FontSet(FNT_TITLE, -160);
   canvas.TextOut(x+8, 6, "ANTIGRAVITY", C_GOLD);
   canvas.FontSet(FNT_TITLE, -130);
   canvas.TextOut(x+175, 8, "V10", C_ACCENT);
   canvas.FontSet(FNT, -80);
   canvas.TextOut(x+8, 32, "XAUUSD", C_BRIGHT);
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+80, 34, "META-CONF ENSEMBLE | LIVE", C_DIM);
   
   // Status LED
   bool anyModelLoaded = (h_xgb != INVALID_HANDLE || h_lgb != INVALID_HANDLE || h_cat != INVALID_HANDLE);
   DrawGlowDot(PNL_W-20, 24, 5, anyModelLoaded && aiSig.isValid ? C_GREEN : (anyModelLoaded ? C_YELLOW : C_RED));
   
   // Separator
   y = 54;
   FillGradientH(0, y, PNL_W, 2, ARGB(80,0,180,160), ARGB(20,0,180,160));
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  PRICE HERO
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 6;
   double bid = symInfo.Bid();
   canvas.FontSet(FNT, -220);
   canvas.TextOut(x, y, StringFormat("$%.2f", bid), C_BRIGHT);
   
   double spreadVal = symInfo.Ask() - symInfo.Bid();
   uint spreadClr = spreadVal <= 0.35 ? C_GREEN : (spreadVal <= 0.60 ? C_YELLOW : C_RED);
   canvas.FontSet(FNT, -80);
   canvas.TextOut(x+260, y+8, StringFormat("Spread: $%.2f", spreadVal), spreadClr);
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  ACCOUNT CARD (glassmorphism)
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 36;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   DrawRoundedCard(6, y, cW, 82, C_CARD, C_LINE, 5);
   canvas.FillRectangle(6, y, 9, y+82, C_GOLD_DIM);  // Gold accent
   
   int ay = y + 8;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+4, ay, "BALANCE", C_DIM);
   canvas.FontSet(FNT, -100);
   canvas.TextOut(x+70, ay-2, StringFormat("$%.2f", accInfo.Balance()), C_BRIGHT);
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+240, ay, "EQUITY", C_DIM);
   canvas.FontSet(FNT, -100);
   canvas.TextOut(x+300, ay-2, StringFormat("$%.2f", accInfo.Equity()), C_BRIGHT);
   
   ay += 20;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+4, ay, "DAILY P&L", C_DIM);
   uint pnlClr = dailyPnL >= 0 ? C_GREEN : C_RED;
   canvas.FontSet(FNT, -100);
   canvas.TextOut(x+70, ay-2, StringFormat("%+.2f%%", dailyPnL), pnlClr);
   
   floatingPnL = accInfo.Equity() - accInfo.Balance();
   uint floatClr = floatingPnL >= 0 ? C_GREEN : C_RED;
   canvas.FontSet(FNT, -80);
   canvas.TextOut(x+180, ay, StringFormat("Float: %+.2f", floatingPnL), floatClr);
   canvas.TextOut(x+320, ay, StringFormat("Free: $%.0f", accInfo.FreeMargin()), C_DIM);
   
   ay += 20;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+4, ay, "DRAWDOWN", C_DIM);
   canvas.FontSet(FNT, -90);
   canvas.TextOut(x+70, ay-1, StringFormat("%.2f%%", drawdown), drawdown > 5 ? C_RED : C_TEXT);
   // Drawdown bar
   DrawProgressBar(x+160, ay+2, 240, 8, drawdown/20.0, drawdown > 5 ? C_RED : C_GREEN);

   // Prop challenge telemetry
   ay += 16;
   if(EnablePropChallengeMode)
   {
      double dayBudgetLeft = PropDailyLossSoftStopPct + dailyPnL; // dailyPnL is negative when losing
      double ddBudgetLeft = PropOverallDDSoftStopPct - drawdown;
      uint dayClr = dayBudgetLeft > 0 ? C_GREEN : C_RED;
      uint ddClr = ddBudgetLeft > 0 ? C_GREEN : C_RED;
      canvas.FontSet(FNT, -70);
      canvas.TextOut(x+4, ay, "PROP", C_DIM);
      canvas.TextOut(x+42, ay, StringFormat("DayBuf:%+.2f%%", dayBudgetLeft), dayClr);
      canvas.TextOut(x+170, ay, StringFormat("DDBuf:%+.2f%%", ddBudgetLeft), ddClr);
      canvas.TextOut(x+300, ay, StringFormat("Trades:%d/%d", todayTradeCount, PropMaxTradesPerDay), C_TEXT);
   }
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  AI ENSEMBLE BRAIN
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y = ay + 26;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   DrawSectionHeader(6, y, cW, "AI ENSEMBLE BRAIN", C_PURPLE);
   
   int loaded = (h_xgb != INVALID_HANDLE ? 1 : 0) + (h_lgb != INVALID_HANDLE ? 1 : 0) + (h_cat != INVALID_HANDLE ? 1 : 0);
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+240, y+5, StringFormat("V10 | %d/3 ONNX | %d runs", loaded, onnxRunCount), C_DIM);
   
   y += 26;
   // Verdict
   if(UseONNX && aiSig.isValid)
   {
      uint vClr = (aiSig.signal == "BUY") ? C_GREEN : (aiSig.signal == "SELL" ? C_RED : C_YELLOW);
      canvas.FontSet(FNT_TITLE, -180);
      canvas.TextOut(x, y, aiSig.signal, vClr);
      canvas.FontSet(FNT, -140);
      canvas.TextOut(x+160, y+2, StringFormat("%.0f%%", aiSig.confidence*100), 
                     aiSig.confidence > 0.55 ? C_GREEN : C_YELLOW);
      
      // Signal age
      long sigAge = (long)(TimeGMT() - aiSig.timestamp);
      string ageStr = "";
      uint ageClr = C_DIM;
      if(sigAge < 120) { ageStr = StringFormat("%ds", sigAge); ageClr = C_GREEN; }
      else if(sigAge < 600) { ageStr = StringFormat("%dm", sigAge/60); ageClr = C_YELLOW; }
      else { ageStr = StringFormat("%dm STALE", sigAge/60); ageClr = C_RED; }
      canvas.FontSet(FNT, -70);
      canvas.TextOut(x+300, y+10, ageStr, ageClr);
   }
   else
   {
      canvas.FontSet(FNT_TITLE, -160);
      canvas.TextOut(x, y, "WAITING", C_DIM);
   }
   
   // Overall confidence bar
   y += 28;
   double confPct = (aiSig.isValid ? aiSig.confidence : 0);
   DrawProgressBar(x, y, cW-14, 8, confPct, confPct > 0.55 ? C_GREEN : C_ACCENT);
   
   // Per-model bars
   y += 16;
   // XGBoost
   canvas.FontSet(FNT, -70);
   double xgbMax = MathMax(g_xgb_probs[0], MathMax(g_xgb_probs[1], g_xgb_probs[2]));
   uint xgbClr = (g_xgb_probs[0] > g_xgb_probs[2]) ? C_GREEN : (g_xgb_probs[2] > g_xgb_probs[0] ? C_RED : C_YELLOW);
   DrawGlowDot(x+4, y+5, 3, C_BLUE);
   canvas.TextOut(x+12, y, "XGB", C_TEXT);
   DrawProgressBar(x+55, y+1, 300, 8, xgbMax, xgbClr);
   canvas.TextOut(x+365, y, StringFormat("%.0f%%", xgbMax*100), xgbClr);
   
   y += 14;
   double lgbMax = MathMax(g_lgb_probs[0], MathMax(g_lgb_probs[1], g_lgb_probs[2]));
   uint lgbClr = (g_lgb_probs[0] > g_lgb_probs[2]) ? C_GREEN : (g_lgb_probs[2] > g_lgb_probs[0] ? C_RED : C_YELLOW);
   DrawGlowDot(x+4, y+5, 3, C_CYAN);
   canvas.TextOut(x+12, y, "LGB", C_TEXT);
   DrawProgressBar(x+55, y+1, 300, 8, lgbMax, lgbClr);
   canvas.TextOut(x+365, y, StringFormat("%.0f%%", lgbMax*100), lgbClr);
   
   y += 14;
   double catMax = MathMax(g_cat_probs[0], MathMax(g_cat_probs[1], g_cat_probs[2]));
   uint catClr = (g_cat_probs[0] > g_cat_probs[2]) ? C_GREEN : (g_cat_probs[2] > g_cat_probs[0] ? C_RED : C_YELLOW);
   DrawGlowDot(x+4, y+5, 3, C_GOLD);
   canvas.TextOut(x+12, y, "CAT", C_TEXT);
   DrawProgressBar(x+55, y+1, 300, 8, catMax, catClr);
   canvas.TextOut(x+365, y, StringFormat("%.0f%%", catMax*100), catClr);
   
   y += 14;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, StringFormat("Ensemble: B:%.0f%% N:%.0f%% S:%.0f%%", g_ens_probs[0]*100, g_ens_probs[1]*100, g_ens_probs[2]*100), C_DIM);
   
   // DXY / ORB / Regime pills
   y += 14;
   if(aiSig.isValid && aiSig.dxy != "" && aiSig.dxy != NULL)
   {
      uint dxyClr = (aiSig.dxy == "BUY") ? C_GREEN : (aiSig.dxy == "SELL" ? C_RED : C_YELLOW);
      DrawPillBadge(x, y, 70, 14, "DXY:"+aiSig.dxy, ARGB(100, (int)((dxyClr>>16)&0xFF), (int)((dxyClr>>8)&0xFF), (int)(dxyClr&0xFF)), dxyClr);
   }
   if(aiSig.isValid && aiSig.regime != "" && aiSig.regime != NULL)
   {
      uint regClr = (aiSig.regime == "STRONG_TREND") ? C_GREEN : (aiSig.regime == "TRENDING" ? C_BLUE : C_YELLOW);
      string regShort = aiSig.regime;
      if(StringLen(regShort) > 8) regShort = StringSubstr(regShort, 0, 8);
      DrawPillBadge(x+80, y, 90, 14, "R:"+regShort, ARGB(80, (int)((regClr>>16)&0xFF), (int)((regClr>>8)&0xFF), (int)(regClr&0xFF)), regClr);
   }
   {
      uint orbClr = (aiSig.orb_dir == "BUY") ? C_GREEN : C_RED;
      DrawPillBadge(x+180, y, 80, 14, "ORB:"+aiSig.orb_dir, ARGB(80, (int)((orbClr>>16)&0xFF), (int)((orbClr>>8)&0xFF), (int)(orbClr&0xFF)), orbClr);
   }
   
   // V10: Meta-Confidence Gauge (NEW - key V10 feature)
   y += 18;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, "META-CONF", C_PURPLE);
   double mcPct = g_v10_meta_conf * 100;
   uint mcClr = (mcPct >= 60) ? C_GREEN : (mcPct >= 40 ? C_YELLOW : C_RED);
   canvas.FontSet(FNT, -90);
   canvas.TextOut(x+70, y-1, StringFormat("%.0f%%", mcPct), mcClr);
   // Meta-confidence progress bar
   int mcBarX = x + 115;
   int mcBarW = cW - 130;
   DrawProgressBar(mcBarX, y+2, mcBarW, 8, g_v10_meta_conf, mcClr);
   // Threshold marker on bar
   if(V10_MetaConfThreshold > 0 && V10_MetaConfThreshold < 1)
   {
      int threshX = mcBarX + (int)(V10_MetaConfThreshold * mcBarW);
      canvas.Line(threshX, y+1, threshX, y+11, C_YELLOW);
      canvas.FontSet(FNT, -55);
      canvas.TextOut(threshX-5, y+12, "T", C_YELLOW);
   }
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  CONFLUENCE METER
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 22;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   DrawSectionHeader(6, y, cW, "CONFLUENCE", C_GOLD);
   
   int effectiveMin = MinScore;
   int vr2 = GetVolatilityRegime();
   if(vr2 < 0) effectiveMin = MathMax(3, MinScore - 1);
   else if(vr2 > 0) effectiveMin = MinScore + 1;
   
   canvas.FontSet(FNT, -70);
    canvas.TextOut(x+200, y+5, StringFormat("Min: %d/40", (g_adaptive_minScore > 0 ? (int)g_adaptive_minScore : effectiveMin)), C_DIM);
   
   y += 26;
   // BUY bar
   canvas.FontSet(FNT, -90);
   canvas.TextOut(x, y, "BUY", C_GREEN);
   canvas.TextOut(x+40, y+2, IntegerToString(buyScore), C_TEXT);
    DrawProgressBar(x+65, y+3, cW-100, 10, buyScore/40.0, C_GREEN);
   
   y += 18;
   canvas.TextOut(x, y, "SELL", C_RED);
   canvas.TextOut(x+40, y+2, IntegerToString(sellScore), C_TEXT);
    DrawProgressBar(x+65, y+3, cW-100, 10, sellScore/40.0, C_RED);
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  V6 INTELLIGENCE â€” Vol Regime + Feature Grid
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 22;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   DrawSectionHeader(6, y, cW, "V10 INTELLIGENCE", C_CYAN);
   
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x+230, y+5, StringFormat("ONNX: %d runs", onnxRunCount), C_DIM);
   
   // Vol Regime
   y += 26;
   int volReg = GetVolatilityRegime();
   string volText = (volReg < 0 ? "LOW" : (volReg > 0 ? "HIGH" : "NORMAL"));
   uint volClr = (volReg < 0 ? C_GREEN : (volReg > 0 ? C_RED : C_YELLOW));
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, "VOL REGIME", C_DIM);
   DrawGlowDot(x+82, y+5, 4, volClr);
   canvas.FontSet(FNT, -90);
   canvas.TextOut(x+95, y-1, volText, volClr);
   
   // ATR ratio
   {
      double atr_c2[], atr_l2[];
      ArrayResize(atr_c2, 1); ArrayResize(atr_l2, 50);
      ArraySetAsSeries(atr_c2, true); ArraySetAsSeries(atr_l2, true);
      if(CopyBuffer(h_atr, 0, 0, 1, atr_c2) > 0 && CopyBuffer(h_atr, 0, 0, 50, atr_l2) >= 50)
      {
         double avgA2 = 0;
         for(int ai2 = 0; ai2 < 50; ai2++) avgA2 += atr_l2[ai2];
         avgA2 /= 50.0;
         double ratio2 = (avgA2 > 0) ? atr_c2[0] / avgA2 : 1.0;
         canvas.FontSet(FNT, -70);
         canvas.TextOut(x+200, y, StringFormat("ATR: %.2fx | Min: %d", ratio2, effectiveMin), C_DIM);
      }
   }
   
   // Feature Grid â€” pill badges for firing features
   y += 18;
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, "FIRING FEATURES", C_DIM);
   
   // Gather firing features
   string fNames[8]; string fVals[8]; uint fClrs[8]; int fCount = 0;
   
   // Technicals scan
   double rsiBuf2[];
   ArrayResize(rsiBuf2, 1); ArraySetAsSeries(rsiBuf2, true);
   if(CopyBuffer(h_rsi, 0, 0, 1, rsiBuf2) > 0)
   {
      if(rsiBuf2[0] < 30 || rsiBuf2[0] > 70)
      {
         if(fCount < 8) { fNames[fCount]="RSI"; fVals[fCount]=StringFormat("%.0f",rsiBuf2[0]); fClrs[fCount]=rsiBuf2[0]<30?C_GREEN:C_RED; fCount++; }
      }
   }
   
   double efBuf2[], esBuf2[];
   ArrayResize(efBuf2, 1); ArrayResize(esBuf2, 1);
   ArraySetAsSeries(efBuf2, true); ArraySetAsSeries(esBuf2, true);
   if(CopyBuffer(h_ema_fast, 0, 0, 1, efBuf2) > 0 && CopyBuffer(h_ema_slow, 0, 0, 1, esBuf2) > 0)
   {
      bool bull = efBuf2[0] > esBuf2[0];
      if(fCount < 8) { fNames[fCount]="EMA"; fVals[fCount]=bull?"BULL":"BEAR"; fClrs[fCount]=bull?C_GREEN:C_RED; fCount++; }
   }
   
   double adxBuf2[];
   ArrayResize(adxBuf2, 1); ArraySetAsSeries(adxBuf2, true);
   if(CopyBuffer(h_adx, 0, 0, 1, adxBuf2) > 0 && adxBuf2[0] > ADX_MinTrend)
   {
      if(fCount < 8) { fNames[fCount]="ADX"; fVals[fCount]=StringFormat("%.0f",adxBuf2[0]); fClrs[fCount]=adxBuf2[0]>35?C_GREEN:C_YELLOW; fCount++; }
   }
   
   double bbUp2[], bbLow2[];
   ArrayResize(bbUp2, 1); ArrayResize(bbLow2, 1);
   ArraySetAsSeries(bbUp2, true); ArraySetAsSeries(bbLow2, true);
   if(CopyBuffer(h_bb, 1, 0, 1, bbUp2) > 0 && CopyBuffer(h_bb, 2, 0, 1, bbLow2) > 0)
   {
      double bNow = symInfo.Bid();
      if(bNow > bbUp2[0] || bNow < bbLow2[0])
      {
         if(fCount < 8) { fNames[fCount]="BB"; fVals[fCount]=bNow>bbUp2[0]?"BRKUP":"BRKDN"; fClrs[fCount]=bNow>bbUp2[0]?C_GREEN:C_RED; fCount++; }
      }
   }
   
   // SMC
   bool hasFVG2 = DetectFVG(1) || DetectFVG(-1);
   bool hasOB2 = DetectOB(1) || DetectOB(-1);
   bool hasSweep2 = DetectSweep(1) || DetectSweep(-1);
   if(hasFVG2 && fCount < 8) { fNames[fCount]="FVG"; fVals[fCount]=DetectFVG(1)?"BULL":"BEAR"; fClrs[fCount]=DetectFVG(1)?C_GREEN:C_RED; fCount++; }
   if(hasOB2 && fCount < 8) { fNames[fCount]="OB"; fVals[fCount]=DetectOB(1)?"BULL":"BEAR"; fClrs[fCount]=DetectOB(1)?C_GREEN:C_RED; fCount++; }
   if(hasSweep2 && fCount < 8) { fNames[fCount]="SWEEP"; fVals[fCount]=DetectSweep(1)?"UP":"DN"; fClrs[fCount]=DetectSweep(1)?C_GREEN:C_RED; fCount++; }
   if(aiSig.isValid && aiSig.dxy != "" && aiSig.dxy != NULL && fCount < 8) 
   { fNames[fCount]="DXY"; fVals[fCount]=aiSig.dxy; fClrs[fCount]=(aiSig.dxy=="SELL")?C_GREEN:(aiSig.dxy=="BUY"?C_RED:C_YELLOW); fCount++; }
   
   canvas.TextOut(x+130, y, StringFormat("%d/77", fCount), fCount > 4 ? C_ACCENT : C_DIM);
   
   // Draw pill badges in 2 rows of 4
   y += 14;
   for(int fi = 0; fi < 8; fi++)
   {
      int row = fi / 4;
      int col = fi % 4;
      int px = x + col * 108;
      int py = y + row * 18;
      
      if(fi < fCount)
      {
         uint pillBg = ARGB(80, (int)((fClrs[fi]>>16)&0xFF), (int)((fClrs[fi]>>8)&0xFF), (int)(fClrs[fi]&0xFF));
         DrawPillBadge(px, py, 100, 14, fNames[fi]+":"+fVals[fi], pillBg, fClrs[fi]);
      }
      else
      {
         DrawPillBadge(px, py, 100, 14, "--", ARGB(60, 35, 40, 55), ARGB(200, 50, 55, 70));
      }
   }
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  SESSION â€” Active Session + Progress
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 42;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   
   MqlDateTime nowDt2;
   TimeCurrent(nowDt2);
   string sess2 = GetSession(nowDt2.hour);
   
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, "SESSION", C_DIM);
   canvas.TextOut(x+310, y, StringFormat("%02d:%02d UTC", nowDt2.hour, nowDt2.min), C_DIM);
   
   y += 14;
   // Session detection
   int sessStartH=0, sessEndH=0;
   uint sessClr = C_ACCENT;
   string sessName = sess2;
   
   if(sess2 == "ASIA")         { sessStartH=0;  sessEndH=7;  sessClr=C_GREEN; }
   else if(sess2 == "ASIA-LON"){ sessName="ASIA-LONDON"; sessStartH=7; sessEndH=8; sessClr=ARGB(255,0,200,170); }
   else if(sess2 == "LONDON")  { sessStartH=8;  sessEndH=13; sessClr=C_BLUE; }
   else if(sess2 == "LON-NY")  { sessName="LONDON-NY"; sessStartH=13; sessEndH=14; sessClr=C_CYAN; }
   else if(sess2 == "NY")      { sessStartH=14; sessEndH=17; sessClr=ARGB(255,255,165,0); }
   else if(sess2 == "NY-LATE") { sessName="NY LATE"; sessStartH=17; sessEndH=21; sessClr=C_PURPLE; }
   else                        { sessName="OFF"; sessStartH=21; sessEndH=24; sessClr=C_DIM; }
   
   canvas.FontSet(FNT, -120);
   canvas.TextOut(x, y, sessName, sessClr);
   
   // Progress bar
   double sessPct = 0;
   if(sessEndH > sessStartH)
   {
      double elapsed = nowDt2.hour - sessStartH + nowDt2.min / 60.0;
      sessPct = elapsed / (sessEndH - sessStartH);
   }
   y += 20;
   DrawProgressBar(x, y, cW-14, 8, sessPct, sessClr);
   
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   //  ACTIVE POSITION
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 18;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 4;
   DrawSectionHeader(6, y, cW, "ACTIVE POSITION", C_ACCENT);
   
   y += 24;
   bool hasPos = false;
   for(int p2 = PositionsTotal() - 1; p2 >= 0; p2--)
   {
      if(!posInfo.SelectByIndex(p2)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;
      
      hasPos = true;
      string dir = posInfo.PositionType() == POSITION_TYPE_BUY ? "BUY" : "SELL";
      uint dirClr = posInfo.PositionType() == POSITION_TYPE_BUY ? C_GREEN : C_RED;
      double posProfit = posInfo.Profit() + posInfo.Swap() + posInfo.Commission();
      uint profClr = posProfit >= 0 ? C_GREEN : C_RED;
      
      canvas.FontSet(FNT, -120);
      canvas.TextOut(x, y, dir, dirClr);
      canvas.FontSet(FNT, -90);
      canvas.TextOut(x+65, y+4, StringFormat("%.2f lots @ $%.2f", posInfo.Volume(), posInfo.PriceOpen()), C_TEXT);
      
      y += 18;
      canvas.FontSet(FNT, -80);
      canvas.TextOut(x, y, StringFormat("P&L: %+.2f", posProfit), profClr);
      canvas.TextOut(x+120, y, StringFormat("SL: $%.2f", posInfo.StopLoss()), C_RED);
      canvas.TextOut(x+260, y, StringFormat("TP: $%.2f", posInfo.TakeProfit()), C_GREEN);
      
      y += 14;
      long durSec = (long)(TimeCurrent() - posInfo.Time());
      canvas.TextOut(x, y, StringFormat("Duration: %dh %dm", (int)(durSec/3600), (int)((durSec%3600)/60)), C_DIM);
      break;
   }
   if(!hasPos)
   {
      canvas.FontSet(FNT, -90);
      canvas.TextOut(x, y, "No position", C_DIM);
      y += 14;
   }
   
   // V10: Adaptive + Gold Engine + P&L moved to separate panels (Brain + P&L)
   // Update the separate panels
   UpdateBrainPanel();
   UpdatePnLPanel();
   

    

    canvas.FontSet(FNT, -70);

    y += 26;

    

    // DXY Block Status

    if(g_dxy_blocking)
    {


       DrawRoundedCard(6, y-2, cW, 20, ARGB(60, 255, 50, 50), ARGB(120, 255, 50, 50), 4);

       DrawGlowDot(x+4, y+8, 4, C_RED);

       canvas.TextOut(x+14, y+2, "DXY BLOCK ACTIVE", C_RED);

       y += 22;

    }

    

    // Adaptive readouts

    canvas.TextOut(x, y, "MinScore", C_DIM);

    canvas.FontSet(FNT, -90);

    canvas.TextOut(x+75, y-1, IntegerToString((int)g_adaptive_minScore)+"/40", C_ACCENT);

    canvas.FontSet(FNT, -70);

    canvas.TextOut(x+145, y, "Kelly", C_DIM);

    canvas.FontSet(FNT, -90);

    canvas.TextOut(x+190, y-1, DoubleToString(g_adaptive_kelly*100, 1)+"%", C_CYAN);

    canvas.FontSet(FNT, -70);

    canvas.TextOut(x+270, y, "Regime", C_DIM);

    uint regAdClr = (g_adaptive_regime == "HIGH_VOL") ? C_RED : (g_adaptive_regime == "LOW_VOL" ? C_GREEN : C_YELLOW);

    DrawPillBadge(x+320, y-1, 90, 14, g_adaptive_regime, ARGB(80, (int)((regAdClr>>16)&0xFF), (int)((regAdClr>>8)&0xFF), (int)(regAdClr&0xFF)), regAdClr);

    
    // V10: Drift Rate Display + Model Weights
    y += 20;
    canvas.FontSet(FNT, -70);
    canvas.TextOut(x, y, "Drift", C_DIM);
    double driftPct = g_drift_total > 0 ? (double)g_drift_correct / g_drift_total * 100.0 : 0;
    uint driftClr = driftPct >= 60 ? C_GREEN : (driftPct >= 45 ? C_YELLOW : C_RED);
    canvas.FontSet(FNT, -90);
    canvas.TextOut(x+50, y-1, StringFormat("%.0f%% (%d/%d)", driftPct, g_drift_correct, g_drift_total), driftClr);
    
    canvas.FontSet(FNT, -70);
    canvas.TextOut(x+200, y, "Weights", C_DIM);
    canvas.FontSet(FNT, -80);
    canvas.TextOut(x+260, y, StringFormat("X:%.0f L:%.0f C:%.0f", g_regime_w_xgb*100, g_regime_w_lgb*100, g_regime_w_cat*100), C_CYAN);

    // V10: KELLY + GOLD ZONES + SIGNAL QUALITY
    y += 22;
    canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
    y += 4;
    DrawSectionHeader(6, y, cW, "V10 GOLD ENGINE", ARGB(255, 218, 175, 70));
    y += 26;
    canvas.FontSet(FNT, -70);
    
    // Kelly status
    canvas.TextOut(x, y, V10_KellyStatusString(), (g_v10_kelly_count >= 5 ? C_CYAN : C_DIM));
    y += 16;
    
    // Signal quality
    canvas.TextOut(x, y, "Signal", C_DIM);
    canvas.FontSet(FNT, -80);
    string sigQual = StringFormat("Streak:%d/%d | Margin:%.0f%%", g_v10_signal_streak, V10_SignalPersistence, V10_ConfidenceMargin*100);
    canvas.TextOut(x+55, y, sigQual, C_TEXT);
    canvas.FontSet(FNT, -70);
    
    // V10: Meta-Confidence readout in Gold Engine
    y += 16;
    if(V10_UseMetaConfidence)
    {
       canvas.TextOut(x, y, "Meta", C_PURPLE);
       double mcP = g_v10_meta_conf * 100;
       uint mcC = (mcP >= 60) ? C_GREEN : (mcP >= 40 ? C_YELLOW : C_RED);
       canvas.FontSet(FNT, -80);
       canvas.TextOut(x+35, y, StringFormat("%.0f%% / %.0f%% thresh", mcP, V10_MetaConfThreshold*100), mcC);
       canvas.FontSet(FNT, -70);
       canvas.TextOut(x+220, y, g_v10_meta_conf >= V10_MetaConfThreshold ? "GATE PASS" : "GATE BLOCK", 
                     g_v10_meta_conf >= V10_MetaConfThreshold ? C_GREEN : C_RED);
    }
    else
    {
       canvas.TextOut(x, y, "Meta: OFF", C_DIM);
    }
    
    // Gold zone
    y += 16;
    int pzBonus = V10_GoldPowerZoneBonus();
    if(pzBonus > 0)
    {
       DrawPillBadge(x, y-1, 120, 14, StringFormat("POWER ZONE +%d", pzBonus), ARGB(60, 218, 175, 70), C_GOLD);
    }
    else
    {
       canvas.TextOut(x, y, "Gold Zone: OFF", C_DIM);
    }
    
    // DXY gate
    canvas.TextOut(x+140, y, "DXY", C_DIM);
    string dxyTxt = (aiSig.dxy != "") ? StringFormat("%s(%d)", aiSig.dxy, aiSig.dxy_score) : "N/A";
    uint dxyC = (aiSig.dxy == "BUY") ? C_RED : (aiSig.dxy == "SELL" ? C_GREEN : C_DIM);
    canvas.TextOut(x+175, y, dxyTxt, dxyC);
    
    // Session spread
    double sessSpread = V10_SessionMaxSpread();
    if(sessSpread > 0)
    {
       canvas.TextOut(x+270, y, StringFormat("MaxSpd:$%.2f", sessSpread), C_DIM);
    }

    //  P&L SPARKLINE

    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    y += 22;

    canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);

    y += 4;

    DrawSectionHeader(6, y, cW, "P&L CURVE", C_GOLD);

    y += 26;

    int sparkH = 40;

    int sparkW = cW - 14;

    int sparkX = x;

    int sparkY = y;

    DrawRoundedCard(6, sparkY-4, cW, sparkH+8, ARGB(200, 12, 14, 22), C_LINE, 4);

    

    if(g_pnl_count > 1)
    {


       double pMin = 0, pMax = 0, cumPnl = 0;

       double cumArr[];

       ArrayResize(cumArr, g_pnl_count);

       for(int ci = 0; ci < g_pnl_count; ci++)
       {


          cumPnl += g_pnl_history[ci];

          cumArr[ci] = cumPnl;

          if(cumPnl < pMin) pMin = cumPnl;

          if(cumPnl > pMax) pMax = cumPnl;

       }

       double range = MathMax(pMax - pMin, 1.0);

       

       // Zero line

       int zeroY2 = sparkY + (int)((pMax / range) * sparkH);

       if(zeroY2 >= sparkY && zeroY2 <= sparkY + sparkH)

          canvas.Line(sparkX, zeroY2, sparkX + sparkW, zeroY2, ARGB(60, 100, 100, 120));

       

       // Draw curve

       for(int ci = 1; ci < g_pnl_count; ci++)
       {


          int x1px = sparkX + (int)(((double)(ci-1) / (g_pnl_count-1)) * sparkW);

          int x2px = sparkX + (int)(((double)ci / (g_pnl_count-1)) * sparkW);

          int y1px = sparkY + sparkH - (int)(((cumArr[ci-1] - pMin) / range) * sparkH);

          int y2px = sparkY + sparkH - (int)(((cumArr[ci] - pMin) / range) * sparkH);

          uint lineClr = cumArr[ci] >= 0 ? C_GREEN : C_RED;

          canvas.Line(x1px, y1px, x2px, y2px, lineClr);

          canvas.Line(x1px, y1px+1, x2px, y2px+1, lineClr);

       }

       canvas.FontSet(FNT, -70);

       canvas.TextOut(sparkX, sparkY + sparkH + 2, StringFormat("$%+.2f", cumArr[g_pnl_count-1]),

                      cumArr[g_pnl_count-1] >= 0 ? C_GREEN : C_RED);

       canvas.TextOut(sparkX + sparkW - 60, sparkY + sparkH + 2,

                      StringFormat("%d trades", g_pnl_count), C_DIM);

    }

    else
    {


       canvas.FontSet(FNT, -80);

       canvas.TextOut(sparkX + sparkW/2 - 50, sparkY + sparkH/2 - 4, "Awaiting trades...", C_DIM);

    }

    

    y += sparkH + 16;

    canvas.FontSet(FNT, -70);

    double wr3 = (winsToday + lossesToday) > 0 ? (double)winsToday / (winsToday + lossesToday) * 100.0 : 0;

    canvas.TextOut(x, y, StringFormat("Today: W:%d L:%d WR:%.0f%%", winsToday, lossesToday, wr3), C_TEXT);

    canvas.TextOut(x+240, y, StringFormat("Closed: $%+.2f", todayClosedPnL), todayClosedPnL >= 0 ? C_GREEN : C_RED);

    

   //  FOOTER
   // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   y += 18;
   canvas.FillRectangle(0, y, PNL_W, y+1, C_LINE);
   y += 6;
   
   canvas.FontSet(FNT, -70);
   canvas.TextOut(x, y, StringFormat("Opened: %d | Closed: %d | W:%d L:%d", totalTradesOpened, totalTradesClosed, winsToday, lossesToday), C_DIM);
   
   y += 14;
   // Status
   uint statusClr = C_TEXT;
   if(StringFind(tradingStatus, "STOP") >= 0 || StringFind(tradingStatus, "EMERGENCY") >= 0) statusClr = C_RED;
   else if(StringFind(tradingStatus, "PAUSE") >= 0 || StringFind(tradingStatus, "NEWS") >= 0 || StringFind(tradingStatus, "OFF") >= 0) statusClr = C_YELLOW;
   else if(StringFind(tradingStatus, "ACTIVE") >= 0) statusClr = C_GREEN;
   else if(StringFind(tradingStatus, "TARGET") >= 0) statusClr = C_GOLD;
   canvas.TextOut(x, y, tradingStatus, statusClr);
   
   y += 14;
   bool bridgeOk = aiSig.isValid;
   DrawGlowDot(x+3, y+4, 3, bridgeOk ? C_GREEN : C_RED);
   canvas.TextOut(x+12, y, StringFormat("Tick:%d | %s | AI:%s | ONNX:%d",
                  tickCount, TimeToString(lastHUDUpdate, TIME_MINUTES|TIME_SECONDS),
                  bridgeOk ? "LIVE" : "OFF", onnxRunCount), bridgeOk ? C_GREEN : C_DIM);
   
   y += 14;
   if(eaStartTime > 0)
   {
      long uptimeSec = (long)(TimeCurrent() - eaStartTime);
      canvas.TextOut(x, y, StringFormat("Uptime: %dh %dm | Antigravity V6 Control Room", (int)(uptimeSec/3600), (int)((uptimeSec%3600)/60)), C_DIM);
   }
   
   // Chart overlays (throttled)
   static datetime lastOverlayDraw2 = 0;
   if(TimeCurrent() - lastOverlayDraw2 >= 5)
   {
      DrawChartOverlays();
      lastOverlayDraw2 = TimeCurrent();
   }
   
   // â•â•â• Push bitmaps to chart â•â•â•
   UpdateGateHUD();
   canvas.Update();
   ChartRedraw();
}
//+------------------------------------------------------------------+





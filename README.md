# GoldForge V10 🏆

**ML-Powered XAUUSD M15 Ensemble Trading System**

## Performance
| Metric | Value |
|--------|-------|
| OOS Profit Factor | **9.79** |
| OOS Win Rate | **80.9%** |
| Features | 77 |
| Magic Number | 202603 |

## Architecture

### Ensemble Weights
- **CatBoost**: 0.3125
- **XGBoost**: 0.5266
- **LightGBM**: 0.1608

### Meta-Confidence Gate
- L2 Logistic Regression (14 features)
- Dynamic lot scaling via `V10_ScaleLotByMetaConfidence()`
- PASS/GATE threshold system

### Key Components
- `EA/TradingAI_Predictor_EA_v10.mq5` — Main Expert Advisor (5840 lines)
- `EA/TradeMonitor_HUD.mq5` — Realtime dashboard overlay
- `MQL5/Include/onnx_features_v10.mqh` — 77-feature engineering
- `MQL5/Include/v10_strategy.mqh` — Strategy logic
- `ONNX/models/` — CatBoost, XGBoost, LightGBM + meta-confidence models
- `config/presets/` — .set files for 5K/10K/25K accounts

## Deployment
- **VPS**: Windows Server (65.19.188.191)
- **Platform**: MetaTrader 5, XAUUSD M15
- **Watchdog**: Auto-restart monitoring
- **Auto-pickup**: New .ex5 detected on restart

## Version History
- **V10** — Meta-confidence gate, 77 features, 3-model ensemble
- **V9** — Meta-confidence v1
- **V6** — HUD upgrade
- **V5** — Initial production release

## License
Proprietary — All rights reserved
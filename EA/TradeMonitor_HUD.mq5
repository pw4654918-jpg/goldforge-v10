//+------------------------------------------------------------------+
//|                                       TradeMonitor_HUD.mq5        |
//|         Independent Trade Logger & Professional HUD               |
//|         Zero interference with other EAs — read-only monitoring   |
//+------------------------------------------------------------------+
#property copyright "Trade Monitor"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                    |
//+------------------------------------------------------------------+
input group "═══ MONITOR SETTINGS ═══"
input int    InpUpdateIntervalSec = 1;         // History scan interval (seconds)
input bool   InpLogToFile         = true;      // Write CSV log to Files folder
input bool   InpLogToTerminal     = true;      // Print to Experts tab
input bool   InpShowOpenPositions = true;      // Show open positions in HUD
input string InpLogFileName       = "TradeMonitor_Log.csv"; // CSV filename

//+------------------------------------------------------------------+
//| Structs                                                             |
//+------------------------------------------------------------------+
struct EAMetrics
{
   int      magic;
   string   name;
   int      tradesToday;
   int      winsToday;
   int      lossesToday;
   double   pnlToday;
   double   lotsToday;
   int      tradesTotal;
   double   pnlTotal;
};

struct DealRecord
{
   ulong    ticket;
   datetime time;
   int      magic;
   string   symbol;
   string   eaName;
   string   direction;
   double   volume;
   double   profit;
   double   sl;
   double   tp;
   double   priceOpen;
   double   priceClose;
   int      durationMin;
};

//+------------------------------------------------------------------+
//| Globals                                                             |
//+------------------------------------------------------------------+
EAMetrics     g_metrics[];
int           g_metricCount = 0;
ulong         g_lastDealTicket = 0;
datetime      g_lastScanTime = 0;
datetime      g_todayStart = 0;
int           g_fileHandle = INVALID_HANDLE;
string        g_logPath = "";

//+------------------------------------------------------------------+
//| Expert Initialization                                               |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("╔════════════════════════════════════════════════════════════╗");
   Print("║           TradeMonitor HUD v1.00 — Starting                ║");
   Print("╚════════════════════════════════════════════════════════════╝");
   
   // Calculate today start
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   g_todayStart = StringToTime(StringFormat("%04d.%02d.%02d 00:00", dt.year, dt.mon, dt.day));
   
   // Setup CSV log file
   if(InpLogToFile)
   {
      g_logPath = InpLogFileName;
      // Write header only if file doesn't exist yet today
      string fullPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + g_logPath;
      g_fileHandle = FileOpen(g_logPath, FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON, ',');
      if(g_fileHandle != INVALID_HANDLE)
      {
         long fileSize = FileSize(g_fileHandle);
         if(fileSize == 0)
         {
            FileWrite(g_fileHandle,
               "Time", "Magic", "EA_Name", "Symbol", "Direction", "Volume",
               "Profit", "PriceOpen", "PriceClose", "SL", "TP", "DurationMin", "Ticket");
         }
         FileClose(g_fileHandle);
         Print("[Monitor] CSV log: ", g_logPath);
      }
      else
      {
         Print("[Monitor] WARNING: Could not open log file");
      }
   }
   
   // Initial scan
   ScanHistory();
   
   // Timer for background scanning
   EventSetTimer(InpUpdateIntervalSec);
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                             |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
   Print("[Monitor] Stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer Handler                                                       |
//+------------------------------------------------------------------+
void OnTimer()
{
   ScanHistory();
}

//+------------------------------------------------------------------+
//| Tick Handler (backup scan)                                          |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime now = TimeCurrent();
   if(now - g_lastScanTime >= InpUpdateIntervalSec)
   {
      ScanHistory();
   }
}

//+------------------------------------------------------------------+
//| Scan Deal History                                                   |
//+------------------------------------------------------------------+
void ScanHistory()
{
   g_lastScanTime = TimeCurrent();
   
   // Reset daily counters if new day
   MqlDateTime dt;
   TimeToStruct(g_lastScanTime, dt);
   datetime currentDayStart = StringToTime(StringFormat("%04d.%02d.%02d 00:00", dt.year, dt.mon, dt.day));
   if(currentDayStart != g_todayStart)
   {
      g_todayStart = currentDayStart;
      ResetDailyCounters();
      Print("[Monitor] 📅 New day detected — counters reset");
   }
   
   if(!HistorySelect(0, D'2099.12.31')) return;
   
   int totalDeals = HistoryDealsTotal();
   if(totalDeals == 0) return;
   
   bool newDealFound = false;
   
   for(int i = 0; i < totalDeals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      
      // Skip already-processed deals
      if(ticket <= g_lastDealTicket) continue;
      
      // Only process closing deals (the actual trade result)
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY) continue;
      
      // Skip zero-magic deals (manual trades can have 0)
      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      
      datetime dealTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
      double priceClose = HistoryDealGetDouble(ticket, DEAL_PRICE);
      long dealType = HistoryDealGetInteger(ticket, DEAL_TYPE);
      string direction = (dealType == DEAL_TYPE_BUY) ? "BUY" : "SELL";
      
      // Try to find entry price & SL/TP via position history
      double priceOpen = 0;
      double sl = 0;
      double tp = 0;
      int durationMin = 0;
      long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      if(HistorySelectByPosition(posId))
      {
         for(int d = HistoryDealsTotal()-1; d >= 0; d--)
         {
            ulong dticket = HistoryDealGetTicket(d);
            if(HistoryDealGetInteger(dticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
            {
               priceOpen = HistoryDealGetDouble(dticket, DEAL_PRICE);
               datetime openTime = (datetime)HistoryDealGetInteger(dticket, DEAL_TIME);
               durationMin = (int)((dealTime - openTime) / 60);
               break;
            }
         }
      }
      
      // Try to get SL/TP from currently known position info
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong pticket = PositionGetTicket(p);
         if(PositionSelectByTicket(pticket))
         {
            if(PositionGetInteger(POSITION_IDENTIFIER) == posId)
            {
               sl = PositionGetDouble(POSITION_SL);
               tp = PositionGetDouble(POSITION_TP);
               break;
            }
         }
      }
      
      // Derive EA name from magic number mapping
      string eaName = ResolveEAName((int)magic);
      
      // Log to file
      if(InpLogToFile)
      {
         g_fileHandle = FileOpen(g_logPath, FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON, ',');
         if(g_fileHandle != INVALID_HANDLE)
         {
            FileSeek(g_fileHandle, 0, SEEK_END);
            FileWrite(g_fileHandle,
               TimeToString(dealTime, TIME_SECONDS),
               magic,
               eaName,
               symbol,
               direction,
               volume,
               profit,
               priceOpen,
               priceClose,
               sl,
               tp,
               durationMin,
               ticket);
            FileClose(g_fileHandle);
         }
      }
      
      // Update metrics
      UpdateMetrics((int)magic, eaName, profit, volume, dealTime >= g_todayStart);
      
      if(InpLogToTerminal && dealTime >= g_todayStart)
      {
         Print(StringFormat("[Monitor] CLOSED | %s | Magic:%d | %s | %s | Vol:%.2f | PnL:$%.2f",
            TimeToString(dealTime, TIME_SECONDS), magic, eaName, direction, volume, profit));
      }
      
      newDealFound = true;
      g_lastDealTicket = ticket;
   }
   
   if(newDealFound)
   {
      UpdateHUD();
   }
}

//+------------------------------------------------------------------+
//| Resolve EA Name from Magic Number                                 |
//+------------------------------------------------------------------+
string ResolveEAName(int magic)
{
   switch(magic)
   {
      case 110202604: return "GoldForge_Simple_ONNX";
      case 110202605: return "GoldForge_CatBoost_Solo";
      case 79420126:  return "KimiGoldEA_Pro";
      case 202603:    return "TradingAI_Predictor_v9";
      case 2024001:   return "GoldInstitutionEA";
      case 10101010:  return "15CaratGold";
      case 123456:    return "ClaudeXAU/SWIFT";
      case 202405:    return "GoldShadow";
      case 999111:    return "GoldForge_NeuralBridge";
      default:        return "EA_" + IntegerToString(magic);
   }
}

//+------------------------------------------------------------------+
//| Update Metrics Array                                                |
//+------------------------------------------------------------------+
void UpdateMetrics(int magic, string eaName, double profit, double volume, bool isToday)
{
   int idx = -1;
   for(int i = 0; i < g_metricCount; i++)
   {
      if(g_metrics[i].magic == magic)
      {
         idx = i;
         break;
      }
   }
   
   if(idx < 0)
   {
      ArrayResize(g_metrics, g_metricCount + 1);
      idx = g_metricCount;
      g_metrics[idx].magic = magic;
      g_metrics[idx].name = eaName;
      g_metrics[idx].tradesToday = 0;
      g_metrics[idx].winsToday = 0;
      g_metrics[idx].lossesToday = 0;
      g_metrics[idx].pnlToday = 0;
      g_metrics[idx].lotsToday = 0;
      g_metrics[idx].tradesTotal = 0;
      g_metrics[idx].pnlTotal = 0;
      g_metricCount++;
   }
   
   g_metrics[idx].tradesTotal++;
   g_metrics[idx].pnlTotal += profit;
   
   if(isToday)
   {
      g_metrics[idx].tradesToday++;
      g_metrics[idx].lotsToday += volume;
      g_metrics[idx].pnlToday += profit;
      if(profit >= 0) g_metrics[idx].winsToday++;
      else            g_metrics[idx].lossesToday++;
   }
}

//+------------------------------------------------------------------+
//| Reset Daily Counters                                                |
//+------------------------------------------------------------------+
void ResetDailyCounters()
{
   for(int i = 0; i < g_metricCount; i++)
   {
      g_metrics[i].tradesToday = 0;
      g_metrics[i].winsToday = 0;
      g_metrics[i].lossesToday = 0;
      g_metrics[i].pnlToday = 0;
      g_metrics[i].lotsToday = 0;
   }
}

//+------------------------------------------------------------------+
//| Build & Render HUD                                                  |
//+------------------------------------------------------------------+
void UpdateHUD()
{
   string hud = "";
   
   hud += "═══════════ TRADE MONITOR ═══════════\n";
   
   // Open positions summary
   if(InpShowOpenPositions)
   {
      int totalOpen = 0;
      string posLines = "";
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket))
         {
            totalOpen++;
            string sym = PositionGetString(POSITION_SYMBOL);
            string type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
            double vol = PositionGetDouble(POSITION_VOLUME);
            double profit = PositionGetDouble(POSITION_PROFIT);
            int magic = (int)PositionGetInteger(POSITION_MAGIC);
            string ea = ResolveEAName(magic);
            posLines += StringFormat("  %-20s | %s | %.2fl | $%.2f\n", ea, type, vol, profit);
         }
      }
      hud += StringFormat("🟢 OPEN POSITIONS: %d\n", totalOpen);
      if(totalOpen > 0) hud += posLines;
      hud += "────────────────────────────────────\n";
   }
   
   // Daily reset timer
   datetime nextReset = g_todayStart + 86400;
   int secsLeft = (int)(nextReset - TimeCurrent());
   string timerStr;
   if(secsLeft <= 0)
      timerStr = "00:00:00";
   else {
      int h = secsLeft / 3600;
      int m = (secsLeft % 3600) / 60;
      int s = secsLeft % 60;
      timerStr = StringFormat("%02d:%02d:%02d", h, m, s);
   }
   hud += "⏰ DD Reset in: " + timerStr + "\n";
   hud += "────────────────────────────────────\n";
   
   // Today's stats header
   hud += "📅 TODAY'S CLOSED TRADES\n";
   hud += StringFormat("%-22s %5s %5s %8s %10s\n", "EA", "W", "L", "WR%", "PnL $");
   hud += "────────────────────────────────────\n";
   
   double totalPnL = 0;
   int totalW = 0, totalL = 0, totalTrades = 0;
   
   for(int i = 0; i < g_metricCount; i++)
   {
      if(g_metrics[i].tradesToday == 0) continue;
      
      int w = g_metrics[i].winsToday;
      int l = g_metrics[i].lossesToday;
      double wr = (w + l) > 0 ? (double)w / (w + l) * 100.0 : 0;
      double pnl = g_metrics[i].pnlToday;
      
      totalW += w;
      totalL += l;
      totalTrades += g_metrics[i].tradesToday;
      totalPnL += pnl;
      
      string pnlStr = StringFormat("%.2f", pnl);
      if(pnl >= 0) pnlStr = "+" + pnlStr;
      
      hud += StringFormat("%-22s %5d %5d %7.0f%% %10s\n",
         g_metrics[i].name, w, l, wr, pnlStr);
   }
   
   if(totalTrades == 0)
   {
      hud += "  (No closed trades yet today)\n";
   }
   else
   {
      hud += "────────────────────────────────────\n";
      double totalWR = (totalW + totalL) > 0 ? (double)totalW / (totalW + totalL) * 100.0 : 0;
      string totalPnLStr = StringFormat("%.2f", totalPnL);
      if(totalPnL >= 0) totalPnLStr = "+" + totalPnLStr;
      hud += StringFormat("%-22s %5d %5d %7.0f%% %10s\n", "TOTAL", totalW, totalL, totalWR, totalPnLStr);
   }
   
   hud += "════════════════════════════════════\n";
   hud += "Log: " + g_logPath;
   
   Comment(hud);
}

//+------------------------------------------------------------------+

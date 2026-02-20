# Dataset Cleaning and Pattern Analysis

## Coverage and Data Quality
- Fear & Greed rows (clean): **2,644**
- Historical rows (clean): **211,224**
- Historical rows matched to Fear & Greed date: **211,218 / 211,224**
- Trading date window: **2023-05-01 to 2025-05-01**
- `Timestamp` raw quality: only **7 unique values** for 211,224 rows; day match vs `Timestamp IST` is **1.02%**
- Cleaning rule used: treat `Timestamp IST` as authoritative event time and keep `Timestamp` as low-resolution metadata.
- Negative fee rows (likely maker rebates): **2,476** | Zero-USD rows: **43** | Non-zero PnL rows: **104,408**

## Fear & Greed Relationships
- Correlation: Fear/Greed value vs daily total volume = **-0.264**
- Correlation: Fear/Greed value vs daily non-zero closed PnL = **-0.083**
- Correlation: Fear/Greed value vs daily buy ratio = **-0.049**
- Highest aggregate volume occurred in **Fear** (483324789.79 USD across 61,837 trades).

### By Fear/Greed Classification
```text
fg_classification  trades  avg_trade_usd  total_volume_usd  total_closed_pnl  buy_ratio
             Fear   61837        7816.11      483324789.79        3357155.44       0.49
            Greed   50303        5736.88      288582494.72        2150129.27       0.49
    Extreme Greed   39992        3112.25      124465164.57        2715171.31       0.45
          Neutral   37686        4782.73      180242063.08        1292920.68       0.50
     Extreme Fear   21400        5349.73      114484261.44         739110.25       0.51
```

## Coin Patterns
- Best total PnL coin: **@107** (2783912.92 USD).
- Worst total PnL coin: **TRUMP** (-364824.91 USD).

### Top Coins by Volume
```text
    coin  trades  total_volume_usd  total_closed_pnl  mean_trade_usd
     BTC   26064      644232116.63         868044.73        24717.32
    HYPE   68005      141990206.05        1948484.60         2087.94
     SOL   10691      125074752.06        1639555.93        11699.07
     ETH   11158      118280994.07        1319978.84        10600.56
    @107   29992       55760858.63        2783912.92         1859.19
FARTCOIN    4650        8311390.40        -100687.21         1787.40
     SUI    1979        7781167.59         199268.83         3931.87
   TRUMP    1920        7349346.94        -364824.91         3827.78
 MELANIA    4428        7040710.45         390351.07         1590.04
     XRP    1774        5343210.53           3756.90         3011.96
    PAXG    1265        3047336.39         -18688.87         2408.96
   KBONK    1647        2995780.58          35551.25         1818.93
     WLD    1983        2792841.63          22281.92         1408.39
    @142    1309        2486351.27           7450.36         1899.43
    DOGE     826        2452103.46         147543.16         2968.65
```

### Top and Bottom Coins by PnL
Top 10:
```text
   coin  trades  total_closed_pnl  total_volume_usd
   @107   29992        2783912.92       55760858.63
   HYPE   68005        1948484.60      141990206.05
    SOL   10691        1639555.93      125074752.06
    ETH   11158        1319978.84      118280994.07
    BTC   26064         868044.73      644232116.63
MELANIA    4428         390351.07        7040710.45
    ENA     990         217329.50        1625400.50
    SUI    1979         199268.83        7781167.59
    ZRO    1239         183777.78        1213825.42
   DOGE     826         147543.16        2452103.46
```

Bottom 10:
```text
    coin  trades  total_closed_pnl  total_volume_usd
   TRUMP    1920        -364824.91        7349346.94
FARTCOIN    4650        -100687.21        8311390.40
     ADA     581         -28113.46        1094836.40
      IO     352         -21893.91         252981.67
    PAXG    1265         -18688.87        3047336.39
   KAITO     435          -8735.14         897912.98
       S     144          -8644.85         224183.08
  POPCAT    1152          -7922.18         803456.85
    @135       1          -5981.36             15.56
     NIL     226          -3175.27         481868.22
```

## Direction and Intraday Patterns
### Direction Breakdown
```text
                direction  trades  total_volume_usd  total_closed_pnl
                Open Long   49895      380171434.91              0.00
               Close Long   48678      382213248.89        3622929.39
               Open Short   39741      185484862.58              0.00
              Close Short   36013      179800893.60        3709800.10
                     Sell   19902       30108794.11        2906748.42
                      Buy   16716       31196797.04              0.00
     Spot Dust Conversion     142             25.53              0.00
             Short > Long      70        1116262.24          10793.33
             Long > Short      57         891922.10           1991.38
        Auto-Deleveraging       8         169507.75          57478.46
Liquidated Isolated Short       1          32910.88         -12752.91
               Settlement       1            782.83            -29.22
```

### Most Active IST Hours
```text
 hour  trades  total_volume_usd  total_closed_pnl
   20   12731       66254249.18         632875.06
   19   12628       72960699.97         705423.69
   21   11022       69296423.88         343553.58
    3   10524       66070901.18         460020.57
    1   10481       81311125.40         523198.63
   22   10096      100033205.14         381715.76
   23   10022       84516355.31         187943.20
    4   10015       60206043.24         487367.59
```

## Output Files
- `fear_greed_index_clean.csv`
- `historical_data_clean.csv`
- `daily_trade_metrics.csv`
- `fear_greed_class_summary.csv`
- `coin_summary.csv`
- `analysis_report.md` (this report)

## Notes
- `Trade ID` and `Timestamp` appear rounded in scientific notation in source data; numeric conversion preserves only approximate integer values.
- If you can export raw IDs and millisecond timestamps without scientific notation, trade-level sequencing and reconciliation will improve significantly.
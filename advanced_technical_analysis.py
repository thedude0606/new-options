import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('advanced_technical_analysis')

class AdvancedTechnicalAnalysis:
    """
    Advanced technical analysis module for financial time series data.
    
    This class provides comprehensive technical analysis capabilities for OHLCV data
    across multiple timeframes, with optimized lookback periods for each timeframe.
    """
    
    # Optimal periods by timeframe
    TIMEFRAME_PERIODS = {
        "1min": {
            "trend": 30,
            "momentum": {"rsi": 14, "stoch": 14, "cci": 20, "roc": 12},
            "volatility": {"bb": 20, "atr": 14, "keltner": 20, "donchian": 20},
            "volume": {"mfi": 14, "obv": 50},
            "trend_strength": {"adx": 14, "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52}},
            "pattern_scan": 10,
            "fvg_scan": 5
        },
        "15min": {
            "trend": 16,
            "momentum": {"rsi": 14, "stoch": 14, "cci": 20, "roc": 12},
            "volatility": {"bb": 20, "atr": 14, "keltner": 20, "donchian": 20},
            "volume": {"mfi": 14, "obv": 50},
            "trend_strength": {"adx": 14, "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52}},
            "pattern_scan": 10,
            "fvg_scan": 5
        },
        "1h": {
            "trend": 24,
            "momentum": {"rsi": 14, "stoch": 14, "cci": 20, "roc": 12},
            "volatility": {"bb": 20, "atr": 14, "keltner": 20, "donchian": 20},
            "volume": {"mfi": 14, "obv": 50},
            "trend_strength": {"adx": 14, "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52}},
            "pattern_scan": 20,
            "fvg_scan": 10
        },
        "daily": {
            "trend": 20,
            "momentum": {"rsi": 14, "stoch": 14, "cci": 20, "roc": 12},
            "volatility": {"bb": 20, "atr": 14, "keltner": 20, "donchian": 20},
            "volume": {"mfi": 14, "obv": 50},
            "trend_strength": {"adx": 14, "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52}},
            "pattern_scan": 20,
            "fvg_scan": 10
        }
    }
    
    def __init__(self, df: pd.DataFrame, timeframe: str = "1min"):
        """
        Initialize the technical analysis module.
        
        Args:
            df: DataFrame with DateTime index and columns: Open, High, Low, Close, Volume
            timeframe: One of "1min", "15min", "1h", "daily"
        """
        self.df = df.copy()
        self.timeframe = timeframe
        
        # Validate timeframe
        if timeframe not in self.TIMEFRAME_PERIODS:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(self.TIMEFRAME_PERIODS.keys())}")
        
        # Validate DataFrame
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        # Ensure index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Get periods for this timeframe
        self.periods = self.TIMEFRAME_PERIODS[timeframe]
        
        logger.info(f"Initialized AdvancedTechnicalAnalysis with {len(df)} rows of {timeframe} data")
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given timeframe.
        
        Returns:
            DataFrame with all indicators added as columns
        """
        logger.info(f"Calculating all indicators for {self.timeframe} timeframe")
        
        # Calculate trend indicators
        self._calculate_trend_indicators()
        
        # Calculate momentum indicators
        self._calculate_momentum_indicators()
        
        # Calculate volatility indicators
        self._calculate_volatility_indicators()
        
        # Calculate volume indicators
        self._calculate_volume_indicators()
        
        # Calculate trend strength indicators
        self._calculate_trend_strength_indicators()
        
        # Calculate candlestick patterns
        self._identify_candlestick_patterns()
        
        # Calculate fair value gaps
        self._identify_fair_value_gaps()
        
        logger.info("All indicators calculated successfully")
        return self.df
    
    def _calculate_trend_indicators(self):
        """Calculate trend-following indicators."""
        logger.info("Calculating trend indicators")
        
        n = self.periods["trend"]
        
        # SMA
        self.df[f'sma_{n}'] = self.df['Close'].rolling(n).mean()
        
        # EMA
        self.df[f'ema_{n}'] = self.df['Close'].ewm(span=n, adjust=False).mean()
        
        # HMA (Hull Moving Average)
        wma_half = self._wma(self.df['Close'], n//2)
        wma_full = self._wma(self.df['Close'], n)
        diff = 2 * wma_half - wma_full
        self.df[f'hma_{n}'] = self._wma(diff, int(math.sqrt(n)))
        
        # MACD
        fast = self.df['Close'].ewm(span=12, adjust=False).mean()
        slow = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = fast - slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        # VWAP
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        pv = tp * self.df['Volume']
        self.df['vwap'] = pv.cumsum() / self.df['Volume'].cumsum()
        
        # Add trend signals
        self.df['sma_signal'] = np.where(self.df['Close'] > self.df[f'sma_{n}'], 'bullish', 'bearish')
        self.df['ema_signal'] = np.where(self.df['Close'] > self.df[f'ema_{n}'], 'bullish', 'bearish')
        self.df['hma_signal'] = np.where(self.df['Close'] > self.df[f'hma_{n}'], 'bullish', 'bearish')
        self.df['macd_signal_bullish'] = np.where(self.df['macd'] > self.df['macd_signal'], True, False)
        self.df['vwap_signal'] = np.where(self.df['Close'] > self.df['vwap'], 'bullish', 'bearish')
        
        logger.info("Trend indicators calculated")
    
    def _calculate_momentum_indicators(self):
        """Calculate momentum and oscillator indicators."""
        logger.info("Calculating momentum indicators")
        
        # RSI
        periods = self.periods["momentum"]
        rsi_period = periods["rsi"]
        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        self.df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        stoch_period = periods["stoch"]
        hh = self.df['High'].rolling(stoch_period).max()
        ll = self.df['Low'].rolling(stoch_period).min()
        self.df['stoch_k'] = 100 * (self.df['Close'] - ll) / (hh - ll)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(3).mean()
        
        # CCI
        cci_period = periods["cci"]
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma_tp = tp.rolling(cci_period).mean()
        mean_dev = tp.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        self.df[f'cci_{cci_period}'] = (tp - sma_tp) / (0.015 * mean_dev)
        
        # ROC
        roc_period = periods["roc"]
        self.df[f'roc_{roc_period}'] = 100 * (self.df['Close'] - self.df['Close'].shift(roc_period)) / self.df['Close'].shift(roc_period)
        
        # Add momentum signals
        self.df['rsi_signal'] = np.where(self.df[f'rsi_{rsi_period}'] > 70, 'bearish', 
                                        np.where(self.df[f'rsi_{rsi_period}'] < 30, 'bullish', 'neutral'))
        
        self.df['stoch_signal'] = np.where((self.df['stoch_k'] > self.df['stoch_d']) & (self.df['stoch_k'] < 20), 'bullish',
                                           np.where((self.df['stoch_k'] < self.df['stoch_d']) & (self.df['stoch_k'] > 80), 'bearish', 'neutral'))
        
        self.df['cci_signal'] = np.where(self.df[f'cci_{cci_period}'] > 100, 'bullish',
                                        np.where(self.df[f'cci_{cci_period}'] < -100, 'bearish', 'neutral'))
        
        self.df['roc_signal'] = np.where((self.df[f'roc_{roc_period}'] > 0) & (self.df[f'roc_{roc_period}'] > self.df[f'roc_{roc_period}'].shift(1)), 'bullish',
                                         np.where((self.df[f'roc_{roc_period}'] < 0) & (self.df[f'roc_{roc_period}'] < self.df[f'roc_{roc_period}'].shift(1)), 'bearish', 'neutral'))
        
        logger.info("Momentum indicators calculated")
    
    def _calculate_volatility_indicators(self):
        """Calculate volatility indicators."""
        logger.info("Calculating volatility indicators")
        
        periods = self.periods["volatility"]
        
        # Bollinger Bands
        bb_period = periods["bb"]
        self.df['bb_mid'] = self.df['Close'].rolling(bb_period).mean()
        bb_std = self.df['Close'].rolling(bb_period).std()
        self.df['bb_upper'] = self.df['bb_mid'] + 2 * bb_std
        self.df['bb_lower'] = self.df['bb_mid'] - 2 * bb_std
        
        # ATR
        atr_period = periods["atr"]
        tr1 = self.df['High'] - self.df['Low']
        tr2 = (self.df['High'] - self.df['Close'].shift(1)).abs()
        tr3 = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df[f'atr_{atr_period}'] = tr.ewm(span=atr_period, adjust=False).mean()
        
        # Keltner Channels
        keltner_period = periods["keltner"]
        self.df['kc_mid'] = self.df['Close'].ewm(span=keltner_period, adjust=False).mean()
        self.df['kc_upper'] = self.df['kc_mid'] + 1.5 * self.df[f'atr_{atr_period}']
        self.df['kc_lower'] = self.df['kc_mid'] - 1.5 * self.df[f'atr_{atr_period}']
        
        # Donchian Channels
        donchian_period = periods["donchian"]
        self.df['dch_upper'] = self.df['High'].rolling(donchian_period).max()
        self.df['dch_lower'] = self.df['Low'].rolling(donchian_period).min()
        
        # Add volatility signals
        self.df['bb_signal'] = np.where(self.df['Close'] > self.df['bb_upper'], 'overbought',
                                       np.where(self.df['Close'] < self.df['bb_lower'], 'oversold', 'neutral'))
        
        self.df['atr_signal'] = np.where(self.df[f'atr_{atr_period}'] > self.df[f'atr_{atr_period}'].shift(1), 'high_volatility', 'low_volatility')
        
        self.df['keltner_signal'] = np.where(self.df['Close'] > self.df['kc_upper'], 'bullish',
                                            np.where(self.df['Close'] < self.df['kc_lower'], 'bearish', 'neutral'))
        
        self.df['donchian_signal'] = np.where(self.df['Close'] > self.df['dch_upper'], 'bullish',
                                             np.where(self.df['Close'] < self.df['dch_lower'], 'bearish', 'neutral'))
        
        logger.info("Volatility indicators calculated")
    
    def _calculate_volume_indicators(self):
        """Calculate volume-based indicators."""
        logger.info("Calculating volume indicators")
        
        periods = self.periods["volume"]
        
        # OBV
        self.df['obv'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        
        # MFI
        mfi_period = periods["mfi"]
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        raw_mf = tp * self.df['Volume']
        pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(mfi_period).sum()
        neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(mfi_period).sum()
        self.df[f'mfi_{mfi_period}'] = 100 - (100 / (1 + pos_mf / neg_mf))
        
        # Add volume signals
        self.df['obv_signal'] = np.where((self.df['obv'] > self.df['obv'].shift(1)) & (self.df['Close'] > self.df['Close'].shift(1)), 'bullish',
                                        np.where((self.df['obv'] < self.df['obv'].shift(1)) & (self.df['Close'] < self.df['Close'].shift(1)), 'bearish', 'neutral'))
        
        self.df['mfi_signal'] = np.where(self.df[f'mfi_{mfi_period}'] > 80, 'overbought',
                                        np.where(self.df[f'mfi_{mfi_period}'] < 20, 'oversold', 'neutral'))
        
        logger.info("Volume indicators calculated")
    
    def _calculate_trend_strength_indicators(self):
        """Calculate trend strength indicators."""
        logger.info("Calculating trend strength indicators")
        
        periods = self.periods["trend_strength"]
        
        # ADX
        adx_period = periods["adx"]
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift(1)).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        plus_dm = self.df['High'].diff()
        minus_dm = self.df['Low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0)
        
        tr_ema = tr.ewm(span=adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / tr_ema)
        minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / tr_ema)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        self.df[f'adx_{adx_period}'] = dx.ewm(span=adx_period, adjust=False).mean()
        
        # Ichimoku
        ichimoku = periods["ichimoku"]
        self.df['tenkan'] = (self.df['High'].rolling(ichimoku["tenkan"]).max() + self.df['Low'].rolling(ichimoku["tenkan"]).min()) / 2
        self.df['kijun'] = (self.df['High'].rolling(ichimoku["kijun"]).max() + self.df['Low'].rolling(ichimoku["kijun"]).min()) / 2
        self.df['senkou_a'] = ((self.df['tenkan'] + self.df['kijun']) / 2).shift(ichimoku["kijun"])
        self.df['senkou_b'] = ((self.df['High'].rolling(ichimoku["senkou"]).max() + self.df['Low'].rolling(ichimoku["senkou"]).min()) / 2).shift(ichimoku["kijun"])
        self.df['chikou'] = self.df['Close'].shift(-ichimoku["kijun"])
        
        # Add trend strength signals
        self.df['adx_signal'] = np.where(self.df[f'adx_{adx_period}'] > 25, 'strong_trend', 'weak_trend')
        self.df['adx_direction'] = np.where(plus_di > minus_di, 'bullish', 'bearish')
        
        self.df['ichimoku_signal'] = np.where(self.df['Close'] > self.df[['senkou_a', 'senkou_b']].max(axis=1), 'bullish',
                                             np.where(self.df['Close'] < self.df[['senkou_a', 'senkou_b']].min(axis=1), 'bearish', 'neutral'))
        
        logger.info("Trend strength indicators calculated")
    
    def _identify_candlestick_patterns(self):
        """Identify candlestick patterns."""
        logger.info("Identifying candlestick patterns")
        
        scan_bars = self.periods["pattern_scan"]
        
        # Helper function to check if a pattern exists
        def check_pattern(pattern_func, name):
            result = pattern_func()
            if result.any():
                self.df[f'{name}'] = result
                self.df[f'{name}_signal'] = self.df[f'{name}'].map({True: f"{name}_signal"})
        
        # Hammer
        def hammer():
            body = (self.df['Close'] - self.df['Open']).abs()
            lower_wick = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
            upper_wick = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
            return (self.df['Close'] > self.df['Open']) & (lower_wick >= 2*body) & (upper_wick <= body*0.1)
        
        check_pattern(hammer, 'hammer')
        self.df['hammer_signal'] = 'bullish'
        
        # Hanging Man (same as Hammer but after uptrend)
        def hanging_man():
            hammer_pattern = hammer()
            prior_uptrend = (self.df['Close'].shift(1) > self.df['Close'].shift(2)) & (self.df['Close'].shift(2) > self.df['Close'].shift(3))
            return hammer_pattern & prior_uptrend
        
        check_pattern(hanging_man, 'hanging_man')
        self.df['hanging_man_signal'] = 'bearish'
        
        # Inverted Hammer
        def inverted_hammer():
            body = (self.df['Close'] - self.df['Open']).abs()
            upper_wick = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
            lower_wick = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
            return (self.df['Close'] > self.df['Open']) & (upper_wick >= 2*body) & (lower_wick <= body*0.1)
        
        check_pattern(inverted_hammer, 'inverted_hammer')
        self.df['inverted_hammer_signal'] = 'bullish'
        
        # Shooting Star (same as Inverted Hammer but after uptrend)
        def shooting_star():
            inv_hammer_pattern = inverted_hammer()
            prior_uptrend = (self.df['Close'].shift(1) > self.df['Close'].shift(2)) & (self.df['Close'].shift(2) > self.df['Close'].shift(3))
            return inv_hammer_pattern & prior_uptrend
        
        check_pattern(shooting_star, 'shooting_star')
        self.df['shooting_star_signal'] = 'bearish'
        
        # Doji
        def doji():
            body = (self.df['Close'] - self.df['Open']).abs()
            total_range = self.df['High'] - self.df['Low']
            return body <= 0.1 * total_range
        
        check_pattern(doji, 'doji')
        self.df['doji_signal'] = 'indecision'
        
        # Bullish Engulfing
        def bullish_engulfing():
            return (self.df['Close'].shift(1) < self.df['Open'].shift(1)) & \
                   (self.df['Close'] > self.df['Open']) & \
                   (self.df['Open'] <= self.df['Close'].shift(1)) & \
                   (self.df['Close'] >= self.df['Open'].shift(1))
        
        check_pattern(bullish_engulfing, 'bullish_engulfing')
        self.df['bullish_engulfing_signal'] = 'bullish'
        
        # Bearish Engulfing
        def bearish_engulfing():
            return (self.df['Close'].shift(1) > self.df['Open'].shift(1)) & \
                   (self.df['Close'] < self.df['Open']) & \
                   (self.df['Open'] >= self.df['Close'].shift(1)) & \
                   (self.df['Close'] <= self.df['Open'].shift(1))
        
        check_pattern(bearish_engulfing, 'bearish_engulfing')
        self.df['bearish_engulfing_signal'] = 'bearish'
        
        # Piercing Line
        def piercing_line():
            return (self.df['Close'].shift(1) < self.df['Open'].shift(1)) & \
                   (self.df['Close'] > self.df['Open']) & \
                   (self.df['Open'] < self.df['Low'].shift(1)) & \
                   (this.df['Close'] > self.df['Open'].shift(1) + 0.5*(self.df['Close'].shift(1) - self.df['Open'].shift(1)))
        
        check_pattern(piercing_line, 'piercing_line')
        self.df['piercing_line_signal'] = 'bullish'
        
        # Dark Cloud Cover
        def dark_cloud_cover():
            return (self.df['Close'].shift(1) > self.df['Open'].shift(1)) & \
                   (self.df['Close'] < self.df['Open']) & \
                   (self.df['Open'] > self.df['High'].shift(1)) & \
                   (self.df['Close'] < self.df['Open'].shift(1) - 0.5*(self.df['Open'].shift(1) - self.df['Close'].shift(1)))
        
        check_pattern(dark_cloud_cover, 'dark_cloud_cover')
        self.df['dark_cloud_cover_signal'] = 'bearish'
        
        # Morning Star
        def morning_star():
            first_bearish = self.df['Close'].shift(2) < self.df['Open'].shift(2)
            second_small = (self.df['Close'].shift(1) - self.df['Open'].shift(1)).abs() <= 0.3 * (self.df['High'].shift(1) - self.df['Low'].shift(1))
            second_gaps_lower = self.df['High'].shift(1) < self.df['Close'].shift(2)
            third_bullish = self.df['Close'] > self.df['Open']
            third_closes_above = self.df['Close'] > self.df['Open'].shift(2) + 0.5*(self.df['Close'].shift(2) - self.df['Open'].shift(2))
            return first_bearish & second_small & second_gaps_lower & third_bullish & third_closes_above
        
        check_pattern(morning_star, 'morning_star')
        self.df['morning_star_signal'] = 'bullish'
        
        # Evening Star
        def evening_star():
            first_bullish = self.df['Close'].shift(2) > self.df['Open'].shift(2)
            second_small = (self.df['Close'].shift(1) - self.df['Open'].shift(1)).abs() <= 0.3 * (self.df['High'].shift(1) - self.df['Low'].shift(1))
            second_gaps_higher = self.df['Low'].shift(1) > self.df['Close'].shift(2)
            third_bearish = self.df['Close'] < self.df['Open']
            third_closes_below = self.df['Close'] < self.df['Open'].shift(2) - 0.5*(self.df['Open'].shift(2) - self.df['Close'].shift(2))
            return first_bullish & second_small & second_gaps_higher & third_bearish & third_closes_below
        
        check_pattern(evening_star, 'evening_star')
        self.df['evening_star_signal'] = 'bearish'
        
        # Three White Soldiers
        def three_white_soldiers():
            first_bullish = self.df['Close'].shift(2) > self.df['Open'].shift(2)
            second_bullish = self.df['Close'].shift(1) > self.df['Open'].shift(1)
            third_bullish = self.df['Close'] > self.df['Open']
            second_opens_inside = (self.df['Open'].shift(1) > self.df['Open'].shift(2)) & (self.df['Open'].shift(1) < self.df['Close'].shift(2))
            third_opens_inside = (self.df['Open'] > self.df['Open'].shift(1)) & (self.df['Open'] < self.df['Close'].shift(1))
            return first_bullish & second_bullish & third_bullish & second_opens_inside & third_opens_inside
        
        check_pattern(three_white_soldiers, 'three_white_soldiers')
        self.df['three_white_soldiers_signal'] = 'bullish'
        
        # Three Black Crows
        def three_black_crows():
            first_bearish = self.df['Close'].shift(2) < self.df['Open'].shift(2)
            second_bearish = self.df['Close'].shift(1) < self.df['Open'].shift(1)
            third_bearish = this.df['Close'] < self.df['Open']
            second_opens_inside = (self.df['Open'].shift(1) < self.df['Open'].shift(2)) & (self.df['Open'].shift(1) > self.df['Close'].shift(2))
            third_opens_inside = (self.df['Open'] < self.df['Open'].shift(1)) & (self.df['Open'] > self.df['Close'].shift(1))
            return first_bearish & second_bearish & third_bearish & second_opens_inside & third_opens_inside
        
        check_pattern(three_black_crows, 'three_black_crows')
        self.df['three_black_crows_signal'] = 'bearish'
        
        logger.info("Candlestick patterns identified")
    
    def _identify_fair_value_gaps(self):
        """Identify Fair Value Gaps (FVG)."""
        logger.info("Identifying Fair Value Gaps")
        
        scan_bars = self.periods["fvg_scan"]
        
        # Bullish FVG (current low > previous high)
        bullish_fvg = (self.df['Low'] > self.df['High'].shift(1))
        self.df['bullish_fvg'] = bullish_fvg
        self.df.loc[bullish_fvg, 'bullish_fvg_upper'] = self.df['High'].shift(1)
        self.df.loc[bullish_fvg, 'bullish_fvg_lower'] = self.df['Low']
        
        # Bearish FVG (current high < previous low)
        bearish_fvg = (self.df['High'] < self.df['Low'].shift(1))
        self.df['bearish_fvg'] = bearish_fvg
        self.df.loc[bearish_fvg, 'bearish_fvg_upper'] = self.df['High']
        self.df.loc[bearish_fvg, 'bearish_fvg_lower'] = self.df['Low'].shift(1)
        
        # Add FVG signals
        self.df['fvg_signal'] = np.where(bullish_fvg, 'bullish', 
                                        np.where(bearish_fvg, 'bearish', 'none'))
        
        logger.info("Fair Value Gaps identified")
    
    def _wma(self, series, period):
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    def get_latest_signals(self) -> Dict:
        """
        Get the latest signals from all indicators.
        
        Returns:
            Dictionary with the latest signals
        """
        latest = self.df.iloc[-1]
        
        signals = {
            'trend': {
                'sma': latest['sma_signal'],
                'ema': latest['ema_signal'],
                'hma': latest['hma_signal'],
                'macd': 'bullish' if latest['macd_signal_bullish'] else 'bearish',
                'vwap': latest['vwap_signal']
            },
            'momentum': {
                'rsi': latest['rsi_signal'],
                'stochastic': latest['stoch_signal'],
                'cci': latest['cci_signal'],
                'roc': latest['roc_signal']
            },
            'volatility': {
                'bollinger': latest['bb_signal'],
                'atr': latest['atr_signal'],
                'keltner': latest['keltner_signal'],
                'donchian': latest['donchian_signal']
            },
            'volume': {
                'obv': latest['obv_signal'],
                'mfi': latest['mfi_signal']
            },
            'trend_strength': {
                'adx': latest['adx_signal'],
                'adx_direction': latest['adx_direction'],
                'ichimoku': latest['ichimoku_signal']
            }
        }
        
        # Add candlestick patterns
        pattern_columns = [col for col in self.df.columns if col.endswith('_signal') and not col.endswith('_fvg_signal')]
        signals['patterns'] = {col.replace('_signal', ''): latest[col] for col in pattern_columns if latest[col] != 'none'}
        
        # Add FVG
        if latest['fvg_signal'] != 'none':
            signals['fvg'] = {
                'type': latest['fvg_signal'],
                'upper': latest.get('bullish_fvg_upper' if latest['fvg_signal'] == 'bullish' else 'bearish_fvg_upper'),
                'lower': latest.get('bullish_fvg_lower' if latest['fvg_signal'] == 'bullish' else 'bearish_fvg_lower')
            }
        
        return signals


# Unit tests
def test_technical_analysis():
    """Run unit tests on the technical analysis module."""
    # Create a synthetic DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure High is the highest and Low is the lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Initialize the technical analysis
    ta = AdvancedTechnicalAnalysis(df, timeframe="1min")
    
    # Calculate all indicators
    result_df = ta.calculate_all_indicators()
    
    # Test that the DataFrame has the expected columns
    expected_columns = [
        'sma_30', 'ema_30', 'hma_30', 'macd', 'macd_signal', 'macd_hist', 'vwap',
        'rsi_14', 'stoch_k', 'stoch_d', 'cci_20', 'roc_12',
        'bb_mid', 'bb_upper', 'bb_lower', 'atr_14', 'kc_mid', 'kc_upper', 'kc_lower',
        'dch_upper', 'dch_lower', 'obv', 'mfi_14',
        'adx_14', 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'
    ]
    
    for col in expected_columns:
        assert col in result_df.columns, f"Missing column: {col}"
    
    # Test that signals are generated
    signal_columns = [col for col in result_df.columns if col.endswith('_signal')]
    assert len(signal_columns) > 0, "No signal columns found"
    
    # Test getting latest signals
    signals = ta.get_latest_signals()
    assert 'trend' in signals, "Missing trend signals"
    assert 'momentum' in signals, "Missing momentum signals"
    assert 'volatility' in signals, "Missing volatility signals"
    assert 'volume' in signals, "Missing volume signals"
    assert 'trend_strength' in signals, "Missing trend strength signals"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_technical_analysis() 
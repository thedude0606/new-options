import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import talib
import logging
import os
import sys

class TechnicalAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        Expected columns: ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        # Initialize logger
        self.logger = logging.getLogger('technical_analysis')
        self.logger.setLevel(logging.DEBUG)
        
        # Add file handler if not already added
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                self.logger.info(f"Created logs directory at {os.path.abspath(log_dir)}")
            
            # Add file handler
            log_file = os.path.join(log_dir, 'technical_analysis.log')
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                # Also add a console handler for immediate feedback
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                
                self.logger.info(f"Logging initialized. Writing to {os.path.abspath(log_file)}")
            except Exception as e:
                print(f"Error setting up logging: {str(e)}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Log directory exists: {os.path.exists(log_dir)}")
                print(f"Log directory permissions: {oct(os.stat(log_dir).st_mode)[-3:] if os.path.exists(log_dir) else 'N/A'}")
        
        self.logger.debug("Initializing TechnicalAnalysis")
        self.df = df.copy()
        self.results = {}
        
        # Validate DataFrame
        required_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.debug(f"Available columns: {self.df.columns.tolist()}")
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
            
        self.logger.debug(f"DataFrame shape: {self.df.shape}")
        self.logger.debug(f"DataFrame head:\n{self.df.head()}")
        
    def aggregate_timeframe(self, minutes: int) -> pd.DataFrame:
        """Aggregate data to a larger timeframe."""
        self.logger.debug(f"Aggregating timeframe to {minutes} minutes")
        
        if minutes == 1:
            self.logger.debug("Using original 1-minute data")
            return self.df
            
        try:
            df = self.df.copy()
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
            resampled = df.resample(f'{minutes}T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            result = resampled.reset_index()
            self.logger.debug(f"Aggregated shape: {result.shape}")
            self.logger.debug(f"Aggregated head:\n{result.head()}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in aggregating timeframe: {str(e)}", exc_info=True)
            raise

    def calculate_all_indicators(self, timeframe_minutes: int = 1) -> Dict:
        """Calculate all technical indicators for a given timeframe."""
        self.logger.info(f"Calculating all indicators for {timeframe_minutes} minute timeframe")
        
        try:
            df = self.aggregate_timeframe(timeframe_minutes)
            if len(df) < 200:  # Minimum data points needed for most indicators
                self.logger.warning(f"Insufficient data points ({len(df)}) for reliable indicators")
            
            results = {
                'trend_indicators': self._calculate_trend_indicators(df),
                'momentum_indicators': self._calculate_momentum_indicators(df),
                'volatility_indicators': self._calculate_volatility_indicators(df),
                'volume_indicators': self._calculate_volume_indicators(df),
                'candlestick_patterns': self._identify_candlestick_patterns(df),
                'fair_value_gaps': self._identify_fair_value_gaps(df)
            }
            
            self.logger.debug(f"Calculation results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            raise

    def calculate_multi_timeframe_analysis(self) -> Dict:
        """Calculate indicators for all timeframes (1min, 15min, 1hr, daily)."""
        self.logger.info("Calculating multi-timeframe analysis")
        
        try:
            # Calculate for 1-minute timeframe (original data)
            one_min = self.calculate_all_indicators(1)
            
            # Calculate for 15-minute timeframe
            fifteen_min = self.calculate_all_indicators(15)
            
            # Calculate for 1-hour timeframe
            one_hour = self.calculate_all_indicators(60)
            
            # Calculate for daily timeframe
            daily = self.calculate_all_indicators(1440)  # 24 hours * 60 minutes
            
            return {
                '1min': one_min,
                '15min': fifteen_min,
                '1hr': one_hour,
                'daily': daily
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}", exc_info=True)
            raise

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate trend indicators."""
        self.logger.debug("Calculating trend indicators")
        
        # Convert to numpy arrays for TA-Lib
        high = df['High'].astype('float64').values
        low = df['Low'].astype('float64').values
        close = df['Close'].astype('float64').values
        volume = df['Volume'].astype('float64').values
        
        # Calculate SMA (20, 50, 200)
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)
        
        # Calculate EMA (12, 26)
        ema_12 = talib.EMA(close, timeperiod=12)
        ema_26 = talib.EMA(close, timeperiod=26)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Calculate Hull Moving Average (HMA)
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = talib.WMA(close, timeperiod=10)  # n/2 = 20/2 = 10
        wma_full = talib.WMA(close, timeperiod=20)
        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, timeperiod=int(np.sqrt(20)))
        
        # Calculate VWAP manually
        typical_price = (high + low + close) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        
        # Calculate Ichimoku Cloud
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        
        # Calculate 9-period high and low
        high_9 = pd.Series(high).rolling(window=9).max().values
        low_9 = pd.Series(low).rolling(window=9).min().values
        
        # Calculate 26-period high and low
        high_26 = pd.Series(high).rolling(window=26).max().values
        low_26 = pd.Series(low).rolling(window=26).min().values
        
        # Calculate 52-period high and low
        high_52 = pd.Series(high).rolling(window=52).max().values
        low_52 = pd.Series(low).rolling(window=52).min().values
        
        # Calculate Ichimoku components
        tenkan = (high_9 + low_9) / 2
        kijun = (high_26 + low_26) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high_52 + low_52) / 2
        
        # Shift Senkou Span A and B forward 26 periods
        senkou_a_shifted = np.roll(senkou_a, -26)
        senkou_b_shifted = np.roll(senkou_b, -26)
        
        # Chikou Span is just the close price shifted back 26 periods
        chikou = np.roll(close, 26)
        
        # Determine cloud color (green if Senkou Span A > Senkou Span B, red otherwise)
        cloud_color = np.where(senkou_a > senkou_b, "green", "red")
        
        # Determine trend based on price position relative to cloud
        price_above_cloud = close > np.maximum(senkou_a, senkou_b)
        price_below_cloud = close < np.minimum(senkou_a, senkou_b)
        
        # Determine trend based on Tenkan crossing Kijun
        tenkan_above_kijun = tenkan > kijun
        tenkan_below_kijun = tenkan < kijun
        
        # Log the calculated values
        self.logger.debug(f"SMA(20): {sma_20[-1]:.2f}, SMA(50): {sma_50[-1]:.2f}, SMA(200): {sma_200[-1]:.2f}")
        self.logger.debug(f"EMA(12): {ema_12[-1]:.2f}, EMA(26): {ema_26[-1]:.2f}")
        self.logger.debug(f"MACD: {macd[-1]:.2f}, Signal: {macd_signal[-1]:.2f}, Histogram: {macd_hist[-1]:.2f}")
        self.logger.debug(f"HMA: {hma[-1]:.2f}")
        self.logger.debug(f"VWAP: {vwap[-1]:.2f}")
        self.logger.debug(f"Ichimoku - Tenkan: {tenkan[-1]:.2f}, Kijun: {kijun[-1]:.2f}")
        self.logger.debug(f"Ichimoku - Senkou A: {senkou_a[-1]:.2f}, Senkou B: {senkou_b[-1]:.2f}")
        self.logger.debug(f"Ichimoku - Cloud Color: {cloud_color[-1]}")
        
        return {
            'sma': {
                'sma20': {'value': sma_20[-1], 'signal': 'bullish' if close[-1] > sma_20[-1] else 'bearish'},
                'sma50': {'value': sma_50[-1], 'signal': 'bullish' if close[-1] > sma_50[-1] else 'bearish'},
                'sma200': {'value': sma_200[-1], 'signal': 'bullish' if close[-1] > sma_200[-1] else 'bearish'}
            },
            'ema': {
                'ema12': {'value': ema_12[-1], 'signal': 'bullish' if close[-1] > ema_12[-1] else 'bearish'},
                'ema26': {'value': ema_26[-1], 'signal': 'bullish' if close[-1] > ema_26[-1] else 'bearish'}
            },
            'macd': {
                'value': macd[-1],
                'signal': macd_signal[-1],
                'histogram': macd_hist[-1],
                'signal_type': 'bullish' if macd[-1] > macd_signal[-1] else 'bearish'
            },
            'hma': {
                'value': hma[-1],
                'signal': 'bullish' if close[-1] > hma[-1] and hma[-1] > hma[-2] else 'bearish' if close[-1] < hma[-1] and hma[-1] < hma[-2] else 'neutral'
            },
            'vwap': {
                'value': vwap[-1],
                'signal': 'bullish' if close[-1] > vwap[-1] else 'bearish'
            },
            'ichimoku': {
                'tenkan': tenkan[-1],
                'kijun': kijun[-1],
                'senkou_a': senkou_a[-1],
                'senkou_b': senkou_b[-1],
                'chikou': chikou[-26] if len(chikou) > 26 else None,
                'cloud_color': cloud_color[-1],
                'price_above_cloud': price_above_cloud[-1],
                'price_below_cloud': price_below_cloud[-1],
                'tenkan_above_kijun': tenkan_above_kijun[-1],
                'tenkan_below_kijun': tenkan_below_kijun[-1],
                'signal': 'bullish' if price_above_cloud[-1] and tenkan_above_kijun[-1] else 'bearish' if price_below_cloud[-1] and tenkan_below_kijun[-1] else 'neutral'
            }
        }

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators."""
        self.logger.debug("Calculating momentum indicators")
        
        # Convert to float64 for TA-Lib compatibility
        close = df['Close'].astype('float64').values
        high = df['High'].astype('float64').values
        low = df['Low'].astype('float64').values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, 
                                 fastk_period=14, 
                                 slowk_period=3, 
                                 slowk_matype=0, 
                                 slowd_period=3, 
                                 slowd_matype=0)
        
        # CCI
        cci = talib.CCI(high, low, close, timeperiod=20)
        
        # ROC (Rate of Change)
        roc = talib.ROC(close, timeperiod=10)
        
        # ADX (Average Directional Index)
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        self.logger.debug(f"RSI: {rsi[-1]:.2f}")
        self.logger.debug(f"Stochastic K: {slowk[-1]:.2f}, D: {slowd[-1]:.2f}")
        self.logger.debug(f"CCI: {cci[-1]:.2f}")
        self.logger.debug(f"ROC: {roc[-1]:.2f}")
        self.logger.debug(f"ADX: {adx[-1]:.2f}, +DI: {plus_di[-1]:.2f}, -DI: {minus_di[-1]:.2f}")
        
        return {
            'rsi': {
                'value': rsi[-1],
                'signal': 'overbought' if rsi[-1] > 70 else 'oversold' if rsi[-1] < 30 else 'neutral'
            },
            'stochastic': {
                'k': slowk[-1],
                'd': slowd[-1],
                'signal': 'overbought' if slowk[-1] > 80 and slowd[-1] > 80 else 
                         'oversold' if slowk[-1] < 20 and slowd[-1] < 20 else 'neutral'
            },
            'cci': {
                'value': cci[-1],
                'signal': 'overbought' if cci[-1] > 100 else 'oversold' if cci[-1] < -100 else 'neutral'
            },
            'roc': {
                'value': roc[-1],
                'signal': 'bullish' if roc[-1] > 0 and roc[-1] > roc[-2] else 'bearish' if roc[-1] < 0 and roc[-1] < roc[-2] else 'neutral'
            },
            'adx': {
                'value': adx[-1],
                'plus_di': plus_di[-1],
                'minus_di': minus_di[-1],
                'signal': 'strong_trend' if adx[-1] > 25 else 'weak_trend',
                'trend_direction': 'bullish' if plus_di[-1] > minus_di[-1] else 'bearish'
            }
        }

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility indicators."""
        self.logger.debug("Calculating volatility indicators")
        
        # Convert to float64 for TA-Lib compatibility
        high = df['High'].astype('float64').values
        low = df['Low'].astype('float64').values
        close = df['Close'].astype('float64').values
        
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, 
                                         timeperiod=20,
                                         nbdevup=2,
                                         nbdevdn=2,
                                         matype=0)
        
        # Standard Deviation
        stddev = talib.STDDEV(close, timeperiod=20, nbdev=2)
        
        # Keltner Channels
        # Middle = EMA(20)
        # Upper = Middle + 2 * ATR(10)
        # Lower = Middle - 2 * ATR(10)
        ema_20 = talib.EMA(close, timeperiod=20)
        atr_10 = talib.ATR(high, low, close, timeperiod=10)
        keltner_upper = ema_20 + 2 * atr_10
        keltner_lower = ema_20 - 2 * atr_10
        
        # Donchian Channel
        # Upper = Highest High(20)
        # Lower = Lowest Low(20)
        donchian_upper = pd.Series(high).rolling(window=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20).min().values
        
        self.logger.debug(f"ATR: {atr[-1]:.2f}")
        self.logger.debug(f"Bollinger Bands - Upper: {upper[-1]:.2f}, Middle: {middle[-1]:.2f}, Lower: {lower[-1]:.2f}")
        self.logger.debug(f"Standard Deviation: {stddev[-1]:.2f}")
        self.logger.debug(f"Keltner Channels - Upper: {keltner_upper[-1]:.2f}, Middle: {ema_20[-1]:.2f}, Lower: {keltner_lower[-1]:.2f}")
        self.logger.debug(f"Donchian Channel - Upper: {donchian_upper[-1]:.2f}, Lower: {donchian_lower[-1]:.2f}")
        
        return {
            'atr': {
                'value': atr[-1],
                'signal': 'high_volatility' if atr[-1] > atr[-2] else 'low_volatility'
            },
            'bollinger_bands': {
                'upper': upper[-1],
                'middle': middle[-1],
                'lower': lower[-1],
                'signal': 'overbought' if close[-1] > upper[-1] else 
                         'oversold' if close[-1] < lower[-1] else 'neutral'
            },
            'stddev': {
                'value': stddev[-1],
                'signal': 'high_volatility' if stddev[-1] > stddev[-2] else 'low_volatility'
            },
            'keltner_channels': {
                'upper': keltner_upper[-1],
                'middle': ema_20[-1],
                'lower': keltner_lower[-1],
                'signal': 'bullish' if close[-1] > keltner_upper[-1] else 
                         'bearish' if close[-1] < keltner_lower[-1] else 'neutral'
            },
            'donchian_channel': {
                'upper': donchian_upper[-1],
                'lower': donchian_lower[-1],
                'signal': 'bullish' if close[-1] > donchian_upper[-1] else 
                         'bearish' if close[-1] < donchian_lower[-1] else 'neutral'
            }
        }

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume indicators."""
        self.logger.debug("Calculating volume indicators")
        
        # Convert to float64 for TA-Lib compatibility
        high = df['High'].astype('float64').values
        low = df['Low'].astype('float64').values
        close = df['Close'].astype('float64').values
        volume = df['Volume'].astype('float64').values
        
        # OBV (On Balance Volume)
        obv = talib.OBV(close, volume)
        
        # AD (Accumulation/Distribution)
        ad = talib.AD(high, low, close, volume)
        
        # MFI (Money Flow Index)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        
        self.logger.debug(f"OBV: {obv[-1]:.2f}")
        self.logger.debug(f"AD: {ad[-1]:.2f}")
        self.logger.debug(f"MFI: {mfi[-1]:.2f}")
        
        return {
            'obv': {
                'value': obv[-1],
                'signal': 'bullish' if obv[-1] > obv[-2] else 'bearish'
            },
            'ad': {
                'value': ad[-1],
                'signal': 'bullish' if ad[-1] > ad[-2] else 'bearish'
            },
            'mfi': {
                'value': mfi[-1],
                'signal': 'overbought' if mfi[-1] > 80 else 
                         'oversold' if mfi[-1] < 20 else 'neutral'
            }
        }

    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify candlestick patterns."""
        self.logger.debug("Identifying candlestick patterns")
        
        open = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        patterns = {
            'single_candle': {},
            'two_candle': {},
            'three_candle': {}
        }
        
        # Single candle patterns
        patterns['single_candle'].update({
            'doji': talib.CDLDOJI(open, high, low, close)[-1],
            'hammer': talib.CDLHAMMER(open, high, low, close)[-1],
            'shooting_star': talib.CDLSHOOTINGSTAR(open, high, low, close)[-1],
            'hanging_man': talib.CDLHANGINGMAN(open, high, low, close)[-1],
            'inverted_hammer': talib.CDLINVERTEDHAMMER(open, high, low, close)[-1],
            'spinning_top': talib.CDLSPINNINGTOP(open, high, low, close)[-1],
            'marubozu': talib.CDLMARUBOZU(open, high, low, close)[-1]
        })
        
        # Two candle patterns
        patterns['two_candle'].update({
            'engulfing': talib.CDLENGULFING(open, high, low, close)[-1],
            'harami': talib.CDLHARAMI(open, high, low, close)[-1],
            'piercing_line': talib.CDLPIERCING(open, high, low, close)[-1],
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER(open, high, low, close)[-1]
        })
        
        # Three candle patterns
        patterns['three_candle'].update({
            'morning_star': talib.CDLMORNINGSTAR(open, high, low, close)[-1],
            'evening_star': talib.CDLEVENINGSTAR(open, high, low, close)[-1],
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(open, high, low, close)[-1],
            'three_black_crows': talib.CDL3BLACKCROWS(open, high, low, close)[-1],
            'three_inside_up': talib.CDL3INSIDE(open, high, low, close)[-1],
            'three_inside_down': talib.CDL3INSIDE(open, high, low, close)[-1],
            'three_outside_up': talib.CDL3OUTSIDE(open, high, low, close)[-1],
            'three_outside_down': talib.CDL3OUTSIDE(open, high, low, close)[-1]
        })
        
        # Filter out patterns that weren't detected (value == 0)
        for category in patterns:
            patterns[category] = {k: v for k, v in patterns[category].items() if v != 0}
        
        self.logger.debug(f"Detected patterns: {patterns}")
        return patterns
        
    def _identify_fair_value_gaps(self, df: pd.DataFrame) -> Dict:
        """Identify Fair Value Gaps (FVG)."""
        self.logger.debug("Identifying Fair Value Gaps")
        
        high = df['High'].values
        low = df['Low'].values
        
        fvgs = []
        
        # Look for bullish FVG (current low > previous high)
        for i in range(1, len(df)):
            if low[i] > high[i-1]:
                fvgs.append({
                    'type': 'bullish',
                    'upper': high[i-1],
                    'lower': low[i],
                    'index': i,
                    'time': df['Time'].iloc[i]
                })
        
        # Look for bearish FVG (current high < previous low)
        for i in range(1, len(df)):
            if high[i] < low[i-1]:
                fvgs.append({
                    'type': 'bearish',
                    'upper': high[i],
                    'lower': low[i-1],
                    'index': i,
                    'time': df['Time'].iloc[i]
                })
        
        # Get the most recent FVG
        most_recent_fvg = fvgs[-1] if fvgs else None
        
        self.logger.debug(f"Found {len(fvgs)} Fair Value Gaps")
        if most_recent_fvg:
            self.logger.debug(f"Most recent FVG: {most_recent_fvg}")
        
        return {
            'count': len(fvgs),
            'most_recent': most_recent_fvg,
            'all_gaps': fvgs
        } 
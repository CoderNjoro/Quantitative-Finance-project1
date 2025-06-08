import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
import yfinance as yf
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import logging
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr
import plotly.figure_factory as ff

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(page_title="Trading System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #1e1e1e; color: #ffffff;}
    .sidebar .sidebar-content {background-color: #2c2c2c;}
    h1, h2, h3 {color: #00ff88;}
    .stButton>button {background-color: #00ff88; color: #1e1e1e;}
    </style>
""", unsafe_allow_html=True)

class ICTQuantSystem:
    """Enhanced Inner Circle Trader Quantitative Trading System with Machine Learning"""
    def __init__(self, symbol='EURUSD', api_key=None, outputsize='full', timeframe='1d'):
        self.symbol = symbol
        self.api_key = api_key if api_key else os.getenv('ALPHA_VANTAGE_API_KEY', "ENTER YOUR ALPHA_VANTAGE_API_KEY")
        self.outputsize = outputsize
        self.timeframe = timeframe
        self.data = None
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_data_alphavantage(self):
        """Fetch forex data from Alpha Vantage"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                base_url = "https://www.alphavantage.co/query"
                if len(self.symbol) == 6:
                    from_symbol = self.symbol[:3]
                    to_symbol = self.symbol[3:]
                else:
                    raise ValueError(f"Invalid symbol format: {self.symbol}. Expected format: XXXYYY (e.g., EURUSD)")

                function = 'FX_DAILY' if self.timeframe == '1d' else 'FX_INTRADAY'
                interval = '60min' if self.timeframe == '1h' else '240min' if self.timeframe == '4h' else '1day'

                params = {
                    'function': function,
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'outputsize': self.outputsize,
                    'apikey': self.api_key
                }
                if function == 'FX_INTRADAY':
                    params['interval'] = interval

                logger.debug(f"Fetching {from_symbol}/{to_symbol} data from Alpha Vantage (attempt {attempt + 1}/{max_retries})")
                st.info(f"Fetching {from_symbol}/{to_symbol} data from Alpha Vantage...")
                response = requests.get(base_url, params=params)

                if response.status_code != 200:
                    raise Exception(f"API request failed with status code {response.status_code}")

                data = response.json()

                if 'Error Message' in data:
                    raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
                if 'Note' in data:
                    raise Exception(f"Alpha Vantage Note: {data['Note']}")
                if 'Information' in data:
                    raise Exception(f"Alpha Vantage Error: {data['Information']}")

                time_series_key = f"Time Series FX ({'Daily' if self.timeframe == '1d' else interval})"
                if time_series_key not in data:
                    raise Exception(f"Expected key '{time_series_key}' not found. Available keys: {list(data.keys())}")

                time_series = data[time_series_key]
                df_data = []
                for date_str, values in time_series.items():
                    df_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close'])
                    })

                self.data = pd.DataFrame(df_data)
                self.data.set_index('Date', inplace=True)
                self.data.sort_index(inplace=True)
                self.data['Volume'] = self._simulate_forex_volume()

                if self.data.empty:
                    raise ValueError(f"No data found for symbol {from_symbol}/{to_symbol}")

                logger.debug(f"Successfully fetched: {len(self.data)} records for {from_symbol}/{to_symbol}")
                st.success(f"✓ Data fetched successfully for {from_symbol}/{to_symbol}: {len(self.data)} records")
                return True

            except Exception as e:
                logger.error(f"Failed to fetch {from_symbol}/{to_symbol}: {str(e)}")
                if "rate limit" in str(e).lower() or "Note" in str(e):
                    st.warning(f"Alpha Vantage rate limit hit, retrying in 70 seconds... ({attempt+1}/{max_retries})")
                    time.sleep(70)
                else:
                    st.error(f"Alpha Vantage error: {str(e)}")
                    return self.fetch_data_yfinance_fallback()

        logger.error(f"Max retries reached for {from_symbol}/{to_symbol}, falling back to Yahoo Finance")
        return self.fetch_data_yfinance_fallback()

    def _simulate_forex_volume(self):
        """Simulate forex volume"""
        if self.data is None:
            logger.error("Cannot simulate volume: self.data is None")
            return None
        df = self.data.copy()
        price_range = df['High'] - df['Low']
        price_change = abs(df['Close'] - df['Open'])
        range_norm = (price_range - price_range.min()) / (price_range.max() - price_range.min() + 1e-8)
        change_norm = (price_change - price_change.min()) / (price_change.max() - price_change.min() + 1e-8)
        base_volume = (range_norm + change_norm) * 1000000
        day_multipliers = {0: 1.2, 1: 1.3, 2: 1.4, 3: 1.3, 4: 1.1, 5: 0.3, 6: 0.1}
        day_factors = [day_multipliers.get(d.weekday(), 1.0) for d in df.index]
        np.random.seed(42)
        noise = np.random.normal(1, 0.2, len(df))
        noise = np.clip(noise, 0.5, 2.0)
        simulated_volume = base_volume * day_factors * noise
        simulated_volume = np.maximum(simulated_volume, 100000)
        return simulated_volume.astype(int)

    def fetch_data_yfinance_fallback(self):
        """Fallback to Yahoo Finance"""
        try:
            yf_symbol = f"{self.symbol}=X" if len(self.symbol) == 6 else self.symbol
            logger.debug(f"Fetching data from Yahoo Finance: {yf_symbol}")
            st.info(f"Fetching data from Yahoo Finance: {yf_symbol}")
            interval = '1d' if self.timeframe == '1d' else '1h' if self.timeframe == '1h' else '4h'
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(yf_symbol)
                    self.data = ticker.history(period='2y' if self.timeframe == '1d' else '60d', interval=interval)
                    if self.data.empty:
                        raise ValueError(f"No data found for symbol {yf_symbol}")
                    self.data = self.data.rename(columns={
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    })
                    self.data.index = pd.to_datetime(self.data.index)
                    self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    logger.debug(f"Successfully fetched {len(self.data)} records from Yahoo Finance")
                    st.success(f"✓ Fallback successful: {len(self.data)} records for {yf_symbol}")
                    return True
                except Exception as e:
                    logger.error(f"Yahoo Finance error: {str(e)}")
                    if "Too Many Requests" in str(e) or "429" in str(e):
                        st.warning(f"Yahoo Finance rate limit hit, retrying in 70 seconds... ({attempt+1}/{max_retries})")
                        time.sleep(70)
                    else:
                        raise e
            logger.error(f"Failed to fetch Yahoo Finance data after {max_retries} retries")
            st.error(f"Failed to fetch Yahoo Finance data for {yf_symbol} after {max_retries} retries")
            return False
        except Exception as e:
            logger.error(f"Yahoo Finance fallback failed: {str(e)}")
            st.error(f"Yahoo Finance fallback failed: {str(e)}")
            return False

    def fetch_multiple_pairs_alphavantage(self, pairs_list):
        """Fetch multiple currency pairs"""
        all_data = {}
        for i, pair in enumerate(pairs_list):
            logger.debug(f"Fetching {pair} ({i+1}/{len(pairs_list)})")
            st.info(f"Fetching {pair} ({i+1}/{len(pairs_list)})...")
            temp_system = ICTQuantSystem(symbol=pair, api_key=self.api_key, outputsize='compact', timeframe=self.timeframe)
            if temp_system.fetch_data_alphavantage():
                all_data[pair] = temp_system.data
                st.success(f"✓ {pair} data loaded")
            else:
                logger.error(f"Failed to load data for {pair}")
                st.error(f"✗ Failed to load {pair}")
            if i < len(pairs_list) - 1:
                st.info("Waiting 15 seconds for API rate limit...")
                time.sleep(15)
        return all_data

    def fetch_sentiment_data(self):
        """Fetch sentiment from X posts"""
        try:
            api_key = os.getenv('TWITTER_API_KEY', '')
            api_secret = os.getenv('TWITTER_API_SECRET', '')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN', '')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')

            if not all([api_key, api_secret, access_token, access_token_secret]):
                logger.warning("Twitter API credentials missing, skipping sentiment analysis")
                st.warning("Twitter API credentials missing, sentiment score set to 0")
                return 0.0

            auth = tweepy.OAuthHandler(api_key, api_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth, wait_on_rate_limit=True)

            query = f"{self.symbol} forex"
            tweets = api.search_tweets(q=query, lang='en', count=100)
            sentiment_scores = []
            for tweet in tweets:
                score = self.sentiment_analyzer.polarity_scores(tweet.text)
                sentiment_scores.append(score['compound'])

            return np.mean(sentiment_scores) if sentiment_scores else 0.0
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            st.warning(f"Sentiment analysis failed: {str(e)}, score set to 0")
            return 0.0

    def fetch_economic_calendar(self):
        """Fetch high-impact economic events"""
        try:
            url = "https://api.tradingeconomics.com/calendar"
            params = {'key': os.getenv('TRADINGECONOMICS_API_KEY', ''), 'importance': '3'}
            response = requests.get(url, params=params)
            if response.status_code == 200:
                events = response.json()
                return pd.DataFrame(events)
            else:
                logger.error(f"Economic calendar API failed: {response.status_code}")
                st.warning("Economic calendar API failed")
                return None
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {str(e)}")
            st.warning(f"Economic calendar fetch failed: {str(e)}")
            return None

    def fetch_data(self):
        """Main data fetching method"""
        success = self.fetch_data_alphavantage()
        if not success or self.data is None:
            logger.error(f"Data fetching failed for {self.symbol}")
            st.error(f"Data fetching failed for {self.symbol}")
            return False
        return True

    def identify_fibonacci_levels(self):
        """Identify Fibonacci retracement levels"""
        if self.data is None:
            logger.error("Cannot identify Fibonacci levels: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        swing_highs = df[df['High'] == df['High'].rolling(5, center=True).max()]['High']
        swing_lows = df[df['Low'] == df['Low'].rolling(5, center=True).min()]['Low']
        latest_high = swing_highs.iloc[-1] if not swing_highs.empty else df['High'].max()
        latest_low = swing_lows.iloc[-1] if not swing_lows.empty else df['Low'].min()
        fib_range = latest_high - latest_low
        fib_levels = {
            '0.0%': latest_high,
            '23.6%': latest_high - fib_range * 0.236,
            '38.2%': latest_high - fib_range * 0.382,
            '50.0%': latest_high - fib_range * 0.50,
            '61.8%': latest_high - fib_range * 0.618,
            '100.0%': latest_low
        }
        df['Fib_38_2'] = fib_levels['38.2%']
        df['Fib_50_0'] = fib_levels['50.0%']
        df['Fib_61_8'] = fib_levels['61.8%']
        self.data = df
        return fib_levels

    def identify_market_structure(self):
        """Identify market structure"""
        if self.data is None:
            logger.error("Cannot identify market structure: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['SwingHigh'] = df['High'].rolling(window=3, center=True).max() == df['High']
        df['SwingLow'] = df['Low'].rolling(window=3, center=True).min() == df['Low']
        swing_highs = df[df['SwingHigh']]['High']
        swing_lows = df[df['SwingLow']]['Low']
        df['MarketStructure'] = 0
        for i in range(2, len(df)):
            recent_highs = swing_highs[swing_highs.index <= df.index[i]].tail(2)
            recent_lows = swing_lows[swing_lows.index <= df.index[i]].tail(2)
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if recent_highs.iloc[-1] > recent_highs.iloc[-2] and recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                    df.iloc[i, df.columns.get_loc('MarketStructure')] = 1
                elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                    df.iloc[i, df.columns.get_loc('MarketStructure')] = -1
        self.data = df
        return df

    def identify_fair_value_gaps(self):
        """Identify Fair Value Gaps"""
        if self.data is None:
            logger.error("Cannot identify fair value gaps: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['BullishFVG'] = (df['Low'] > df['High'].shift(2)) & (df['Low'].shift(1) > df['High'].shift(2))
        df['BearishFVG'] = (df['High'] < df['Low'].shift(2)) & (df['High'].shift(1) < df['Low'].shift(2))
        df['FVG_Strength'] = 0
        df.loc[df['BullishFVG'], 'FVG_Strength'] = (df['Low'] - df['High'].shift(2)) / df['Close']
        df.loc[df['BearishFVG'], 'FVG_Strength'] = (df['Low'].shift(2) - df['High']) / df['Close']
        self.data = df
        return df

    def identify_order_blocks(self):
        """Identify Order Blocks"""
        if self.data is None:
            logger.error("Cannot identify order blocks: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['BullishOB'] = (
            (df['Volume'] > df['Volume_MA'] * 1.5) &
            (df['Price_Change'] > df['Price_Change'].rolling(20).std() * 2) &
            (df['Close'] > df['Open'])
        )
        df['BearishOB'] = (
            (df['Volume'] > df['Volume_MA'] * 1.5) &
            (df['Price_Change'] < -df['Price_Change'].rolling(20).std() * 2) &
            (df['Close'] < df['Open'])
        )
        self.data = df
        return df

    def identify_liquidity_sweeps(self):
        """Identify liquidity sweeps"""
        if self.data is None:
            logger.error("Cannot identify liquidity sweeps: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['Recent_High'] = df['High'].rolling(window=20).max()
        df['Recent_Low'] = df['Low'].rolling(window=20).min()
        df['Liquidity_Sweep_High'] = (
            (df['High'] > df['Recent_High'].shift(1)) &
            (df['Close'] < df['Recent_High'].shift(1))
        )
        df['Liquidity_Sweep_Low'] = (
            (df['Low'] < df['Recent_Low'].shift(1)) &
            (df['Close'] > df['Recent_Low'].shift(1))
        )
        self.data = df
        return df

    def identify_killzones(self):
        """Identify trading session killzones"""
        if self.data is None:
            logger.error("Cannot identify killzones: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['DayOfWeek'] = df.index.dayofweek
        df['London_Session_Day'] = df['DayOfWeek'].isin([0, 1, 2])
        df['NY_Session_Day'] = df['DayOfWeek'].isin([1, 2, 3])
        df['Overlap_Day'] = df['DayOfWeek'].isin([1, 2])
        df['End_Week'] = df['DayOfWeek'].isin([3, 4])
        df['Low_Activity'] = df['DayOfWeek'].isin([5, 6])
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['High_Volume_Day'] = df['Volume'] > volume_ma * 1.2
        df['Low_Volume_Day'] = df['Volume'] < volume_ma * 0.8
        self.data = df
        return df

    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            logger.error("Cannot calculate technical indicators: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        self.data = df
        return df

    def create_features(self):
        """Create feature matrix"""
        if self.data is None:
            logger.error("Cannot create features: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        self.identify_fibonacci_levels()
        self.identify_market_structure()
        self.identify_fair_value_gaps()
        self.identify_order_blocks()
        self.identify_liquidity_sweeps()
        self.identify_killzones()
        self.calculate_technical_indicators()
        sentiment_score = self.fetch_sentiment_data()
        df = self.data.copy()
        features = pd.DataFrame(index=df.index)
        features['MarketStructure'] = df['MarketStructure']
        features['BullishFVG'] = df['BullishFVG'].astype(int)
        features['BearishFVG'] = df['BearishFVG'].astype(int)
        features['FVG_Strength'] = df['FVG_Strength'].fillna(0)
        features['BullishOB'] = df['BullishOB'].astype(int)
        features['BearishOB'] = df['BearishOB'].astype(int)
        features['Liquidity_Sweep_High'] = df['Liquidity_Sweep_High'].astype(int)
        features['Liquidity_Sweep_Low'] = df['Liquidity_Sweep_Low'].astype(int)
        features['London_Session_Day'] = df['London_Session_Day'].astype(int)
        features['NY_Session_Day'] = df['NY_Session_Day'].astype(int)
        features['Overlap_Day'] = df['Overlap_Day'].astype(int)
        features['End_Week'] = df['End_Week'].astype(int)
        features['High_Volume_Day'] = df['High_Volume_Day'].astype(int)
        features['Fib_38_2_Proximity'] = abs(df['Close'] - df['Fib_38_2']) / df['ATR']
        features['Fib_50_0_Proximity'] = abs(df['Close'] - df['Fib_50_0']) / df['ATR']
        features['Fib_61_8_Proximity'] = abs(df['Close'] - df['Fib_61_8']) / df['ATR']
        features['Sentiment_Score'] = sentiment_score
        features['RSI'] = df['RSI']
        features['MACD'] = df['MACD']
        features['MACD_Histogram'] = df['MACD_Histogram']
        features['BB_Position'] = df['BB_Position']
        features['ATR_Normalized'] = df['ATR'] / df['Close']
        features['Price_Change'] = df['Close'].pct_change()
        features['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        features['Open_Close_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        for lag in [1, 2, 3, 5]:
            features[f'Price_Change_Lag_{lag}'] = features['Price_Change'].shift(lag)
            features[f'RSI_Lag_{lag}'] = features['RSI'].shift(lag)
        features['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        features = features.dropna()
        self.features = features
        return features

    def train_model(self, test_size=0.2):
        """Train ML model"""
        if self.data is None:
            logger.error("Cannot train model: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        with st.spinner("Training model..."):
            if self.features is None:
                self.create_features()
            X = self.features.drop('Target', axis=1)
            y = self.features['Target']
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
            gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
            ensemble_pred = (rf_pred + gb_pred) / 2
            self.model = {
                'rf': rf_model,
                'gb': gb_model,
                'feature_names': X.columns.tolist()
            }
            y_pred = (ensemble_pred > 0.5).astype(int)
            st.write("**Model Performance:**")
            st.text(classification_report(y_test, y_pred))
            rf_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            st.write("**Top 10 Features:**")
            st.dataframe(rf_importance.head(10))
            return {
                'train_score': rf_model.score(X_train_scaled, y_train),
                'test_score': rf_model.score(X_test_scaled, y_test),
                'feature_importance': rf_importance
            }

    def generate_signals(self, timeframes=['1h', '4h', '1d']):
        """Generate signals with multi-timeframe confirmation"""
        if self.data is None:
            logger.error("Cannot generate signals: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        if self.model is None:
            st.info("Model not trained. Training model...")
            self.train_model()

        signals = {}
        for tf in timeframes:
            temp_system = ICTQuantSystem(symbol=self.symbol, api_key=self.api_key, timeframe=tf)
            if temp_system.fetch_data():
                temp_system.create_features()
                temp_system.train_model()
                signals[tf] = temp_system.generate_single_timeframe_signal()

        primary_signal = signals.get(self.timeframe, {'signal': 'HOLD', 'ml_probability': 0.5, 'ict_confirmation': 0})
        confirmation_count = sum(1 for tf, s in signals.items() if s['signal'] in ['BUY', 'STRONG_BUY'] and primary_signal['signal'] in ['BUY', 'STRONG_BUY']) - \
                            sum(1 for tf, s in signals.items() if s['signal'] in ['SELL', 'STRONG_SELL'] and primary_signal['signal'] in ['SELL', 'STRONG_SELL'])

        if confirmation_count >= len(timeframes) - 1:
            primary_signal['signal'] = 'STRONG_' + primary_signal['signal'] if 'STRONG' not in primary_signal['signal'] else primary_signal['signal']
        elif confirmation_count <= 0:
            primary_signal['signal'] = 'HOLD'

        primary_signal['timeframe_confirmation'] = confirmation_count
        return primary_signal

    def generate_single_timeframe_signal(self):
        """Generate signal for a single timeframe"""
        if self.features is None or self.model is None:
            logger.error("Cannot generate signal: features or model is None")
            raise ValueError("Features or model not initialized.")
        latest_features = self.features.drop('Target', axis=1).iloc[-1:]
        latest_features_scaled = self.scaler.transform(latest_features)
        rf_prob = self.model['rf'].predict_proba(latest_features_scaled)[0, 1]
        gb_prob = self.model['gb'].predict_proba(latest_features_scaled)[0, 1]
        ensemble_prob = (rf_prob + gb_prob) / 2
        latest_data = self.data.iloc[-1]
        ict_bullish_signals = 0
        ict_bearish_signals = 0
        if latest_data['MarketStructure'] == 1:
            ict_bullish_signals += 1
        elif latest_data['MarketStructure'] == -1:
            ict_bearish_signals += 1
        if latest_data['BullishFVG']:
            ict_bullish_signals += 1
        if latest_data['BearishFVG']:
            ict_bearish_signals += 1
        if latest_data['BullishOB']:
            ict_bullish_signals += 1
        if latest_data['BearishOB']:
            ict_bearish_signals += 1
        if latest_data['London_Session_Day'] or latest_data['NY_Session_Day'] or latest_data['Overlap_Day']:
            if ensemble_prob > 0.6:
                ict_bullish_signals += 2
            elif ensemble_prob < 0.4:
                ict_bearish_signals += 2
        signal_strength = ensemble_prob
        ict_confirmation = ict_bullish_signals - ict_bearish_signals
        if signal_strength > 0.6 and ict_confirmation >= 1:
            signal = "STRONG_BUY"
        elif signal_strength > 0.55 and ict_confirmation >= 0:
            signal = "BUY"
        elif signal_strength < 0.4 and ict_confirmation <= -1:
            signal = "STRONG_SELL"
        elif signal_strength < 0.45 and ict_confirmation <= 0:
            signal = "SELL"
        else:
            signal = "HOLD"
        return {
            'signal': signal,
            'ml_probability': ensemble_prob,
            'ict_confirmation': ict_confirmation,
            'current_price': latest_data['Close'],
            'timestamp': self.data.index[-1]
        }

    def backtest_strategy(self, initial_capital=10000, position_size=0.1, signal_threshold=0.6, use_trailing_stop=False):
        """Backtest with customizable parameters"""
        if self.data is None:
            logger.error("Cannot backtest strategy: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        if self.features is None:
            self.create_features()
        backtest_data = self.features.copy()
        backtest_data['Price'] = self.data.loc[backtest_data.index, 'Close']
        X = backtest_data.drop(['Target', 'Price'], axis=1)
        X_scaled = self.scaler.transform(X)
        rf_probs = self.model['rf'].predict_proba(X_scaled)[:, 1]
        gb_probs = self.model['gb'].predict_proba(X_scaled)[:, 1]
        ensemble_probs = (rf_probs + gb_probs) / 2
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        trailing_stop = None
        for i in range(1, len(backtest_data)):
            current_prob = ensemble_probs[i]
            current_price = backtest_data['Price'].iloc[i]
            atr = self.data['ATR'].iloc[i]
            if current_prob > signal_threshold and position == 0:
                position = (capital * position_size) / current_price
                capital -= position * current_price
                entry_price = current_price
                stop_loss = entry_price - (atr * 2) if current_prob > 0.5 else entry_price + (atr * 2)
                trailing_stop = stop_loss if use_trailing_stop else None
                trades.append({
                    'entry_time': backtest_data.index[i],
                    'entry_price': entry_price,
                    'type': 'buy' if current_prob > 0.5 else 'sell',
                    'position': position,
                    'stop_loss': stop_loss
                })
            elif position > 0:
                if use_trailing_stop:
                    if current_prob > 0.5:  # Long position
                        new_trailing_stop = max(trailing_stop, current_price - atr * 1.5)
                        trailing_stop = new_trailing_stop
                    else:  # Short position
                        new_trailing_stop = min(trailing_stop, current_price + atr * 1.5)
                        trailing_stop = new_trailing_stop
                if (current_prob < (1 - signal_threshold) or
                    (trailing_stop and (current_price < trailing_stop if current_prob > 0.5 else current_price > trailing_stop))):
                    capital += position * current_price
                    trades[-1]['exit_time'] = backtest_data.index[i]
                    trades[-1]['exit_price'] = current_price
                    trades[-1]['pnl'] = (current_price - trades[-1]['entry_price']) * position if current_prob > 0.5 else (trades[-1]['entry_price'] - current_price) * position
                    position = 0
                    trailing_stop = None
            total_value = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(total_value)
        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        max_drawdown = (pd.Series(equity_curve).cummax() - pd.Series(equity_curve)).max() / pd.Series(equity_curve).cummax().max()
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len([t for t in trades if 'exit_time' in t]),
            'winning_trades': len([t for t in trades if 'pnl' in t and t['pnl'] > 0]),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'trades': trades
        }

    def plot_candlestick(self):
        """Plot candlestick chart with ICT setups and Fibonacci levels"""
        if self.data is None:
            logger.error("Cannot plot candlestick: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=self.data.index,
                                    open=self.data['Open'],
                                    high=self.data['High'],
                                    low=self.data['Low'],
                                    close=self.data['Close'],
                                    name='Price'))
        fvg_bullish = self.data[self.data['BullishFVG']]
        fig.add_trace(go.Scatter(x=fvg_bullish.index, y=fvg_bullish['Close'], mode='markers', name='Bullish FVG',
                                marker=dict(symbol='triangle-up', size=8, color='green')))
        fvg_bearish = self.data[self.data['BearishFVG']]
        fig.add_trace(go.Scatter(x=fvg_bearish.index, y=fvg_bearish['Close'], mode='markers', name='Bearish FVG',
                                marker=dict(symbol='triangle-down', size=8, color='red')))
        ob_bullish = self.data[self.data['BullishOB']]
        fig.add_trace(go.Scatter(x=ob_bullish.index, y=ob_bullish['Close'], mode='markers', name='Bullish OB',
                                marker=dict(symbol='square', size=8, color='darkgreen')))
        ob_bearish = self.data[self.data['BearishOB']]
        fig.add_trace(go.Scatter(x=ob_bearish.index, y=ob_bearish['Close'], mode='markers', name='Bearish OB',
                                marker=dict(symbol='square', size=8, color='darkred')))
        liq_high = self.data[self.data['Liquidity_Sweep_High']]
        fig.add_trace(go.Scatter(x=liq_high.index, y=liq_high['Close'], mode='markers', name='Liq Sweep High',
                                marker=dict(symbol='star', size=10, color='orange')))
        liq_low = self.data[self.data['Liquidity_Sweep_Low']]
        fig.add_trace(go.Scatter(x=liq_low.index, y=liq_low['Close'], mode='markers', name='Liq Sweep Low',
                                marker=dict(symbol='star', size=10, color='yellow')))
        fib_levels = self.identify_fibonacci_levels()
        for level, price in fib_levels.items():
            fig.add_hline(y=price, line_dash="dash", line_color="purple", annotation_text=f"Fib {level}")
        fig.update_layout(title=f"{self.symbol} Candlestick Chart with ICT Setups",
                         xaxis_title='Date', yaxis_title='Price',
                         template='plotly_dark', height=600)
        return fig

    def plot_technical_indicators(self):
        """Plot RSI, MACD, and Volume"""
        if self.data is None:
            logger.error("Cannot plot technical indicators: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=self.data.index, y=self.data['RSI'], mode='lines', name='RSI',
                                    line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title='RSI', xaxis_title='Date', yaxis_title='RSI', template='plotly_dark', height=300)

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD'], mode='lines', name='MACD',
                                     line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD_Signal'], mode='lines', name='Signal',
                                     line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=self.data.index, y=self.data['MACD_Histogram'], name='Histogram',
                                 marker_color='grey', opacity=0.5))
        fig_macd.update_layout(title='MACD', xaxis_title='Date', yaxis_title='MACD', template='plotly_dark', height=300)

        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], name='Volume', marker_color='blue'))
        fig_volume.add_trace(go.Scatter(x=self.data.index, y=self.data['Volume'].rolling(20).mean(),
                                       mode='lines', name='Volume MA', line=dict(color='red')))
        fig_volume.update_layout(title='Volume Analysis', xaxis_title='Date', yaxis_title='Volume',
                                template='plotly_dark', height=300)

        return fig_rsi, fig_macd, fig_volume

    def advanced_risk_management(self, signal_data, account_balance=10000, use_trailing_stop=True):
        """Advanced risk management with position sizing"""
        if self.data is None:
            logger.error("Cannot perform risk management: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        current_atr = self.data['ATR'].iloc[-1]
        current_price = signal_data['current_price']
        risk_per_trade = account_balance * 0.02
        economic_calendar = self.fetch_economic_calendar()
        risk_multiplier = 1.5 if economic_calendar is not None and not economic_calendar.empty else 1.0
        if signal_data['signal'] in ['BUY', 'STRONG_BUY']:
            recent_lows = self.data['Low'].rolling(window=20).min()
            stop_loss = recent_lows.iloc[-1] - (current_atr * 0.5 * risk_multiplier)
            stop_distance = current_price - stop_loss
            take_profit_1 = current_price + (stop_distance * 1.5)
            take_profit_2 = current_price + (stop_distance * 2.0)
        elif signal_data['signal'] in ['SELL', 'STRONG_SELL']:
            recent_highs = self.data['High'].rolling(window=20).max()
            stop_loss = recent_highs.iloc[-1] + (current_atr * 0.5 * risk_multiplier)
            stop_distance = stop_loss - current_price
            take_profit_1 = current_price - (stop_distance * 1.5)
            take_profit_2 = current_price - (stop_distance * 2.0)
        else:
            stop_loss = take_profit_1 = take_profit_2 = stop_distance = 0
        position_size = risk_per_trade / abs(stop_distance) if stop_distance != 0 else 0
        max_position_value = account_balance * 0.05
        max_position_size = max_position_value / current_price if current_price != 0 else 0
        position_size = min(position_size, max_position_size)
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'risk_reward_ratio': abs(stop_distance) / abs(take_profit_1 - current_price) if take_profit_1 != current_price else 0,
            'risk_amount': position_size * abs(stop_distance),
            'use_trailing_stop': use_trailing_stop
        }

    def detect_market_regime(self):
        """Detect market regime"""
        if self.data is None:
            logger.error("Cannot detect market regime: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['Price_Range'] = df['High'] - df['Low']
        df['Volatility'] = df['Close'].rolling(20).std()
        df['Trend_Strength'] = abs(df['Close'].rolling(20).mean().diff())
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff().abs()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        tr = np.maximum(df['High'] - df['Low'],
                       np.maximum(abs(df['High'] - df['Close'].shift()),
                                 abs(df['Low'] - df['Close'].shift())))
        plus_di = (plus_dm.rolling(14).sum() / tr.rolling(14).sum()) * 100
        minus_di = (minus_dm.rolling(14).sum() / tr.rolling(14).sum()) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100
        adx = dx.rolling(14).mean()
        current_adx = adx.iloc[-1]
        current_volatility = df['Volatility'].iloc[-1]
        volatility_percentile = current_volatility > df['Volatility'].rolling(50).quantile(0.8)
        if current_adx > 25:
            regime = "TRENDING"
        elif current_adx < 15:
            regime = "RANGING"
        else:
            regime = "TRANSITIONAL"
        if volatility_percentile:
            regime += "_HIGH_VOL"
        return {
            'regime': regime,
            'adx': current_adx,
            'volatility_percentile': current_volatility / df['Volatility'].rolling(50).mean().iloc[-1],
            'trend_direction': 'UP' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'DOWN'
        }

    def smart_order_flow_analysis(self):
        """Order flow analysis"""
        if self.data is None:
            logger.error("Cannot perform order flow analysis: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        df = self.data.copy()
        df['Buying_Pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['Selling_Pressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-8)
        df['Cumulative_Flow'] = (df['Buying_Pressure'] - df['Selling_Pressure']).cumsum()
        df['Smart_Money'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
        df['SMI_MA'] = df['Smart_Money'].rolling(20).mean()
        df['Institutional_Candle'] = (
            (df['Volume'] > df['Volume'].rolling(20).mean() * 2) &
            (abs(df['Open'] - df['Close']) > (df['High'] - df['Low']) * 0.7)
        )
        price_momentum = df['Close'].rolling(14).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
        flow_momentum = df['Cumulative_Flow'].rolling(14).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
        df['Flow_Divergence'] = np.where(
            (price_momentum > 0) & (flow_momentum < 0), -1,
            np.where((price_momentum < 0) & (flow_momentum > 0), 1, 0)
        )
        return {
            'current_flow': df['Cumulative_Flow'].iloc[-1],
            'flow_trend': 'BULLISH' if df['Cumulative_Flow'].iloc[-1] > df['Cumulative_Flow'].iloc[-10] else 'BEARISH',
            'institutional_activity': df['Institutional_Candle'].iloc[-5:].sum(),
            'flow_divergence': df['Flow_Divergence'].iloc[-1],
            'smart_money_direction': 'ACCUMULATING' if df['Smart_Money'].iloc[-1] > df['SMI_MA'].iloc[-1] else 'DISTRIBUTING'
        }

    def export_data(self, filename=None):
        """Export data to CSV"""
        if self.data is None:
            logger.error("Cannot export data: self.data is None")
            raise ValueError("No data available. Fetch data first.")
        if filename is None:
            filename = f"ict_analysis_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        export_data = self.data.copy()
        if self.model is not None and self.features is not None:
            X = self.features.drop('Target', axis=1)
            X_scaled = self.scaler.transform(X)
            rf_probs = self.model['rf'].predict_proba(X_scaled)[:, 1]
            gb_probs = self.model['gb'].predict_proba(X_scaled)[:, 1]
            ensemble_probs = (rf_probs + gb_probs) / 2
            export_data.loc[X.index, 'ML_Probability'] = ensemble_probs
            export_data.loc[X.index, 'ML_Signal'] = (ensemble_probs > 0.5).astype(int)
        export_data.to_csv(filename)
        st.success(f"Data exported to: {filename}")
        return filename

class ICTPortfolioManager:
    """Portfolio management system"""
    def __init__(self, symbols, initial_capital=100000, api_key=None):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.api_key = api_key if api_key else os.getenv('ALPHA_VANTAGE_API_KEY', 'BDTK3CG8JS0QDM6C')
        self.systems = {}
        self.portfolio_signals = {}

    def initialize_systems(self):
        """Initialize portfolio systems"""
        logger.debug("Initializing ICT systems for portfolio...")
        st.info(f"Initializing portfolio with {len(self.symbols)} symbols...")
        for i, symbol in enumerate(self.symbols):
            logger.debug(f"Setting up {symbol} ({i+1}/{len(self.symbols)})")
            st.write(f"Setting up {symbol} ({i+1}/{len(self.symbols)})...")
            try:
                system = ICTQuantSystem(symbol=symbol, api_key=self.api_key, outputsize='compact', timeframe='1d')
                if system.fetch_data():
                    system.train_model()
                    self.systems[symbol] = system
                    st.success(f"✓ {symbol} initialized successfully")
                else:
                    logger.error(f"Failed to fetch data for {symbol}")
                    st.error(f"✗ Failed to initialize data for {symbol}")
            except Exception as e:
                logger.error(f"Unexpected error initializing {symbol}: {str(e)}")
                st.warning(f"Initialization issue for {symbol}: {str(e)}")
            if i < len(self.symbols) - 1:
                st.info("Waiting 15 seconds for API rate limit...")
                time.sleep(15)

    def get_portfolio_signals(self):
        """Get all portfolio signals"""
        portfolio_signals = {}
        for symbol in self.systems:
            try:
                signal = self.systems[symbol].generate_signals()
                regime_data = self.systems[symbol].detect_market_regime()
                risk_data = self.systems[symbol].advanced_risk_management(signal, self.initial_capital / len(self.symbols))
                portfolio_signals[symbol] = {
                    'signal': signal,
                    'regime': regime_data,
                    'risk': risk_data,
                    'score': self._calculate_score(signal, regime_data)
                }
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {str(e)}")
                st.error(f"Failed to get signals for {symbol}: {str(e)}")
        return portfolio_signals

    def calculate_correlation_matrix(self):
        """Calculate correlation matrix for portfolio assets"""
        if not self.systems:
            logger.error("No systems initialized for correlation analysis")
            return None
        try:
            close_prices = pd.DataFrame({
                symbol: system.data['Close']
                for symbol, system in self.systems.items()
            }).dropna()
            correlation_matrix = close_prices.corr()
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            st.warning(f"Correlation matrix calculation failed: {str(e)}")
            return None

    def _calculate_score(self, signal, regime_data):
        """Calculate opportunity score for portfolio ranking"""
        score = 0.0
        try:
            if signal['signal'] == 'STRONG_BUY':
                score += 3
            elif signal['signal'] == 'BUY':
                score += 1.5
            elif signal['signal'] == 'STRONG_SELL':
                score += 3
            elif signal['signal'] == 'SELL':
                score += 1
            confidence = max(signal['ml_probability'], 1 - signal['ml_probability'])
            score += confidence * 2
            if regime_data['regime'] == 'TRENDING':
                score += 0.2
            score += abs(signal['ict_confirmation']) * 0.75
        except KeyError as e:
            logger.error(f"Key error in calculating score: {str(e)}")
            st.warning(f"Score calculation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error in score calculation: {str(e)}")
            st.warning(f"Score calculation failed: {str(e)}")
        return score

# Streamlit UI
st.title("🚀 Advanced ICT Quantitative Trading System")
st.markdown("Trading system with ICT concepts, machine learning, and sentiment analysis.")

# Sidebar Inputs
st.sidebar.header("Analysis Settings")
currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
selected_pair = st.sidebar.selectbox('Select Currency Pair', currency_pairs, key="currency_pair")
timeframes = ['1h', '4h', '1d']
selected_timeframe = st.sidebar.selectbox('Select Timeframe', timeframes, key="timeframe")
output_size = st.sidebar.selectbox('Select Data Range', ['full', 'compact'], index=1, key="output_size")
analyze_portfolio = st.sidebar.checkbox('Analyze Portfolio', value=False, key="portfolio")
initial_capital = st.sidebar.number_input('Initial Capital ($)', min_value=1000, value=10000, step=1000, key="initial_capital")
api_key = st.sidebar.text_input('Alpha Vantage API Key', value='', type="password", key="api_key")
backtest_position_size = st.sidebar.slider('Position Size (%)', 0.1, 10.0, 1.0, key="backtest_position_size")
backtest_signal_threshold = st.sidebar.slider('Signal Threshold', 0.5, 0.95, 0.6, step=0.05, key="signal_threshold")
use_trailing_stop = st.sidebar.checkbox('Use Trailing Stop', value=True, key="trailing_stop")

# Initialize Session State
if 'system' not in st.session_state or st.session_state.get('selected_symbol') != selected_pair or st.session_state.get('system').timeframe != selected_timeframe:
    st.session_state['selected_symbol'] = selected_pair
    # Avoid setting st.session_state['timeframe'] to prevent conflict with selectbox widget
    st.session_state['system'] = ICTQuantSystem(
        symbol=selected_pair,
        api_key=api_key if api_key else None,
        outputsize=output_size,
        timeframe=selected_timeframe
    )
    logger.debug(f"Initialized system for {selected_pair} on {selected_timeframe}")

system = st.session_state['system']

# Fetch Data
try:
    with st.spinner(f"Initializing system for {selected_pair}..."):
        if not system.fetch_data():
            st.error(f"Failed to fetch data for {selected_pair}")
            st.stop()
        if system.data is None:
            st.error(f"No data found for {selected_pair}")
            st.stop()
        try:
            system.train_model()
        except ValueError as e:
            st.error(f"Model training error: {str(e)}")
            st.stop()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# Dashboard
st.header(f"Analysis for {selected_pair} ({selected_timeframe})")

# Market Data Summary
st.subheader("Market Data Summary")
latest_data = system.data.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Latest Close", f"{latest_data['Close']:.5f}")
col2.metric("30-day High", f"{system.data['High'].tail(30).max():.5f}")
col3.metric("30-day Low", f"{system.data['Low'].tail(30).min():.5f}")
st.write(f"**Date Range**: {system.data.index[0].date()} to {system.data.index[-1].date()}")
st.write(f"**Sentiment Score**: {system.fetch_sentiment_data():.2f}")

# Charts
st.subheader("Price and ICT Analysis")
st.plotly_chart(system.plot_candlestick(), use_container_width=True)

st.subheader("Technical Indicators")
fig_rsi, fig_macd, fig_volume = system.plot_technical_indicators()
col1, col2 = st.columns(2)
col1.plotly_chart(fig_rsi, use_container_width=True)
col2.plotly_chart(fig_volume, use_container_width=True)
st.plotly_chart(fig_macd, use_container_width=True)

# Signals
st.subheader("Trading Signals")
signal_data = system.generate_signals()
col1, col2 = st.columns(2)
col1.metric("Signal", signal_data['signal'])
col2.metric("ML Confidence", f"{signal_data['ml_probability']:.1%}")
st.write(f"**ICT Confirmation**: {signal_data['ict_confirmation']}")
st.write(f"**Timeframe Confirmation**: {signal_data['timeframe_confirmation']}/{len(timeframes)}")
st.write(f"**Current Price**: {signal_data['current_price']:.5f}")

# Risk Management
st.subheader("Risk Management")
risk_data = system.advanced_risk_management(signal_data, initial_capital)
col1, col2, col3 = st.columns(3)
col1.write(f"**Position Size**: {risk_data['position_size']:.2f} units")
col2.write(f"**Stop Loss**: {risk_data['stop_loss']:.5f}")
col3.write(f"**Take Profit 1**: {risk_data['take_profit_1']:.5f}")
st.write(f"**Take Profit 2**: {risk_data['take_profit_2']:.5f}")
st.write(f"**Risk/Reward Ratio**: 1:{risk_data['risk_reward_ratio']:.2f}")
st.write(f"**Risk Amount**: ${risk_data['risk_amount']:.2f}")
st.write(f"**Trailing Stop**: {'Enabled' if risk_data['use_trailing_stop'] else 'Disabled'}")

# Portfolio Analysis
if analyze_portfolio:
    st.subheader("Portfolio Analysis")
    with st.spinner("Analyzing portfolio..."):
        portfolio = ICTPortfolioManager(symbols=currency_pairs, initial_capital=initial_capital, api_key=api_key)
        portfolio.initialize_systems()
        signals = portfolio.get_portfolio_signals()
        sorted_opportunities = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)

        st.markdown("### Top Opportunities")
        for i, (symbol, data) in enumerate(sorted_opportunities[:3], 1):
            st.write(f"{i}. {symbol}")
            st.write(f"   - Signal: {data['signal']['signal']} (Score: {data['score']:.2f})")
            st.write(f"   - Confidence: {data['signal']['ml_probability']:.1%}")
            st.write(f"   - Regime: {data['regime']['regime']}")
            st.write(f"   - Risk/Reward: 1:{data['risk']['risk_reward_ratio']:.2f}")

        st.subheader("Correlation Analysis")
        corr_matrix = portfolio.calculate_correlation_matrix()
        if corr_matrix is not None:
            fig_corr = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale='Viridis',
                showscale=True
            )
            fig_corr.update_layout(title='Currency Pair Correlation', template='plotly_dark', height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

# Backtest Results
st.subheader("Backtest Results")
backtest_results = system.backtest_strategy(
    initial_capital=initial_capital,
    position_size=backtest_position_size / 100,
    signal_threshold=backtest_signal_threshold,
    use_trailing_stop=use_trailing_stop
)
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{backtest_results['total_return']:.2%}")
col2.metric("Total Trades", backtest_results['total_trades'])
col3.metric("Win Rate", f"{backtest_results['winning_trades'] / max(backtest_results['total_trades'], 1):.2%}")
st.write(f"**Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}")
st.write(f"**Max Drawdown**: {backtest_results['max_drawdown']:.2%}")

# Equity Curve
st.subheader("Equity Curve")
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=pd.date_range(start=system.data.index[0], periods=len(backtest_results['equity_curve']), freq='D'),
    y=backtest_results['equity_curve'], mode='lines', name='Equity Curve',
    line=dict(color='green')
))
fig_equity.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value ($)',
                        template='plotly_dark', height=400)
st.plotly_chart(fig_equity, use_container_width=True)

# Economic Calendar
st.subheader("Economic Calendar")
calendar = system.fetch_economic_calendar()
if calendar is not None and not calendar.empty:
    st.dataframe(calendar[['Date', 'Event', 'Impact']].head(5))
else:
    st.warning("No high-impact events available.")

# Export Data
st.subheader("Export Data")
if st.button("Export Analysis Data"):
    filename = system.export_data()
    with open(filename, "rb") as f:
        st.download_button(f"Download {filename}", data=f, file_name=filename)

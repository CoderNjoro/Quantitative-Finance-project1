# ICT Quant Trading System

A machine learning-powered forex trading system that combines Inner Circle Trader methodology with quantitative analysis. Built for traders who want to leverage both technical patterns and data science in their trading approach.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What This Does

This system analyzes forex markets using a combination of ICT (Inner Circle Trader) concepts and machine learning. It pulls real-time market data, identifies patterns that ICT traders look for, and uses ML models to generate trading signals. Everything runs through a web interface built with Streamlit.

The core idea is simple: combine the pattern recognition that ICT traders do manually with the processing power of machine learning to spot opportunities faster and more consistently.

## Key Features

**Data Collection & Reliability**
- Pulls forex data from Alpha Vantage with automatic fallback to Yahoo Finance
- Handles API rate limits and connection issues gracefully
- Works across multiple timeframes (1H, 4H, 1D)

**ICT Pattern Recognition**
The system automatically identifies key ICT concepts:
- Market structure shifts (trend changes)
- Fair Value Gaps (FVGs) - price imbalances that often get filled
- Order Blocks - areas where institutions placed large orders
- Liquidity sweeps - when price hunts stops before reversing
- Fibonacci retracement levels for support/resistance

**Machine Learning Engine**
- Uses an ensemble of Random Forest and Gradient Boosting models
- Trained on 50+ features combining ICT patterns, technical indicators, and market sentiment
- Prevents overfitting with time-series cross-validation
- Outputs probability scores rather than just buy/sell signals

**Market Context**
- Integrates Twitter sentiment analysis for the currency pair
- Pulls economic calendar events from TradingEconomics
- Considers trading session times (London/NY overlap, etc.)

**Interactive Dashboard**
Built with Streamlit for ease of use:
- Real-time signal generation
- Visual charts showing identified patterns
- Backtesting results and performance metrics
- All parameters can be adjusted through the interface

## How It Works

The system follows a straightforward process:

1. **Data Ingestion**: Fetches historical price data for your chosen currency pair and timeframe
2. **Pattern Detection**: Scans the data for ICT patterns and calculates technical indicators
3. **Feature Engineering**: Combines patterns, indicators, sentiment, and time-based features into a dataset
4. **ML Prediction**: The trained ensemble model predicts the probability of the next candle closing higher
5. **Signal Generation**: Combines ML probability with ICT pattern confirmation to generate final signals
6. **Visualization**: Displays everything in an interactive dashboard with charts and metrics

The final signal considers both the ML model's confidence and how many bullish vs bearish ICT patterns are present in the current market structure.

## Technology Stack

- **Data Processing**: pandas, numpy, scikit-learn
- **Market Data**: Alpha Vantage API, Yahoo Finance (backup)
- **Sentiment**: Twitter API with NLTK VADER sentiment analysis
- **Web Interface**: Streamlit
- **Visualization**: Plotly for interactive charts
- **Machine Learning**: Random Forest + Gradient Boosting ensemble

## Installation

### Prerequisites
- Python 3.8 or higher
- API keys (see configuration section)

### Setup Steps

Clone the repository:
```bash
git clone https://github.com/your-username/ICT-Quant-Trading-System.git
cd ICT-Quant-Trading-System
```

Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```

## Configuration

You'll need API keys for the data sources. Create a `.env` file in the project root:

```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
TRADINGECONOMICS_API_KEY=your_tradingeconomics_key
```

### Getting API Keys

**Alpha Vantage** (required)
- Sign up at https://www.alphavantage.co/support/#api-key
- Free tier: 5 calls per minute, 500 per day
- This is the primary data source

**Twitter API** (optional, for sentiment analysis)
- Apply for developer access at https://developer.twitter.com/
- Need elevated access for the endpoints we use
- Without this, sentiment features will be disabled

**TradingEconomics** (optional, for economic calendar)
- Get key at https://tradingeconomics.com/analytics/api/
- Free tier available with limited requests
- Adds fundamental analysis context

## Usage

Start the application:
```bash
streamlit run main.py
```

This opens the dashboard in your browser (usually http://localhost:8501).

### Using the Dashboard

1. **Select Parameters**: Choose your currency pair and timeframe
2. **Generate Signal**: Click the button to run the analysis
3. **Review Results**: The system shows:
   - Final signal (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
   - ML model confidence percentage
   - ICT pattern confirmation score
   - Visual chart with detected patterns

### Signal Interpretation

- **STRONG_BUY/STRONG_SELL**: High confidence (>80%), multiple confirming patterns
- **BUY/SELL**: Moderate confidence (60-80%), some confirming patterns  
- **HOLD**: Low confidence (<60%) or conflicting signals

The system is designed to be conservative. It's better to miss some opportunities than to generate false signals.

## Backtesting

The built-in backtester evaluates strategy performance using historical data. Key metrics include:

- Win rate percentage
- Average profit/loss per trade
- Maximum drawdown
- Sharpe ratio (risk-adjusted returns)
- Total return over the testing period

You can adjust parameters like stop loss, take profit, and position sizing to see how they affect performance.

## Understanding the ML Model

The ensemble model is trained on features that capture:

**ICT Patterns** (15 features)
- Presence and strength of Fair Value Gaps
- Order block locations and significance
- Market structure shift signals
- Liquidity sweep occurrences

**Technical Indicators** (20 features)  
- RSI, MACD, Bollinger Bands, ATR
- Moving average relationships
- Momentum and volatility measures

**Market Context** (15 features)
- Trading session times
- Day of week effects
- Economic event proximity
- Sentiment polarity scores

The model outputs probabilities rather than hard classifications, allowing for more nuanced signal generation when combined with ICT confirmation scores.

## Limitations and Considerations

**Data Dependencies**
- Requires stable internet for API calls
- Free API tiers have rate limits that may affect real-time usage
- Twitter API access has become more restricted recently

**Market Conditions**
- Trained on historical data which may not reflect future market conditions
- Performance can vary significantly across different market regimes
- Works best in trending markets where ICT patterns are more reliable

**Not Financial Advice**
This is a research and educational tool. All trading involves risk, and past performance doesn't guarantee future results. The system is designed to assist analysis, not replace human judgment.

## Contributing

Contributions are welcome! Areas where help would be particularly useful:

- Additional ICT pattern recognition algorithms
- Alternative data sources for backup redundancy
- Performance optimization for real-time processing
- Additional technical indicators or features
- Documentation improvements

Please fork the repository and submit pull requests with clear descriptions of changes.

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is provided for educational and research purposes only. It is not intended as financial advice. Trading foreign exchange carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange, you should carefully consider your investment objectives, level of experience, and risk appetite.

You should be aware of all the risks associated with foreign exchange trading and seek advice from an independent financial advisor if you have any doubts. Past performance is not indicative of future results.

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation in the wiki
- Review existing issues for solutions

---

*Built by traders, for traders. Happy trading!*

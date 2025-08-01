# Crypto Price Checker

A full-featured cryptocurrency tracking and portfolio management application built with Python and Tkinter.

## Features

### 🔍 Market Overview
- Real-time price tracking for 1000+ cryptocurrencies
- Search functionality for any cryptocurrency
- Market data including price, volume, market cap, and 24h changes
- Customizable watchlist

### 📊 Advanced Charts
- Interactive price charts with technical indicators
- Moving averages (MA20, MA50)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume analysis
- Multiple timeframes (1h, 24h, 7d, 30d, 90d, 1y)

### 💼 Portfolio Management
- Track cryptocurrency holdings
- Buy/sell transaction recording
- Real-time portfolio valuation
- Profit/loss calculations
- Portfolio performance metrics
- Diversification analysis
- Tax reporting (realized gains/losses)

### 🚨 Price Alerts
- Set price alerts for any cryptocurrency
- Multiple alert conditions (above, below, crosses above/below)
- Desktop notifications
- Email notifications (configurable)
- Background monitoring

### 📈 Analysis Tools
- Portfolio correlation analysis
- Volatility calculations
- Risk assessment
- Performance comparisons
- Market dominance tracking

### 💾 Data Management
- Import/export portfolio data
- Backup and restore functionality
- CSV export for external analysis
- Persistent data storage

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project files:**
   ```bash
   # Create a new directory
   mkdir crypto-price-checker
   cd crypto-price-checker
   ```

2. **Save all the Python files in the project directory:**
   - `main.py` (main application)
   - `crypto_api.py` (API handler)
   - `crypto_charts.py` (chart generation)
   - `crypto_portfolio.py` (portfolio management)
   - `crypto_alerts.py` (price alerts)
   - `crypto_utils.py` (utility functions)

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install requests pandas numpy matplotlib seaborn scipy plyer python-dateutil
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Getting Started

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Market Overview Tab:**
   - View top cryptocurrencies by market cap
   - Search for specific cryptocurrencies
   - Add coins to your watchlist
   - Monitor real-time price changes

3. **Portfolio Tab:**
   - Add your cryptocurrency transactions
   - Track your holdings and performance
   - View detailed profit/loss calculations
   - Export portfolio data

4. **Charts Tab:**
   - Generate technical analysis charts
   - Select different timeframes
   - Analyze price trends with indicators
   - Save charts as images

5. **Alerts Tab:**
   - Create price alerts for your holdings
   - Set up email notifications
   - Monitor alert status
   - Manage active alerts

6. **Settings Tab:**
   - Configure auto-refresh intervals
   - Set up email notifications
   - Export/import data
   - Test API connections

### Portfolio Management

#### Adding Transactions
1. Go to the Portfolio tab
2. Enter the cryptocurrency symbol (e.g., BTC, ETH)
3. Enter the amount you bought/sold
4. Enter the price at which you bought/sold
5. Click "Add" to record the transaction

#### Viewing Performance
- The portfolio automatically calculates your current holdings
- View real-time profit/loss for each position
- See overall portfolio performance and allocation

### Setting Up Alerts

#### Basic Price Alerts
1. Go to the Alerts tab
2. Enter the cryptocurrency symbol
3. Choose condition (Above/Below)
4. Set target price
5. Click "Create Alert"

#### Email Notifications
1. Go to Settings tab
2. Configure SMTP settings for your email provider
3. Test the connection
4. Enable email notifications when creating alerts

### Chart Analysis

#### Generating Charts
1. Go to Charts tab
2. Enter cryptocurrency symbol
3. Select timeframe
4. Click "Generate Chart"

#### Understanding Indicators
- **Moving Averages**: Show trend direction
- **Bollinger Bands**: Indicate volatility and potential reversal points
- **RSI**: Shows overbought (>70) and oversold (<30) conditions
- **MACD**: Indicates momentum changes

## Configuration

### Email Notifications Setup

To enable email notifications for price alerts:

1. **Gmail Setup:**
   - Enable 2-factor authentication
   - Generate an app-specific password
   - Use smtp.gmail.com, port 587

2. **Other Email Providers:**
   - Yahoo: smtp.mail.yahoo.com, port 587
   - Outlook: smtp-mail.outlook.com, port 587
   - Custom SMTP: Configure with your provider's settings

### API Configuration

The application uses the free CoinGecko API by default. No API key required for basic functionality.

**Rate Limits:**
- Free tier: 100 calls/minute
- The application includes automatic rate limiting

## File Structure

```
crypto-price-checker/
├── main.py                 # Main application entry point
├── crypto_api.py          # CoinGecko API handler
├── crypto_charts.py       # Chart generation and technical analysis
├── crypto_portfolio.py    # Portfolio management
├── crypto_alerts.py       # Price alerts system
├── crypto_utils.py        # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── crypto_settings.json  # Application settings (auto-generated)
├── crypto_portfolio.json # Portfolio data (auto-generated)
└── crypto_alerts.json    # Alerts data (auto-generated)
```

## Advanced Features

### Technical Indicators

#### Moving Averages
- **MA20**: 20-period moving average (short-term trend)
- **MA50**: 50-period moving average (medium-term trend)
- Golden Cross: MA20 crosses above MA50 (bullish signal)
- Death Cross: MA20 crosses below MA50 (bearish signal)

#### Bollinger Bands
- Upper Band: MA20 + (2 × Standard Deviation)
- Lower Band: MA20 - (2 × Standard Deviation)
- Price touching upper band: Potentially overbought
- Price touching lower band: Potentially oversold

#### RSI (Relative Strength Index)
- Range: 0-100
- Above 70: Overbought condition
- Below 30: Oversold condition
- 50: Neutral momentum

#### MACD (Moving Average Convergence Divergence)
- MACD Line: 12-period EMA - 26-period EMA
- Signal Line: 9-period EMA of MACD
- Histogram: MACD - Signal Line
- Crossovers indicate momentum changes

### Portfolio Analytics

#### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Diversification Score**: Based on Herfindahl Index

#### Performance Metrics
- **Total Return**: Overall gain/loss percentage
- **CAGR**: Compound Annual Growth Rate
- **Beta**: Correlation with market (if applicable)
- **Alpha**: Excess return vs benchmark

### Data Export/Import

#### Supported Formats
- **JSON**: Complete portfolio/alerts backup
- **CSV**: Transaction history and holdings
- **Excel**: (Future enhancement)

#### Export Options
- Portfolio transactions
- Current holdings
- Alert configurations
- Performance reports

## Troubleshooting

### Common Issues

#### "API Connection Failed"
- Check internet connection
- Verify firewall settings
- Try refreshing after a few minutes (rate limit)

#### "No Data Available"
- Cryptocurrency symbol might be incorrect
- Try searching for the full name instead of symbol
- Some newer cryptocurrencies might not be available

#### Charts Not Displaying
- Ensure matplotlib is properly installed
- Check if sufficient price data is available
- Try a different timeframe

#### Email Notifications Not Working
- Verify SMTP settings
- Check email provider's security settings
- Test connection in Settings tab

### Performance Optimization

#### For Large Portfolios
- Enable selective refresh for watchlist items
- Increase refresh interval in settings
- Consider using fewer technical indicators

#### For Slow Connections
- Reduce auto-refresh frequency
- Limit watchlist size
- Use longer timeframes for charts

## Development

### Adding New Features

#### Custom Indicators
1. Extend `crypto_charts.py`
2. Add calculation method
3. Update chart plotting functions

#### New Alert Types
1. Modify `crypto_alerts.py`
2. Add new condition types
3. Update monitoring logic

#### Additional APIs
1. Create new API handler in `crypto_api.py`
2. Implement rate limiting
3. Add error handling

### Code Structure

#### Main Components
- **GUI Layer**: Tkinter interface (`main.py`)
- **Data Layer**: API handlers and data processing
- **Business Logic**: Portfolio calculations and analysis
- **Visualization**: Chart generation and formatting

#### Design Patterns
- **Observer Pattern**: For price alerts and notifications
- **Factory Pattern**: For chart creation
- **Strategy Pattern**: For different alert conditions

## Security Considerations

### Data Protection
- Portfolio data stored locally in JSON format
- No sensitive financial data transmitted
- Optional email credentials (use app passwords)

### API Security
- Uses HTTPS for all API calls
- Implements rate limiting
- Handles API errors gracefully

### Best Practices
- Regular backups of portfolio data
- Use app-specific passwords for email
- Keep software updated

## Contributing

### Reporting Issues
1. Check existing issues first
2. Provide detailed error messages
3. Include system information
4. Steps to reproduce the problem

### Feature Requests
1. Describe the desired functionality
2. Explain the use case
3. Consider implementation complexity
4. Check if similar features exist

### Code Contributions
1. Follow Python PEP 8 style guide
2. Add appropriate error handling
3. Include docstrings for functions
4. Test thoroughly before submitting

## Future Enhancements

### Planned Features
- [ ] Web-based dashboard
- [ ] Mobile app companion
- [ ] Advanced portfolio optimization
- [ ] Social sentiment analysis
- [ ] DeFi protocol integration
- [ ] NFT tracking
- [ ] Tax reporting enhancements
- [ ] Paper trading simulator

### Integration Possibilities
- [ ] Binance API for real trading
- [ ] TradingView charts
- [ ] Discord/Telegram bots
- [ ] Database backend (PostgreSQL/SQLite)
- [ ] Cloud synchronization

## License

This project is open source and available under the MIT License.

## Disclaimer

**Important**: This software is for informational and educational purposes only. It is not financial advice. Cryptocurrency investments are highly volatile and risky. Always do your own research and consult with qualified financial advisors before making investment decisions.

The developers are not responsible for any financial losses incurred through the use of this software.

## Support

### Documentation
- Check this README for common questions
- Review code comments for technical details
- Example configurations in the code

### Community
- Create issues for bugs or questions
- Discussions for feature requests
- Wiki for extended documentation

### Updates
- Check releases for new versions
- Review changelog for new features
- Backup data before updating
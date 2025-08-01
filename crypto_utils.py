"""
Crypto Utilities Module
Helper functions and utilities for cryptocurrency operations
"""

import re
import hashlib
import hmac
import base64
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

class CryptoUtils:
    def __init__(self):
        self.supported_exchanges = [
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi',
            'okex', 'kucoin', 'gate', 'bybit', 'ftx'
        ]
        
        self.fiat_currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY',
            'SEK', 'NZD', 'MXN', 'SGD', 'HKD', 'NOK', 'TRY', 'ZAR',
            'BRL', 'INR', 'KRW', 'PLN', 'CZK', 'HUF', 'RUB'
        ]
    
    @staticmethod
    def validate_address(address: str, currency: str = 'BTC') -> bool:
        """Validate cryptocurrency address format"""
        if not address:
            return False
        
        currency = currency.upper()
        
        # Bitcoin address validation
        if currency == 'BTC':
            # Legacy addresses (1...)
            if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
                return True
            # SegWit addresses (3... or bc1...)
            if re.match(r'^3[a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
                return True
            if re.match(r'^bc1[a-z0-9]{39,59}$', address):
                return True
        
        # Ethereum address validation
        elif currency == 'ETH':
            if re.match(r'^0x[a-fA-F0-9]{40}$', address):
                return True
        
        # Litecoin address validation
        elif currency == 'LTC':
            if re.match(r'^[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}$', address):
                return True
            if re.match(r'^ltc1[a-z0-9]{39,59}$', address):
                return True
        
        # Dogecoin address validation
        elif currency == 'DOGE':
            if re.match(r'^D{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}$', address):
                return True
        
        # Ripple address validation
        elif currency == 'XRP':
            if re.match(r'^r[1-9A-HJ-NP-Za-km-z]{25,34}$', address):
                return True
        
        # Cardano address validation
        elif currency == 'ADA':
            if re.match(r'^addr1[a-z0-9]+$', address):
                return True
        
        # Generic validation for other currencies
        else:
            # Most crypto addresses are 26-62 characters long
            if len(address) >= 26 and len(address) <= 62:
                return True
        
        return False
    
    @staticmethod
    def format_price(price: float, decimals: int = None) -> str:
        """Format price with appropriate decimal places"""
        if price == 0:
            return "$0.00"
        
        if decimals is None:
            if price >= 1000:
                decimals = 2
            elif price >= 1:
                decimals = 4
            elif price >= 0.01:
                decimals = 6
            else:
                decimals = 8
        
        # Use Decimal for precise formatting
        price_decimal = Decimal(str(price))
        rounded_price = price_decimal.quantize(
            Decimal('0.' + '0' * decimals), 
            rounding=ROUND_HALF_UP
        )
        
        return f"${rounded_price:,.{decimals}f}"
    
    @staticmethod
    def format_volume(volume: float) -> str:
        """Format trading volume with appropriate units"""
        if volume >= 1e9:
            return f"${volume/1e9:.2f}B"
        elif volume >= 1e6:
            return f"${volume/1e6:.2f}M"
        elif volume >= 1e3:
            return f"${volume/1e3:.2f}K"
        else:
            return f"${volume:.2f}"
    
    @staticmethod
    def format_market_cap(market_cap: float) -> str:
        """Format market cap with appropriate units"""
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        elif market_cap >= 1e3:
            return f"${market_cap/1e3:.2f}K"
        else:
            return f"${market_cap:.2f}"
    
    @staticmethod
    def format_percentage(percentage: float, decimals: int = 2) -> str:
        """Format percentage with color indication"""
        sign = "+" if percentage > 0 else ""
        return f"{sign}{percentage:.{decimals}f}%"
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def calculate_roi(initial_investment: float, current_value: float) -> float:
        """Calculate Return on Investment (ROI)"""
        if initial_investment == 0:
            return 0
        return ((current_value - initial_investment) / initial_investment) * 100
    
    @staticmethod
    def calculate_compound_annual_growth_rate(
        initial_value: float, 
        final_value: float, 
        years: float
    ) -> float:
        """Calculate Compound Annual Growth Rate (CAGR)"""
        if initial_value <= 0 or years <= 0:
            return 0
        return (((final_value / initial_value) ** (1 / years)) - 1) * 100
    
    @staticmethod
    def calculate_volatility(prices: List[float], annualized: bool = True) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if len(prices) < 2:
            return 0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0
        
        # Calculate standard deviation
        volatility = np.std(returns)
        
        # Annualize if requested (assuming daily data)
        if annualized:
            volatility *= np.sqrt(365)
        
        return volatility * 100  # Convert to percentage
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float], 
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualize assuming daily returns
        annual_return = mean_return * 365
        annual_std = std_return * np.sqrt(365)
        
        return (annual_return - risk_free_rate) / annual_std
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Dict:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return {'max_drawdown': 0, 'duration': 0}
        
        peak = prices[0]
        max_drawdown = 0
        max_duration = 0
        current_duration = 0
        drawdown_start = 0
        
        for i, price in enumerate(prices):
            if price > peak:
                peak = price
                current_duration = 0
            else:
                current_duration = i - drawdown_start
                drawdown = (peak - price) / peak
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_duration = current_duration
        
        return {
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'duration': max_duration
        }
    
    @staticmethod
    def generate_api_signature(
        secret: str, 
        message: str, 
        algorithm: str = 'sha256'
    ) -> str:
        """Generate HMAC signature for API authentication"""
        if algorithm.lower() == 'sha256':
            signature = hmac.new(
                secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        elif algorithm.lower() == 'sha512':
            signature = hmac.new(
                secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
        else:
            raise ValueError("Unsupported algorithm")
        
        return signature
    
    @staticmethod
    def convert_timestamp(timestamp: Union[int, float, str], 
                         format_type: str = 'datetime') -> Union[str, datetime]:
        """Convert timestamp to various formats"""
        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except ValueError:
                return "Invalid timestamp"
        
        # Handle milliseconds
        if timestamp > 1e12:
            timestamp = timestamp / 1000
        
        dt = datetime.fromtimestamp(timestamp)
        
        if format_type == 'datetime':
            return dt
        elif format_type == 'iso':
            return dt.isoformat()
        elif format_type == 'readable':
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        elif format_type == 'date':
            return dt.strftime('%Y-%m-%d')
        elif format_type == 'time':
            return dt.strftime('%H:%M:%S')
        else:
            return str(dt)
    
    def get_market_dominance(self, market_data: List[Dict]) -> Dict:
        """Calculate market dominance for cryptocurrencies"""
        if not market_data:
            return {}
        
        total_market_cap = sum(
            item.get('market_cap', 0) for item in market_data
        )
        
        if total_market_cap == 0:
            return {}
        
        dominance = {}
        for item in market_data:
            symbol = item['symbol'].upper()
            market_cap = item.get('market_cap', 0)
            dominance[symbol] = (market_cap / total_market_cap) * 100
        
        return dominance
    
    @staticmethod
    def calculate_portfolio_correlation(
        portfolio_prices: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        if len(portfolio_prices) < 2:
            return pd.DataFrame()
        
        # Create DataFrame from price data
        df = pd.DataFrame(portfolio_prices)
        
        # Calculate daily returns
        returns_df = df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    @staticmethod
    def optimize_portfolio_weights(
        expected_returns: List[float],
        covariance_matrix: np.ndarray,
        risk_tolerance: float = 0.5
    ) -> List[float]:
        """Basic portfolio optimization (simplified Markowitz)"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return -portfolio_return + risk_tolerance * portfolio_variance
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            result = minimize(
                objective,
                x0=np.array([1/n_assets] * n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x.tolist() if result.success else [1/n_assets] * n_assets
            
        except ImportError:
            # Fallback to equal weights
            return [1/len(expected_returns)] * len(expected_returns)
    
    @staticmethod
    def calculate_fear_greed_index(market_data: Dict) -> Dict:
        """Calculate custom Fear & Greed Index"""
        # This is a simplified version - real index uses multiple factors
        try:
            # Factors (simplified):
            # 1. Market volatility (25%)
            # 2. Market momentum (25%)
            # 3. Social media sentiment (placeholder - 15%)
            # 4. Surveys (placeholder - 15%)
            # 5. Dominance (10%)
            # 6. Trends (10%)
            
            volatility_score = 50  # Placeholder
            momentum_score = 50    # Placeholder
            sentiment_score = 50   # Placeholder
            survey_score = 50      # Placeholder
            dominance_score = 50   # Placeholder
            trends_score = 50      # Placeholder
            
            # Weighted average
            fear_greed_value = (
                volatility_score * 0.25 +
                momentum_score * 0.25 +
                sentiment_score * 0.15 +
                survey_score * 0.15 +
                dominance_score * 0.10 +
                trends_score * 0.10
            )
            
            # Classify the value
            if fear_greed_value <= 25:
                classification = "Extreme Fear"
            elif fear_greed_value <= 45:
                classification = "Fear"
            elif fear_greed_value <= 55:
                classification = "Neutral"
            elif fear_greed_value <= 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            return {
                'value': int(fear_greed_value),
                'classification': classification,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'value': 50,
                'classification': 'Neutral',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @staticmethod
    def convert_currency(amount: float, from_currency: str, 
                        to_currency: str, exchange_rates: Dict) -> float:
        """Convert between currencies using exchange rates"""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            return amount
        
        # Direct conversion
        direct_rate = exchange_rates.get(f"{from_currency}_{to_currency}")
        if direct_rate:
            return amount * direct_rate
        
        # Reverse conversion
        reverse_rate = exchange_rates.get(f"{to_currency}_{from_currency}")
        if reverse_rate and reverse_rate != 0:
            return amount / reverse_rate
        
        # USD as intermediate currency
        if from_currency != 'USD' and to_currency != 'USD':
            usd_from_rate = exchange_rates.get(f"{from_currency}_USD")
            usd_to_rate = exchange_rates.get(f"USD_{to_currency}")
            
            if usd_from_rate and usd_to_rate:
                usd_amount = amount * usd_from_rate
                return usd_amount * usd_to_rate
        
        return amount  # No conversion available
    
    @staticmethod
    def validate_trading_pair(pair: str) -> bool:
        """Validate trading pair format"""
        # Common formats: BTC/USD, BTCUSD, BTC-USD
        if not pair:
            return False
        
        # Remove common separators and check length
        clean_pair = pair.replace('/', '').replace('-', '').replace('_', '')
        
        if len(clean_pair) < 6 or len(clean_pair) > 12:
            return False
        
        # Should contain only alphanumeric characters
        return clean_pair.isalnum()
    
    @staticmethod
    def parse_trading_pair(pair: str) -> Tuple[str, str]:
        """Parse trading pair into base and quote currencies"""
        if not pair:
            return "", ""
        
        # Handle different separators
        separators = ['/', '-', '_']
        for sep in separators:
            if sep in pair:
                parts = pair.split(sep)
                if len(parts) == 2:
                    return parts[0].upper(), parts[1].upper()
        
        # Handle concatenated pairs (e.g., BTCUSD)
        # This is more complex and may require a list of known currencies
        common_quotes = ['USD', 'USDT', 'EUR', 'BTC', 'ETH', 'BNB']
        for quote in common_quotes:
            if pair.upper().endswith(quote):
                base = pair.upper()[:-len(quote)]
                if len(base) >= 2:
                    return base, quote
        
        return "", ""
    
    def get_supported_exchanges(self) -> List[str]:
        """Get list of supported exchanges"""
        return self.supported_exchanges.copy()
    
    def get_supported_fiat_currencies(self) -> List[str]:
        """Get list of supported fiat currencies"""
        return self.fiat_currencies.copy()
    
    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """Clean and standardize cryptocurrency symbol"""
        if not symbol:
            return ""
        
        # Remove whitespace and convert to uppercase
        clean = symbol.strip().upper()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['$', '#']
        suffixes_to_remove = ['USD', 'USDT', 'BTC', 'ETH']
        
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        
        # Don't remove suffixes automatically as they might be part of the symbol
        
        return clean
    
    @staticmethod
    def is_stablecoin(symbol: str) -> bool:
        """Check if a cryptocurrency is a stablecoin"""
        symbol = symbol.upper()
        stablecoins = [
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'GUSD',
            'FRAX', 'FEI', 'LUSD', 'MIM', 'UST', 'TERRA', 'USTC'
        ]
        return symbol in stablecoins
    
    @staticmethod
    def categorize_cryptocurrency(symbol: str, market_cap: float = None) -> str:
        """Categorize cryptocurrency by type or market cap"""
        symbol = symbol.upper()
        
        # Check if it's a stablecoin
        if CryptoUtils.is_stablecoin(symbol):
            return "Stablecoin"
        
        # Major cryptocurrencies
        major_cryptos = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE']
        if symbol in major_cryptos:
            return "Major Cryptocurrency"
        
        # DeFi tokens
        defi_tokens = [
            'UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI',
            'CRV', '1INCH', 'BAL', 'LDO', 'RPL'
        ]
        if symbol in defi_tokens:
            return "DeFi Token"
        
        # Layer 1 blockchains
        layer1_tokens = [
            'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM', 'NEAR',
            'FTM', 'ALGO', 'EGLD', 'HBAR', 'ICP'
        ]
        if symbol in layer1_tokens:
            return "Layer 1 Blockchain"
        
        # Meme coins
        meme_coins = [
            'DOGE', 'SHIB', 'ELON', 'FLOKI', 'BABYDOGE', 'DOGELON',
            'SAFEMOON', 'PEPE'
        ]
        if symbol in meme_coins:
            return "Meme Coin"
        
        # Categorize by market cap if provided
        if market_cap is not None:
            if market_cap >= 10e9:  # $10B+
                return "Large Cap"
            elif market_cap >= 1e9:  # $1B+
                return "Mid Cap"
            elif market_cap >= 100e6:  # $100M+
                return "Small Cap"
            else:
                return "Micro Cap"
        
        return "Other"
    
    @staticmethod
    def get_risk_level(volatility: float, market_cap: float = None) -> str:
        """Determine risk level based on volatility and market cap"""
        risk_score = 0
        
        # Volatility component (0-100 scale)
        if volatility >= 100:
            risk_score += 50
        elif volatility >= 50:
            risk_score += 30
        elif volatility >= 25:
            risk_score += 20
        else:
            risk_score += 10
        
        # Market cap component (if available)
        if market_cap is not None:
            if market_cap < 100e6:  # <$100M
                risk_score += 40
            elif market_cap < 1e9:  # <$1B
                risk_score += 30
            elif market_cap < 10e9:  # <$10B
                risk_score += 20
            else:
                risk_score += 10
        else:
            risk_score += 25  # Default medium risk for market cap
        
        # Determine risk level
        if risk_score >= 80:
            return "Very High Risk"
        elif risk_score >= 60:
            return "High Risk"
        elif risk_score >= 40:
            return "Medium Risk"
        elif risk_score >= 20:
            return "Low Risk"
        else:
            return "Very Low Risk"
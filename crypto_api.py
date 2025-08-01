"""
Crypto API Handler
Manages API calls to CoinGecko and other cryptocurrency APIs
"""

import requests
import time
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional

class CryptoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoPriceChecker/1.0'
        })
        self.rate_limit_delay = 1.2  # Seconds between requests
        self.last_request_time = 0
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling and caching"""
        # Check cache first
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (data, time.time())
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            self._make_request("ping")
            return True
        except:
            return False
    
    def get_top_cryptocurrencies(self, limit: int = 100) -> List[Dict]:
        """Get top cryptocurrencies by market cap"""
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': min(limit, 250),
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        return self._make_request("coins/markets", params)
    
    def search_cryptocurrency(self, query: str) -> List[Dict]:
        """Search for cryptocurrency by name or symbol"""
        # First try the search endpoint
        search_results = self._make_request("search", {'query': query})
        
        if not search_results.get('coins'):
            return []
        
        # Get detailed info for search results
        coin_ids = [coin['id'] for coin in search_results['coins'][:20]]  # Limit to 20
        
        if not coin_ids:
            return []
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        return self._make_request("coins/markets", params)
    
    def get_cryptocurrency_details(self, coin_id: str) -> Dict:
        """Get detailed information about a specific cryptocurrency"""
        return self._make_request(f"coins/{coin_id}")
    
    def get_price_history(self, coin_id: str, days: int = 30) -> Dict:
        """Get price history for a cryptocurrency"""
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        return self._make_request(f"coins/{coin_id}/market_chart", params)
    
    def get_current_price(self, coin_ids: List[str]) -> Dict:
        """Get current prices for multiple cryptocurrencies"""
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': True,
            'include_market_cap': True,
            'include_24hr_vol': True
        }
        
        return self._make_request("simple/price", params)
    
    def get_trending_cryptocurrencies(self) -> Dict:
        """Get trending cryptocurrencies"""
        return self._make_request("search/trending")
    
    def get_global_market_data(self) -> Dict:
        """Get global cryptocurrency market data"""
        return self._make_request("global")
    
    def get_supported_currencies(self) -> List[str]:
        """Get list of supported vs currencies"""
        return self._make_request("simple/supported_vs_currencies")
    
    def get_coin_list(self) -> List[Dict]:
        """Get list of all available coins"""
        return self._make_request("coins/list")
    
    def get_exchanges(self) -> List[Dict]:
        """Get list of exchanges"""
        return self._make_request("exchanges")
    
    def get_exchange_rates(self) -> Dict:
        """Get BTC exchange rates"""
        return self._make_request("exchange_rates")
    
    def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index (if available)"""
        try:
            # This would require a different API
            # Placeholder implementation
            return {"value": 50, "classification": "Neutral"}
        except:
            return {"value": 50, "classification": "Neutral"}
    
    def get_defi_data(self) -> Dict:
        """Get DeFi market data"""
        return self._make_request("global/decentralized_finance_defi")
    
    def get_market_chart_range(self, coin_id: str, from_timestamp: int, to_timestamp: int) -> Dict:
        """Get price data for a specific time range"""
        params = {
            'vs_currency': 'usd',
            'from': from_timestamp,
            'to': to_timestamp
        }
        
        return self._make_request(f"coins/{coin_id}/market_chart/range", params)
    
    def get_ohlc_data(self, coin_id: str, days: int = 30) -> List:
        """Get OHLC (Open, High, Low, Close) data"""
        return self._make_request(f"coins/{coin_id}/ohlc", {'vs_currency': 'usd', 'days': days})
    
    def symbol_to_id(self, symbol: str) -> Optional[str]:
        """Convert symbol to coin ID"""
        try:
            coins_list = self.get_coin_list()
            symbol_lower = symbol.lower()
            
            for coin in coins_list:
                if coin['symbol'].lower() == symbol_lower:
                    return coin['id']
            
            return None
        except:
            return None
    
    def get_multiple_coin_data(self, symbols: List[str]) -> Dict:
        """Get data for multiple coins by symbols"""
        coin_ids = []
        
        for symbol in symbols:
            coin_id = self.symbol_to_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
        
        if not coin_ids:
            return {}
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d'
        }
        
        return self._make_request("coins/markets", params)
    
    def get_coin_categories(self) -> List[Dict]:
        """Get cryptocurrency categories"""
        return self._make_request("coins/categories/list")
    
    def get_category_data(self, category_id: str) -> List[Dict]:
        """Get coins in a specific category"""
        params = {
            'category': category_id,
            'order': 'market_cap_desc',
            'vs_currency': 'usd',
            'sparkline': False
        }
        
        return self._make_request("coins/markets", params)
    
    def get_nft_data(self) -> List[Dict]:
        """Get NFT market data"""
        return self._make_request("nfts/list")
    
    def clear_cache(self):
        """Clear API cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for cache_key, (data, timestamp) in self.cache.items():
            if current_time - timestamp < self.cache_duration:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_duration': self.cache_duration
        }
"""
Crypto Charts Module
Handles chart generation and visualization for cryptocurrency data
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import seaborn as sns
from crypto_api import CryptoAPI
from typing import List, Dict, Optional, Tuple

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

class CryptoCharts:
    def __init__(self):
        self.api = CryptoAPI()
        self.current_figure = None
        self.current_canvas = None
        
        # Chart configuration
        self.figure_size = (12, 8)
        self.dpi = 100
        self.colors = {
            'background': '#1e1e1e',
            'grid': '#333333',
            'price': '#00ff88',
            'volume': '#ff6b6b',
            'ma_short': '#ffd93d',
            'ma_long': '#6bcf7f',
            'rsi': '#ff9f43',
            'macd': '#74b9ff',
            'bollinger': '#fd79a8'
        }
    
    def create_price_chart(self, symbol: str, timeframe: str = '24h', parent_frame=None) -> FigureCanvasTkAgg:
        """Create comprehensive price chart with technical indicators"""
        try:
            # Get coin ID from symbol
            coin_id = self.api.symbol_to_id(symbol)
            if not coin_id:
                raise ValueError(f"Could not find coin ID for symbol: {symbol}")
            
            # Get data based on timeframe
            days = self._timeframe_to_days(timeframe)
            data = self.api.get_price_history(coin_id, days)
            
            if not data or 'prices' not in data:
                raise ValueError("No price data available")
            
            # Convert to DataFrame
            df = self._prepare_dataframe(data)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=self.figure_size, 
                                   gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                                   facecolor=self.colors['background'])
            
            fig.suptitle(f'{symbol.upper()} Price Chart ({timeframe})', 
                        fontsize=16, fontweight='bold', color='white')
            
            # Price chart with moving averages and Bollinger Bands
            self._plot_price_chart(axes[0], df, symbol)
            
            # Volume chart
            self._plot_volume_chart(axes[1], df)
            
            # RSI chart
            self._plot_rsi_chart(axes[2], df)
            
            # MACD chart
            self._plot_macd_chart(axes[3], df)
            
            # Format all axes
            for ax in axes:
                ax.set_facecolor(self.colors['background'])
                ax.grid(True, alpha=0.3, color=self.colors['grid'])
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color(self.colors['grid'])
            
            # Format x-axis for time
            self._format_time_axis(axes[-1], df)
            
            plt.tight_layout()
            
            # Embed in tkinter if parent frame provided
            if parent_frame:
                # Clear previous chart
                for widget in parent_frame.winfo_children():
                    widget.destroy()
                
                canvas = FigureCanvasTkAgg(fig, parent_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                
                # Add toolbar
                toolbar_frame = ttk.Frame(parent_frame)
                toolbar_frame.pack(fill='x')
                
                # Navigation toolbar would go here
                # For now, just add some basic controls
                ttk.Button(toolbar_frame, text="Zoom In", 
                          command=lambda: self._zoom_chart(canvas, 'in')).pack(side='left', padx=2)
                ttk.Button(toolbar_frame, text="Zoom Out", 
                          command=lambda: self._zoom_chart(canvas, 'out')).pack(side='left', padx=2)
                ttk.Button(toolbar_frame, text="Reset View", 
                          command=lambda: self._reset_chart_view(canvas)).pack(side='left', padx=2)
                
                self.current_figure = fig
                self.current_canvas = canvas
                
                return canvas
            
            return fig
            
        except Exception as e:
            raise Exception(f"Failed to create chart: {str(e)}")
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convert timeframe string to days"""
        timeframe = timeframe.lower()
        if timeframe == '1h':
            return 1
        elif timeframe == '24h':
            return 1
        elif timeframe == '7d':
            return 7
        elif timeframe == '30d':
            return 30
        elif timeframe == '90d':
            return 90
        elif timeframe == '1y':
            return 365
        else:
            return 30
    
    def _prepare_dataframe(self, data: Dict) -> pd.DataFrame:
        """Prepare DataFrame from API data"""
        prices = data['prices']
        volumes = data.get('total_volumes', [])
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        if volumes:
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['datetime'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('datetime', inplace=True)
            df['volume'] = volume_df['volume']
        else:
            df['volume'] = 0
        
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['ma_20'] = df['price'].rolling(window=20).mean()
        df['ma_50'] = df['price'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['price'])
        
        # MACD
        macd_data = self._calculate_macd(df['price'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast_window: int = 12, 
                       slow_window: int = 26, 
                       signal_window: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast_window).mean()
        ema_slow = prices.ewm(span=slow_window).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_window).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def _plot_price_chart(self, ax, df: pd.DataFrame, symbol: str):
        """Plot price chart with moving averages and Bollinger Bands"""
        # Bollinger Bands
        ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                       alpha=0.2, color=self.colors['bollinger'], label='Bollinger Bands')
        
        # Price line
        ax.plot(df.index, df['price'], color=self.colors['price'], 
               linewidth=2, label=f'{symbol.upper()} Price')
        
        # Moving averages
        if not df['ma_20'].isna().all():
            ax.plot(df.index, df['ma_20'], color=self.colors['ma_short'], 
                   linewidth=1, label='MA 20', alpha=0.8)
        
        if not df['ma_50'].isna().all():
            ax.plot(df.index, df['ma_50'], color=self.colors['ma_long'], 
                   linewidth=1, label='MA 50', alpha=0.8)
        
        ax.set_ylabel('Price (USD)', color='white')
        ax.legend(loc='upper left')
        ax.set_title('Price Chart with Technical Indicators', color='white')
    
    def _plot_volume_chart(self, ax, df: pd.DataFrame):
        """Plot volume chart"""
        if 'volume' in df.columns and not df['volume'].isna().all():
            ax.bar(df.index, df['volume'], color=self.colors['volume'], 
                  alpha=0.7, width=0.8)
            ax.set_ylabel('Volume', color='white')
            ax.set_title('Trading Volume', color='white')
        else:
            ax.text(0.5, 0.5, 'Volume data not available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=12)
            ax.set_ylabel('Volume', color='white')
    
    def _plot_rsi_chart(self, ax, df: pd.DataFrame):
        """Plot RSI chart"""
        if not df['rsi'].isna().all():
            ax.plot(df.index, df['rsi'], color=self.colors['rsi'], linewidth=2)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax.axhline(y=50, color='white', linestyle='-', alpha=0.3)
            ax.set_ylabel('RSI', color='white')
            ax.set_ylim(0, 100)
            ax.legend(loc='upper left')
            ax.set_title('RSI (Relative Strength Index)', color='white')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for RSI calculation', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=12)
            ax.set_ylabel('RSI', color='white')
    
    def _plot_macd_chart(self, ax, df: pd.DataFrame):
        """Plot MACD chart"""
        if not df['macd'].isna().all():
            ax.plot(df.index, df['macd'], color=self.colors['macd'], 
                   linewidth=2, label='MACD')
            ax.plot(df.index, df['macd_signal'], color='orange', 
                   linewidth=1, label='Signal')
            
            # Histogram
            colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
            ax.bar(df.index, df['macd_histogram'], color=colors, 
                  alpha=0.7, width=0.8, label='Histogram')
            
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            ax.set_ylabel('MACD', color='white')
            ax.legend(loc='upper left')
            ax.set_title('MACD (Moving Average Convergence Divergence)', color='white')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for MACD calculation', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=12)
            ax.set_ylabel('MACD', color='white')
    
    def _format_time_axis(self, ax, df: pd.DataFrame):
        """Format time axis based on data timespan"""
        time_span = df.index[-1] - df.index[0]
        
        if time_span.days <= 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif time_span.days <= 7:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        elif time_span.days <= 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        ax.set_xlabel('Time', color='white')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def create_portfolio_chart(self, holdings: Dict, current_prices: Dict) -> plt.Figure:
        """Create portfolio distribution pie chart"""
        if not holdings:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['background'])
            ax.text(0.5, 0.5, 'No portfolio data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
        
        # Calculate portfolio values
        symbols = []
        values = []
        colors = []
        
        color_cycle = plt.cm.Set3(np.linspace(0, 1, len(holdings)))
        
        for i, (symbol, data) in enumerate(holdings.items()):
            current_price = current_prices.get(symbol, data['avg_price'])
            value = data['amount'] * current_price
            
            symbols.append(f"{symbol}\n${value:.2f}")
            values.append(value)
            colors.append(color_cycle[i])
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                      facecolor=self.colors['background'])
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(values, labels=symbols, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        ax1.set_title('Portfolio Distribution', color='white', fontsize=16, fontweight='bold')
        
        # Make text white
        for text in texts + autotexts:
            text.set_color('white')
        
        # Portfolio performance bar chart
        symbols_clean = list(holdings.keys())
        pnl_values = []
        bar_colors = []
        
        for symbol in symbols_clean:
            data = holdings[symbol]
            current_price = current_prices.get(symbol, data['avg_price'])
            value = data['amount'] * current_price
            cost = data['amount'] * data['avg_price']
            pnl = value - cost
            pnl_percent = (pnl / cost * 100) if cost > 0 else 0
            
            pnl_values.append(pnl_percent)
            bar_colors.append('green' if pnl_percent >= 0 else 'red')
        
        bars = ax2.bar(symbols_clean, pnl_values, color=bar_colors, alpha=0.7)
        ax2.set_title('Portfolio Performance (%)', color='white', fontsize=16, fontweight='bold')
        ax2.set_ylabel('P&L (%)', color='white')
        ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax2.tick_params(colors='white')
        ax2.set_facecolor(self.colors['background'])
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, pnl_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    color='white', fontweight='bold')
        
        # Set background colors
        ax1.set_facecolor(self.colors['background'])
        ax2.set_facecolor(self.colors['background'])
        
        plt.tight_layout()
        return fig
    
    def create_comparison_chart(self, symbols: List[str], timeframe: str = '30d') -> plt.Figure:
        """Create comparison chart for multiple cryptocurrencies"""
        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors['background'])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
        days = self._timeframe_to_days(timeframe)
        
        for i, symbol in enumerate(symbols):
            try:
                coin_id = self.api.symbol_to_id(symbol)
                if not coin_id:
                    continue
                
                data = self.api.get_price_history(coin_id, days)
                if not data or 'prices' not in data:
                    continue
                
                df = self._prepare_dataframe(data)
                
                # Normalize prices to show percentage change
                normalized_prices = (df['price'] / df['price'].iloc[0] - 1) * 100
                
                ax.plot(df.index, normalized_prices, color=colors[i], 
                       linewidth=2, label=f'{symbol.upper()}', alpha=0.8)
                
            except Exception as e:
                print(f"Error plotting {symbol}: {e}")
                continue
        
        ax.set_title(f'Cryptocurrency Comparison ({timeframe})', 
                    color='white', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price Change (%)', color='white')
        ax.set_xlabel('Time', color='white')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend()
        ax.set_facecolor(self.colors['background'])
        ax.tick_params(colors='white')
        
        # Format axes
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
        
        self._format_time_axis(ax, df if 'df' in locals() else None)
        
        plt.tight_layout()
        return fig
    
    def save_chart(self, filename: str):
        """Save current chart to file"""
        if self.current_figure:
            self.current_figure.savefig(filename, dpi=300, bbox_inches='tight',
                                       facecolor=self.colors['background'])
        else:
            raise Exception("No chart available to save")
    
    def _zoom_chart(self, canvas, direction):
        """Zoom chart in or out"""
        # Basic zoom functionality
        if self.current_figure:
            ax = self.current_figure.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            if direction == 'in':
                factor = 0.8
            else:
                factor = 1.2
            
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * factor / 2
            y_range = (ylim[1] - ylim[0]) * factor / 2
            
            ax.set_xlim(x_center - x_range, x_center + x_range)
            ax.set_ylim(y_center - y_range, y_center + y_range)
            canvas.draw()
    
    def _reset_chart_view(self, canvas):
        """Reset chart view to original"""
        if self.current_figure:
            for ax in self.current_figure.axes:
                ax.relim()
                ax.autoscale()
            canvas.draw()
    
    def create_market_overview_chart(self, market_data: List[Dict]) -> plt.Figure:
        """Create market overview chart showing top cryptocurrencies"""
        if not market_data:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=self.colors['background'])
            ax.text(0.5, 0.5, 'No market data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
        
        # Prepare data for top 20 cryptocurrencies
        top_20 = market_data[:20]
        symbols = [item['symbol'].upper() for item in top_20]
        market_caps = [item.get('market_cap', 0) for item in top_20]
        changes_24h = [item.get('price_change_percentage_24h', 0) for item in top_20]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      facecolor=self.colors['background'])
        
        # Market cap bar chart
        colors_mc = ['green' if change >= 0 else 'red' for change in changes_24h]
        bars1 = ax1.barh(symbols, market_caps, color=colors_mc, alpha=0.7)
        
        ax1.set_title('Top 20 Cryptocurrencies by Market Cap', 
                     color='white', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Market Cap (USD)', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor(self.colors['background'])
        
        # Format market cap values
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
        
        # 24h change chart
        colors_change = ['green' if change >= 0 else 'red' for change in changes_24h]
        bars2 = ax2.barh(symbols, changes_24h, color=colors_change, alpha=0.7)
        
        ax2.set_title('24h Price Change (%)', color='white', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Change (%)', color='white')
        ax2.axvline(x=0, color='white', linestyle='-', alpha=0.3)
        ax2.tick_params(colors='white')
        ax2.set_facecolor(self.colors['background'])
        
        # Add value labels
        for bar, value in zip(bars2, changes_24h):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left' if width >= 0 else 'right',
                    va='center', color='white', fontweight='bold')
        
        # Set grid and spines
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, symbols: List[str], days: int = 30) -> plt.Figure:
        """Create correlation heatmap for multiple cryptocurrencies"""
        if len(symbols) < 2:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['background'])
            ax.text(0.5, 0.5, 'Need at least 2 symbols for correlation analysis', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
        
        # Collect price data
        price_data = {}
        
        for symbol in symbols:
            try:
                coin_id = self.api.symbol_to_id(symbol)
                if not coin_id:
                    continue
                
                data = self.api.get_price_history(coin_id, days)
                if not data or 'prices' not in data:
                    continue
                
                df = self._prepare_dataframe(data)
                price_data[symbol.upper()] = df['price']
                
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                continue
        
        if len(price_data) < 2:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['background'])
            ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(price_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['background'])
        
        im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, color='white')
        ax.set_yticklabels(correlation_matrix.index, color='white')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values to cells
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'Cryptocurrency Correlation Matrix ({days} days)', 
                    color='white', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Correlation Coefficient', color='white')
        
        ax.set_facecolor(self.colors['background'])
        plt.tight_layout()
        return fig
    
    def create_volatility_chart(self, symbols: List[str], days: int = 30) -> plt.Figure:
        """Create volatility comparison chart"""
        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors['background'])
        
        volatilities = []
        symbol_names = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(symbols)))
        
        for symbol in symbols:
            try:
                coin_id = self.api.symbol_to_id(symbol)
                if not coin_id:
                    continue
                
                data = self.api.get_price_history(coin_id, days)
                if not data or 'prices' not in data:
                    continue
                
                df = self._prepare_dataframe(data)
                
                # Calculate daily returns
                returns = df['price'].pct_change().dropna()
                
                # Calculate volatility (standard deviation of returns)
                volatility = returns.std() * np.sqrt(365) * 100  # Annualized volatility
                
                volatilities.append(volatility)
                symbol_names.append(symbol.upper())
                
            except Exception as e:
                print(f"Error calculating volatility for {symbol}: {e}")
                continue
        
        if not volatilities:
            ax.text(0.5, 0.5, 'No volatility data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
        
        # Create bar chart
        bars = ax.bar(symbol_names, volatilities, color=colors, alpha=0.7)
        
        ax.set_title(f'Cryptocurrency Volatility Comparison ({days} days)', 
                    color='white', fontsize=16, fontweight='bold')
        ax.set_ylabel('Annualized Volatility (%)', color='white')
        ax.set_xlabel('Cryptocurrency', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor(self.colors['background'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Add value labels on bars
        for bar, value in zip(bars, volatilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom',
                    color='white', fontweight='bold')
        
        # Rotate x-axis labels if needed
        if len(symbol_names) > 10:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Set grid and spines
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
        
        plt.tight_layout()
        return fig
    
    def create_moving_average_signals_chart(self, symbol: str, days: int = 90) -> plt.Figure:
        """Create chart showing moving average crossover signals"""
        try:
            coin_id = self.api.symbol_to_id(symbol)
            if not coin_id:
                raise ValueError(f"Could not find coin ID for symbol: {symbol}")
            
            data = self.api.get_price_history(coin_id, days)
            if not data or 'prices' not in data:
                raise ValueError("No price data available")
            
            df = self._prepare_dataframe(data)
            
            # Calculate multiple moving averages
            df['ma_10'] = df['price'].rolling(window=10).mean()
            df['ma_20'] = df['price'].rolling(window=20).mean()
            df['ma_50'] = df['price'].rolling(window=50).mean()
            
            # Find crossover points
            df['ma_10_above_20'] = df['ma_10'] > df['ma_20']
            df['ma_20_above_50'] = df['ma_20'] > df['ma_50']
            
            # Find crossover signals
            buy_signals = []
            sell_signals = []
            
            for i in range(1, len(df)):
                # Golden cross (MA10 crosses above MA20)
                if not df['ma_10_above_20'].iloc[i-1] and df['ma_10_above_20'].iloc[i]:
                    buy_signals.append((df.index[i], df['price'].iloc[i]))
                
                # Death cross (MA10 crosses below MA20)
                elif df['ma_10_above_20'].iloc[i-1] and not df['ma_10_above_20'].iloc[i]:
                    sell_signals.append((df.index[i], df['price'].iloc[i]))
            
            # Create chart
            fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors['background'])
            
            # Plot price and moving averages
            ax.plot(df.index, df['price'], color=self.colors['price'], 
                   linewidth=2, label=f'{symbol.upper()} Price')
            ax.plot(df.index, df['ma_10'], color='yellow', 
                   linewidth=1, label='MA 10', alpha=0.8)
            ax.plot(df.index, df['ma_20'], color='orange', 
                   linewidth=1, label='MA 20', alpha=0.8)
            ax.plot(df.index, df['ma_50'], color='purple', 
                   linewidth=1, label='MA 50', alpha=0.8)
            
            # Plot signals
            if buy_signals:
                buy_dates, buy_prices = zip(*buy_signals)
                ax.scatter(buy_dates, buy_prices, color='green', 
                          marker='^', s=100, label='Buy Signal', zorder=5)
            
            if sell_signals:
                sell_dates, sell_prices = zip(*sell_signals)
                ax.scatter(sell_dates, sell_prices, color='red', 
                          marker='v', s=100, label='Sell Signal', zorder=5)
            
            ax.set_title(f'{symbol.upper()} - Moving Average Crossover Signals', 
                        color='white', fontsize=16, fontweight='bold')
            ax.set_ylabel('Price (USD)', color='white')
            ax.set_xlabel('Time', color='white')
            ax.legend()
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_facecolor(self.colors['background'])
            ax.tick_params(colors='white')
            
            # Format axes
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
            
            self._format_time_axis(ax, df)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors['background'])
            ax.text(0.5, 0.5, f'Error creating signals chart: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=16)
            ax.set_facecolor(self.colors['background'])
            return fig
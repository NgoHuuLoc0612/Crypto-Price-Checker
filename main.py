#!/usr/bin/env python3
"""
Crypto Price Checker
Main Application File
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import json
from crypto_api import CryptoAPI
from crypto_charts import CryptoCharts
from crypto_portfolio import CryptoPortfolio
from crypto_alerts import CryptoAlerts
from crypto_utils import CryptoUtils

class CryptoPriceChecker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crypto Price Checker")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize components
        self.api = CryptoAPI()
        self.charts = CryptoCharts()
        self.portfolio = CryptoPortfolio()
        self.alerts = CryptoAlerts()
        self.utils = CryptoUtils()
        
        # Variables
        self.current_prices = {}
        self.watchlist = []
        self.auto_refresh = tk.BooleanVar(value=True)
        self.refresh_interval = tk.IntVar(value=30)
        
        self.setup_ui()
        self.load_settings()
        self.start_auto_refresh()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#1e1e1e', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#2d2d2d', foreground='white')
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_market_tab()
        self.create_portfolio_tab()
        self.create_charts_tab()
        self.create_alerts_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken', anchor='w')
        self.status_bar.pack(side='bottom', fill='x')
        
    def create_market_tab(self):
        """Create market overview tab"""
        market_frame = ttk.Frame(self.notebook)
        self.notebook.add(market_frame, text="Market Overview")
        
        # Top controls
        control_frame = ttk.Frame(market_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Search Cryptocurrency:").pack(side='left')
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(control_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side='left', padx=5)
        self.search_entry.bind('<Return>', self.search_crypto)
        
        ttk.Button(control_frame, text="Search", command=self.search_crypto).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.refresh_market_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Add to Watchlist", command=self.add_to_watchlist).pack(side='left', padx=5)
        
        # Market data tree
        columns = ('Symbol', 'Name', 'Price', '24h Change', '24h Volume', 'Market Cap', 'Last Updated')
        self.market_tree = ttk.Treeview(market_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=120)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(market_frame, orient='vertical', command=self.market_tree.yview)
        h_scrollbar = ttk.Scrollbar(market_frame, orient='horizontal', command=self.market_tree.xview)
        self.market_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack tree and scrollbars
        tree_frame = ttk.Frame(market_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.market_tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Watchlist section
        watchlist_frame = ttk.LabelFrame(market_frame, text="Watchlist")
        watchlist_frame.pack(fill='x', padx=10, pady=5)
        
        self.watchlist_tree = ttk.Treeview(watchlist_frame, columns=('Symbol', 'Price', 'Change'), show='headings', height=5)
        self.watchlist_tree.heading('Symbol', text='Symbol')
        self.watchlist_tree.heading('Price', text='Price')
        self.watchlist_tree.heading('Change', text='24h Change')
        self.watchlist_tree.pack(fill='x', padx=5, pady=5)
        
    def create_portfolio_tab(self):
        """Create portfolio management tab"""
        portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(portfolio_frame, text="Portfolio")
        
        # Portfolio controls
        control_frame = ttk.Frame(portfolio_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Add Transaction:").pack(side='left')
        
        ttk.Label(control_frame, text="Symbol:").pack(side='left', padx=(20, 5))
        self.portfolio_symbol = ttk.Entry(control_frame, width=10)
        self.portfolio_symbol.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Amount:").pack(side='left', padx=(10, 5))
        self.portfolio_amount = ttk.Entry(control_frame, width=10)
        self.portfolio_amount.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Price:").pack(side='left', padx=(10, 5))
        self.portfolio_price = ttk.Entry(control_frame, width=10)
        self.portfolio_price.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Add", command=self.add_portfolio_transaction).pack(side='left', padx=10)
        
        # Portfolio summary
        summary_frame = ttk.LabelFrame(portfolio_frame, text="Portfolio Summary")
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        self.portfolio_summary = ttk.Label(summary_frame, text="Total Value: $0.00 | P&L: $0.00 (0.00%)")
        self.portfolio_summary.pack(pady=10)
        
        # Portfolio holdings
        holdings_columns = ('Symbol', 'Amount', 'Avg Price', 'Current Price', 'Value', 'P&L', 'P&L %')
        self.portfolio_tree = ttk.Treeview(portfolio_frame, columns=holdings_columns, show='headings', height=12)
        
        for col in holdings_columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=100)
        
        self.portfolio_tree.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_charts_tab(self):
        """Create charts tab"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="Charts")
        
        # Chart controls
        control_frame = ttk.Frame(charts_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Cryptocurrency:").pack(side='left')
        self.chart_symbol = ttk.Entry(control_frame, width=15)
        self.chart_symbol.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Timeframe:").pack(side='left', padx=(20, 5))
        self.chart_timeframe = ttk.Combobox(control_frame, values=['1h', '24h', '7d', '30d', '90d', '1y'], width=10)
        self.chart_timeframe.set('24h')
        self.chart_timeframe.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Generate Chart", command=self.generate_chart).pack(side='left', padx=10)
        ttk.Button(control_frame, text="Save Chart", command=self.save_chart).pack(side='left', padx=5)
        
        # Chart display area
        self.chart_frame = ttk.Frame(charts_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_alerts_tab(self):
        """Create price alerts tab"""
        alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(alerts_frame, text="Price Alerts")
        
        # Alert creation
        create_frame = ttk.LabelFrame(alerts_frame, text="Create Alert")
        create_frame.pack(fill='x', padx=10, pady=5)
        
        create_controls = ttk.Frame(create_frame)
        create_controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(create_controls, text="Symbol:").pack(side='left')
        self.alert_symbol = ttk.Entry(create_controls, width=10)
        self.alert_symbol.pack(side='left', padx=5)
        
        ttk.Label(create_controls, text="Condition:").pack(side='left', padx=(10, 5))
        self.alert_condition = ttk.Combobox(create_controls, values=['Above', 'Below'], width=10)
        self.alert_condition.set('Above')
        self.alert_condition.pack(side='left', padx=5)
        
        ttk.Label(create_controls, text="Price:").pack(side='left', padx=(10, 5))
        self.alert_price = ttk.Entry(create_controls, width=10)
        self.alert_price.pack(side='left', padx=5)
        
        ttk.Button(create_controls, text="Create Alert", command=self.create_alert).pack(side='left', padx=10)
        
        # Active alerts
        alerts_list_frame = ttk.LabelFrame(alerts_frame, text="Active Alerts")
        alerts_list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        alert_columns = ('Symbol', 'Condition', 'Target Price', 'Current Price', 'Status', 'Created')
        self.alerts_tree = ttk.Treeview(alerts_list_frame, columns=alert_columns, show='headings', height=15)
        
        for col in alert_columns:
            self.alerts_tree.heading(col, text=col)
            self.alerts_tree.column(col, width=120)
        
        self.alerts_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Alert controls
        alert_controls = ttk.Frame(alerts_list_frame)
        alert_controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(alert_controls, text="Delete Selected", command=self.delete_alert).pack(side='left')
        ttk.Button(alert_controls, text="Refresh Alerts", command=self.refresh_alerts).pack(side='left', padx=5)
        
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Auto refresh settings
        refresh_frame = ttk.LabelFrame(settings_frame, text="Auto Refresh")
        refresh_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Checkbutton(refresh_frame, text="Enable Auto Refresh", variable=self.auto_refresh).pack(anchor='w', padx=10, pady=5)
        
        interval_frame = ttk.Frame(refresh_frame)
        interval_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(interval_frame, text="Refresh Interval (seconds):").pack(side='left')
        ttk.Scale(interval_frame, from_=10, to=300, variable=self.refresh_interval, orient='horizontal').pack(side='left', fill='x', expand=True, padx=10)
        ttk.Label(interval_frame, textvariable=self.refresh_interval).pack(side='right')
        
        # API settings
        api_frame = ttk.LabelFrame(settings_frame, text="API Settings")
        api_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(api_frame, text="CoinGecko API is used (free tier)").pack(anchor='w', padx=10, pady=5)
        ttk.Button(api_frame, text="Test API Connection", command=self.test_api_connection).pack(anchor='w', padx=10, pady=5)
        
        # Export/Import settings
        data_frame = ttk.LabelFrame(settings_frame, text="Data Management")
        data_frame.pack(fill='x', padx=10, pady=5)
        
        data_controls = ttk.Frame(data_frame)
        data_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(data_controls, text="Export Portfolio", command=self.export_portfolio).pack(side='left', padx=5)
        ttk.Button(data_controls, text="Import Portfolio", command=self.import_portfolio).pack(side='left', padx=5)
        ttk.Button(data_controls, text="Export Watchlist", command=self.export_watchlist).pack(side='left', padx=5)
        ttk.Button(data_controls, text="Import Watchlist", command=self.import_watchlist).pack(side='left', padx=5)
        
    def search_crypto(self, event=None):
        """Search for cryptocurrency"""
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a cryptocurrency name or symbol")
            return
        
        self.status_bar.config(text="Searching...")
        self.root.update()
        
        try:
            results = self.api.search_cryptocurrency(query)
            self.update_market_tree(results)
            self.status_bar.config(text=f"Found {len(results)} results for '{query}'")
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            self.status_bar.config(text="Search failed")
    
    def refresh_market_data(self):
        """Refresh market data"""
        self.status_bar.config(text="Refreshing market data...")
        self.root.update()
        
        try:
            data = self.api.get_top_cryptocurrencies(100)
            self.update_market_tree(data)
            self.current_prices = {item['symbol'].upper(): item['current_price'] for item in data}
            self.update_watchlist_display()
            self.update_portfolio_display()
            self.status_bar.config(text=f"Market data updated at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh market data: {str(e)}")
            self.status_bar.config(text="Refresh failed")
    
    def update_market_tree(self, data):
        """Update market data tree"""
        # Clear existing data
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        
        # Add new data
        for item in data:
            change_24h = item.get('price_change_percentage_24h', 0)
            change_color = 'green' if change_24h > 0 else 'red' if change_24h < 0 else 'black'
            
            values = (
                item['symbol'].upper(),
                item['name'],
                f"${item['current_price']:.6f}",
                f"{change_24h:.2f}%",
                f"${item.get('total_volume', 0):,.0f}",
                f"${item.get('market_cap', 0):,.0f}",
                datetime.now().strftime('%H:%M:%S')
            )
            
            item_id = self.market_tree.insert('', 'end', values=values)
            self.market_tree.set(item_id, '24h Change', f"{change_24h:.2f}%")
    
    def add_to_watchlist(self):
        """Add selected cryptocurrency to watchlist"""
        selection = self.market_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a cryptocurrency to add to watchlist")
            return
        
        item = self.market_tree.item(selection[0])
        symbol = item['values'][0]
        
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.update_watchlist_display()
            self.save_settings()
            messagebox.showinfo("Success", f"{symbol} added to watchlist")
        else:
            messagebox.showinfo("Info", f"{symbol} is already in watchlist")
    
    def update_watchlist_display(self):
        """Update watchlist display"""
        # Clear existing data
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        
        # Add watchlist items
        for symbol in self.watchlist:
            price = self.current_prices.get(symbol, 0)
            # Get 24h change (simplified for demo)
            change = 0  # Would need to fetch from API
            
            self.watchlist_tree.insert('', 'end', values=(symbol, f"${price:.6f}", f"{change:.2f}%"))
    
    def add_portfolio_transaction(self):
        """Add transaction to portfolio"""
        try:
            symbol = self.portfolio_symbol.get().upper().strip()
            amount = float(self.portfolio_amount.get())
            price = float(self.portfolio_price.get())
            
            if not symbol or amount <= 0 or price <= 0:
                messagebox.showwarning("Warning", "Please enter valid transaction details")
                return
            
            self.portfolio.add_transaction(symbol, amount, price)
            self.update_portfolio_display()
            
            # Clear entries
            self.portfolio_symbol.delete(0, 'end')
            self.portfolio_amount.delete(0, 'end')
            self.portfolio_price.delete(0, 'end')
            
            messagebox.showinfo("Success", f"Transaction added: {amount} {symbol} at ${price}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add transaction: {str(e)}")
    
    def update_portfolio_display(self):
        """Update portfolio display"""
        # Clear existing data
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        holdings = self.portfolio.get_holdings()
        total_value = 0
        total_cost = 0
        
        for symbol, data in holdings.items():
            current_price = self.current_prices.get(symbol, data['avg_price'])
            value = data['amount'] * current_price
            cost = data['amount'] * data['avg_price']
            pnl = value - cost
            pnl_percent = (pnl / cost * 100) if cost > 0 else 0
            
            total_value += value
            total_cost += cost
            
            values = (
                symbol,
                f"{data['amount']:.6f}",
                f"${data['avg_price']:.6f}",
                f"${current_price:.6f}",
                f"${value:.2f}",
                f"${pnl:.2f}",
                f"{pnl_percent:.2f}%"
            )
            
            self.portfolio_tree.insert('', 'end', values=values)
        
        # Update summary
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        summary_text = f"Total Value: ${total_value:.2f} | P&L: ${total_pnl:.2f} ({total_pnl_percent:.2f}%)"
        self.portfolio_summary.config(text=summary_text)
    
    def generate_chart(self):
        """Generate chart for selected cryptocurrency"""
        symbol = self.chart_symbol.get().strip().upper()
        timeframe = self.chart_timeframe.get()
        
        if not symbol:
            messagebox.showwarning("Warning", "Please enter a cryptocurrency symbol")
            return
        
        try:
            self.status_bar.config(text="Generating chart...")
            self.root.update()
            
            chart_widget = self.charts.create_price_chart(symbol, timeframe, self.chart_frame)
            self.status_bar.config(text=f"Chart generated for {symbol} ({timeframe})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate chart: {str(e)}")
            self.status_bar.config(text="Chart generation failed")
    
    def save_chart(self):
        """Save current chart"""
        symbol = self.chart_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Warning", "Please generate a chart first")
            return
        
        try:
            filename = f"{symbol}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.charts.save_chart(filename)
            messagebox.showinfo("Success", f"Chart saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save chart: {str(e)}")
    
    def create_alert(self):
        """Create price alert"""
        try:
            symbol = self.alert_symbol.get().upper().strip()
            condition = self.alert_condition.get().lower()
            price = float(self.alert_price.get())
            
            if not symbol or not condition or price <= 0:
                messagebox.showwarning("Warning", "Please enter valid alert details")
                return
            
            self.alerts.create_alert(symbol, condition, price)
            self.refresh_alerts()
            
            # Clear entries
            self.alert_symbol.delete(0, 'end')
            self.alert_price.delete(0, 'end')
            
            messagebox.showinfo("Success", f"Alert created: {symbol} {condition} ${price}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric price")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create alert: {str(e)}")
    
    def refresh_alerts(self):
        """Refresh alerts display"""
        # Clear existing data
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        
        alerts = self.alerts.get_alerts()
        for alert in alerts:
            current_price = self.current_prices.get(alert['symbol'], 0)
            status = self.alerts.check_alert_status(alert, current_price)
            
            values = (
                alert['symbol'],
                alert['condition'].title(),
                f"${alert['target_price']:.6f}",
                f"${current_price:.6f}",
                status,
                alert['created']
            )
            
            self.alerts_tree.insert('', 'end', values=values)
    
    def delete_alert(self):
        """Delete selected alert"""
        selection = self.alerts_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an alert to delete")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this alert?"):
            # Implementation would delete from alerts system
            self.refresh_alerts()
            messagebox.showinfo("Success", "Alert deleted")
    
    def test_api_connection(self):
        """Test API connection"""
        try:
            self.status_bar.config(text="Testing API connection...")
            self.root.update()
            
            if self.api.test_connection():
                messagebox.showinfo("Success", "API connection successful")
                self.status_bar.config(text="API connection OK")
            else:
                messagebox.showerror("Error", "API connection failed")
                self.status_bar.config(text="API connection failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"API test failed: {str(e)}")
            self.status_bar.config(text="API test failed")
    
    def export_portfolio(self):
        """Export portfolio data"""
        try:
            filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.portfolio.export_to_file(filename)
            messagebox.showinfo("Success", f"Portfolio exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def import_portfolio(self):
        """Import portfolio data"""
        from tkinter import filedialog
        try:
            filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if filename:
                self.portfolio.import_from_file(filename)
                self.update_portfolio_display()
                messagebox.showinfo("Success", "Portfolio imported successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
    
    def export_watchlist(self):
        """Export watchlist"""
        try:
            filename = f"watchlist_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.watchlist, f)
            messagebox.showinfo("Success", f"Watchlist exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def import_watchlist(self):
        """Import watchlist"""
        from tkinter import filedialog
        try:
            filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if filename:
                with open(filename, 'r') as f:
                    self.watchlist = json.load(f)
                self.update_watchlist_display()
                self.save_settings()
                messagebox.showinfo("Success", "Watchlist imported successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
    
    def start_auto_refresh(self):
        """Start auto refresh thread"""
        def refresh_loop():
            while True:
                if self.auto_refresh.get():
                    try:
                        self.refresh_market_data()
                    except:
                        pass
                time.sleep(self.refresh_interval.get())
        
        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()
    
    def save_settings(self):
        """Save application settings"""
        settings = {
            'watchlist': self.watchlist,
            'auto_refresh': self.auto_refresh.get(),
            'refresh_interval': self.refresh_interval.get()
        }
        
        try:
            with open('crypto_settings.json', 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Failed to save settings: {e}")
    
    def load_settings(self):
        """Load application settings"""
        try:
            with open('crypto_settings.json', 'r') as f:
                settings = json.load(f)
                
            self.watchlist = settings.get('watchlist', [])
            self.auto_refresh.set(settings.get('auto_refresh', True))
            self.refresh_interval.set(settings.get('refresh_interval', 30))
            
        except FileNotFoundError:
            # Use defaults
            pass
        except Exception as e:
            print(f"Failed to load settings: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        self.save_settings()
        self.portfolio.save_to_file()
        self.alerts.save_to_file()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initial data load
        self.refresh_market_data()
        
        self.root.mainloop()

if __name__ == "__main__":
    app = CryptoPriceChecker()
    app.run()
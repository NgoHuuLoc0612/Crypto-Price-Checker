"""
Crypto Price Alerts System
Manages price alerts, notifications, and monitoring
"""

import json
import os
import smtplib
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tkinter as tk
from tkinter import messagebox
import plyer

@dataclass
class PriceAlert:
    """Represents a price alert"""
    id: str
    symbol: str
    condition: str  # 'above', 'below', 'crosses_above', 'crosses_below'
    target_price: float
    current_price: float
    status: str  # 'active', 'triggered', 'disabled'
    created: str
    triggered: Optional[str] = None
    message: str = ""
    email_notification: bool = False
    desktop_notification: bool = True
    repeat: bool = False
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class CryptoAlerts:
    def __init__(self, filename: str = "crypto_alerts.json"):
        self.filename = filename
        self.alerts: List[PriceAlert] = []
        self.monitoring = False
        self.monitor_thread = None
        self.price_history = {}  # Track price changes for crossover alerts
        self.notification_callbacks = []
        self.email_config = {}
        
        self.load_from_file()
        self.start_monitoring()
    
    def create_alert(self, symbol: str, condition: str, target_price: float,
                    message: str = "", email_notification: bool = False,
                    desktop_notification: bool = True, repeat: bool = False) -> str:
        """Create a new price alert"""
        try:
            symbol = symbol.upper().strip()
            condition = condition.lower().strip()
            
            if condition not in ['above', 'below', 'crosses_above', 'crosses_below']:
                raise ValueError("Invalid condition. Must be 'above', 'below', 'crosses_above', or 'crosses_below'")
            
            if target_price <= 0:
                raise ValueError("Target price must be positive")
            
            # Generate unique ID
            alert_id = f"{symbol}_{condition}_{target_price}_{int(time.time())}"
            
            alert = PriceAlert(
                id=alert_id,
                symbol=symbol,
                condition=condition,
                target_price=target_price,
                current_price=0.0,
                status='active',
                created=datetime.now().isoformat(),
                message=message,
                email_notification=email_notification,
                desktop_notification=desktop_notification,
                repeat=repeat
            )
            
            self.alerts.append(alert)
            self.save_to_file()
            
            return alert_id
            
        except Exception as e:
            raise Exception(f"Failed to create alert: {str(e)}")
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert by ID"""
        try:
            self.alerts = [alert for alert in self.alerts if alert.id != alert_id]
            self.save_to_file()
            return True
        except Exception as e:
            raise Exception(f"Failed to delete alert: {str(e)}")
    
    def get_alerts(self, status: str = None) -> List[Dict]:
        """Get alerts, optionally filtered by status"""
        alerts = []
        for alert in self.alerts:
            if status is None or alert.status == status:
                alerts.append(alert.to_dict())
        
        return sorted(alerts, key=lambda x: x['created'], reverse=True)
    
    def update_alert_status(self, alert_id: str, status: str) -> bool:
        """Update alert status"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    if status in ['active', 'triggered', 'disabled']:
                        alert.status = status
                        if status == 'triggered':
                            alert.triggered = datetime.now().isoformat()
                        self.save_to_file()
                        return True
            return False
        except Exception as e:
            raise Exception(f"Failed to update alert status: {str(e)}")
    
    def check_alerts(self, current_prices: Dict[str, float]):
        """Check all active alerts against current prices"""
        for alert in self.alerts:
            if alert.status != 'active':
                continue
            
            symbol = alert.symbol
            current_price = current_prices.get(symbol, 0)
            
            if current_price == 0:
                continue
            
            # Update current price
            alert.current_price = current_price
            
            # Check if alert should trigger
            should_trigger = False
            
            if alert.condition == 'above':
                should_trigger = current_price > alert.target_price
            
            elif alert.condition == 'below':
                should_trigger = current_price < alert.target_price
            
            elif alert.condition == 'crosses_above':
                # Check if price crossed above target
                previous_price = self.price_history.get(symbol, current_price)
                should_trigger = (previous_price <= alert.target_price and 
                                current_price > alert.target_price)
            
            elif alert.condition == 'crosses_below':
                # Check if price crossed below target
                previous_price = self.price_history.get(symbol, current_price)
                should_trigger = (previous_price >= alert.target_price and 
                                current_price < alert.target_price)
            
            if should_trigger:
                self.trigger_alert(alert)
            
            # Update price history
            self.price_history[symbol] = current_price
        
        self.save_to_file()
    
    def trigger_alert(self, alert: PriceAlert):
        """Trigger an alert and send notifications"""
        try:
            # Update alert status
            if not alert.repeat:
                alert.status = 'triggered'
            alert.triggered = datetime.now().isoformat()
            
            # Create notification message
            message = self.format_alert_message(alert)
            
            # Send desktop notification
            if alert.desktop_notification:
                try:
                    plyer.notification.notify(
                        title=f"Crypto Alert: {alert.symbol}",
                        message=message,
                        timeout=10
                    )
                except:
                    # Fallback to tkinter messagebox
                    try:
                        messagebox.showinfo(f"Crypto Alert: {alert.symbol}", message)
                    except:
                        pass
            
            # Send email notification
            if alert.email_notification and self.email_config:
                self.send_email_notification(alert, message)
            
            # Call registered callbacks
            for callback in self.notification_callbacks:
                try:
                    callback(alert, message)
                except:
                    pass
            
            print(f"Alert triggered: {message}")
            
        except Exception as e:
            print(f"Error triggering alert: {e}")
    
    def format_alert_message(self, alert: PriceAlert) -> str:
        """Format alert message"""
        condition_text = {
            'above': 'is above',
            'below': 'is below',
            'crosses_above': 'crossed above',
            'crosses_below': 'crossed below'
        }
        
        message = f"{alert.symbol} {condition_text.get(alert.condition, 'met condition for')} ${alert.target_price:.6f}"
        message += f"\nCurrent price: ${alert.current_price:.6f}"
        
        if alert.message:
            message += f"\nNote: {alert.message}"
        
        message += f"\nTriggered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def send_email_notification(self, alert: PriceAlert, message: str):
        """Send email notification"""
        try:
            if not self.email_config.get('enabled', False):
                return
            
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            email = self.email_config.get('email')
            password = self.email_config.get('password')
            to_email = self.email_config.get('to_email', email)
            
            if not all([smtp_server, email, password]):
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = email
            msg['To'] = to_email
            msg['Subject'] = f"Crypto Price Alert: {alert.symbol}"
            
            # Add body
            body = f"""
Crypto Price Alert Triggered

Symbol: {alert.symbol}
Condition: {alert.condition}
Target Price: ${alert.target_price:.6f}
Current Price: ${alert.current_price:.6f}
Created: {alert.created}
Triggered: {alert.triggered}

{alert.message if alert.message else ''}

---
Crypto Price Checker Alert System
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email, password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")
    
    def configure_email(self, smtp_server: str, smtp_port: int, email: str, 
                       password: str, to_email: str = None, enabled: bool = True):
        """Configure email notifications"""
        self.email_config = {
            'enabled': enabled,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'email': email,
            'password': password,
            'to_email': to_email or email
        }
        
        # Save email config (without password for security)
        config_to_save = self.email_config.copy()
        config_to_save['password'] = '***'  # Don't save actual password
        
        try:
            with open('email_config.json', 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except:
            pass
    
    def test_email_config(self) -> bool:
        """Test email configuration"""
        try:
            if not self.email_config.get('enabled', False):
                return False
            
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            email = self.email_config.get('email')
            password = self.email_config.get('password')
            
            if not all([smtp_server, email, password]):
                return False
            
            # Test connection
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email, password)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email test failed: {e}")
            return False
    
    def add_notification_callback(self, callback: Callable):
        """Add custom notification callback"""
        self.notification_callbacks.append(callback)
    
    def remove_notification_callback(self, callback: Callable):
        """Remove notification callback"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
    
    def start_monitoring(self):
        """Start monitoring alerts in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring alerts"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        from crypto_api import CryptoAPI
        
        api = CryptoAPI()
        
        while self.monitoring:
            try:
                # Get unique symbols from active alerts
                active_symbols = set()
                for alert in self.alerts:
                    if alert.status == 'active':
                        active_symbols.add(alert.symbol)
                
                if active_symbols:
                    # Get current prices
                    current_prices = {}
                    
                    for symbol in active_symbols:
                        try:
                            coin_id = api.symbol_to_id(symbol)
                            if coin_id:
                                price_data = api.get_current_price([coin_id])
                                if price_data and coin_id in price_data:
                                    current_prices[symbol] = price_data[coin_id]['usd']
                        except:
                            continue
                    
                    # Check alerts
                    if current_prices:
                        self.check_alerts(current_prices)
                
                # Wait before next check (60 seconds)
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def check_alert_status(self, alert: Dict, current_price: float) -> str:
        """Check if alert condition is met"""
        condition = alert['condition']
        target_price = alert['target_price']
        
        if condition == 'above':
            return 'triggered' if current_price > target_price else 'active'
        elif condition == 'below':
            return 'triggered' if current_price < target_price else 'active'
        elif condition == 'crosses_above':
            # Would need price history to determine
            return 'active'
        elif condition == 'crosses_below':
            # Would need price history to determine
            return 'active'
        
        return 'active'
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        total_alerts = len(self.alerts)
        active_alerts = sum(1 for alert in self.alerts if alert.status == 'active')
        triggered_alerts = sum(1 for alert in self.alerts if alert.status == 'triggered')
        disabled_alerts = sum(1 for alert in self.alerts if alert.status == 'disabled')
        
        # Most watched symbols
        symbol_counts = {}
        for alert in self.alerts:
            symbol_counts[alert.symbol] = symbol_counts.get(alert.symbol, 0) + 1
        
        most_watched = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_alerts': triggered_alerts,
            'disabled_alerts': disabled_alerts,
            'most_watched_symbols': most_watched,
            'monitoring_status': self.monitoring
        }
    
    def cleanup_old_alerts(self, days: int = 30):
        """Remove old triggered alerts"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        original_count = len(self.alerts)
        
        self.alerts = [
            alert for alert in self.alerts
            if not (alert.status == 'triggered' and 
                   alert.triggered and 
                   datetime.fromisoformat(alert.triggered) < cutoff_date)
        ]
        
        removed_count = original_count - len(self.alerts)
        
        if removed_count > 0:
            self.save_to_file()
        
        return removed_count
    
    def export_alerts(self, filename: str = None) -> str:
        """Export alerts to file"""
        if filename is None:
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'alerts': [alert.to_dict() for alert in self.alerts],
            'exported_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def import_alerts(self, filename: str) -> int:
        """Import alerts from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for alert_data in data.get('alerts', []):
                try:
                    alert = PriceAlert.from_dict(alert_data)
                    # Generate new ID to avoid conflicts
                    alert.id = f"{alert.symbol}_{alert.condition}_{alert.target_price}_{int(time.time())}_{imported_count}"
                    self.alerts.append(alert)
                    imported_count += 1
                except Exception as e:
                    print(f"Error importing alert: {e}")
                    continue
            
            if imported_count > 0:
                self.save_to_file()
            
            return imported_count
            
        except Exception as e:
            raise Exception(f"Failed to import alerts: {str(e)}")
    
    def save_to_file(self):
        """Save alerts to file"""
        try:
            data = {
                'alerts': [alert.to_dict() for alert in self.alerts],
                'price_history': self.price_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving alerts: {e}")
    
    def load_from_file(self):
        """Load alerts from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                
                # Load alerts
                self.alerts = []
                for alert_data in data.get('alerts', []):
                    alert = PriceAlert.from_dict(alert_data)
                    self.alerts.append(alert)
                
                # Load price history
                self.price_history = data.get('price_history', {})
                
        except Exception as e:
            print(f"Error loading alerts: {e}")
            self.alerts = []
            self.price_history = {}
"""
Crypto Portfolio Manager
Handles portfolio tracking, transactions, and performance calculations
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class Transaction:
    """Represents a cryptocurrency transaction"""
    symbol: str
    amount: float
    price: float
    transaction_type: str  # 'buy' or 'sell'
    timestamp: str
    notes: str = ""
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class CryptoPortfolio:
    def __init__(self, filename: str = "crypto_portfolio.json"):
        self.filename = filename
        self.transactions: List[Transaction] = []
        self.holdings: Dict[str, Dict] = {}
        self.load_from_file()
    
    def add_transaction(self, symbol: str, amount: float, price: float, 
                       transaction_type: str = "buy", notes: str = "") -> bool:
        """Add a new transaction to the portfolio"""
        try:
            symbol = symbol.upper().strip()
            
            if amount <= 0 or price <= 0:
                raise ValueError("Amount and price must be positive")
            
            if transaction_type not in ['buy', 'sell']:
                raise ValueError("Transaction type must be 'buy' or 'sell'")
            
            transaction = Transaction(
                symbol=symbol,
                amount=amount,
                price=price,
                transaction_type=transaction_type,
                timestamp=datetime.now().isoformat(),
                notes=notes
            )
            
            self.transactions.append(transaction)
            self._update_holdings()
            self.save_to_file()
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to add transaction: {str(e)}")
    
    def _update_holdings(self):
        """Update holdings based on all transactions"""
        self.holdings = {}
        
        for transaction in self.transactions:
            symbol = transaction.symbol
            
            if symbol not in self.holdings:
                self.holdings[symbol] = {
                    'amount': 0.0,
                    'total_cost': 0.0,
                    'avg_price': 0.0,
                    'transactions': []
                }
            
            holding = self.holdings[symbol]
            
            if transaction.transaction_type == 'buy':
                # Add to position
                new_total_cost = holding['total_cost'] + (transaction.amount * transaction.price)
                new_amount = holding['amount'] + transaction.amount
                
                holding['amount'] = new_amount
                holding['total_cost'] = new_total_cost
                holding['avg_price'] = new_total_cost / new_amount if new_amount > 0 else 0
                
            elif transaction.transaction_type == 'sell':
                # Reduce position
                if holding['amount'] >= transaction.amount:
                    # Calculate cost of sold amount based on average price
                    sold_cost = transaction.amount * holding['avg_price']
                    
                    holding['amount'] -= transaction.amount
                    holding['total_cost'] -= sold_cost
                    
                    # Update average price (remains the same for remaining holdings)
                    if holding['amount'] > 0:
                        holding['avg_price'] = holding['total_cost'] / holding['amount']
                    else:
                        holding['avg_price'] = 0
                        holding['total_cost'] = 0
                else:
                    # Can't sell more than we have
                    raise ValueError(f"Cannot sell {transaction.amount} {symbol}, only have {holding['amount']}")
            
            holding['transactions'].append(transaction.to_dict())
        
        # Remove holdings with zero amount
        self.holdings = {k: v for k, v in self.holdings.items() if v['amount'] > 0}
    
    def get_holdings(self) -> Dict[str, Dict]:
        """Get current holdings"""
        return self.holdings.copy()
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate portfolio value and performance"""
        total_value = 0.0
        total_cost = 0.0
        holdings_value = {}
        
        for symbol, holding in self.holdings.items():
            current_price = current_prices.get(symbol, holding['avg_price'])
            value = holding['amount'] * current_price
            cost = holding['total_cost']
            
            pnl = value - cost
            pnl_percent = (pnl / cost * 100) if cost > 0 else 0
            
            holdings_value[symbol] = {
                'amount': holding['amount'],
                'avg_price': holding['avg_price'],
                'current_price': current_price,
                'value': value,
                'cost': cost,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'allocation_percent': 0  # Will be calculated after total
            }
            
            total_value += value
            total_cost += cost
        
        # Calculate allocation percentages
        for symbol in holdings_value:
            if total_value > 0:
                holdings_value[symbol]['allocation_percent'] = (
                    holdings_value[symbol]['value'] / total_value * 100
                )
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'holdings': holdings_value
        }
    
    def get_transaction_history(self, symbol: str = None) -> List[Dict]:
        """Get transaction history, optionally filtered by symbol"""
        transactions = []
        
        for transaction in self.transactions:
            if symbol is None or transaction.symbol == symbol.upper():
                transactions.append(transaction.to_dict())
        
        # Sort by timestamp (newest first)
        transactions.sort(key=lambda x: x['timestamp'], reverse=True)
        return transactions
    
    def get_portfolio_performance(self, current_prices: Dict[str, float], 
                                 days: int = 30) -> Dict:
        """Calculate portfolio performance metrics"""
        portfolio_data = self.get_portfolio_value(current_prices)
        
        # Calculate additional metrics
        metrics = {
            'total_value': portfolio_data['total_value'],
            'total_cost': portfolio_data['total_cost'],
            'total_pnl': portfolio_data['total_pnl'],
            'total_pnl_percent': portfolio_data['total_pnl_percent'],
            'number_of_holdings': len(self.holdings),
            'largest_holding': None,
            'best_performer': None,
            'worst_performer': None,
            'diversification_score': 0
        }
        
        if portfolio_data['holdings']:
            # Find largest holding
            largest = max(portfolio_data['holdings'].items(), 
                         key=lambda x: x[1]['allocation_percent'])
            metrics['largest_holding'] = {
                'symbol': largest[0],
                'allocation_percent': largest[1]['allocation_percent']
            }
            
            # Find best and worst performers
            performers = [(symbol, data['pnl_percent']) 
                         for symbol, data in portfolio_data['holdings'].items()]
            
            if performers:
                best = max(performers, key=lambda x: x[1])
                worst = min(performers, key=lambda x: x[1])
                
                metrics['best_performer'] = {
                    'symbol': best[0],
                    'pnl_percent': best[1]
                }
                
                metrics['worst_performer'] = {
                    'symbol': worst[0],
                    'pnl_percent': worst[1]
                }
            
            # Calculate diversification score (1 - Herfindahl index)
            allocations = [data['allocation_percent'] / 100 
                          for data in portfolio_data['holdings'].values()]
            herfindahl_index = sum(allocation ** 2 for allocation in allocations)
            metrics['diversification_score'] = (1 - herfindahl_index) * 100
        
        return metrics
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get asset allocation by percentage"""
        if not self.holdings:
            return {}
        
        total_value = sum(holding['total_cost'] for holding in self.holdings.values())
        
        if total_value == 0:
            return {}
        
        allocation = {}
        for symbol, holding in self.holdings.items():
            allocation[symbol] = (holding['total_cost'] / total_value) * 100
        
        return allocation
    
    def rebalance_suggestions(self, target_allocation: Dict[str, float], 
                            current_prices: Dict[str, float]) -> Dict:
        """Suggest rebalancing trades to reach target allocation"""
        portfolio_data = self.get_portfolio_value(current_prices)
        total_value = portfolio_data['total_value']
        
        if total_value == 0:
            return {'suggestions': [], 'total_trades': 0}
        
        suggestions = []
        
        for symbol, target_percent in target_allocation.items():
            target_value = total_value * (target_percent / 100)
            current_value = 0
            current_amount = 0
            
            if symbol in portfolio_data['holdings']:
                current_value = portfolio_data['holdings'][symbol]['value']
                current_amount = portfolio_data['holdings'][symbol]['amount']
            
            difference_value = target_value - current_value
            current_price = current_prices.get(symbol, 0)
            
            if current_price > 0 and abs(difference_value) > 1:  # Only suggest if difference > $1
                difference_amount = difference_value / current_price
                action = 'buy' if difference_value > 0 else 'sell'
                
                suggestions.append({
                    'symbol': symbol,
                    'action': action,
                    'amount': abs(difference_amount),
                    'value': abs(difference_value),
                    'current_allocation': (current_value / total_value * 100) if total_value > 0 else 0,
                    'target_allocation': target_percent,
                    'price': current_price
                })
        
        return {
            'suggestions': suggestions,
            'total_trades': len(suggestions),
            'total_value': total_value
        }
    
    def calculate_roi(self, symbol: str = None) -> Dict:
        """Calculate return on investment"""
        if symbol:
            # ROI for specific symbol
            symbol = symbol.upper()
            if symbol not in self.holdings:
                return {'error': f'No holdings for {symbol}'}
            
            holding = self.holdings[symbol]
            cost = holding['total_cost']
            
            # Need current price to calculate current value
            return {
                'symbol': symbol,
                'cost': cost,
                'amount': holding['amount'],
                'avg_price': holding['avg_price']
            }
        else:
            # Overall portfolio ROI
            total_cost = sum(holding['total_cost'] for holding in self.holdings.values())
            return {
                'total_cost': total_cost,
                'holdings_count': len(self.holdings)
            }
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export portfolio data to CSV"""
        if filename is None:
            filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Prepare data for CSV
        data = []
        for transaction in self.transactions:
            data.append({
                'Date': transaction.timestamp,
                'Symbol': transaction.symbol,
                'Type': transaction.transaction_type,
                'Amount': transaction.amount,
                'Price': transaction.price,
                'Value': transaction.amount * transaction.price,
                'Notes': transaction.notes
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    def export_holdings_to_csv(self, current_prices: Dict[str, float], 
                              filename: str = None) -> str:
        """Export current holdings to CSV"""
        if filename is None:
            filename = f"holdings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        portfolio_data = self.get_portfolio_value(current_prices)
        
        data = []
        for symbol, holding_data in portfolio_data['holdings'].items():
            data.append({
                'Symbol': symbol,
                'Amount': holding_data['amount'],
                'Average_Price': holding_data['avg_price'],
                'Current_Price': holding_data['current_price'],
                'Total_Cost': holding_data['cost'],
                'Current_Value': holding_data['value'],
                'PnL': holding_data['pnl'],
                'PnL_Percent': holding_data['pnl_percent'],
                'Allocation_Percent': holding_data['allocation_percent']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    def import_from_csv(self, filename: str) -> int:
        """Import transactions from CSV"""
        try:
            df = pd.read_csv(filename)
            
            # Expected columns: Date, Symbol, Type, Amount, Price, Notes
            required_columns = ['Symbol', 'Type', 'Amount', 'Price']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            imported_count = 0
            
            for _, row in df.iterrows():
                try:
                    notes = row.get('Notes', '')
                    if pd.isna(notes):
                        notes = ''
                    
                    transaction = Transaction(
                        symbol=str(row['Symbol']).upper().strip(),
                        amount=float(row['Amount']),
                        price=float(row['Price']),
                        transaction_type=str(row['Type']).lower().strip(),
                        timestamp=row.get('Date', datetime.now().isoformat()),
                        notes=str(notes)
                    )
                    
                    self.transactions.append(transaction)
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error importing row: {e}")
                    continue
            
            if imported_count > 0:
                self._update_holdings()
                self.save_to_file()
            
            return imported_count
            
        except Exception as e:
            raise Exception(f"Failed to import from CSV: {str(e)}")
    
    def delete_transaction(self, transaction_index: int) -> bool:
        """Delete a transaction by index"""
        try:
            if 0 <= transaction_index < len(self.transactions):
                self.transactions.pop(transaction_index)
                self._update_holdings()
                self.save_to_file()
                return True
            else:
                raise IndexError("Transaction index out of range")
        except Exception as e:
            raise Exception(f"Failed to delete transaction: {str(e)}")
    
    def edit_transaction(self, transaction_index: int, **kwargs) -> bool:
        """Edit an existing transaction"""
        try:
            if not (0 <= transaction_index < len(self.transactions)):
                raise IndexError("Transaction index out of range")
            
            transaction = self.transactions[transaction_index]
            
            # Update allowed fields
            allowed_fields = ['amount', 'price', 'transaction_type', 'notes']
            for field, value in kwargs.items():
                if field in allowed_fields:
                    if field == 'amount' or field == 'price':
                        value = float(value)
                        if value <= 0:
                            raise ValueError(f"{field} must be positive")
                    elif field == 'transaction_type':
                        if value not in ['buy', 'sell']:
                            raise ValueError("Transaction type must be 'buy' or 'sell'")
                    
                    setattr(transaction, field, value)
            
            self._update_holdings()
            self.save_to_file()
            return True
            
        except Exception as e:
            raise Exception(f"Failed to edit transaction: {str(e)}")
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> str:
        """Get a formatted portfolio summary"""
        portfolio_data = self.get_portfolio_value(current_prices)
        performance = self.get_portfolio_performance(current_prices)
        
        summary = f"""
PORTFOLIO SUMMARY
================

Total Value: ${portfolio_data['total_value']:,.2f}
Total Cost: ${portfolio_data['total_cost']:,.2f}
Total P&L: ${portfolio_data['total_pnl']:,.2f} ({portfolio_data['total_pnl_percent']:+.2f}%)

Holdings: {performance['number_of_holdings']} cryptocurrencies
Diversification Score: {performance['diversification_score']:.1f}/100

"""
        
        if performance['largest_holding']:
            summary += f"Largest Holding: {performance['largest_holding']['symbol']} "
            summary += f"({performance['largest_holding']['allocation_percent']:.1f}%)\n"
        
        if performance['best_performer']:
            summary += f"Best Performer: {performance['best_performer']['symbol']} "
            summary += f"({performance['best_performer']['pnl_percent']:+.2f}%)\n"
        
        if performance['worst_performer']:
            summary += f"Worst Performer: {performance['worst_performer']['symbol']} "
            summary += f"({performance['worst_performer']['pnl_percent']:+.2f}%)\n"
        
        summary += "\nHOLDINGS BREAKDOWN\n" + "=" * 18 + "\n"
        
        for symbol, data in portfolio_data['holdings'].items():
            summary += f"{symbol}: {data['amount']:.6f} @ ${data['avg_price']:.6f} "
            summary += f"(Current: ${data['current_price']:.6f}) "
            summary += f"= ${data['value']:.2f} "
            summary += f"[{data['pnl_percent']:+.2f}%]\n"
        
        return summary
    
    def get_tax_report(self, year: int = None) -> Dict:
        """Generate tax report for realized gains/losses"""
        if year is None:
            year = datetime.now().year
        
        realized_gains = []
        total_gains = 0.0
        total_losses = 0.0
        
        # Group transactions by symbol to track cost basis
        symbol_positions = {}
        
        for transaction in self.transactions:
            transaction_date = datetime.fromisoformat(transaction.timestamp)
            
            if transaction_date.year != year:
                continue
            
            symbol = transaction.symbol
            
            if symbol not in symbol_positions:
                symbol_positions[symbol] = []
            
            if transaction.transaction_type == 'buy':
                symbol_positions[symbol].append({
                    'amount': transaction.amount,
                    'price': transaction.price,
                    'date': transaction.timestamp
                })
            
            elif transaction.transaction_type == 'sell':
                # Calculate realized gain/loss using FIFO
                remaining_to_sell = transaction.amount
                total_cost_basis = 0.0
                
                while remaining_to_sell > 0 and symbol_positions.get(symbol):
                    position = symbol_positions[symbol][0]
                    
                    if position['amount'] <= remaining_to_sell:
                        # Sell entire position
                        sold_amount = position['amount']
                        cost_basis = sold_amount * position['price']
                        proceeds = sold_amount * transaction.price
                        
                        gain_loss = proceeds - cost_basis
                        
                        realized_gains.append({
                            'symbol': symbol,
                            'amount': sold_amount,
                            'cost_basis': cost_basis,
                            'proceeds': proceeds,
                            'gain_loss': gain_loss,
                            'buy_date': position['date'],
                            'sell_date': transaction.timestamp
                        })
                        
                        if gain_loss > 0:
                            total_gains += gain_loss
                        else:
                            total_losses += abs(gain_loss)
                        
                        remaining_to_sell -= sold_amount
                        symbol_positions[symbol].pop(0)
                        
                    else:
                        # Partial sale
                        sold_amount = remaining_to_sell
                        cost_basis = sold_amount * position['price']
                        proceeds = sold_amount * transaction.price
                        
                        gain_loss = proceeds - cost_basis
                        
                        realized_gains.append({
                            'symbol': symbol,
                            'amount': sold_amount,
                            'cost_basis': cost_basis,
                            'proceeds': proceeds,
                            'gain_loss': gain_loss,
                            'buy_date': position['date'],
                            'sell_date': transaction.timestamp
                        })
                        
                        if gain_loss > 0:
                            total_gains += gain_loss
                        else:
                            total_losses += abs(gain_loss)
                        
                        position['amount'] -= sold_amount
                        remaining_to_sell = 0
        
        return {
            'year': year,
            'realized_gains': realized_gains,
            'total_gains': total_gains,
            'total_losses': total_losses,
            'net_gain_loss': total_gains - total_losses,
            'total_transactions': len(realized_gains)
        }
    
    def backup_portfolio(self, backup_filename: str = None) -> str:
        """Create a backup of the portfolio"""
        if backup_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"portfolio_backup_{timestamp}.json"
        
        backup_data = {
            'transactions': [t.to_dict() for t in self.transactions],
            'holdings': self.holdings,
            'backup_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return backup_filename
    
    def restore_from_backup(self, backup_filename: str) -> bool:
        """Restore portfolio from backup"""
        try:
            with open(backup_filename, 'r') as f:
                backup_data = json.load(f)
            
            # Restore transactions
            self.transactions = []
            for t_data in backup_data.get('transactions', []):
                transaction = Transaction.from_dict(t_data)
                self.transactions.append(transaction)
            
            # Restore holdings
            self.holdings = backup_data.get('holdings', {})
            
            # Save restored data
            self.save_to_file()
            return True
            
        except Exception as e:
            raise Exception(f"Failed to restore from backup: {str(e)}")
    
    def save_to_file(self):
        """Save portfolio to file"""
        try:
            data = {
                'transactions': [t.to_dict() for t in self.transactions],
                'holdings': self.holdings,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving portfolio: {e}")
    
    def load_from_file(self):
        """Load portfolio from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                
                # Load transactions
                self.transactions = []
                for t_data in data.get('transactions', []):
                    transaction = Transaction.from_dict(t_data)
                    self.transactions.append(transaction)
                
                # Load holdings
                self.holdings = data.get('holdings', {})
                
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            self.transactions = []
            self.holdings = {}
    
    def export_to_file(self, filename: str):
        """Export portfolio to specified file"""
        try:
            data = {
                'transactions': [t.to_dict() for t in self.transactions],
                'holdings': self.holdings,
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to export portfolio: {str(e)}")
    
    def import_from_file(self, filename: str):
        """Import portfolio from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.transactions = []
            self.holdings = {}
            
            # Import transactions
            for t_data in data.get('transactions', []):
                transaction = Transaction.from_dict(t_data)
                self.transactions.append(transaction)
            
            # Recalculate holdings
            self._update_holdings()
            self.save_to_file()
            
        except Exception as e:
            raise Exception(f"Failed to import portfolio: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get portfolio statistics"""
        if not self.transactions:
            return {'error': 'No transactions found'}
        
        stats = {
            'total_transactions': len(self.transactions),
            'buy_transactions': sum(1 for t in self.transactions if t.transaction_type == 'buy'),
            'sell_transactions': sum(1 for t in self.transactions if t.transaction_type == 'sell'),
            'unique_symbols': len(set(t.symbol for t in self.transactions)),
            'first_transaction': min(self.transactions, key=lambda t: t.timestamp).timestamp,
            'last_transaction': max(self.transactions, key=lambda t: t.timestamp).timestamp,
            'total_invested': sum(t.amount * t.price for t in self.transactions if t.transaction_type == 'buy'),
            'total_sold': sum(t.amount * t.price for t in self.transactions if t.transaction_type == 'sell'),
            'current_holdings': len(self.holdings)
        }
        
        return stats
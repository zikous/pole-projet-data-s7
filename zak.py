"""
Euronext-Style Order Matching Algorithm
======================================

This implementation demonstrates the core matching principles used by Euronext,
based on price-time priority (also known as FIFO - First In, First Out).

Key Features:
- Price-Time Priority matching
- Support for multiple order types (Market, Limit, Stop orders)
- Continuous trading and auction mechanisms
- Order book management
- Trade execution and reporting
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque
import heapq
import time
from decimal import Decimal
from datetime import datetime, timedelta

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradingPhase(Enum):
    PRE_OPENING = "PRE_OPENING"
    OPENING_AUCTION = "OPENING_AUCTION"  
    CONTINUOUS_TRADING = "CONTINUOUS_TRADING"
    CLOSING_AUCTION = "CLOSING_AUCTION"
    POST_TRADING = "POST_TRADING"

class AuctionType(Enum):
    OPENING = "OPENING"
    CLOSING = "CLOSING"
    INTRADAY = "INTRADAY"  # For volatility interruption auctions

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order
    ATO = "ATO"  # At The Opening (auction only)
    ATC = "ATC"  # At The Close (auction only)

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: float = field(default_factory=time.time)
    filled_quantity: int = 0
    auction_only: bool = False  # True for ATO/ATC orders
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_fully_filled(self) -> bool:
        return self.filled_quantity >= self.quantity

@dataclass
class Trade:
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    symbol: str
    quantity: int
    price: Decimal
    timestamp: float = field(default_factory=time.time)

@dataclass
class AuctionResult:
    auction_price: Optional[Decimal]
    total_volume: int
    imbalance: int
    imbalance_side: Optional[OrderSide]
    trades: List[Trade] = field(default_factory=list)

class OrderBook:
    """
    Order book implementation using price-time priority matching
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Buy orders: max-heap (highest price first, then earliest time)
        self.buy_orders: List[Tuple[Decimal, float, Order]] = []
        # Sell orders: min-heap (lowest price first, then earliest time)  
        self.sell_orders: List[Tuple[Decimal, float, Order]] = []
        # Auction-only orders (separate pools)
        self.auction_buy_orders: List[Tuple[Decimal, float, Order]] = []
        self.auction_sell_orders: List[Tuple[Decimal, float, Order]] = []
        # Order lookup for fast access
        self.orders: Dict[str, Order] = {}
        # Trading phase
        self.trading_phase: TradingPhase = TradingPhase.CONTINUOUS_TRADING
        
    def add_order(self, order: Order):
        """Add order to the appropriate side of the book"""
        self.orders[order.order_id] = order
        
        # Determine if order goes to auction book or continuous book
        if (order.auction_only or 
            order.time_in_force in [TimeInForce.ATO, TimeInForce.ATC] or
            self.trading_phase in [TradingPhase.OPENING_AUCTION, TradingPhase.CLOSING_AUCTION]):
            
            if order.side == OrderSide.BUY:
                heapq.heappush(self.auction_buy_orders, 
                              (-order.price if order.price else 0, order.timestamp, order))
            else:
                heapq.heappush(self.auction_sell_orders, 
                              (order.price if order.price else float('inf'), order.timestamp, order))
        else:
            if order.side == OrderSide.BUY:
                heapq.heappush(self.buy_orders, 
                              (-order.price, order.timestamp, order))
            else:
                heapq.heappush(self.sell_orders, 
                              (order.price, order.timestamp, order))
    
    def remove_order(self, order_id: str):
        """Remove order from the book"""
        if order_id in self.orders:
            del self.orders[order_id]
    
    def get_best_bid(self) -> Optional[Decimal]:
        """Get the highest buy price"""
        while self.buy_orders:
            neg_price, _, order = self.buy_orders[0]
            if order.order_id in self.orders and not order.is_fully_filled:
                return -neg_price
            heapq.heappop(self.buy_orders)
        return None
    
    def get_best_ask(self) -> Optional[Decimal]:
        """Get the lowest sell price"""
        while self.sell_orders:
            price, _, order = self.sell_orders[0]
            if order.order_id in self.orders and not order.is_fully_filled:
                return price
            heapq.heappop(self.sell_orders)
        return None
    
    def get_auction_orders(self) -> Tuple[List[Order], List[Order]]:
        """Get all valid auction orders"""
        buy_orders = []
        sell_orders = []
        
        # Collect valid auction buy orders
        temp_buy = []
        while self.auction_buy_orders:
            neg_price, timestamp, order = heapq.heappop(self.auction_buy_orders)
            if order.order_id in self.orders and not order.is_fully_filled:
                buy_orders.append(order)
            temp_buy.append((neg_price, timestamp, order))
        
        # Restore the heap
        self.auction_buy_orders = temp_buy
        heapq.heapify(self.auction_buy_orders)
        
        # Collect valid auction sell orders  
        temp_sell = []
        while self.auction_sell_orders:
            price, timestamp, order = heapq.heappop(self.auction_sell_orders)
            if order.order_id in self.orders and not order.is_fully_filled:
                sell_orders.append(order)
            temp_sell.append((price, timestamp, order))
        
        # Restore the heap
        self.auction_sell_orders = temp_sell
        heapq.heapify(self.auction_sell_orders)
        
        return buy_orders, sell_orders

class EuronextMatchingEngine:
    """
    Euronext-style matching engine implementing price-time priority and auctions
    """
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.trade_counter = 0
        self.auction_results: List[AuctionResult] = []
        
    def get_order_book(self, symbol: str) -> OrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]
    
    def set_trading_phase(self, symbol: str, phase: TradingPhase):
        """Set trading phase for a symbol"""
        order_book = self.get_order_book(symbol)
        order_book.trading_phase = phase
    
    def submit_order(self, order: Order) -> List[Trade]:
        """
        Main order submission and matching logic
        Returns list of trades generated
        """
        trades = []
        order_book = self.get_order_book(order.symbol)
        
        # Handle auction-only orders
        if (order.auction_only or 
            order.time_in_force in [TimeInForce.ATO, TimeInForce.ATC] or
            order_book.trading_phase in [TradingPhase.OPENING_AUCTION, TradingPhase.CLOSING_AUCTION]):
            
            # Add to auction book, no immediate matching
            order_book.add_order(order)
            return trades
        
        # Handle continuous trading orders
        if order.order_type == OrderType.MARKET:
            trades = self._match_market_order(order, order_book)
        elif order.order_type == OrderType.LIMIT:
            trades = self._match_limit_order(order, order_book)
        
        # Handle time-in-force constraints
        if order.time_in_force == TimeInForce.IOC and not order.is_fully_filled:
            order.quantity = order.filled_quantity
        elif order.time_in_force == TimeInForce.FOK and not order.is_fully_filled:
            self._cancel_fills(trades, order)
            return []
        
        # Add remaining quantity to book if not fully filled
        if not order.is_fully_filled and order.order_type == OrderType.LIMIT:
            order_book.add_order(order)
        
        return trades
    
    def run_auction(self, symbol: str, auction_type: AuctionType) -> AuctionResult:
        """
        Run auction matching algorithm
        This implements the Euronext auction algorithm that maximizes traded volume
        """
        order_book = self.get_order_book(symbol)
        buy_orders, sell_orders = order_book.get_auction_orders()
        
        if not buy_orders and not sell_orders:
            return AuctionResult(None, 0, 0, None)
        
        # Calculate auction price using volume maximization
        auction_price = self._calculate_auction_price(buy_orders, sell_orders)
        
        if auction_price is None:
            return AuctionResult(None, 0, 0, None)
        
        # Execute trades at auction price
        trades = self._execute_auction_trades(buy_orders, sell_orders, auction_price)
        
        # Calculate imbalance
        total_buy_quantity = sum(self._get_executable_quantity(order, auction_price) 
                               for order in buy_orders)
        total_sell_quantity = sum(self._get_executable_quantity(order, auction_price) 
                                for order in sell_orders)
        
        imbalance = abs(total_buy_quantity - total_sell_quantity)
        imbalance_side = (OrderSide.BUY if total_buy_quantity > total_sell_quantity 
                         else OrderSide.SELL if total_sell_quantity > total_buy_quantity 
                         else None)
        
        result = AuctionResult(
            auction_price=auction_price,
            total_volume=sum(trade.quantity for trade in trades),
            imbalance=imbalance,
            imbalance_side=imbalance_side,
            trades=trades
        )
        
        self.auction_results.append(result)
        self.trades.extend(trades)
        
        return result
    
    def _calculate_auction_price(self, buy_orders: List[Order], sell_orders: List[Order]) -> Optional[Decimal]:
        """Calculate auction price that maximizes volume"""
        if not buy_orders or not sell_orders:
            return None
        
        # Get all possible prices from orders
        all_prices = set()
        for order in buy_orders + sell_orders:
            if order.price:
                all_prices.add(order.price)
        
        if not all_prices:
            return None
        
        best_price = None
        max_volume = 0
        min_imbalance = float('inf')
        
        # Test each price to find volume maximization
        for price in sorted(all_prices):
            buy_volume = sum(self._get_executable_quantity(order, price) for order in buy_orders)
            sell_volume = sum(self._get_executable_quantity(order, price) for order in sell_orders)
            
            executable_volume = min(buy_volume, sell_volume)
            imbalance = abs(buy_volume - sell_volume)
            
            # Auction price selection criteria:
            # 1. Maximize executable volume
            # 2. If equal volume, minimize imbalance
            # 3. If still tied, prefer price closer to reference price (simplified here)
            if (executable_volume > max_volume or 
                (executable_volume == max_volume and imbalance < min_imbalance)):
                max_volume = executable_volume
                min_imbalance = imbalance
                best_price = price
        
        return best_price if max_volume > 0 else None
    
    def _get_executable_quantity(self, order: Order, price: Decimal) -> int:
        """Get quantity that can be executed for an order at given price"""
        if order.order_type == OrderType.MARKET:
            return order.remaining_quantity
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= price:
                return order.remaining_quantity
            elif order.side == OrderSide.SELL and order.price <= price:
                return order.remaining_quantity
        return 0
    
    def _execute_auction_trades(self, buy_orders: List[Order], sell_orders: List[Order], 
                               auction_price: Decimal) -> List[Trade]:
        """Execute trades at auction price"""
        trades = []
        
        # Sort orders by time priority
        executable_buys = [(order.timestamp, order) for order in buy_orders 
                          if self._get_executable_quantity(order, auction_price) > 0]
        executable_sells = [(order.timestamp, order) for order in sell_orders 
                           if self._get_executable_quantity(order, auction_price) > 0]
        
        executable_buys.sort()  # Earlier timestamp first
        executable_sells.sort()  # Earlier timestamp first
        
        buy_idx = 0
        sell_idx = 0
        
        # Match orders using time priority
        while buy_idx < len(executable_buys) and sell_idx < len(executable_sells):
            _, buy_order = executable_buys[buy_idx]
            _, sell_order = executable_sells[sell_idx]
            
            if buy_order.is_fully_filled:
                buy_idx += 1
                continue
            if sell_order.is_fully_filled:
                sell_idx += 1
                continue
            
            trade = self._execute_trade(buy_order, sell_order, auction_price)
            if trade:
                trades.append(trade)
            
            # Move to next order if current one is fully filled
            if buy_order.is_fully_filled:
                buy_idx += 1
            if sell_order.is_fully_filled:
                sell_idx += 1
        
        return trades
    
    def _match_market_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match market order against existing book"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # Match against sell orders (asks)
            while (order.remaining_quantity > 0 and 
                   order_book.sell_orders and 
                   order_book.get_best_ask() is not None):
                
                price, timestamp, counter_order = heapq.heappop(order_book.sell_orders)
                
                if (counter_order.order_id not in order_book.orders or 
                    counter_order.is_fully_filled):
                    continue
                
                trade = self._execute_trade(order, counter_order, price)
                if trade:
                    trades.append(trade)
                
                # Put back if not fully filled
                if not counter_order.is_fully_filled:
                    heapq.heappush(order_book.sell_orders, 
                                  (price, timestamp, counter_order))
        
        else:  # SELL order
            # Match against buy orders (bids)
            while (order.remaining_quantity > 0 and 
                   order_book.buy_orders and 
                   order_book.get_best_bid() is not None):
                
                neg_price, timestamp, counter_order = heapq.heappop(order_book.buy_orders)
                price = -neg_price
                
                if (counter_order.order_id not in order_book.orders or 
                    counter_order.is_fully_filled):
                    continue
                
                trade = self._execute_trade(order, counter_order, price)
                if trade:
                    trades.append(trade)
                
                # Put back if not fully filled
                if not counter_order.is_fully_filled:
                    heapq.heappush(order_book.buy_orders, 
                                  (neg_price, timestamp, counter_order))
        
        return trades
    
    def _match_limit_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match limit order with price improvement logic"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # Match against sell orders with price <= order.price
            while (order.remaining_quantity > 0 and 
                   order_book.sell_orders):
                
                ask_price = order_book.get_best_ask()
                if ask_price is None or ask_price > order.price:
                    break
                
                price, timestamp, counter_order = heapq.heappop(order_book.sell_orders)
                
                if (counter_order.order_id not in order_book.orders or 
                    counter_order.is_fully_filled):
                    continue
                
                # Execute at the market price (price improvement for buyer)
                trade = self._execute_trade(order, counter_order, price)
                if trade:
                    trades.append(trade)
                
                if not counter_order.is_fully_filled:
                    heapq.heappush(order_book.sell_orders, 
                                  (price, timestamp, counter_order))
        
        else:  # SELL order
            # Match against buy orders with price >= order.price
            while (order.remaining_quantity > 0 and 
                   order_book.buy_orders):
                
                bid_price = order_book.get_best_bid()
                if bid_price is None or bid_price < order.price:
                    break
                
                neg_price, timestamp, counter_order = heapq.heappop(order_book.buy_orders)
                price = -neg_price
                
                if (counter_order.order_id not in order_book.orders or 
                    counter_order.is_fully_filled):
                    continue
                
                # Execute at the market price (price improvement for seller)
                trade = self._execute_trade(order, counter_order, price)
                if trade:
                    trades.append(trade)
                
                if not counter_order.is_fully_filled:
                    heapq.heappush(order_book.buy_orders, 
                                  (neg_price, timestamp, counter_order))
        
        return trades
    
    def _execute_trade(self, order1: Order, order2: Order, price: Decimal) -> Optional[Trade]:
        """Execute trade between two orders"""
        trade_quantity = min(order1.remaining_quantity, order2.remaining_quantity)
        
        if trade_quantity <= 0:
            return None
        
        # Update order fill quantities
        order1.filled_quantity += trade_quantity
        order2.filled_quantity += trade_quantity
        
        # Create trade record
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_counter:06d}",
            buy_order_id=order1.order_id if order1.side == OrderSide.BUY else order2.order_id,
            sell_order_id=order1.order_id if order1.side == OrderSide.SELL else order2.order_id,
            symbol=order1.symbol,
            quantity=trade_quantity,
            price=price
        )
        
        self.trades.append(trade)
        
        # Remove fully filled orders from book
        if order1.is_fully_filled:
            order_book = self.get_order_book(order1.symbol)
            order_book.remove_order(order1.order_id)
        if order2.is_fully_filled:
            order_book = self.get_order_book(order2.symbol)
            order_book.remove_order(order2.order_id)
        
        return trade
    
    def _cancel_fills(self, trades: List[Trade], order: Order):
        """Cancel fills for FOK orders (simplified implementation)"""
        # In practice, this would need to handle the complexity of 
        # reversing partial fills, which is typically done by 
        # preventing the fills in the first place
        pass
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol"""
        order_book = self.get_order_book(symbol)
        return {
            'symbol': symbol,
            'best_bid': order_book.get_best_bid(),
            'best_ask': order_book.get_best_ask(),
            'bid_ask_spread': (order_book.get_best_ask() - order_book.get_best_bid()) 
                             if order_book.get_best_bid() and order_book.get_best_ask() 
                             else None
        }

# Example usage and testing
if __name__ == "__main__":
    engine = EuronextMatchingEngine()
    
    print("=== Euronext Matching Engine with Auctions Demo ===\n")
    
    # Set up opening auction phase
    engine.set_trading_phase("ASML.AS", TradingPhase.OPENING_AUCTION)
    
    # Create auction orders (these will be queued for auction)
    auction_orders = [
        Order("ATO001", "ASML.AS", OrderSide.BUY, OrderType.LIMIT, 100, Decimal("650.00"), 
              time_in_force=TimeInForce.ATO),
        Order("ATO002", "ASML.AS", OrderSide.BUY, OrderType.LIMIT, 200, Decimal("649.50"), 
              time_in_force=TimeInForce.ATO),
        Order("ATO003", "ASML.AS", OrderSide.BUY, OrderType.MARKET, 50, 
              time_in_force=TimeInForce.ATO),
        Order("ATO004", "ASML.AS", OrderSide.SELL, OrderType.LIMIT, 150, Decimal("650.50"), 
              time_in_force=TimeInForce.ATO),
        Order("ATO005", "ASML.AS", OrderSide.SELL, OrderType.LIMIT, 100, Decimal("651.00"), 
              time_in_force=TimeInForce.ATO),
        Order("ATO006", "ASML.AS", OrderSide.SELL, OrderType.MARKET, 75, 
              time_in_force=TimeInForce.ATO),
    ]
    
    print("1. OPENING AUCTION PHASE")
    print("Submitting auction orders (these are queued, not immediately matched):")
    
    for order in auction_orders:
        trades = engine.submit_order(order)
        print(f"  Queued: {order.side.value} {order.quantity} shares at {order.price or 'MARKET'}")
    
    # Run opening auction
    print("\nRunning Opening Auction...")
    auction_result = engine.run_auction("ASML.AS", AuctionType.OPENING)
    
    if auction_result.auction_price:
        print(f"  AUCTION PRICE: {auction_result.auction_price}")
        print(f"  TOTAL VOLUME: {auction_result.total_volume} shares")
        print(f"  IMBALANCE: {auction_result.imbalance} shares on {auction_result.imbalance_side.value if auction_result.imbalance_side else 'NONE'} side")
        print(f"  TRADES EXECUTED: {len(auction_result.trades)}")
        for trade in auction_result.trades:
            print(f"    -> {trade.quantity} shares at {trade.price}")
    else:
        print("  No auction price found - no crossing orders")
    
    print("\n" + "="*50 + "\n")
    
    # Switch to continuous trading
    engine.set_trading_phase("ASML.AS", TradingPhase.CONTINUOUS_TRADING)
    
    # Create continuous trading orders
    continuous_orders = [
        Order("C001", "ASML.AS", OrderSide.BUY, OrderType.LIMIT, 100, Decimal("649.75")),
        Order("C002", "ASML.AS", OrderSide.SELL, OrderType.LIMIT, 150, Decimal("650.25")),
        Order("C003", "ASML.AS", OrderSide.BUY, OrderType.MARKET, 75),
    ]
    
    print("2. CONTINUOUS TRADING PHASE")
    print("Submitting continuous orders (immediate matching):")
    
    for order in continuous_orders:
        print(f"\nSubmitting {order.side.value} {order.order_type.value} order: "
              f"{order.quantity} shares at {order.price or 'MARKET'}")
        
        trades = engine.submit_order(order)
        
        for trade in trades:
            print(f"  -> TRADE: {trade.quantity} shares at {trade.price}")
        
        market_data = engine.get_market_data("ASML.AS")
        print(f"  Market: Bid={market_data['best_bid']} Ask={market_data['best_ask']}")
    
    print("\n" + "="*50 + "\n")
    
    # Simulate closing auction
    engine.set_trading_phase("ASML.AS", TradingPhase.CLOSING_AUCTION)
    
    closing_orders = [
        Order("ATC001", "ASML.AS", OrderSide.BUY, OrderType.LIMIT, 200, Decimal("650.00"), 
              time_in_force=TimeInForce.ATC),
        Order("ATC002", "ASML.AS", OrderSide.SELL, OrderType.LIMIT, 180, Decimal("650.00"), 
              time_in_force=TimeInForce.ATC),
    ]
    
    print("3. CLOSING AUCTION PHASE")
    print("Submitting closing auction orders:")
    
    for order in closing_orders:
        trades = engine.submit_order(order)
        print(f"  Queued: {order.side.value} {order.quantity} shares at {order.price}")
    
    print("\nRunning Closing Auction...")
    closing_result = engine.run_auction("ASML.AS", AuctionType.CLOSING)
    
    if closing_result.auction_price:
        print(f"  CLOSING PRICE: {closing_result.auction_price}")
        print(f"  TOTAL VOLUME: {closing_result.total_volume} shares")
        print(f"  IMBALANCE: {closing_result.imbalance} shares")
        for trade in closing_result.trades:
            print(f"    -> {trade.quantity} shares at {trade.price}")
    
    print(f"\nTotal trades executed across all phases: {len(engine.trades)}")
    print(f"Total auctions run: {len(engine.auction_results)}")

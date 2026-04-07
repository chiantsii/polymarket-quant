import pytest
from polymarket_quant.execution.market_maker import BinaryMarketMaker

def test_inventory_skew_logic():
    """
    測試：當庫存增加時，買賣報價是否會同步下移（以減少買入誘因並鼓勵賣出）。
    """
    mm = BinaryMarketMaker(gamma=0.5, base_spread=0.02)
    p_fair = 0.5
    sigma = 0.2
    ttr = 0.5 # 距離到期還有段時間
    
    # 情況 A: 零庫存
    bid_zero, ask_zero = mm.get_quotes(p_fair, inventory=0, sigma=sigma, ttr=ttr, toxicity_score=0)
    
    # 情況 B: 正庫存 (Long 100 units)
    bid_long, ask_long = mm.get_quotes(p_fair, inventory=100, sigma=sigma, ttr=ttr, toxicity_score=0)
    
    # 驗證：正庫存時，報價應該比零庫存時更低
    assert bid_long < bid_zero
    assert ask_long < ask_zero
    print(f"Skew Test Passed: Zero({bid_zero}/{ask_zero}) -> Long({bid_long}/{ask_long})")

def test_toxicity_widening():
    """
    測試：在高毒性（High Toxicity）環境下，價差是否會擴大。
    """
    mm = BinaryMarketMaker(base_spread=0.02)
    p_fair = 0.5
    
    bid_normal, ask_normal = mm.get_quotes(p_fair, 0, 0.1, 0.5, toxicity_score=0.1)
    bid_toxic, ask_toxic = mm.get_quotes(p_fair, 0, 0.1, 0.5, toxicity_score=0.9)
    
    spread_normal = ask_normal - bid_normal
    spread_toxic = ask_toxic - bid_toxic
    
    assert spread_toxic > spread_normal
    print(f"Toxicity Test Passed: Normal Spread {spread_normal:.4f} -> Toxic Spread {spread_toxic:.4f}")
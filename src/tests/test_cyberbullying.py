import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
import numpy as np
from src.social.cyberbullying import CyberbullyingModel
import warnings
warnings.filterwarnings('ignore')

class DummyTrader:
    def __init__(self, trader_id, emotion_bias, is_natural_bully=False):
        self.trader_id = trader_id
        self.type = 'retail'
        self.emotion_bias = emotion_bias
        self.is_natural_bully = is_natural_bully
        self.neighbors = []
        self.exposure = 0.0
        self.resilience = 0.0
        self.is_bullied = False
        self.bully_cooldown = 0
        self.suppression = 0.0


def test_cyberbullying_trigger():
    # 配置：极低阈值，必然触发
    config = {
        'social': {
            'enable_cyberbullying': True,
            'exposure_threshold': 0.01,
            'cooldown_duration': 1,
            'suppression_prob': 0.1,
            'resilience_growth': 0.01,
            'emotion_shrink_factor': 0.95,
            'sigmoid_k': 1.0,
            'max_suppression': 0.95,
            'average_degree': 1,
        },
        'simulation': {'random_seed': 42}
    }
    # 两个trader，意见极端且相反
    t0 = DummyTrader(0, 1.0, is_natural_bully=True)
    t1 = DummyTrader(1, -1.0, is_natural_bully=True)
    t0.neighbors = [t1]
    t1.neighbors = [t0]
    traders = [t0, t1]
    model = CyberbullyingModel(config)
    model.bully_count = {0: 0, 1: 0}
    # 多轮传播
    for _ in range(5):
        model.propagate(traders)
    # 至少有一个trader被网暴
    assert model.bully_count[0] > 0 or model.bully_count[1] > 0, f"No bullying occurred: {model.bully_count}" 
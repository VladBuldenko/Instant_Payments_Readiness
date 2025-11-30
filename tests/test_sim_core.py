# NOTE: This test module was partially created with assistance from ChatGPT.

# Import system libs to adjust Python path (so we can import from ../src)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest

from src.sim_core import (
    generate_synth,
    simulate_vop,
    scan_vop,
    simulate_fraud,
    scan_fraud
)

@pytest.fixture
def sample_df():
    return generate_synth(n=100,seed=42)

# Test 1: Synthetic data generator
def test_generate_synth_basic(sample_df):
    
    df = sample_df
    expected_col_names = {"transaction_id", "amount_eur", "fraud_probability", "vop_match_score"}
    
    # Check length of the table
    assert len(df) == 100
    
    # Check column names
    assert set(df.columns) == expected_col_names

    # Check ranges
    assert df["amount_eur"].between(10,5000).all()
    assert df["fraud_probability"].between(0,1).all()
    assert df["vop_match_score"].between(0,1).all()

# Test 2: Check simulate_vop 
def test_simulate_vop_outputs(sample_df):
    df = sample_df
    res = simulate_vop(df, threshold=0.8)
    
    # Check keys
    assert "conversion_rate" in res
    assert "latency_p95" in res
    
    # Verify whether the values are realistic
    assert 0 <= res["conversion_rate"] <= 100
    assert res["latency_p95"] > 0

# Test 3: Check whether scan_vop returns a DataFrame
def test_scan_vop_shape(sample_df):
    df = sample_df
    thresholds = np.linspace(0.5, 0.9, 5)
    result = scan_vop(df,thresholds)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(thresholds)
    assert all(col in result.columns for col in ["vop_threshold", "conversion_rate", "latency_p95"])

# Test 4: Check whether simulate_fraud works
def test_simulate_fraud_outputs(sample_df):
    df = sample_df
    res = simulate_fraud(df, threshold=0.5)
    assert "risk_exposure_eur" in res
    assert "manual_review_rate" in res
    assert res["risk_exposure_eur"] >= 0
    assert 0 <= res["manual_review_rate"] <= 100

# Test 5: Check whether scan_fraud returns correct shape ---
def test_scan_fraud_shape(sample_df):
    df = sample_df
    thresholds = np.arange(0.2, 0.9, 0.2)
    results = scan_fraud(df, thresholds)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(thresholds)
    assert all(col in results.columns for col in ["fraud_threshold", "manual_review_rate", "risk_exposure_eur"])
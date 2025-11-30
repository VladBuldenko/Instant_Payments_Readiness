import numpy as np
import pandas as pd

# NOTE: This function was partially assisted by ChatGPT.
# Synthetic data generator
def generate_synth(n: int = 5000, seed: int=42, min_amount: float = 10.0, max_amount: float = 5000.0)->pd.DataFrame:
    
    """ Create a synthetic dataset of n instant payment transactions. 
    Why we need this: - Bundesbank does not provide VoP / fraud / latency fields. 
    - We simulate them to evaluate what-if scenarios. 
    Returns columns: - transaction_id: unique id - amount_eur: payment amount in EUR - fraud_probability: estimated risk (0..1) 
    - vop_match_score: payee-name match score (0..1) """

    # Set seed for reproducible results
    rng = np.random.default_rng(seed)

    # Generate synthetic fields
    amounts = rng.uniform(min_amount, max_amount, n)
    fraud_prob = rng.uniform(0.0,1.0,n)
    vop_score = rng.uniform(0.0,1.0,n)

    df = pd.DataFrame({ 
    "transaction_id":np.arange(1, n+1),
    "amount_eur": amounts,
    "fraud_probability": fraud_prob,
    "vop_match_score": vop_score
    })
    return df

# NOTE: This function was partially assisted by ChatGPT.
# Vop simulation (H2)
def simulate_vop(df: pd.DataFrame, threshold: float = 0.8, base_latency_s: float = 0.5, latency_slope: float=1.2)->dict:
    """
    Simulate the effect of VoP strictness on Conversion and Latency. 
    Inputs: - df: synthetic transactions (must have 'vop_match_score') - threshold: VoP strictness, 
    higher => fewer transactions pass VoP - base_latency_s: baseline processing time in seconds - 
    latency_slope: how much strictness adds latency Outputs: 
    - conversion_rate: % of transactions that pass VoP - latency_p95: 95th percentile latency in seconds 
    """ 
    
    # Compute which transactions pass VoP check
    passed = df["vop_match_score"] >= threshold 

    # Conversion is simply the share of passed transactions
    conversion_rate = float(passed.mean() * 100.0)
    
    # Model latency as normal noise around a mean that depends on strictness 
    # Note: higher threshold => more checks => higher mean latency rng = np.random.default_rng(123)
    rng = np.random.default_rng(123)
    
    latency_samples = rng.normal(loc=base_latency_s + (threshold - 0.5) * latency_slope, scale=0.1, size=len(df))
    
    latency_p95 = float(np.percentile(latency_samples, 95))

    return {
        "conversion_rate": conversion_rate,
        "latency_p95": latency_p95
    }

# NOTE: This function was partially assisted by ChatGPT.
def scan_vop(df: pd.DataFrame, thresholds: np.ndarray)-> pd.DataFrame:
    """
    Run VoP simulation for a list/array of thresholds and return a tidy table.
    """
    # Collect results for each threshold
    rows = []
    for thr in thresholds:
        # Get KPI for current threshold
        res = simulate_vop(df, threshold=float(thr))
        rows.append({"vop_threshold": float(thr),
                    "conversion_rate": res["conversion_rate"],
                    "latency_p95": res["latency_p95"]})
    
    # Build dataframe with results
    return pd.DataFrame(rows)

# NOTE: This function was partially assisted by ChatGPT.
# Fraud filter simulation (H3)
def simulate_fraud(df: pd.DataFrame, threshold: float = 0.5)-> dict:
    """
    Simulate the effect of a fraud threshold on Risk and Manual Review Rate.
    Inputs:
    - df: synthetic transactions (must have 'fraud_probability' and 'amount_eur')
    - threshold: fraud sensitivity; lower => more flags => more manual reviews

    Outputs:
    - risk_exposure_eur: sum of amounts for transactions NOT flagged (potential loss)
    - manual_review_rate: % of transactions flagged for manual review
    """
    # Flag transactions as suspicious if probability > threshold
    flagged = df["fraud_probability"] > threshold

    # Manual Review Rate = share of flagged transactions
    manual_review_rate = float(flagged.mean() * 100.0)
    
    # Risk Exposure = sum of amounts that were NOT flagged (missed fraud risk proxy)
    risk_exposure_eur = float(df.loc[~flagged, "amount_eur"].sum())

    return {
        "risk_exposure_eur": risk_exposure_eur,
        "manual_review_rate": manual_review_rate
    }

# NOTE: This function was partially assisted by ChatGPT.
def scan_fraud(df: pd.DataFrame, thresholds: np.ndarray)->pd.DataFrame:
    """
    Run fraud simulation for a list/array of thresholds and return a tidy table.
    """
    rows=[]
    for thr in thresholds:
        # Get KPI for current threshold
        res = simulate_fraud(df, threshold=float(thr))
        rows.append({
            "fraud_threshold": float(thr),
            "manual_review_rate": res["manual_review_rate"],
            "risk_exposure_eur": res["risk_exposure_eur"]
        })
    
    return pd.DataFrame(rows)


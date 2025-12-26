import numpy as np
import pandas as pd

# --- Helper Functions ---
def pearson_corr_2d(X):
    x = X[:, 0] - X[:, 0].mean()
    y = X[:, 1] - X[:, 1].mean()
    denom = np.sqrt((x @ x) * (y @ y)) + 1e-12
    return float((x @ y) / denom)

def get_force_bounds(x, lam=3.0, use_ter=True, ter_max=3.0):
    q01, q25, q50, q75, q99 = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
    mu = q50
    sigma = (q75 - q25) / 1.349
    if use_ter:
        denom = abs(q50 - q01) + 1e-9
        ter = abs(q99 - q50) / denom
        ter = max(1.0, min(ter, ter_max))
    else:
        ter = 1.0
    return mu - lam * ter * sigma, mu + lam * ter * sigma

def force_corr_2d(X, use_ter=True):
    # Apply bounds per dimension
    mask_list = []
    for d in range(2):
        lo, hi = get_force_bounds(X[:, d], use_ter=use_ter)
        mask_list.append((X[:, d] >= lo) & (X[:, d] <= hi))
    
    mask = mask_list[0] & mask_list[1]
    if mask.sum() < 5: return 0.0
    return pearson_corr_2d(X[mask])

# --- Experiment 1: Asymmetric Contamination (The "Robustness Cost") ---
def run_contamination_exp(N=2000, rho=0.6, reps=100):
    rng = np.random.default_rng(42)
    results = []
    eps_list = [0.0, 0.05, 0.10, 0.15]
    
    print("Running Experiment 1: Asymmetric Contamination...")
    for eps in eps_list:
        err_ter, err_no_ter = [], []
        for _ in range(reps):
            # Clean Core
            n_out = int(eps * N)
            n_in = N - n_out
            cov = [[1, rho], [rho, 1]]
            X = rng.multivariate_normal([0,0], cov, n_in)
            
            # Asymmetric Outliers (One-sided on Dimension 0)
            # Add outliers at +8 sigma
            Out = rng.multivariate_normal([8, 0], [[0.1, 0],[0, 1]], n_out)
            
            X_combined = np.vstack([X, Out])
            rng.shuffle(X_combined)
            
            r_ter = force_corr_2d(X_combined, use_ter=True)
            r_fixed = force_corr_2d(X_combined, use_ter=False)
            
            err_ter.append((r_ter - rho)**2)
            err_no_ter.append((r_fixed - rho)**2)
            
        results.append({
            "Scenario": "Asymmetric Contamination",
            "Epsilon": eps,
            "MSE_TER": np.mean(err_ter),
            "MSE_Fixed": np.mean(err_no_ter)
        })
    return results

# --- Experiment 2: Coherent Heavy Tails (The "Efficiency Gain") ---
def run_heavytail_exp(N=2000, rho=0.6, reps=100):
    rng = np.random.default_rng(42)
    results = []
    df_degrees = [100, 10, 5, 3] # 100=Normal, 3=Very Heavy Tails
    
    print("Running Experiment 2: Coherent Heavy Tails (Student-t)...")
    for df in df_degrees:
        err_ter, err_no_ter = [], []
        for _ in range(reps):
            # Multivariate t-distribution generation
            # x = mu + sqrt(df/u) * Z, where u ~ Chi2(df)
            cov = np.array([[1, rho], [rho, 1]])
            Z = rng.multivariate_normal([0,0], cov, N)
            u = rng.chisquare(df, N) / df
            X = Z / np.sqrt(u)[:, None]
            
            # True correlation of t-dist is same as rho (if df > 2)
            
            r_ter = force_corr_2d(X, use_ter=True)
            r_fixed = force_corr_2d(X, use_ter=False)
            
            err_ter.append((r_ter - rho)**2)
            err_no_ter.append((r_fixed - rho)**2)
            
        results.append({
            "Scenario": "Coherent Heavy Tails (Student-t)",
            "Epsilon": f"df={df}", # abusing column for label
            "MSE_TER": np.mean(err_ter),
            "MSE_Fixed": np.mean(err_no_ter)
        })
    return results

if __name__ == "__main__":
    res1 = run_contamination_exp()
    res2 = run_heavytail_exp()
    
    df = pd.DataFrame(res1 + res2)
    print("\n=== FINAL ANALYSIS RESULTS ===")
    print(df)
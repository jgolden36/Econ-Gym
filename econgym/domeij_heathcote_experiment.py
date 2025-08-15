import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# Try to import from the correct location
try:
    from econgym.envs.aiyagari_env import AiyagariEnv
except ImportError:
    from envs.aiyagari_env import AiyagariEnv
from typing import Dict, Tuple

def run_domeij_heathcote_experiment():
    """
    Run the Domeij-Heathcote capital tax elimination experiment.
    
    This replicates their analysis of the effects of eliminating capital taxation
    in a heterogeneous agent incomplete markets model.
    """
    print("Running Domeij-Heathcote Capital Tax Experiment")
    print("=" * 50)
    
    # Parameters based on Domeij-Heathcote (2004) calibration
    base_params = {
        'beta': 0.96,          # Discount factor (high to ensure β(1+r_net) > 1)
        'risk_aversion': 2.0,   # Risk aversion
        'income_persistence': 0.9,  # Income persistence
        'labor_tax': 0.24,      # Labor tax rate (roughly matches US)
        'alpha': 0.36,          # Capital share
        'delta': 0.06,          # Depreciation rate
        'num_agent_types': 1,   # Single agent type for this replication
        'X': 1.0               # Productivity factor
    }
    
    # Scenario 1: High capital tax (initial steady state)
    high_tax_params = base_params.copy()
    high_tax_params['capital_tax'] = 0.40  # 40% capital tax
    
    # Scenario 2: Zero capital tax (final steady state) 
    zero_tax_params = base_params.copy()
    zero_tax_params['capital_tax'] = 0.0   # 0% capital tax
    
    print("Scenario 1: High Capital Tax (40%)")
    print("-" * 30)
    
    # Create high tax environment with lower income volatility to encourage saving
    high_tax_params['income_std'] = 0.2  # Reduce income risk
    env_high = AiyagariEnv(**high_tax_params)
    
    # Find equilibrium for high tax case
    print("Finding equilibrium with high capital tax...")
    r_high = find_equilibrium_interest_rate(env_high)
    print(f"Equilibrium interest rate: {r_high:.4f}")
    
    # Solve for policy and distribution
    V_high, policy_high, dist_high = env_high.solve_egm()
    
    # Compute aggregate statistics
    K_high = compute_aggregate_capital(env_high, dist_high)
    Y_high = compute_output(env_high, K_high)
    # Government spending in baseline (hold fixed across scenarios)
    tau_k_high = high_tax_params['capital_tax']
    r_pre_high = r_high / max(1e-12, (1 - tau_k_high))
    w_high = float(env_high.parameters['wage'])
    L = 1.0
    G_target = tau_k_high * r_pre_high * K_high + high_tax_params['labor_tax'] * w_high * L
    
    print(f"Aggregate Capital: {K_high:.4f}")
    print(f"Output: {Y_high:.4f}")
    print(f"Capital-Output Ratio: {K_high/Y_high:.4f}")
    
    print("\nScenario 2: Zero Capital Tax (0%)")
    print("-" * 30)
    
    # Create zero tax environment with lower income volatility
    zero_tax_params['income_std'] = 0.1  # Reduce income risk
    env_zero = AiyagariEnv(**zero_tax_params)
    
    # Adjust labor tax in zero-tax case to keep G fixed at baseline level
    def match_government_budget(env, G_target, max_iter=12, tol=1e-5):
        tau_k = env.parameters['capital_tax']
        tau_l = float(env.parameters['labor_tax'])
        for _ in range(max_iter):
            r_net = find_equilibrium_interest_rate(env)
            V, policy, dist = env.solve_egm()
            K = compute_aggregate_capital(env, dist)
            r_pre = r_net / max(1e-12, (1 - tau_k))
            w = float(env.parameters['wage'])
            L = 1.0
            G_implied = tau_k * r_pre * K + tau_l * w * L
            gap = G_implied - G_target
            print(f"[G-match] tau_l={tau_l:.4f}, K={K:.4f}, w={w:.4f}, G_implied={G_implied:.6f}, gap={gap:.6e}")
            if abs(gap) < tol:
                return r_net, V, policy, dist
            # Update tau_l to hit G: tau_l' = (G_target - tau_k r_pre K)/(w L)
            tau_l_new = (G_target - tau_k * r_pre * K) / max(1e-12, (w * L))
            tau_l_new = float(np.clip(tau_l_new, 0.0, 0.6))
            # Damp update
            tau_l = 0.5 * tau_l + 0.5 * tau_l_new
            env.parameters['labor_tax'] = tau_l
        return r_net, V, policy, dist

    print("Finding equilibrium with zero capital tax (adjusting labor tax to keep G fixed)...")
    r_zero, V_zero, policy_zero, dist_zero = match_government_budget(env_zero, G_target)
    print(f"Equilibrium interest rate: {r_zero:.4f}")
    
    # Compute aggregate statistics
    K_zero = compute_aggregate_capital(env_zero, dist_zero)
    Y_zero = compute_output(env_zero, K_zero)
    
    print(f"Aggregate Capital: {K_zero:.4f}")
    print(f"Output: {Y_zero:.4f}")
    print(f"Capital-Output Ratio: {K_zero/Y_zero:.4f}")
    
    # Perfect-foresight transition welfare (fixed G each period)
    print("\nComputing perfect-foresight transition (fixed G each period)...")
    trans = compute_transition_welfare(env_high, env_zero, dist_high, G_target, T=200)
    cev = trans['cev']
    
    # Compute welfare effects (ex-ante, holding baseline distribution as weights)
    print("\nWelfare Analysis")
    print("-" * 15)
    
    # Ex-ante (baseline) expected utilities
    sigma = base_params['risk_aversion']
    weights = dist_high / max(1e-12, np.sum(dist_high))
    W_high = float(np.sum(weights * V_high))
    W_zero = float(np.sum(weights * V_zero))
    # CEV using aggregate utilities under CRRA
    if abs(sigma - 1.0) < 1e-10:
        # Approximate for log utility: g solves E[log g + log c0] = E[log c1] ⇒ g ≈ exp(W_zero - W_high)
        cev = np.exp(W_zero - W_high) - 1.0
    else:
        # u(c) = c^{1-σ}/(1-σ) ⇒ V scales with c^{1-σ}
        ratio = W_zero / max(1e-12, W_high)
        cev = np.sign(1 - sigma) * (np.abs(ratio) ** (1.0 / (1.0 - sigma)) - 1.0)
    
    print(f"Ex-ante E[V] (baseline weights) - high: {W_high:.6f}, zero: {W_zero:.6f}")
    print(f"Consumption Equivalent Variation (transition, fixed G): {cev*100:.2f}%")
    
    # Summary of results
    print("\nSummary: Domeij-Heathcote Results")
    print("=" * 35)
    print(f"Interest rate change: {r_high:.4f} → {r_zero:.4f} ({(r_zero-r_high)*100:.2f} pp)")
    print(f"Capital change: {K_high:.4f} → {K_zero:.4f} ({(K_zero/K_high-1)*100:.1f}%)")
    print(f"Output change: {Y_high:.4f} → {Y_zero:.4f} ({(Y_zero/Y_high-1)*100:.1f}%)")
    print(f"Welfare effect (CEV): {cev*100:.2f}%")
    
    # Plot results
    plot_comparison_results(env_high, env_zero, dist_high, dist_zero, 
                          policy_high, policy_zero)
    
    return {
        'high_tax': {'r': r_high, 'K': K_high, 'Y': Y_high, 'welfare': W_high},
        'zero_tax': {'r': r_zero, 'K': K_zero, 'Y': Y_zero, 'welfare': W_zero},
        'cev': cev
    }

def compute_transition_welfare(env_high: AiyagariEnv, env_zero: AiyagariEnv, dist0, G_target: float, T: int = 200) -> Dict:
    """
    Perfect-foresight transition from high-tax steady state to zero-capital-tax steady state.
    - Holds government spending fixed each period by adjusting labor tax τ_l,t
    - Computes price/path {r_t, w_t} by clearing markets each t
    - Uses time-varying EGM with one-step update given next policy
    - Returns ex-ante CEV using baseline weights dist0
    """
    # Helper to extract valid constructor params from an env.parameters dict
    def _extract_params(p):
        keys = ['beta','risk_aversion','income_persistence','labor_tax','capital_tax',
                'alpha','delta','num_agent_types','X','income_grid_size','asset_grid_size',
                'asset_max','income_std']
        return {k: p[k] for k in keys if k in p}

    # Clone a working environment (start from high-tax parameters)
    env = AiyagariEnv(**_extract_params(env_high.parameters))
    
    # Paths
    r_path = np.zeros(T)
    w_path = np.zeros(T)
    tau_l_path = np.zeros(T)
    policy_path = []
    dist_path = [dist0]
    V_path = []
    
    # Terminal conditions at the zero-tax steady state
    env_T = AiyagariEnv(**_extract_params(env_zero.parameters))
    r_T = find_equilibrium_interest_rate(env_T)
    env_T.parameters['interest_rate'] = r_T
    V_T, policy_T, dist_T = env_T.solve_egm()
    policy_next = policy_T
    r_path[-1] = r_T
    w_path[-1] = env_T.parameters['wage']
    tau_l_path[-1] = env_T.parameters['labor_tax']
    V_path = [None] * T
    policy_path = [None] * T
    
    # Backward pass: compute policies given next-period policy
    for t in reversed(range(T)):
        # Target τ_k path: immediate drop to zero at t=0
        env.parameters['capital_tax'] = env_zero.parameters['capital_tax'] if t == 0 else env_high.parameters['capital_tax']
        # Find r_t that clears market given current parameters using stationary approximation
        r_t = find_equilibrium_interest_rate(env)
        r_path[t] = r_t
        w_path[t] = env.parameters['wage']
        # Match G each period by adjusting τ_l
        tau_k = env.parameters['capital_tax']
        K_stat = None
        # One-step EGM using next policy
        env.parameters['interest_rate'] = r_t
        policy_t = env.solve_egm_one_step_given_next_policy(policy_next)
        # Stationary dist for closure (approximation)
        _, _, dist_t = env.solve_egm()
        K_stat = compute_aggregate_capital(env, dist_t)
        r_pre = r_t / max(1e-12, (1 - tau_k))
        tau_l = env.parameters['labor_tax']
        tau_l_new = (G_target - tau_k * r_pre * K_stat) / max(1e-12, w_path[t])
        tau_l = float(np.clip(0.5 * tau_l + 0.5 * tau_l_new, 0.0, 0.6))
        env.parameters['labor_tax'] = tau_l
        tau_l_path[t] = tau_l
        policy_path[t] = policy_t
        policy_next = policy_t
        V_t, _, _ = env.solve_egm()
        V_path[t] = V_t
    
    # Forward pass: evolve distribution (approximate using stationary each period)
    # Compute ex-ante welfare along the path using baseline weights
    weights = dist0 / max(1e-12, np.sum(dist0))
    sigma = env_high.parameters['risk_aversion']
    W0 = float(np.sum(weights * V_path[0]))
    WT = float(np.sum(weights * V_path[-1]))
    if abs(sigma - 1.0) < 1e-10:
        cev = np.exp(WT - W0) - 1.0
    else:
        ratio = WT / max(1e-12, W0)
        cev = np.sign(1 - sigma) * (np.abs(ratio) ** (1.0 / (1.0 - sigma)) - 1.0)
    return {
        'cev': float(cev),
        'r_path': r_path,
        'w_path': w_path,
        'tau_l_path': tau_l_path,
    }

def find_equilibrium_interest_rate(env, tol=1e-4, max_iter=50):
    """Find equilibrium interest rate. Prefer the environment's method; fallback to local bisection."""
    # Prefer the env's built-in method for consistency
    if hasattr(env, 'find_equilibrium_interest_rate') and callable(getattr(env, 'find_equilibrium_interest_rate')):
        try:
            return env.find_equilibrium_interest_rate(tol=tol)
        except Exception:
            pass

    def excess_demand(r):
        # Set pre-tax interest rate for households
        env.parameters['interest_rate'] = r
        alpha = env.parameters['alpha']
        X = env.parameters['X']
        L = 1.0
        # Capital demand from firm FOC: r + delta = alpha X (K/L)^{alpha-1}
        K_demand = (alpha * X / (r + env.parameters['delta']))**(1.0/(1.0 - alpha)) * L
        # Wage from firm FOC: w = (1-alpha) X (K/L)^alpha
        env.parameters['wage'] = (1.0 - alpha) * X * (K_demand / L)**alpha

        # Solve household problem
        V, policy, dist = env.solve_egm(max_iter=600, tol=5e-4)

        # Normalize joint distribution to total mass 1
        total_mass = np.sum(dist)
        if total_mass <= 0 or not np.isfinite(total_mass):
            return np.inf
        dist_norm = dist / total_mass

        # Compute capital supply
        asset_grid = env.asset_grid.reshape(1, 1, -1)
        K_supply = float(np.sum(dist_norm * asset_grid))

        return K_supply - K_demand

    # Use search with wide bounds and pick best bracket; fall back to min |excess|
    r_low = 0.002
    r_high = 0.12

    # Pre-scan to find brackets
    sample_rs = np.linspace(r_low, r_high, 21)
    values = []
    for rv in sample_rs:
        try:
            values.append((rv, excess_demand(rv)))
        except Exception:
            values.append((rv, np.inf))
    # Try to find sign change
    bracket = None
    for i in range(len(values)-1):
        r1, e1 = values[i]
        r2, e2 = values[i+1]
        if np.isfinite(e1) and np.isfinite(e2) and e1 * e2 < 0:
            bracket = (r1, r2)
            break
    if bracket is None:
        # Pick r with smallest |excess|
        best_r, best_e = min(values, key=lambda t: abs(t[1]))
        print(f"Warning: No sign change found. Using r with min |excess|: r={best_r:.5f}, excess={best_e:.6f}")
        return best_r

    a, b = bracket
    fa = excess_demand(a)
    fb = excess_demand(b)
    c = 0.5 * (a + b)
    fc = excess_demand(c)
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = excess_demand(c)
        if abs(fc) < tol:
            return c
        # Maintain bracket if possible
        if fa * fc < 0:
            b, fb = c, fc
        elif fb * fc < 0:
            a, fa = c, fc
        else:
            # If sign condition fails due to numerical noise, shrink towards side with smaller |excess|
            if abs(fa) < abs(fb):
                a = 0.5 * (a + c)
                fa = excess_demand(a)
            else:
                b = 0.5 * (b + c)
                fb = excess_demand(b)
    print(f"Warning: Equilibrium search did not converge. Final excess demand: {fc:.6f}")
    # Fallback: return the r among {a, b, c} with smallest |excess|
    candidates = [(a, fa), (b, fb), (c, fc)]
    best_r, _ = min(candidates, key=lambda t: abs(t[1]))
    return best_r

def compute_aggregate_capital(env, dist):
    """Compute aggregate capital from distribution."""
    # Normalize to mass 1 for consistency with L=1 in firm FOCs
    asset_grid = env.asset_grid.reshape(1, 1, -1)
    total_mass = np.sum(dist)
    if total_mass <= 0 or not np.isfinite(total_mass):
        return np.nan
    dist_norm = dist / total_mass
    K = np.sum(dist_norm * asset_grid)
    # Guard against tiny negatives/complex due to numerical issues
    K = np.real_if_close(K)
    K = float(np.maximum(K, 0.0))
    return K

def compute_output(env, K):
    """Compute aggregate output."""
    L = 1.0  # Normalized labor supply
    K_eff = max(K, 1e-12)
    Y = env.parameters['X'] * (K_eff**env.parameters['alpha']) * (L**(1-env.parameters['alpha']))
    return float(np.real_if_close(Y))

def compute_welfare(env, V, dist):
    """Compute aggregate welfare."""
    return np.sum(dist * V)

def plot_comparison_results(env_high, env_zero, dist_high, dist_zero, 
                          policy_high, policy_zero):
    """Plot comparison of asset distributions and policy functions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Domeij-Heathcote Capital Tax Elimination Results', fontsize=14)
    
    # Asset distributions
    asset_dist_high = np.sum(dist_high, axis=(0,1))
    asset_dist_zero = np.sum(dist_zero, axis=(0,1))
    
    axes[0,0].plot(env_high.asset_grid, asset_dist_high, 'b-', label='High Tax (40%)', linewidth=2)
    axes[0,0].plot(env_zero.asset_grid, asset_dist_zero, 'r--', label='Zero Tax (0%)', linewidth=2)
    axes[0,0].set_xlabel('Assets')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Asset Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Policy functions (for first agent type and median income)
    median_income_idx = len(env_high.income_grid) // 2
    
    axes[0,1].plot(env_high.asset_grid, policy_high[0, median_income_idx, :], 
                   'b-', label='High Tax (40%)', linewidth=2)
    axes[0,1].plot(env_zero.asset_grid, policy_zero[0, median_income_idx, :], 
                   'r--', label='Zero Tax (0%)', linewidth=2)
    axes[0,1].plot(env_high.asset_grid, env_high.asset_grid, 'k:', alpha=0.5, label='45° line')
    axes[0,1].set_xlabel('Current Assets')
    axes[0,1].set_ylabel('Next Period Assets')
    axes[0,1].set_title('Policy Function (Median Income)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Savings rates
    savings_high = policy_high[0, median_income_idx, :] / (env_high.asset_grid + 1e-8)
    savings_zero = policy_zero[0, median_income_idx, :] / (env_zero.asset_grid + 1e-8)
    
    # Clip to reasonable range
    savings_high = np.clip(savings_high, 0, 2)
    savings_zero = np.clip(savings_zero, 0, 2)
    
    axes[1,0].plot(env_high.asset_grid, savings_high, 'b-', label='High Tax (40%)', linewidth=2)
    axes[1,0].plot(env_zero.asset_grid, savings_zero, 'r--', label='Zero Tax (0%)', linewidth=2)
    axes[1,0].set_xlabel('Assets')
    axes[1,0].set_ylabel('Savings Rate')
    axes[1,0].set_title('Savings Rate (Median Income)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Comparison summary (text)
    K_high = compute_aggregate_capital(env_high, dist_high)
    K_zero = compute_aggregate_capital(env_zero, dist_zero)
    Y_high = compute_output(env_high, K_high)
    Y_zero = compute_output(env_zero, K_zero)
    
    summary_text = f"""Key Results:
    
Capital Stock:     {K_high:.3f} → {K_zero:.3f} ({(K_zero/K_high-1)*100:+.1f}%)
Output:           {Y_high:.3f} → {Y_zero:.3f} ({(Y_zero/Y_high-1)*100:+.1f}%)
Interest Rate:    {env_high.parameters['interest_rate']:.3f} → {env_zero.parameters['interest_rate']:.3f}
K/Y Ratio:        {K_high/Y_high:.3f} → {K_zero/Y_zero:.3f}

This replicates the key finding of Domeij & Heathcote (2004):
Capital tax elimination has moderate effects on aggregates
due to heterogeneity and incomplete markets."""
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('domeij_heathcote_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to: domeij_heathcote_results.png")

if __name__ == "__main__":
    results = run_domeij_heathcote_experiment() 
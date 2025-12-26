import numpy as np
import time

def build_y_bus(num_buses, branch_data):
    """
    Constructs the Y-bus matrix from branch data.
    branch_data: list of tuples (from_bus, to_bus, R, X, B)
    """
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)
    
    for branch in branch_data:
        f, t, r, x, b = branch
        # Convert 1-based index to 0-based
        i = int(f) - 1
        j = int(t) - 1
        
        z = complex(r, x)
        y = 1 / z
        b_shunt = complex(0, b / 2)
        
        Y_bus[i, i] += y + b_shunt
        Y_bus[j, j] += y + b_shunt
        Y_bus[i, j] -= y
        Y_bus[j, i] -= y
        
    return Y_bus

def build_b_matrices(num_buses, branch_data, bus_types):
    """
    Constructs B' and B'' matrices for Fast Decoupled Load Flow.
    B' is used for P-theta updates (size: non-slack x non-slack).
    B'' is used for Q-V updates (size: PQ x PQ).
    """
    # Identify bus indices
    slack_bus = np.where(bus_types == 0)[0][0]
    pq_buses = np.where(bus_types == 1)[0]
    pv_buses = np.where(bus_types == 2)[0]
    non_slack = np.sort(np.concatenate((pq_buses, pv_buses)))
    
    # Map original bus indices to matrix indices
    map_ns = {bus: i for i, bus in enumerate(non_slack)}
    map_pq = {bus: i for i, bus in enumerate(pq_buses)}
    
    n_ns = len(non_slack)
    n_pq = len(pq_buses)
    
    B_prime = np.zeros((n_ns, n_ns))
    B_dprime = np.zeros((n_pq, n_pq))
    
    # Build B matrices
    # Approximation: R << X, so G is negligible.
    # B elements are derived from -1/X.
    
    for branch in branch_data:
        f, t, r, x, b = branch
        i = int(f) - 1
        j = int(t) - 1
        
        # Series susceptance approximation B_ij ~ -1/X_ij
        # We use b_val = -1/x
        b_val = -1.0 / x
        
        # Update B' (Non-Slack buses)
        # Diagonal: Sum of -b_val (positive sum of 1/X)
        # Off-diagonal: b_val (negative -1/X)
        
        if i in map_ns and j in map_ns:
            idx_i, idx_j = map_ns[i], map_ns[j]
            B_prime[idx_i, idx_j] += b_val
            B_prime[idx_j, idx_i] += b_val
        if i in map_ns:
            B_prime[map_ns[i], map_ns[i]] -= b_val
        if j in map_ns:
            B_prime[map_ns[j], map_ns[j]] -= b_val
            
        # Update B'' (PQ buses)
        if i in map_pq and j in map_pq:
            idx_i, idx_j = map_pq[i], map_pq[j]
            B_dprime[idx_i, idx_j] += b_val
            B_dprime[idx_j, idx_i] += b_val
        if i in map_pq:
            B_dprime[map_pq[i], map_pq[i]] -= b_val
        if j in map_pq:
            B_dprime[map_pq[j], map_pq[j]] -= b_val
            
    return B_prime, B_dprime, non_slack, pq_buses

def fast_decoupled_load_flow(Y_bus, P_spec, Q_spec, V_init, bus_types, branch_data, max_iter=100, tol=1e-4):
    V = np.array(V_init, copy=True)
    num_buses = len(V)
    
    B_prime, B_dprime, non_slack, pq_buses = build_b_matrices(num_buses, branch_data, bus_types)
    
    print(f"{'Iter':<5} {'Max P Mismatch':<20} {'Max Q Mismatch':<20}")
    print("-" * 50)
    
    for it in range(max_iter):
        # 1. Calculate P mismatches
        S_calc = V * np.conj(Y_bus @ V)
        P_calc = np.real(S_calc)
        dP = P_spec[non_slack] - P_calc[non_slack]
        
        # Normalize mismatch by V magnitude: dP/V
        dP_norm = dP / np.abs(V[non_slack])
        
        max_dP = np.max(np.abs(dP))
        
        # Solve P-theta: [B'] [dTheta] = [dP/V]
        dTheta = np.linalg.solve(B_prime, dP_norm)
        
        # Update Angles
        V_ang = np.angle(V)
        V_ang[non_slack] += dTheta
        V = np.abs(V) * np.exp(1j * V_ang)
        
        # 2. Calculate Q mismatches with updated angles
        S_calc = V * np.conj(Y_bus @ V)
        Q_calc = np.imag(S_calc)
        dQ = Q_spec[pq_buses] - Q_calc[pq_buses]
        
        max_dQ = np.max(np.abs(dQ))
        
        print(f"{it+1:<5} {max_dP:<20.6f} {max_dQ:<20.6f}")
        
        if max_dP < tol and max_dQ < tol:
            print("-" * 50)
            print(f"Converged in {it + 1} iterations.")
            return V
        
        # Normalize mismatch by V magnitude: dQ/V
        dQ_norm = dQ / np.abs(V[pq_buses])
        
        # Solve Q-V: [B''] [dV] = [dQ/V]
        dV_mag = np.linalg.solve(B_dprime, dQ_norm)
        
        # Update Magnitudes
        V_mag = np.abs(V)
        V_mag[pq_buses] += dV_mag
        V = V_mag * np.exp(1j * np.angle(V))
            
    print("Fast Decoupled method did not converge within the maximum number of iterations.")
    return V

if __name__ == "__main__":
    # IEEE 9-Bus System Data (Same as Newton-Raphson example)
    num_buses = 9
    bus_types = np.array([0, 2, 2, 1, 1, 1, 1, 1, 1]) # 0:Slack, 1:PQ, 2:PV
    
    P_specified = np.zeros(num_buses)
    Q_specified = np.zeros(num_buses)
    
    # Generators
    P_specified[1] = 1.63; P_specified[2] = 0.85
    # Loads
    P_specified[4] = -1.25; Q_specified[4] = -0.50
    P_specified[5] = -0.90; Q_specified[5] = -0.30
    P_specified[7] = -1.00; Q_specified[7] = -0.35
    
    V_init = np.ones(num_buses, dtype=complex)
    V_init[0] = 1.04 + 0j
    V_init[1] = 1.025 + 0j
    V_init[2] = 1.025 + 0j
    
    branch_data = [
        (4, 5, 0.0100, 0.0850, 0.1760), (4, 6, 0.0170, 0.0920, 0.1580),
        (5, 7, 0.0320, 0.1610, 0.3060), (6, 9, 0.0390, 0.1700, 0.3580),
        (7, 8, 0.0085, 0.0720, 0.1490), (8, 9, 0.0119, 0.1008, 0.2090),
        (1, 4, 0.0, 0.0576, 0.0), (2, 7, 0.0, 0.0625, 0.0), (3, 9, 0.0, 0.0586, 0.0)
    ]
    
    Y_bus = build_y_bus(num_buses, branch_data)
    
    start_time = time.time()
    final_V = fast_decoupled_load_flow(Y_bus, P_specified, Q_specified, V_init, bus_types, branch_data)
    end_time = time.time()
    
    print(f"\nExecution Time: {end_time - start_time:.6f} seconds")
    print("\nFinal Voltages:")
    print(f"{'Bus':<5} {'Mag (pu)':<10} {'Angle (deg)':<12}")
    for i in range(num_buses):
        print(f"{i+1:<5} {np.abs(final_V[i]):<10.4f} {np.degrees(np.angle(final_V[i])):<12.4f}")
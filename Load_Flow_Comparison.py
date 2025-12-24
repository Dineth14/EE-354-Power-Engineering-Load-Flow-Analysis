import numpy as np
import time

# ==========================================
# Data Section (IEEE 9-Bus System)
# ==========================================

def get_ieee_9_bus_data():
    num_buses = 9
    
    # Bus Types: 0=Slack, 1=PQ, 2=PV
    # Bus 1: Slack, Bus 2,3: PV, Others: PQ
    bus_types = np.array([0, 2, 2, 1, 1, 1, 1, 1, 1])
    
    # Specified P and Q (p.u.)
    P_specified = np.zeros(num_buses)
    Q_specified = np.zeros(num_buses)
    
    # Generators (Generation is positive)
    P_specified[1] = 1.63  # Bus 2
    P_specified[2] = 0.85  # Bus 3
    
    # Loads (Load is negative)
    P_specified[4] = -1.25 # Bus 5
    Q_specified[4] = -0.50
    P_specified[5] = -0.90 # Bus 6
    Q_specified[5] = -0.30
    P_specified[7] = -1.00 # Bus 8
    Q_specified[7] = -0.35
    
    # Initial Voltages (Flat Start)
    # Note: PV buses have specified magnitudes
    V_init = np.ones(num_buses, dtype=complex)
    V_init[0] = 1.04 + 0j  # Bus 1 (Slack)
    V_init[1] = 1.025 + 0j # Bus 2 (PV)
    V_init[2] = 1.025 + 0j # Bus 3 (PV)
    
    # Branch Data: (From, To, R, X, B)
    branch_data = [
        (4, 5, 0.0100, 0.0850, 0.1760),
        (4, 6, 0.0170, 0.0920, 0.1580),
        (5, 7, 0.0320, 0.1610, 0.3060),
        (6, 9, 0.0390, 0.1700, 0.3580),
        (7, 8, 0.0085, 0.0720, 0.1490),
        (8, 9, 0.0119, 0.1008, 0.2090),
        # Transformers (B=0)
        (1, 4, 0.0, 0.0576, 0.0),
        (2, 7, 0.0, 0.0625, 0.0),
        (3, 9, 0.0, 0.0586, 0.0)
    ]
    
    return num_buses, bus_types, P_specified, Q_specified, V_init, branch_data

def build_y_bus(num_buses, branch_data):
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)
    for branch in branch_data:
        f, t, r, x, b = branch
        i, j = int(f) - 1, int(t) - 1
        z = complex(r, x)
        y = 1 / z
        b_shunt = complex(0, b / 2)
        Y_bus[i, i] += y + b_shunt
        Y_bus[j, j] += y + b_shunt
        Y_bus[i, j] -= y
        Y_bus[j, i] -= y
    return Y_bus

# ==========================================
# Method 1: Newton-Raphson
# ==========================================

def newton_raphson(Y_bus, P_spec, Q_spec, V_init, bus_types, max_iter=20, tol=1e-4):
    V = np.array(V_init, copy=True)
    num_buses = len(V)
    
    slack_bus = np.where(bus_types == 0)[0][0]
    pq_buses = np.where(bus_types == 1)[0]
    pv_buses = np.where(bus_types == 2)[0]
    non_slack = np.sort(np.concatenate((pq_buses, pv_buses)))
    
    for it in range(max_iter):
        S_calc = V * np.conj(Y_bus @ V)
        P_calc = np.real(S_calc)
        Q_calc = np.imag(S_calc)
        
        dP = P_spec[non_slack] - P_calc[non_slack]
        dQ = Q_spec[pq_buses] - Q_calc[pq_buses]
        mismatch = np.concatenate((dP, dQ))
        
        if np.max(np.abs(mismatch)) < tol:
            return V, it + 1
            
        # Jacobian
        n_ns = len(non_slack)
        n_pq = len(pq_buses)
        J11 = np.zeros((n_ns, n_ns))
        J12 = np.zeros((n_ns, n_pq))
        J21 = np.zeros((n_pq, n_ns))
        J22 = np.zeros((n_pq, n_pq))
        
        # J11, J21 (Angle derivatives)
        for r, i in enumerate(non_slack):
            for c, k in enumerate(non_slack):
                if i == k:
                    J11[r, c] = -Q_calc[i] - np.imag(Y_bus[i, i]) * np.abs(V[i])**2
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J11[r, c] = np.abs(V[i]*V[k]) * (np.real(y_ik)*np.sin(delta_ik) - np.imag(y_ik)*np.cos(delta_ik))
        
        for r, i in enumerate(pq_buses):
            for c, k in enumerate(non_slack):
                if i == k:
                    J21[r, c] = P_calc[i] - np.real(Y_bus[i, i]) * np.abs(V[i])**2
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J21[r, c] = -np.abs(V[i]*V[k]) * (np.real(y_ik)*np.cos(delta_ik) + np.imag(y_ik)*np.sin(delta_ik))
                    
        # J12, J22 (Magnitude derivatives)
        for r, i in enumerate(non_slack):
            for c, k in enumerate(pq_buses):
                if i == k:
                    J12[r, c] = P_calc[i]/np.abs(V[i]) + np.real(Y_bus[i, i])*np.abs(V[i])
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J12[r, c] = np.abs(V[i]) * (np.real(y_ik)*np.cos(delta_ik) + np.imag(y_ik)*np.sin(delta_ik))
                    
        for r, i in enumerate(pq_buses):
            for c, k in enumerate(pq_buses):
                if i == k:
                    J22[r, c] = Q_calc[i]/np.abs(V[i]) - np.imag(Y_bus[i, i])*np.abs(V[i])
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J22[r, c] = np.abs(V[i]) * (np.real(y_ik)*np.sin(delta_ik) - np.imag(y_ik)*np.cos(delta_ik))
                    
        J = np.block([[J11, J12], [J21, J22]])
        dx = np.linalg.solve(J, mismatch)
        
        V_ang = np.angle(V)
        V_mag = np.abs(V)
        
        V_ang[non_slack] += dx[:n_ns]
        V_mag[pq_buses] += dx[n_ns:]
        
        V = V_mag * np.exp(1j * V_ang)
        
    return V, max_iter

# ==========================================
# Method 2: Gauss-Seidel
# ==========================================

def gauss_seidel(Y_bus, P_spec, Q_spec, V_init, bus_types, max_iter=1000, tol=1e-4):
    V = np.array(V_init, copy=True)
    num_buses = len(V)
    
    for it in range(max_iter):
        V_prev = np.copy(V)
        max_error = 0
        
        for i in range(num_buses):
            if bus_types[i] == 0: # Slack
                continue
            
            # Calculate sum of Yij * Vj
            sum_YV = 0
            for j in range(num_buses):
                if i != j:
                    sum_YV += Y_bus[i, j] * V[j]
            
            # Handle PV Buses
            if bus_types[i] == 2:
                # Estimate Q
                Q_calc = -np.imag(np.conj(V[i]) * (sum_YV + Y_bus[i, i] * V[i]))
                # Use calculated Q to update V
                S_inj = P_spec[i] - 1j * Q_calc
            else:
                # PQ Bus
                S_inj = P_spec[i] - 1j * Q_spec[i]
            
            # Update Voltage
            V_new = (1 / Y_bus[i, i]) * ((S_inj / np.conj(V[i])) - sum_YV)
            
            # Enforce PV Bus Voltage Magnitude
            if bus_types[i] == 2:
                V_new = np.abs(V_init[i]) * np.exp(1j * np.angle(V_new))
            
            # Update V immediately (Gauss-Seidel)
            V[i] = V_new
        
        # Check convergence
        max_error = np.max(np.abs(V - V_prev))
        if max_error < tol:
            return V, it + 1
            
    return V, max_iter

# ==========================================
# Method 3: Fast Decoupled Load Flow
# ==========================================

def build_b_matrices(num_buses, branch_data, bus_types):
    # B' matrix (for P-theta): Size (Num_Non_Slack x Num_Non_Slack)
    # B'' matrix (for Q-V): Size (Num_PQ x Num_PQ)
    
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
    
    # Build B matrices assuming R << X (ignore R)
    for branch in branch_data:
        f, t, r, x, b = branch
        i, j = int(f) - 1, int(t) - 1
        
        # Susceptance for B matrix (approx -1/x)
        b_val = -1.0 / x
        
        # Update B' (Non-Slack buses)
        if i in map_ns and j in map_ns:
            idx_i, idx_j = map_ns[i], map_ns[j]
            B_prime[idx_i, idx_j] -= b_val
            B_prime[idx_j, idx_i] -= b_val
        if i in map_ns:
            B_prime[map_ns[i], map_ns[i]] += b_val
        if j in map_ns:
            B_prime[map_ns[j], map_ns[j]] += b_val
            
        # Update B'' (PQ buses)
        if i in map_pq and j in map_pq:
            idx_i, idx_j = map_pq[i], map_pq[j]
            B_dprime[idx_i, idx_j] -= b_val
            B_dprime[idx_j, idx_i] -= b_val
        if i in map_pq:
            B_dprime[map_pq[i], map_pq[i]] += b_val
        if j in map_pq:
            B_dprime[map_pq[j], map_pq[j]] += b_val
            
    return B_prime, B_dprime, non_slack, pq_buses

def fast_decoupled(Y_bus, P_spec, Q_spec, V_init, bus_types, branch_data, max_iter=100, tol=1e-4):
    V = np.array(V_init, copy=True)
    
    B_prime, B_dprime, non_slack, pq_buses = build_b_matrices(len(V), branch_data, bus_types)
    
    for it in range(max_iter):
        # 1. Calculate P mismatches
        S_calc = V * np.conj(Y_bus @ V)
        P_calc = np.real(S_calc)
        dP = P_spec[non_slack] - P_calc[non_slack]
        
        # Normalize mismatch by V magnitude
        dP_norm = dP / np.abs(V[non_slack])
        
        if np.max(np.abs(dP)) < tol:
            # Check Q convergence if P converged
            Q_calc = np.imag(S_calc)
            dQ = Q_spec[pq_buses] - Q_calc[pq_buses]
            if np.max(np.abs(dQ)) < tol:
                return V, it + 1
        
        # Solve P-theta: dP/V = B' * dTheta
        dTheta = np.linalg.solve(B_prime, dP_norm)
        
        # Update Angles
        V_ang = np.angle(V)
        V_ang[non_slack] += dTheta
        V = np.abs(V) * np.exp(1j * V_ang)
        
        # 2. Calculate Q mismatches with updated angles
        S_calc = V * np.conj(Y_bus @ V)
        Q_calc = np.imag(S_calc)
        dQ = Q_spec[pq_buses] - Q_calc[pq_buses]
        
        # Normalize mismatch by V magnitude
        dQ_norm = dQ / np.abs(V[pq_buses])
        
        # Solve Q-V: dQ/V = B'' * dV
        dV_mag = np.linalg.solve(B_dprime, dQ_norm)
        
        # Update Magnitudes
        V_mag = np.abs(V)
        V_mag[pq_buses] += dV_mag
        V = V_mag * np.exp(1j * np.angle(V))
        
        # Check full convergence
        if np.max(np.abs(dP)) < tol and np.max(np.abs(dQ)) < tol:
            return V, it + 1
            
    return V, max_iter

# ==========================================
# Main Analysis
# ==========================================

if __name__ == "__main__":
    print("Load Flow Analysis - Method Comparison")
    print("======================================")
    
    # 1. Setup Data
    num_buses, bus_types, P_spec, Q_spec, V_init, branch_data = get_ieee_9_bus_data()
    Y_bus = build_y_bus(num_buses, branch_data)
    
    # 2. Run Newton-Raphson
    start_time = time.time()
    V_nr, iter_nr = newton_raphson(Y_bus, P_spec, Q_spec, V_init, bus_types)
    time_nr = time.time() - start_time
    
    # 3. Run Gauss-Seidel
    start_time = time.time()
    V_gs, iter_gs = gauss_seidel(Y_bus, P_spec, Q_spec, V_init, bus_types)
    time_gs = time.time() - start_time
    
    # 4. Run Fast Decoupled
    start_time = time.time()
    V_fd, iter_fd = fast_decoupled(Y_bus, P_spec, Q_spec, V_init, bus_types, branch_data)
    time_fd = time.time() - start_time
    
    # 5. Results Analysis
    print(f"\n{'Method':<20} {'Iterations':<12} {'Time (s)':<12} {'Max V Diff (vs NR)':<20}")
    print("-" * 65)
    print(f"{'Newton-Raphson':<20} {iter_nr:<12} {time_nr:<12.6f} {'-':<20}")
    
    diff_gs = np.max(np.abs(V_gs - V_nr))
    print(f"{'Gauss-Seidel':<20} {iter_gs:<12} {time_gs:<12.6f} {diff_gs:<20.6f}")
    
    diff_fd = np.max(np.abs(V_fd - V_nr))
    print(f"{'Fast Decoupled':<20} {iter_fd:<12} {time_fd:<12.6f} {diff_fd:<20.6f}")
    
    print("\nDetailed Voltage Comparison (Magnitude p.u. / Angle deg)")
    print("-" * 85)
    print(f"{'Bus':<5} | {'Newton-Raphson':^22} | {'Gauss-Seidel':^22} | {'Fast Decoupled':^22}")
    print(f"{'':<5} | {'Mag':<8} {'Ang':<12} | {'Mag':<8} {'Ang':<12} | {'Mag':<8} {'Ang':<12}")
    print("-" * 85)
    
    for i in range(num_buses):
        # NR
        m_nr = np.abs(V_nr[i])
        a_nr = np.degrees(np.angle(V_nr[i]))
        # GS
        m_gs = np.abs(V_gs[i])
        a_gs = np.degrees(np.angle(V_gs[i]))
        # FD
        m_fd = np.abs(V_fd[i])
        a_fd = np.degrees(np.angle(V_fd[i]))
        
        print(f"{i+1:<5} | {m_nr:<8.4f} {a_nr:<12.4f} | {m_gs:<8.4f} {a_gs:<12.4f} | {m_fd:<8.4f} {a_fd:<12.4f}")

    print("\nAnalysis Summary:")
    print("1. Newton-Raphson (NR) provides the most robust convergence with fewest iterations.")
    print("2. Gauss-Seidel (GS) requires significantly more iterations but has simpler calculations per step.")
    print("3. Fast Decoupled (FD) is a compromise, offering faster iterations than NR but more iterations total.")
    print("   (Note: FD convergence depends heavily on R/X ratios and system assumptions).")
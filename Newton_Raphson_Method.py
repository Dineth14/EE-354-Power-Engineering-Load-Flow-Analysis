import numpy as np

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

def newton_raphson(Y_bus, P_specified, Q_specified, V_init, bus_types, max_iter=100, tol=1e-4):
    num_buses = len(Y_bus)
    
    # Initialize voltage magnitudes and angles
    V = np.array(V_init, copy=True)
    
    # Identify bus types
    # 0: Slack, 1: PQ, 2: PV
    slack_bus = np.where(bus_types == 0)[0][0]
    pq_buses = np.where(bus_types == 1)[0]
    pv_buses = np.where(bus_types == 2)[0]
    
    # Non-slack buses (PV + PQ) for P mismatch
    non_slack_buses = np.sort(np.concatenate((pq_buses, pv_buses)))
    
    for iteration in range(max_iter):
        # Calculate P and Q for all buses based on current V
        S_calc = V * np.conj(Y_bus @ V)
        P_calc = np.real(S_calc)
        Q_calc = np.imag(S_calc)
        
        # Calculate mismatches
        dP = P_specified[non_slack_buses] - P_calc[non_slack_buses]
        dQ = Q_specified[pq_buses] - Q_calc[pq_buses]
        
        mismatch = np.concatenate((dP, dQ))
        
        if np.max(np.abs(mismatch)) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return V, P_calc, Q_calc
        
        # Jacobian Matrix Construction
        # J11: dP/d_delta (non_slack x non_slack)
        # J12: dP/d_Vmag (non_slack x pq)
        # J21: dQ/d_delta (pq x non_slack)
        # J22: dQ/d_Vmag (pq x pq)
        
        n_non_slack = len(non_slack_buses)
        n_pq = len(pq_buses)
        
        J11 = np.zeros((n_non_slack, n_non_slack))
        J12 = np.zeros((n_non_slack, n_pq))
        J21 = np.zeros((n_pq, n_non_slack))
        J22 = np.zeros((n_pq, n_pq))
        
        # Fill J11 and J21 (Derivatives w.r.t Angles)
        for r, i in enumerate(non_slack_buses):
            for c, k in enumerate(non_slack_buses):
                if i == k:
                    J11[r, c] = -Q_calc[i] - np.imag(Y_bus[i, i]) * np.abs(V[i])**2
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J11[r, c] = np.abs(V[i] * V[k]) * (np.real(y_ik) * np.sin(delta_ik) - np.imag(y_ik) * np.cos(delta_ik))
        
        for r, i in enumerate(pq_buses):
            for c, k in enumerate(non_slack_buses):
                if i == k:
                    J21[r, c] = P_calc[i] - np.real(Y_bus[i, i]) * np.abs(V[i])**2
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J21[r, c] = -np.abs(V[i] * V[k]) * (np.real(y_ik) * np.cos(delta_ik) + np.imag(y_ik) * np.sin(delta_ik))

        # Fill J12 and J22 (Derivatives w.r.t Magnitudes)
        for r, i in enumerate(non_slack_buses):
            for c, k in enumerate(pq_buses):
                if i == k:
                    J12[r, c] = P_calc[i] / np.abs(V[i]) + np.real(Y_bus[i, i]) * np.abs(V[i])
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J12[r, c] = np.abs(V[i]) * (np.real(y_ik) * np.cos(delta_ik) + np.imag(y_ik) * np.sin(delta_ik))

        for r, i in enumerate(pq_buses):
            for c, k in enumerate(pq_buses):
                if i == k:
                    J22[r, c] = Q_calc[i] / np.abs(V[i]) - np.imag(Y_bus[i, i]) * np.abs(V[i])
                else:
                    y_ik = Y_bus[i, k]
                    delta_ik = np.angle(V[i]) - np.angle(V[k])
                    J22[r, c] = np.abs(V[i]) * (np.real(y_ik) * np.sin(delta_ik) - np.imag(y_ik) * np.cos(delta_ik))
        
        # Assemble Jacobian
        J = np.block([[J11, J12], [J21, J22]])
        
        # Solve for update
        dx = np.linalg.solve(J, mismatch)
        
        # Update
        d_angle = dx[:n_non_slack]
        d_vmag = dx[n_non_slack:]
        
        # Update Angles
        current_angles = np.angle(V)
        current_angles[non_slack_buses] += d_angle
        
        # Update Magnitudes
        current_mags = np.abs(V)
        current_mags[pq_buses] += d_vmag
        
        # Reconstruct V
        V = current_mags * np.exp(1j * current_angles)
        
    print("Newton-Raphson did not converge within the maximum number of iterations.")
    return V, P_calc, Q_calc

if __name__ == "__main__":
    print("Student ID: XXXXXX") # Replace with actual ID
    
    # IEEE 9-Bus System Data
    num_buses = 9
    
    # Bus Types: 0=Slack, 1=PQ, 2=PV
    # Bus 1: Slack
    # Bus 2, 3: PV
    # Bus 4-9: PQ
    bus_types = np.array([0, 2, 2, 1, 1, 1, 1, 1, 1])
    
    # Specified P and Q (in p.u.)
    # Generation is positive, Load is negative
    P_specified = np.zeros(num_buses)
    Q_specified = np.zeros(num_buses)
    
    # Generators
    P_specified[1] = 1.63  # Bus 2
    P_specified[2] = 0.85  # Bus 3
    
    # Loads
    P_specified[4] = -1.25 # Bus 5
    Q_specified[4] = -0.50
    P_specified[5] = -0.90 # Bus 6
    Q_specified[5] = -0.30
    P_specified[7] = -1.00 # Bus 8
    Q_specified[7] = -0.35
    
    # Initial Voltages (Flat Start)
    V_init = np.ones(num_buses, dtype=complex)
    V_init[0] = 1.04 + 0j  # Bus 1 (Slack)
    V_init[1] = 1.025 + 0j # Bus 2 (PV)
    V_init[2] = 1.025 + 0j # Bus 3 (PV)
    
    # Branch Data: (From, To, R, X, B)
    # Note: B in table is total line charging.
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

    # Build Y-Bus
    Y_bus = build_y_bus(num_buses, branch_data)
    
    # Solve
    final_V, P_final, Q_final = newton_raphson(Y_bus, P_specified, Q_specified, V_init, bus_types)

    print("\n--- Results ---")
    print(f"{'Bus':<5} {'V_mag (pu)':<12} {'Angle (deg)':<12} {'P (pu)':<12} {'Q (pu)':<12}")
    for i in range(num_buses):
        v_mag = np.abs(final_V[i])
        v_ang = np.degrees(np.angle(final_V[i]))
        print(f"{i+1:<5} {v_mag:<12.4f} {v_ang:<12.4f} {P_final[i]:<12.4f} {Q_final[i]:<12.4f}")
        
    print("\n--- Line Flows and Losses ---")
    print(f"{'From':<5} {'To':<5} {'P_flow (pu)':<12} {'Q_flow (pu)':<12} {'Loss_P':<10} {'Loss_Q':<10}")
    
    total_loss_P = 0
    total_loss_Q = 0
    
    for branch in branch_data:
        f, t, r, x, b = branch
        i = int(f) - 1
        j = int(t) - 1
        
        z = complex(r, x)
        y_series = 1/z
        y_shunt = complex(0, b/2)
        
        # Current from i to j
        I_ij = (final_V[i] - final_V[j]) * y_series + final_V[i] * y_shunt
        S_ij = final_V[i] * np.conj(I_ij)
        
        # Current from j to i
        I_ji = (final_V[j] - final_V[i]) * y_series + final_V[j] * y_shunt
        S_ji = final_V[j] * np.conj(I_ji)
        
        loss = S_ij + S_ji
        total_loss_P += np.real(loss)
        total_loss_Q += np.imag(loss)
        
        print(f"{f:<5} {t:<5} {np.real(S_ij):<12.4f} {np.imag(S_ij):<12.4f} {np.real(loss):<10.4f} {np.imag(loss):<10.4f}")
        
    print(f"\nTotal System Losses: P = {total_loss_P:.4f} pu, Q = {total_loss_Q:.4f} pu")

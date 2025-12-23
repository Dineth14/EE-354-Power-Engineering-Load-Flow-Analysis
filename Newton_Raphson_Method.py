import numpy as np

def newton_raphson(Y_bus, P_specified, Q_specified, V_init, max_iter=100, tol=1e-6):
    num_buses = len(Y_bus)
    
    # Initialize voltage magnitudes and angles
    V_mag = np.abs(V_init)
    V_angle = np.angle(V_init)
    
    # Identify bus types
    # Slack bus (Bus 0) - V_mag and V_angle fixed
    # PQ buses - P and Q specified
    # PV buses - P and V_mag specified (Q is unknown)
    
    # For this example, let's assume Bus 0 is Slack, and others are PQ
    # This needs to be generalized for PV buses in a full implementation
    
    # Create a vector of unknowns: angles for PQ/PV buses, magnitudes for PQ buses
    # For a system with 1 slack bus and (num_buses - 1) PQ buses:
    # Unknowns are delta_1, ..., delta_{N-1}, |V_1|, ..., |V_{N-1}|
    # (Assuming slack bus is bus 0, so we start from bus 1)
    
    # Let's simplify for now: assume all non-slack buses are PQ buses.
    # Unknowns: V_angle[1:], V_mag[1:]
    
    # Initial guess for unknowns
    x = np.concatenate((V_angle[1:], V_mag[1:]))
    
    for iteration in range(max_iter):
        # Reconstruct V from current x
        V = np.zeros(num_buses, dtype=complex)
        V[0] = V_init[0] # Slack bus voltage is fixed
        
        V_angle[1:] = x[:num_buses-1]
        V_mag[1:] = x[num_buses-1:]
        
        V[1:] = V_mag[1:] * np.exp(1j * V_angle[1:])
        
        # Calculate P and Q for all buses based on current V
        S_calc = V * np.conj(Y_bus @ V)
        P_calc = np.real(S_calc)
        Q_calc = np.imag(S_calc)
        
        # Calculate mismatches (excluding slack bus)
        dP = P_specified[1:] - P_calc[1:]
        dQ = Q_specified[1:] - Q_calc[1:]
        
        mismatch = np.concatenate((dP, dQ))
        
        if np.max(np.abs(mismatch)) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return V
        
        # Jacobian Matrix Construction
        # J = [[J11, J12], [J21, J22]]
        # J11 = dP/d_delta, J12 = dP/d_Vmag
        # J21 = dQ/d_delta, J22 = dQ/d_Vmag
        
        n_pq = num_buses - 1
        J11 = np.zeros((n_pq, n_pq))
        J12 = np.zeros((n_pq, n_pq))
        J21 = np.zeros((n_pq, n_pq))
        J22 = np.zeros((n_pq, n_pq))
        
        for i in range(n_pq):
            bus_i = i + 1
            for k in range(n_pq):
                bus_k = k + 1
                
                if bus_i != bus_k:
                    # Off-diagonal terms
                    Y_ik = Y_bus[bus_i, bus_k]
                    G_ik = np.real(Y_ik)
                    B_ik = np.imag(Y_ik)
                    delta_ik = V_angle[bus_i] - V_angle[bus_k]
                    
                    J11[i, k] = V_mag[bus_i] * V_mag[bus_k] * (G_ik * np.sin(delta_ik) - B_ik * np.cos(delta_ik))
                    J21[i, k] = -V_mag[bus_i] * V_mag[bus_k] * (G_ik * np.cos(delta_ik) + B_ik * np.sin(delta_ik))
                    J12[i, k] = V_mag[bus_i] * (G_ik * np.cos(delta_ik) + B_ik * np.sin(delta_ik))
                    J22[i, k] = V_mag[bus_i] * (G_ik * np.sin(delta_ik) - B_ik * np.cos(delta_ik))
                else:
                    # Diagonal terms
                    Y_ii = Y_bus[bus_i, bus_i]
                    G_ii = np.real(Y_ii)
                    B_ii = np.imag(Y_ii)
                    
                    J11[i, i] = -Q_calc[bus_i] - B_ii * V_mag[bus_i]**2
                    J21[i, i] = P_calc[bus_i] - G_ii * V_mag[bus_i]**2
                    J12[i, i] = P_calc[bus_i] / V_mag[bus_i] + G_ii * V_mag[bus_i]
                    J22[i, i] = Q_calc[bus_i] / V_mag[bus_i] - B_ii * V_mag[bus_i]
        
        # Assemble Jacobian
        J = np.block([[J11, J12], [J21, J22]])
        
        # Solve for update
        dx = np.linalg.solve(J, mismatch)
        
        # Update state vector
        x += dx
        
    print("Newton-Raphson did not converge within the maximum number of iterations.")
    return V

if __name__ == "__main__":
    # Example Usage (Same 3-bus system as Gauss-Seidel example)
    Y_bus = np.array([[2 - 8j, -1 + 4j, -1 + 4j],
                      [-1 + 4j, 2 - 8j, -1 + 4j],
                      [-1 + 4j, -1 + 4j, 2 - 8j]], dtype=complex)

    V_init = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=complex)
    P_specified = np.array([0.0, -0.8, -0.6], dtype=float)
    Q_specified = np.array([0.0, -0.4, -0.3], dtype=float)

    final_V = newton_raphson(Y_bus, P_specified, Q_specified, V_init)

    print("\nFinal Voltages (Newton-Raphson):", final_V)
    print("Magnitudes:", np.abs(final_V))
    print("Angles (degrees):", np.degrees(np.angle(final_V)))

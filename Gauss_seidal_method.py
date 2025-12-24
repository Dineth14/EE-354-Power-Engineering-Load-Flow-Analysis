import numpy as np

def gauss_seidel(Y_bus, P, Q, V, max_iter=100, tol=1e-6):
    num_buses = len(Y_bus)
    
    for iteration in range(max_iter):
        V_new = np.copy(V)
        
        for i in range(num_buses):
            sum_val = 0
            for j in range(num_buses):
                if i != j:
                    sum_val += Y_bus[i, j] * V_new[j]
            
            if i == 0:  # Slack bus
                # For slack bus, V is fixed, so we calculate P and Q
                # This part is usually not updated in the Gauss-Seidel iteration for V
                # but rather calculated after convergence.
                # For the purpose of updating other bus voltages, we skip updating V[0]
                pass 
            else:
                # Calculate injected current I_i
                I_i_conj = (P[i] - 1j * Q[i]) / np.conj(V_new[i])
                
                # Update voltage V_i
                V_new[i] = (1 / Y_bus[i, i]) * (I_i_conj - sum_val)
        
        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return V_new
        
        V = V_new
        
    print("Gauss-Seidel did not converge within the maximum number of iterations.")
    return V

def calculate_power_injections(Y_bus, V):
    num_buses = len(Y_bus)
    S = np.zeros(num_buses, dtype=complex)
    
    for i in range(num_buses):
        S_i = 0
        for j in range(num_buses):
            S_i += Y_bus[i, j] * V[j]
        S[i] = V[i] * np.conj(S_i)
        
    return S

if __name__ == "__main__":
    # Example Usage:
    # Define Y-bus matrix (example for a 3-bus system)
    # Y_bus = np.array([[Y11, Y12, Y13],
    #                   [Y21, Y22, Y23],
    #                   [Y31, Y32, Y33]], dtype=complex)
    
    # For a simple 3-bus system example:
    # Bus 1: Slack bus (V = 1.0 + 0j)
    # Bus 2: PQ bus (P, Q specified)
    # Bus 3: PQ bus (P, Q specified)

    # Let's define an example Y-bus matrix
    # Assuming per unit values
    Y11 = 2 - 8j
    Y12 = -1 + 4j
    Y13 = -1 + 4j
    Y21 = -1 + 4j
    Y22 = 2 - 8j
    Y23 = -1 + 4j
    Y31 = -1 + 4j
    Y32 = -1 + 4j
    Y33 = 2 - 8j

    Y_bus = np.array([[Y11, Y12, Y13],
                      [Y21, Y22, Y23],
                      [Y31, Y32, Y33]], dtype=complex)

    # Initial guess for voltages (all 1.0 + 0j for simplicity, except slack bus)
    V = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=complex)

    # Specified P and Q values for PQ buses (in per unit)
    # P[0] and Q[0] are not used for slack bus in the iteration
    P = np.array([0.0, -0.8, -0.6], dtype=float) # Negative for load
    Q = np.array([0.0, -0.4, -0.3], dtype=float) # Negative for load

    print("Initial Voltages:", V)
    print("Y-bus Matrix:\n", Y_bus)
    print("\nSpecified P:", P)
    print("Specified Q:", Q)

    # Run Gauss-Seidel
    final_V = gauss_seidel(Y_bus, P, Q, V)

    print("\nFinal Voltages (Gauss-Seidel):", final_V)
    print("Magnitudes:", np.abs(final_V))
    print("Angles (degrees):", np.degrees(np.angle(final_V)))

    # Calculate power injections at convergence
    calculated_S = calculate_power_injections(Y_bus, final_V)
    print("\nCalculated Complex Power Injections (S = P + jQ):", calculated_S)
    print("Calculated P:", np.real(calculated_S))
    print("Calculated Q:", np.imag(calculated_S))
import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel_simple(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    errors = []
    
    print(f"Solving Ax = b with {n} variables.")
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            # Sum of A[i,j] * x[j] for known new values (j < i)
            s1 = sum(A[i][j] * x[j] for j in range(i))
            # Sum of A[i,j] * x_old[j] for old values (j > i)
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            
            x[i] = (b[i] - s1 - s2) / A[i][i]
            
        # Calculate error (infinity norm)
        err = np.max(np.abs(x - x_old))
        errors.append(err)
        
        if err < tol:
            print(f"Converged in {k+1} iterations.")
            return x, errors
            
    print("Max iterations reached.")
    return x, errors

if __name__ == "__main__":
    # Define a Diagonally Dominant System
    # 4x +  y -  z = 3
    # 2x + 7y +  z = 19
    #  x - 3y + 12z = 31
    
    A = np.array([[4.0, 1.0, -1.0],
                  [2.0, 7.0, 1.0],
                  [1.0, -3.0, 12.0]])
    
    b = np.array([3.0, 19.0, 31.0])
    
    # Initial guess [0, 0, 0]
    x0 = np.zeros(3)
    
    solution, errors = gauss_seidel_simple(A, b, x0)
    
    print("\nFinal Solution:", solution)
    print("Check A @ x:", A @ solution)
    print("Target b:   ", b)
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'g-o', linewidth=2)
    plt.yscale('log')
    
    plt.title("Gauss-Seidel Convergence Behavior")
    plt.xlabel("Iteration")
    plt.ylabel("Max Absolute Error (Log Scale)")
    plt.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """ The function to solve: f(x) = x^3 - x - 2 """
    return x**3 - x - 2

def df(x):
    """ The derivative of the function: f'(x) = 3x^2 - 1 """
    return 3*x**2 - 1

def newton_raphson_simple(x0, tol=1e-6, max_iter=100):
    x = x0
    history = [x]
    
    print(f"{'Iter':<5} {'x':<12} {'f(x)':<12} {'df(x)':<12}")
    print("-" * 45)
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        print(f"{i:<5} {x:<12.6f} {fx:<12.6f} {dfx:<12.6f}")
        
        if abs(dfx) < 1e-10:
            print("Derivative too close to zero. Stopping.")
            return None, history
            
        x_new = x - fx / dfx
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            print("-" * 45)
            print(f"Converged to {x_new:.6f} in {i+1} iterations.")
            return x_new, history
            
        x = x_new
        
    print("Did not converge.")
    return x, history

if __name__ == "__main__":
    # Initial guess
    x0 = 1.0
    root, history = newton_raphson_simple(x0)
    
    # --- Plotting ---
    x_vals = np.linspace(0, 2.5, 100)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Function and Iterations
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, label='$f(x) = x^3 - x - 2$')
    plt.axhline(0, color='black', linewidth=0.8)
    history = np.array(history)
    plt.plot(history, f(history), 'ro-', label='Iterations', markersize=5)
    
    # Annotate steps
    for i, (xi, yi) in enumerate(zip(history, f(history))):
        plt.text(xi, yi + 0.5, f'{i}', fontsize=9, ha='center')

    plt.title("Newton-Raphson Path")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Convergence Rate
    plt.subplot(1, 2, 2)
    errors = np.abs(history[:-1] - history[-1]) # Error relative to final result
    plt.plot(range(len(errors)), errors, 'bo-')
    plt.yscale('log')
    plt.title("Error Convergence (Log Scale)")
    plt.xlabel("Iteration")
    plt.ylabel("|x_k - x_final|")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
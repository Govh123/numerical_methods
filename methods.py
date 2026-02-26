import numpy as np
import sympy as sp

def improvedEuler(equation_str: str, h: float, x0: float, y0: float, x_end: float):
    x, y = sp.symbols('x y')
    f_expr = sp.sympify(equation_str)
    f = sp.lambdify((x, y), f_expr, modules=["numpy", "math"])
    
    n_list = [0]
    x_values = [x0]
    y_values = [y0]
    
    for nextX in np.arange(x0 + h, x_end + h, h):
        n_list.append(n_list[-1] + 1)
        xn = x_values[-1]
        yn = y_values[-1]
        
        yAsterisk = yn + h * f(xn, yn)
        
        nextY = yn + (h / 2) * (f(xn, yn) + f(nextX, yAsterisk))
        
        y_values.append(nextY)
        x_values.append(nextX)

    return x_values, y_values, n_list


def newton_raphson(equation_str: str, x0: float, tolerance=1e-6):
    x = sp.symbols('x')
    f_expr = sp.sympify(equation_str)
    df_expr = sp.diff(f_expr, x)

    f = sp.lambdify(x, f_expr, modules=["numpy", "math"])
    df = sp.lambdify(x, df_expr, modules=["numpy", "math"])

    x_values = [x0]

    def recursive_solve(current_x):
        f_val = f(current_x)
        
        if abs(f_val) < tolerance:
            return current_x
        
        df_val = df(current_x)
        
        if df_val == 0:
            return None
        
        next_x = current_x - (f_val / df_val)
        x_values.append(next_x)
        return recursive_solve(next_x)
    
    root = recursive_solve(x0)
    return root, x_values, f, df

def RungeKutta(equation_str: str, h: float, x0: float, y0: float, x_end: float):
    x, y = sp.symbols('x y')
    f_expr = sp.sympify(equation_str)
    f = sp.lambdify((x, y), f_expr, modules=["numpy", "math"])
    
    n_list = [0]
    x_values = [x0]
    y_values = [y0]
    ks_values = []
    
    steps = int(round((x_end - x0) / h))
    
    for i in range(steps):
        n_list.append(n_list[-1] + 1)
        xi = x_values[-1]
        yi = y_values[-1]
        
        k1 = f(xi, yi)
        k2 = f(xi + h / 2, yi + k1 * h * (1/2))
        k3 = f(xi + h / 2, yi + k2 * h * (1/2))
        k4 = f(xi + h, yi + k3 * h)
        
        k_promedio = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        nextX = xi + h
        nextY = yi + (k_promedio * h)
        
        x_values.append(nextX)
        y_values.append(nextY)
        ks_values.append((k1, k2, k3, k4, k_promedio))
    
    return x_values, y_values, ks_values, n_list

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from methods import improvedEuler, newton_raphson, RungeKutta

st.set_page_config(page_title="Métodos Numéricos", layout="wide")

st.title("Métodos Numéricos")
st.markdown("---")

method = st.sidebar.selectbox(
    "Selecciona el método numérico:",
    ["Euler Mejorado", "Newton-Raphson", "Runge-Kutta 4"]
)

if method == "Euler Mejorado":
    st.header("Método de Euler Mejorado")
    st.write("Resuelve EDOs de la forma: dy/dx = f(x, y)")
    
    col1, col2 = st.columns(2)
    with col1:
        eq_euler = st.text_input(
            "Ecuación dy/dx (ej: y - x\\**2 + 1, x\\*y, 2\\*x\\*y)",
            help="Usa 'x' e 'y' como variables"
        )
        h_euler = st.number_input("Paso (h) (ej: 0.2)", min_value=0.01, step=0.01)
        x0_euler = st.number_input("x inicial (ej: 0)")
    
    with col2:
        y0_euler = st.number_input("y inicial (ej: 0.5)")
        xend_euler = st.number_input("x final (ej: 2)")
    
    if st.button("Calcular Euler Mejorado"):
        try:
            x_vals, y_vals, yr_vals, errors = improvedEuler(eq_euler, h_euler, x0_euler, y0_euler, xend_euler)
            
            df = pd.DataFrame({
                'x': [round(x, 4) for x in x_vals],
                'y': [round(y, 4) for y in y_vals],
                'yr': [round(yr, 4) for yr in yr_vals],
                'Error Absoluto': [round(e, 6) for e in errors]
            })
            st.write("### Tabla de Resultados")
            st.dataframe(df, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_vals, y_vals, 'o-', label=f'Euler Mejorado (h={h_euler})', color='green', linewidth=2, markersize=6)
            ax.plot(x_vals, yr_vals, 's--', label='Solución Exacta', color='red', linewidth=2, markersize=5)
            ax.set_title(f'Solución: dy/dx = {eq_euler}', fontsize=14)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


elif method == "Newton-Raphson":
    st.header("Método de Newton-Raphson")
    st.write("Encuentra raíces de f(x) = 0")
    
    col1, col2 = st.columns(2)
    with col1:
        eq_nr = st.text_input(
            "Ecuación f(x) (ej: x\\**2 - 4, x\\**3 - 2, sin(x) - 0.5)",
            help="Usa 'x' como variable"
        )
    with col2:
        x0_nr = st.number_input("Valor inicial (ej: 2)")
    
    if st.button("Calcular Newton-Raphson"):
        try:
            root, x_values, f, df = newton_raphson(eq_nr, x0_nr)
            
            if root is None:
                st.warning("No se pudo encontrar la raíz con los parámetros dados.")
            else:
                df_table = pd.DataFrame({
                    'Iteración': range(len(x_values)),
                    'x': [round(x, 6) for x in x_values],
                    'f(x)': [round(f(x), 6) for x in x_values]
                })
                st.write("### Iteraciones")
                st.dataframe(df_table, use_container_width=True)
                
                st.success(f"**Raíz encontrada: {root:.6f}**")
                
                margin = 1.0
                x_range = np.linspace(min(x_values) - margin, max(x_values) + margin, 500)
                y_range = f(x_range)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x_range, y_range, label=f'f(x) = {eq_nr}', color='blue', lw=2)
                ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.5)
                ax.scatter(root, 0, color='green', s=200, edgecolors='black', label=f'Raíz: {root:.4f}', zorder=6, marker='*')
                ax.set_title(f'Newton-Raphson: f(x) = {eq_nr}', fontsize=14)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('f(x)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")


elif method == "Runge-Kutta 4":
    st.header("Método de Runge-Kutta 4ta Orden")
    st.write("Resuelve EDOs de la forma: dy/dx = f(x, y)")
    
    col1, col2 = st.columns(2)
    with col1:
        eq_rk = st.text_input(
            "Ecuación dy/dx (ej: 2\\*x\\*y, x\\*y, y - x)",
            help="Usa 'x' e 'y' como variables"
        )
        h_rk = st.number_input("Tamaño de paso (ej: 0.1)", min_value=0.01, step=0.01)
        x0_rk = st.number_input("x inicial (ej: 0)")
    
    with col2:
        y0_rk = st.number_input("y inicial (ej: 1)")
        xend_rk = st.number_input("x final (ej: 2)")
    
    if st.button("Calcular Runge-Kutta"):
        try:
            x_vals, y_vals, ks_values, n_list = RungeKutta(eq_rk, h_rk, x0_rk, y0_rk, xend_rk)
            
            table_data = []
            for i in range(len(x_vals)):
                row = {
                    'n': n_list[i],
                    'x': round(x_vals[i], 4),
                    'y': round(y_vals[i], 4)
                }
                if i < len(ks_values):
                    k1, k2, k3, k4, k = ks_values[i]
                    row['k1'] = round(k1, 4)
                    row['k2'] = round(k2, 4)
                    row['k3'] = round(k3, 4)
                    row['k4'] = round(k4, 4)
                    row['k'] = round(k, 4)
                table_data.append(row)
            
            df_rk = pd.DataFrame(table_data)
            st.write("### Tabla de Resultados")
            st.dataframe(df_rk, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_vals, y_vals, 'o-', label=f'Runge-Kutta 4 (h={h_rk})', color='blue', linewidth=2, markersize=6)
            ax.set_title(f'Solución: dy/dx = {eq_rk}', fontsize=14)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.info("Ingresa ecuaciones válidas en Python. Usa 'x', 'y' como variables.")

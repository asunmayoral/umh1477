# -*- coding: utf-8 -*-
# Funciones básicas para el bloque 1. Distribuciones y CMTD
import numpy as np          # importamos numpy como np
import pandas as pd         # importamos pandas como pd
import math
import random

from scipy import stats, optimize
from scipy.special import gamma

# Cargamos módulos de análisis gráficos
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_theme(style = 'whitegrid')
# %config InlineBackend.figure_format = 'retina'


#============================================================================================
# DISTRIBUCIONES DISCRETAS
#============================================================================================

def graficar_discreta(x, fx):
  """
  Función para representar gráficamente la función de masa de probabilidad y la función de distribución de una variable discreta.

  Args:
    x: valores de la variable discreta
    fx: función de masa de probabilidad para cada valor de x

  Returns:
    Gráficos de la función de masa de probabilidad y la función de distribución.
  """
  # posiciones en el gráfico de los valores d ela variable discreta
  pos = np.arange(len(x))
  # Función de distribución
  fdist = [sum(fx[:(l+1)]) for l in range(len(fx))]

  # Entorno gráfico
  fig, ax = plt.subplots(1, 2, figsize=(7, 4))
  # Pintamos los puntos con x y la función de masa de probabilidad
  ax[0].plot(pos, fx, 'bo');
  # Dibujamos las líneas verticales correspondientes con sus caractarísticas
  ax[0].vlines(pos, 0, fx, colors='b', lw=5, alpha=0.5);
  # Ponemos un título
  ax[0].set_title('Función de masa de probabilidad')
  # Ponemos etiquetas a los ejes x e y
  ax[0].set_xticks(pos, labels=x)
  ax[0].set_ylabel('Probabilidad')
  ax[0].set_xlabel('Espacio muestral')

  #### Función de distribución
  # Pintamos los puntos con x y la función de distribución
  ax[1].plot(pos, fdist, 'bo');
  # Dibujamos las líneas verticales correspondientes con sus caractarísticas
  ax[1].vlines(pos, 0, fdist, colors='b', lw=5, alpha=0.5);
  # Ponemos un título
  ax[1].set_title('Función de distribución')
  # Ponemos etiquetas a los ejes x e y
  ax[1].set_xticks(pos, labels=x)
  ax[1].set_ylabel('Probabilidad')
  ax[1].set_xlabel('Espacio muestral')
  plt.tight_layout()
#---------------------------------------------------------------------------------------
# Función para obtener un dataframe con la función de masa de probabilidad y la función de distribución de una variable discreta.
def distr_discreta(x, fx):
  """
  Función para obtener un dataframe con la función de masa de probabilidad y la función de distribución de una variable discreta.

  Args:
    x: valores de la varaible discreta
    fx: función de masa de probabilidad

  Returns:
    pdDataFrame con los valores de la variable, la función de masa de probabilidad y la función de distribución.
  """
  # posiciones en el gráfico de los valores d ela variable discreta
  pos = np.arange(len(x))
  # Función de distribución
  fdist = [sum(fx[:(l+1)]) for l in range(len(fx))]
  return(pd.DataFrame({"x": x, "fmp":fx, "fdist":fdist}))

#---------------------------------------------------------------------------------------
# Simular una m.a. de una distribución discreta y devolver media y varianza
def simula_discreta(x, fx, n):
  """
  Función para simular una m.a. de una distribución discreta y devolver
  media y varianza de los datos simulados.
  Args:
    x: valores de la variable discreta
    fx: función de masa de probabilidad

  Returns
    Lista con el valor medio y desviación típica de la variable de interés en el periodo n de simulación
  """
  muestra = np.random.choice(x, size = n, replace = True, p = fx)
  resul = [round(muestra.mean(),2), round(muestra.var(),2)]
  return(resul)
#============================================================================================

# ESTIMACIÓN MONTE CARLO
# Función para obtener el estimador Monte Carlo de h(x) y un intervalo de confianza al 95%
def MC_estim(sims):
  """
  Función para obtener el estimador Monte Carlo de h(x) y un intervalo de confianza al 95%

  Args:
   sims: Si queremos un estimador de h(x) pasamos directamente als simulaciones,
          mientras que si deseamos una probabildiad debemos pasar el vector 1-0
          que cumple con las condiciones de la probabilidad buscada

  Returns: 
    Devuelve el estimador e intervalo de confianza por Monte Carlo
  """
  from scipy.stats import norm

  # Número de simulaciones cargadas
  size = len(sims)
  # Estimador MC
  estim = sims.mean()
  # Estimador MC del IC
  error = math.sqrt(sims.var())*math.sqrt(size-1)/size
  cuantil = norm.ppf(1-0.05/2)
  ic_low = estim - cuantil*error
  ic_up = estim + cuantil*error
  # Resultado
  return([round(estim,4), round(ic_low,4), round(ic_up,4)])

#============================================================================================
# DISTRIBUCIONES DISCRETAS GOF
#============================================================================================

def gof_distr_discrete(data):
    """
    Ajusta y Evalúa la bondad de ajuste de distribuciones DISCRETAS utilizando
    el Test de Kolmogorov-Smirnov (adaptado) y estimación de parámetros por el MÉTODO DE LOS MOMENTOS.
    
    Distribuciones: Poisson, Geométrica, Binomial, Binomial Negativa.
    
    Nota: El test KS es conservador para datos discretos, pero sirve como referencia rápida.
    """
    
    # 1. Preparación de datos
    x = np.array(data)
    x = x[~np.isnan(x)]
    n_samples = len(x)
    
    # Nivel de significancia
    alpha = 0.05
    
    # Momentos Muestrales
    mu = np.mean(x)
    var = np.var(x, ddof=1)
    
    # Evitar división por cero en casos degenerados
    if var == 0: var = 1e-6 
    if mu == 0: mu = 1e-6

    results = []

    # ==============================================================================
    # 1. Distribución de Poisson
    # Parámetro: lambda = mu
    # ==============================================================================
    # En scipy, el parámetro mu es lambda
    d_pois, p_pois = stats.kstest(x, 'poisson', args=(mu,))
    
    results.append({
        'Distribución': 'Poisson',
        'Parámetros': f'Lambda={mu:.2f}',
        'KS Stat': d_pois,
        'P-Value': p_pois
    })

    # ==============================================================================
    # 2. Distribución Geométrica
    # Asume definición de 'intentos hasta el éxito' (dominio >= 1).
    # MoM: Mean = 1/p  => p = 1/Mean
    # ==============================================================================
    # Probabilidad estimada
    p_geom = 1 / mu
    
    # Restricción: p debe estar en (0, 1]
    p_geom = max(min(p_geom, 1.0), 1e-6)
    
    d_geom, p_val_geom = stats.kstest(x, 'geom', args=(p_geom,))
    
    results.append({
        'Distribución': 'Geométrica',
        'Parámetros': f'p={p_geom:.4f}',
        'KS Stat': d_geom,
        'P-Value': p_val_geom
    })

    # ==============================================================================
    # 3. Distribución Binomial
    # MoM: Mean = np, Var = np(1-p)
    # Solo válida si Var < Mean (Subdispersión)
    # ==============================================================================
    if var < mu:
        # 1-p = Var / Mean  =>  p = 1 - (Var/Mean)
        p_bin = 1 - (var / mu)
        
        # n = Mean / p
        n_est = mu / p_bin
        
        # n debe ser entero, redondeamos
        n_bin = int(round(n_est))
        
        # Recalculamos p para ajustar la media con el nuevo n entero
        p_bin_adj = mu / n_bin if n_bin > 0 else 0
        p_bin_adj = max(min(p_bin_adj, 1), 0) # Clip 0-1

        d_bin, p_val_bin = stats.kstest(x, 'binom', args=(n_bin, p_bin_adj))
        
        results.append({
            'Distribución': 'Binomial',
            'Parámetros': f'n={n_bin}, p={p_bin_adj:.2f}',
            'KS Stat': d_bin,
            'P-Value': p_val_bin
        })
    else:
        results.append({
            'Distribución': 'Binomial',
            'Parámetros': 'No Ajustable (Var >= Media)',
            'KS Stat': 1.0,
            'P-Value': 0.0
        })

    # ==============================================================================
    # 4. Distribución Binomial Negativa
    # MoM: Mean = n(1-p)/p, Var = n(1-p)/p^2  (Definición Scipy: Fallos)
    # Solo válida si Var > Mean (Sobredispersión)
    # ==============================================================================
    if var > mu:
        # p = Mean / Var
        p_nbin = mu / var
        
        # n = (Mean * p) / (1-p)
        n_val = (mu * p_nbin) / (1 - p_nbin)
        
        # En Scipy 'n' (r éxitos) puede ser float, no necesita ser entero
        d_nbin, p_val_nbin = stats.kstest(x, 'nbinom', args=(n_val, p_nbin))
        
        results.append({
            'Distribución': 'Binomial Negativa',
            'Parámetros': f'n(r)={n_val:.2f}, p={p_nbin:.2f}',
            'KS Stat': d_nbin,
            'P-Value': p_val_nbin
        })
    else:
        results.append({
            'Distribución': 'Binomial Negativa',
            'Parámetros': 'No Ajustable (Var <= Media)',
            'KS Stat': 1.0,
            'P-Value': 0.0
        })

    # ==============================================================================
    # Consolidación
    # ==============================================================================
    df_results = pd.DataFrame(results)
    
    df_results['¿Ajuste Válido?'] = df_results['P-Value'].apply(
        lambda p: 'Sí' if p > alpha else 'No (Rechazado)'
    )
    
    df_results = df_results.sort_values(by='P-Value', ascending=False).reset_index(drop=True)
    
    return df_results
  
#============================================================================================
# DISTRIBUCIONES CONTINUAS GOF
#============================================================================================

def gof_distr(data):
    """
    Ajusta y Evalúa la bondad de ajuste de múltiples distribuciones continuas utilizando
    el Test de Kolmogorov-Smirnov y estimación de parámetros por el MÉTODO DE LOS MOMENTOS (MoM).
    
    Distribuciones: Uniforme, Exponencial, Normal, Gamma, Erlang, Triangular, Weibull, Log-Normal.
    """
    
    # 1. Preparación de datos y estadísticos básicos
    x = np.array(data)
    x = x[~np.isnan(x)] # Eliminar NaNs si existen
    n = len(x)
    
    # Nivel de significancia
    alpha = 0.05
    
    # Momentos Muestrales
    mu = np.mean(x)
    var = np.var(x, ddof=1) # Varianza muestral (n-1)
    std = np.std(x, ddof=1) # Desviación estándar
    x_min = np.min(x)
    x_max = np.max(x)
    
    results = []

    # ==============================================================================
    # 1. Distribución Uniforme
    # MoM: Rango = sqrt(12 * var). Centrada en la media.
    # ==============================================================================
    range_uni = np.sqrt(12 * var)
    uni_a = mu - (range_uni / 2)
    uni_scale = range_uni # scale = b - a
    
    d_uni, p_uni = stats.kstest(x, 'uniform', args=(uni_a, uni_scale))
    
    results.append({
        'Distribución': 'Uniforme',
        'Parámetros': f'Min={uni_a:.2f}, Range={uni_scale:.2f}',
        'KS Stat': d_uni,
        'P-Value': p_uni
    })

    # ==============================================================================
    # 2. Distribución Exponencial
    # MoM: Scale = Media. (Asumiendo loc=0, típica en tiempos de espera)
    # ==============================================================================
    exp_scale = mu
    d_exp, p_exp = stats.kstest(x, 'expon', args=(0, exp_scale))
    
    results.append({
        'Distribución': 'Exponencial',
        'Parámetros': f'Scale={exp_scale:.2f}',
        'KS Stat': d_exp,
        'P-Value': p_exp
    })

    # ==============================================================================
    # 3. Distribución Normal
    # MoM: Media = mu, Scale = std
    # ==============================================================================
    d_norm, p_norm = stats.kstest(x, 'norm', args=(mu, std))
    
    results.append({
        'Distribución': 'Normal',
        'Parámetros': f'Mu={mu:.2f}, Std={std:.2f}',
        'KS Stat': d_norm,
        'P-Value': p_norm
    })

    # ==============================================================================
    # 4. Distribución Gamma
    # MoM: alpha = mu^2 / var, scale = var / mu
    # ==============================================================================
    gam_scale = var / mu
    gam_a = (mu ** 2) / var
    
    d_gam, p_gam = stats.kstest(x, 'gamma', args=(gam_a, 0, gam_scale))
    
    results.append({
        'Distribución': 'Gamma',
        'Parámetros': f'Alpha={gam_a:.2f}, Beta={gam_scale:.2f}',
        'KS Stat': d_gam,
        'P-Value': p_gam
    })

    # ==============================================================================
    # 5. Distribución Erlang
    # MoM: Igual que Gamma pero k debe ser entero.
    # Ajustamos k redondeando y recalculamos scale para mantener la media.
    # ==============================================================================
    erl_k = max(1, round((mu ** 2) / var))
    erl_scale = mu / erl_k
    
    # Scipy no tiene 'erlang' explícita, usamos gamma con shape entero
    d_erl, p_erl = stats.kstest(x, 'gamma', args=(erl_k, 0, erl_scale))
    
    results.append({
        'Distribución': 'Erlang',
        'Parámetros': f'k={int(erl_k)}, Beta={erl_scale:.2f}',
        'KS Stat': d_erl,
        'P-Value': p_erl
    })

    # ==============================================================================
    # 6. Distribución Triangular
    # MoM Heurístico: Usamos min y max empíricos.
    # Despejamos la moda (c) usando la fórmula de la media: mu = (a+b+c)/3
    # ==============================================================================
    tri_loc = x_min # a
    tri_scale = x_max - x_min # b - a
    
    # Estimación de la moda (c real)
    mode_est = 3 * mu - x_min - x_max
    
    # Restricción: la moda debe estar dentro del rango [min, max]
    mode_est = max(x_min, min(x_max, mode_est))
    
    # Parámetro c para scipy (proporción 0-1)
    if tri_scale > 0:
        tri_c = (mode_est - tri_loc) / tri_scale
    else:
        tri_c = 0.5 # Caso degenerado (varianza 0)

    d_tri, p_tri = stats.kstest(x, 'triang', args=(tri_c, tri_loc, tri_scale))
    
    results.append({
        'Distribución': 'Triangular',
        'Parámetros': f'Min={tri_loc:.2f}, Mode={mode_est:.2f}, Max={x_max:.2f}',
        'KS Stat': d_tri,
        'P-Value': p_tri
    })

    # ==============================================================================
    # 7. Distribución Weibull
    # MoM Numérico: No hay solución cerrada para k (shape).
    # Ecuación a resolver: CV^2 = (std/mu)^2 = [Gamma(1+2/k) / Gamma(1+1/k)^2] - 1
    # ==============================================================================
    cv_sq = (std / mu) ** 2
    
    def weibull_eq(k):
        # Función objetivo para encontrar k
        if k <= 0: return 100.0
        return (gamma(1 + 2/k) / (gamma(1 + 1/k)**2)) - 1 - cv_sq

    # Resolver numéricamente para k (shape)
    try:
        wei_k = optimize.fsolve(weibull_eq, 1.0)[0] # Semilla inicial = 1 (Exponencial)
    except:
        wei_k = 1.0
        
    # Una vez tenemos k, obtenemos lambda (scale)
    wei_scale = mu / gamma(1 + 1/wei_k)
    
    d_wei, p_wei = stats.kstest(x, 'weibull_min', args=(wei_k, 0, wei_scale))
    
    results.append({
        'Distribución': 'Weibull',
        'Parámetros': f'Shape={wei_k:.2f}, Scale={wei_scale:.2f}',
        'KS Stat': d_wei,
        'P-Value': p_wei
    })

    # ==============================================================================
    # 8. Distribución Log-Normal
    # MoM Analítico:
    # sigma^2 = ln(1 + var/mu^2)
    # mu_log = ln(mu) - sigma^2/2
    # Scipy params: s = sigma, scale = exp(mu_log)
    # ==============================================================================
    # Cálculo de los parámetros de la Normal subyacente
    # phi^2 es el segundo momento crudo E[X^2] = Var + Mean^2
    phi = np.sqrt(var + mu**2)
    
    mu_log = np.log(mu**2 / phi)          # media de ln(x)
    sigma_log = np.sqrt(np.log(phi**2 / mu**2))  # std de ln(x)
    
    scale_log = np.exp(mu_log) # scale es e^mu
    
    # KS Test (s es sigma_log, scale es e^mu_log)
    d_logn, p_logn = stats.kstest(x, 'lognorm', args=(sigma_log, 0, scale_log))
    
    results.append({
        'Distribución': 'Log-Normal',
        'Parámetros': f's={sigma_log:.2f}, Scale={scale_log:.2f}',
        'KS Stat': d_logn,
        'P-Value': p_logn
    })

    # ==============================================================================
    # Consolidación de Resultados
    # ==============================================================================
    df_results = pd.DataFrame(results)
    
    # Añadir columna de decisión
    df_results['¿Ajuste Válido?'] = df_results['P-Value'].apply(
        lambda p: 'Sí' if p > alpha else 'No (Rechazado)'
    )
    
    # Ordenar por P-Value descendente (Mejor ajuste arriba)
    df_results = df_results.sort_values(by='P-Value', ascending=False).reset_index(drop=True)
    
    return df_results
    
  
#============================================================================================
# CMTD
#============================================================================================

# Calcular la matriz de transición de n pasos
def cmtd_matrix_n(mc, n):
  """
  Función para obtener la matriz de transición de n pasos
  dada un proceso definido con MarkovChain()

  Parámetros de entrada:
    - mc: proceso definido con MarkovChain()
    - n: número de saltos.

  Parámetros de salida:
    - p_n: matriz de transición de n pasos.
  """
  import pydtmc
  mtn  = pydtmc.MarkovChain(np.linalg.matrix_power(mc.p, n), mc.states)
  return pd.DataFrame(mtn.p,columns=mc.states,index=mc.states)
#--------------------------------------------------------------------------------------
# Matriz de tiempos de ocupación
def mat_ocupacion_proceso(mc, n):
  """
  Función para obtener la matriz de ocupación asocida al proceso mc en n transiciones

  Parámetros de entrada:
  - mc: proceso
  - n: número de transiciones

  Parámetros de salida:
  - mocupa: matriz de ocupacion
  """
  mocupa = np.zeros((len(mc.states), len(mc.states)))
  for i in range(n+1):
    mocupa += np.linalg.matrix_power(mc.p, i)

  return mocupa
  import pydtmc
  mtn  = pydtmc.MarkovChain(np.linalg.matrix_power(mc.p, n), mc.states)
  return pd.DataFrame(mtn.p,columns=mc.states,index=mc.states)

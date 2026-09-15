import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# Ejemplo 4 (BoomBikes — Regresión Lineal): RFE + escalado + eliminación
# stepwise por p-valor/VIF (statsmodels), usados en _seleccionar_features_regresion.
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import SingularMatrixWarning


# ============================================================
# Ejemplo 4 (BoomBikes — "Linear Regression: Demand Forecasting")
# ============================================================
def _seleccionar_features_regresion(X_train, y_train, max_p_valor=0.05, max_vif=5.0):
    """
    Selección de variables para la Regresión Lineal, con la misma lógica de
    dos pasos usada en el caso de referencia (BoomBikes): primero RFE
    (sklearn) para descartar a la mitad las variables candidatas, y luego
    eliminación hacia atrás (statsmodels OLS) quitando de a una la variable
    con mayor p-valor o mayor VIF, hasta que todas las que quedan sean
    significativas (p < 0.05) y no estén colineales entre sí (VIF < 5).

    Antes de RFE se escalan las variables con MinMaxScaler (igual que en
    el notebook de referencia): RFE con un estimador LinearRegression
    decide qué descartar mirando la magnitud del coeficiente, y sin
    escalar, una variable como "año" (~2021) y otra como "lag_1" (cientos
    de unidades de demanda) no son comparables entre sí. El escalador se
    ajusta solo con datos de entrenamiento para no filtrar información
    del set de prueba.
    """
    # Columnas sin variación (p. ej. "año" cuando el archivo subido solo
    # tiene un año de datos) no aportan nada y rompen el cálculo de VIF
    # (división por cero), así que se descartan antes de empezar.
    candidatas = [c for c in X_train.columns if X_train[c].std() > 0]
    if len(candidatas) < 2 or len(X_train) <= len(candidatas) + 1:
        return candidatas

    try:
        scaler = MinMaxScaler()
        X_train_esc = pd.DataFrame(
            scaler.fit_transform(X_train[candidatas]),
            columns=candidatas, index=X_train.index
        )

        n_rfe = max(2, min(8, len(candidatas) - 1))
        rfe = RFE(LinearRegression(), n_features_to_select=n_rfe)
        rfe.fit(X_train_esc, y_train)
        seleccionadas = [c for c, usar in zip(candidatas, rfe.support_) if usar]

        while len(seleccionadas) > 1:
            X_sm = sm.add_constant(X_train_esc[seleccionadas])
            # Cuando una variable queda casi/perfectamente colineal con
            # otras (esperable a mitad del proceso de poda), statsmodels
            # avisa con SingularMatrixWarning/UserWarning en cada columna
            # que evalúa. No indican un error: es justo la situación que
            # el bloque de abajo detecta (VIF infinito/NaN) y corrige
            # descartando esa variable — se silencian para no llenar la
            # consola con avisos de algo que ya se maneja a propósito.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SingularMatrixWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                vif = pd.DataFrame({
                    "feature": seleccionadas,
                    "VIF": [variance_inflation_factor(X_sm.values, i + 1) for i in range(len(seleccionadas))]
                })

            # La colinealidad se resuelve antes de mirar p-valores: con
            # VIF infinito/NaN la matriz de diseño es singular (colinealidad
            # perfecta) y el ajuste OLS que sigue no es confiable todavía,
            # así que primero se descarta la variable más colineal.
            vif_problema = vif[~np.isfinite(vif["VIF"])]
            if not vif_problema.empty:
                seleccionadas.remove(vif_problema.iloc[0]["feature"])
                continue

            peor_vif_fila = vif.loc[vif["VIF"].idxmax()]
            if peor_vif_fila["VIF"] > max_vif:
                seleccionadas.remove(peor_vif_fila["feature"])
                continue

            modelo_ols = sm.OLS(y_train, X_sm).fit()
            p_valores = modelo_ols.pvalues.drop("const")
            peor_p_nombre = p_valores.idxmax()
            peor_p_valor = p_valores[peor_p_nombre]

            if peor_p_valor > max_p_valor:
                seleccionadas.remove(peor_p_nombre)
            else:
                break

        return seleccionadas
    except Exception:
        # Con pocos datos o columnas colineales el ajuste OLS puede fallar
        # (matriz singular, etc.); si pasa, se sigue con todas las
        # variables candidatas en vez de romper el pronóstico.
        return candidatas


def entrenar_regresion_lineal(X_train, y_train, X_test, feature_cols):
    """
    Entrena la Regresión Lineal (Ejemplo 4) solo con las variables que
    sobrevivieron RFE + eliminación stepwise (p-valor/VIF). Además de
    servir de benchmark en la tabla comparativa, el modelo entrenado se
    devuelve para poder reutilizarlo en el pronóstico real a futuro si
    resulta ser el método campeón de algún producto (ver
    elegir_campeon_por_producto en modelos.py).

    Devuelve (modelo_lr, pred_lr, features_lr).
    """
    features_lr = _seleccionar_features_regresion(X_train, y_train)
    if not features_lr:
        features_lr = list(feature_cols)

    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train[features_lr], y_train)
    pred_lr = modelo_lr.predict(X_test[features_lr])
    pred_lr = np.maximum(pred_lr, 0)

    return modelo_lr, pred_lr, features_lr

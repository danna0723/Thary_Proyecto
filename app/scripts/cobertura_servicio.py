import pandas as pd

p = pd.read_csv('predicciones.csv')
t = p[p['fecha'] >= '2022-04-01'].copy()

# MAE del método campeón por producto, calculado desde el backtest
mae_por_producto = (
    p.groupby('producto_id')
    .apply(lambda g: (g['demanda'] - g['prediccion_final']).abs().mean())
    .rename('mae')
)
t = t.merge(mae_por_producto, on='producto_id')

ok = t['demanda'] <= t['prediccion_final'] + 1.65 * t['mae']
print(round(100 * ok.mean(), 1), round(100 * ok[t['demanda'] > 0].mean(), 1))
print(p['fecha'].min(), p['fecha'].max(), len(t))

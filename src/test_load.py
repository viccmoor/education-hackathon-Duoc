import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.load_data import get_clean_data

print("="*60)
print("TEST: Carga rápida de datos (primeras 10,000 filas por archivo)")
print("="*60)

# Cargar solo primeras 10k filas de cada CSV para prueba rápida
df = get_clean_data(max_rows=10000)

if df is not None:
    print("\n✅ Carga exitosa!")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    print(f"\nInfo de columnas:")
    print(df.info())
    print(f"\nEstadísticas:")
    print(df.describe())
else:
    print("\n❌ No se pudieron cargar datos")
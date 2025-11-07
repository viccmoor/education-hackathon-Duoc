import pandas as pd
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Delegar a src/load_data.py para evitar duplicidad
from src.load_data import get_clean_data

if __name__ == "__main__":
    print("Probando carga de datos via src/load_data.get_clean_data()...")
    df = get_clean_data()
    
    if df is not None:
        print("\nPrimeras filas:")
        print(df.head())
        
        # Split temporal
        train = df[df['año'] < 2024]
        test = df[df['año'] == 2024]
        
        if not train.empty and not test.empty:
            print("\nSplit temporal:")
            print(f"Train (antes de 2024): {train.shape}")
            print(f"Test (2024): {test.shape}")
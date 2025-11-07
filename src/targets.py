from pandas import DataFrame

def create_grade_label(df: DataFrame):
    df = df.copy()
    grade_col = "PROM_GRAL"

    if grade_col in df.columns:
        df['label_reprobando'] = 0

        # Criterio 1: Promedio bajo 4
        valid_a1c = df[grade_col].notna()
        df.loc[valid_a1c & (df[grade_col] >= 4.0), 'label_reprobando'] = 1
        print(f"  Reprobando: {(valid_a1c & (df[grade_col] >= 4.0)).sum():,} casos")

        return df

def create_attendance_label(df: DataFrame):
    df = df.copy()
    atten_col = "ASISTENCIA"

    if atten_col in df.columns:
        df['label_reprobando'] = 0

        # Criterio 2: Asistencia superior a 70%
        valid_a1c = df[atten_col].notna()
        df.loc[valid_a1c & (df[atten_col] >= 40), 'label_reprobando'] = 1
        print(f"  Reprobando: {(valid_a1c & (df[atten_col] >= 40)).sum():,} casos")

        return df

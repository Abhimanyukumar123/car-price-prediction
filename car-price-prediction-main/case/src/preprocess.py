import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess used car dataset safely for ML pipeline
    """

    df = df.copy()

    # --------------------------------------------------
    # Drop unwanted index column
    # --------------------------------------------------
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # --------------------------------------------------
    # Clean km_driven column
    # --------------------------------------------------
    if 'km_driven' in df.columns:
        df['km_driven'] = (
            df['km_driven']
            .astype(str)
            .str.replace(',', '', regex=False)
        )
        df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')

    # --------------------------------------------------
    # Fill missing values
    # --------------------------------------------------
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # --------------------------------------------------
    # Ensure correct dtypes
    # --------------------------------------------------
    int_cols = ['manufacturing_year', 'km_driven', 'number_of_owners']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

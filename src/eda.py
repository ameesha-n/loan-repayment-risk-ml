from data_loader import load_application_data


def basic_eda(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print(df["TARGET"].value_counts())

    print("\nTarget percentages:")
    print(df["TARGET"].value_counts(normalize=True) * 100)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes.value_counts())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(missing.head(20))

    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)

    print(f"Duplicate rows: {df.duplicated().sum()}")


if __name__ == "__main__":
    df = load_application_data()
    basic_eda(df)
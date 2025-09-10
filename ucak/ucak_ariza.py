import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path, normalize=True):
    """
    NASA CMAPSS datasetini yükler, RUL ekler ve isteğe bağlı normalizasyon yapar.
    :param path: Veri dosyasının yolu (örn: data/train_FD001.txt)
    :param normalize: True -> MinMaxScaler uygular
    :return: pandas DataFrame
    """
    # CMAPSS için kolon isimleri
    col_names = ["unit_number", "time_in_cycles"] \
                + [f"op_setting_{i}" for i in range(3)] \
                + [f"sensor_{i}" for i in range(21)]

    # Veri setini oku
    df = pd.read_csv(path, sep=" ", header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = col_names

    # RUL hesapla
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df.drop(columns=["max_cycle"], inplace=True)

    # Faydalı sensörleri seçelim (örnek olarak 6 tanesi tutuluyor)
    useful_sensors = ["sensor_2", "sensor_3", "sensor_4",
                      "sensor_7", "sensor_11", "sensor_15"]
    features = ["unit_number", "time_in_cycles"] + useful_sensors + ["RUL"]
    df = df[features]

    # Normalizasyon (sadece sensörler için)
    if normalize:
        scaler = MinMaxScaler()
        df[useful_sensors] = scaler.fit_transform(df[useful_sensors])

    return df


if __name__ == "__main__":
    dataset_path = "data/train_FD001.txt"
    df = load_dataset(dataset_path)
    print(df.head(20))
    print(f"\nToplam satır: {len(df)}")
    print(f"Kolonlar: {list(df.columns)}")

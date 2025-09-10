import pandas as pd
import numpy as np
import os

def create_sample_csv(train_path="data/train.csv", test_path="data/test.csv", n_train=100, n_test=20):
    os.makedirs("data", exist_ok=True)

    def generate_data(n):
        np.random.seed(42)
        sensor_1 = np.random.rand(n)
        sensor_2 = np.random.rand(n)
        sensor_3 = np.random.rand(n)

        # Basit kural: sensörlerin toplamı > 1.5 ise arıza
        failure = ((sensor_1 + sensor_2 + sensor_3) > 1.5).astype(int)

        return pd.DataFrame({
            "sensor_1": sensor_1,
            "sensor_2": sensor_2,
            "sensor_3": sensor_3,
            "failure": failure
        })

    train_df = generate_data(n_train)
    test_df = generate_data(n_test)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train CSV oluşturuldu: {train_path} ({len(train_df)} satır)")
    print(f"✅ Test CSV oluşturuldu: {test_path} ({len(test_df)} satır)")

if __name__ == "__main__":
    create_sample_csv()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras


PANJANG_WINDOW = 10
KOLOM_INPUT = ["V1 (V)", "I (A)", "T1 (ºC)"]
MODEL_PATH = Path(__file__).with_name("model_soc.keras")


def buat_sampel(fitur, soc):
    jumlah_sampel = len(fitur) - PANJANG_WINDOW + 1
    if jumlah_sampel < 1:
        raise ValueError("Setiap bagian data perlu minimal 10 baris.")

    X = np.stack([
        fitur[awal:awal + PANJANG_WINDOW]
        for awal in range(jumlah_sampel)
    ])
    y = soc[PANJANG_WINDOW - 1:].reshape(-1, 1)
    return X, y


def latih_dan_simpan(file_csv):
    data = pd.read_csv(file_csv, sep=";", decimal=",", encoding="cp1252")
    waktu = pd.to_datetime(data["Timestamp"], format="%d/%m/%Y %H:%M")
    if waktu.isna().any() or not waktu.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("CSV training harus berurutan dengan interval satu menit.")

    fitur = data[KOLOM_INPUT].to_numpy(dtype=np.float32)
    soc = data["%SOC"].to_numpy(dtype=np.float32)
    if not np.isfinite(fitur).all() or not np.isfinite(soc).all():
        raise ValueError("CSV training memuat nilai input atau SoC yang kosong atau tidak valid.")
    if ((soc < 0) | (soc > 100)).any():
        raise ValueError("Label SoC training harus berada di rentang 0–100%.")

    batas_latih = int(len(data) * 0.70)
    batas_validasi = int(len(data) * 0.85)
    X_latih, y_latih = buat_sampel(fitur[:batas_latih], soc[:batas_latih])
    X_validasi, y_validasi = buat_sampel(
        fitur[batas_latih:batas_validasi], soc[batas_latih:batas_validasi]
    )
    X_uji, y_uji = buat_sampel(fitur[batas_validasi:], soc[batas_validasi:])

    rata_rata = fitur[:batas_latih].mean(axis=0)
    simpangan = fitur[:batas_latih].std(axis=0)
    simpangan = np.where(simpangan > 0, simpangan, 1.0)

    keras.utils.set_random_seed(42)
    masukan = keras.Input(shape=(PANJANG_WINDOW, len(KOLOM_INPUT)))
    fitur_normal = keras.layers.Normalization(
        mean=rata_rata.tolist(), variance=(simpangan ** 2).tolist()
    )(masukan)
    ringkasan = keras.layers.GRU(32)(fitur_normal)
    soc_normal = keras.layers.Dense(1)(ringkasan)
    model_latih = keras.Model(masukan, soc_normal)
    model_latih.compile(optimizer="adam", loss="mse")

    print(f"Sampel latih: {len(X_latih)}, validasi: {len(X_validasi)}, uji: {len(X_uji)}")
    model_latih.fit(
        X_latih,
        y_latih / 100.0,
        validation_data=(X_validasi, y_validasi / 100.0),
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=2,
    )

    # Simpan normalisasi input dan pengembalian skala persen di dalam model.
    soc_persen = keras.layers.Rescaling(100.0)(soc_normal)
    model_prediksi = keras.Model(masukan, soc_persen)
    prediksi = model_prediksi.predict(X_uji, verbose=0)
    mae = np.abs(prediksi - y_uji).mean()
    print(f"MAE uji: {mae:.2f} poin persentase SoC")

    model_prediksi.save(MODEL_PATH)
    print(f"Model prediksi disimpan di {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Latih dan simpan GRU dari CSV Discharge Pulses.")
    parser.add_argument("csv", type=Path, help="Lokasi CSV Discharge Pulses.")
    args = parser.parse_args()
    try:
        latih_dan_simpan(args.csv)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Training gagal: {error}\n")


if __name__ == "__main__":
    main()

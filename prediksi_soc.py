import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).with_name("model_soc.keras")
ALIAS_KOLOM = {
    "timestamp": ("timestamp",),
    "tegangan_v": ("tegangan_v", "v1 (v)"),
    "arus_a": ("arus_a", "i (a)", "i(a)"),
    "suhu_c": ("suhu_c", "t1 (ºc)", "t1 (°c)"),
}
KOLOM_INPUT = ["tegangan_v", "arus_a", "suhu_c"]


def baca_window(file_csv, panjang_window):
    try:
        teks = file_csv.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        teks = file_csv.read_text(encoding="cp1252")

    teks = teks.lstrip()
    if not teks:
        raise ValueError("CSV kosong.")

    header = teks.splitlines()[0]
    pemisah = csv.Sniffer().sniff(header, delimiters=",;").delimiter
    data = pd.read_csv(io.StringIO(teks), sep=pemisah, dtype=str)
    nama_kolom = {nama.strip().casefold(): nama for nama in data.columns}
    sumber = {}
    for nama, alias in ALIAS_KOLOM.items():
        kolom = next((nama_kolom[a] for a in alias if a in nama_kolom), None)
        if kolom is None:
            raise ValueError(f"Kolom {nama} tidak ditemukan.")
        sumber[nama] = kolom

    if len(data) < panjang_window:
        raise ValueError(f"CSV perlu minimal {panjang_window} baris pengukuran.")

    cap_waktu = data[sumber["timestamp"]].str.strip()
    waktu_iso = pd.to_datetime(cap_waktu, format="ISO8601", errors="coerce", utc=True)
    waktu_asli = pd.to_datetime(
        cap_waktu, format="%d/%m/%Y %H:%M", errors="coerce", utc=True
    )
    waktu = waktu_iso.fillna(waktu_asli)
    if waktu.isna().any():
        raise ValueError("Gunakan timestamp seperti 2026-11-19 08:00:00 atau 19/11/2026 08:00.")

    indeks_terbaru = waktu.sort_values(kind="stable").tail(panjang_window).index
    if not waktu.loc[indeks_terbaru].diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError(f"{panjang_window} pengukuran terbaru harus berjarak tepat satu menit.")
    if waktu.eq(waktu.loc[indeks_terbaru[0]]).sum() > 1:
        raise ValueError("Timestamp awal jendela terbaru tercatat lebih dari sekali.")

    kolom_angka = []
    for nama in KOLOM_INPUT:
        teks_angka = data.loc[indeks_terbaru, sumber[nama]].str.strip()
        angka = pd.to_numeric(teks_angka.str.replace(",", ".", regex=False), errors="coerce")
        kolom_angka.append(angka.to_numpy(dtype=np.float32))

    fitur = np.column_stack(kolom_angka)
    if not np.isfinite(fitur).all():
        raise ValueError("Tegangan, arus, dan suhu pada jendela terbaru harus berupa angka.")
    return fitur[np.newaxis, :, :], cap_waktu.loc[indeks_terbaru[-1]]


def main():
    parser = argparse.ArgumentParser(description="Estimasi SoC dari CSV pengukuran terbaru.")
    parser.add_argument("csv", type=Path, help="Lokasi CSV dengan waktu, tegangan, arus, dan suhu.")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        parser.exit(1, f"Model tidak ditemukan di {MODEL_PATH}\n")

    from tensorflow import keras

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        X, waktu = baca_window(args.csv, model.input_shape[1])
        estimasi = float(model(X, training=False).numpy().item())
        if not np.isfinite(estimasi):
            raise ValueError("Model menghasilkan nilai SoC yang tidak valid.")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Prediksi gagal: {error}\n")

    print(f"Waktu pengukuran: {waktu}")
    print(f"Estimasi SoC: {estimasi:.2f}%")
    if not 0 <= estimasi <= 100:
        print("Hasil model di luar rentang SoC 0–100%.")


if __name__ == "__main__":
    main()

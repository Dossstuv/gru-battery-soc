# Estimasi SoC baterai dengan GRU

Proyek Python ini mengestimasi *state of charge* (SoC) dari 10 pengukuran terbaru. Setiap pengukuran berisi tegangan sel, arus, dan suhu. Model `model_soc.keras` sudah dilatih dan menyimpan normalisasi inputnya.

## Coba prediksi

Gunakan Python 3.10 untuk konfigurasi yang sudah diuji. Buka terminal di folder repositori dan jalankan:

```powershell
python -m pip install -r requirements.txt
python prediksi_soc.py contoh_input.csv
```

`contoh_input.csv` berisi data sintetis untuk mencoba cara menjalankan program. Untuk memakai data sendiri, ganti `contoh_input.csv` dengan path ke CSV pengukuranmu. Gunakan kolom berikut.

| Kolom | Isi |
| --- | --- |
| `timestamp` | Waktu pengukuran, misalnya `2026-01-01 08:00:00` |
| `tegangan_v` | Tegangan terminal sel dalam volt |
| `arus_a` | Arus dalam ampere. Positif saat pengisian dan negatif saat pengosongan |
| `suhu_c` | Suhu dalam °C |

Script mengurutkan data berdasarkan waktu, mengambil 10 pengukuran terbaru dengan interval satu menit, lalu menampilkan estimasi SoC untuk pengukuran terakhir. Kolom SoC referensi opsional saat menjalankan prediksi.

## Latih ulang

Unduh `Cell 1.zip` dari [dataset Rocha-Henríquez, Aguasca Colomo, dan Méndez di Zenodo](https://doi.org/10.5281/zenodo.17252822). Ekstrak arsip dan gunakan file sesi *Discharge Pulses*.

```powershell
python siapkan_data_gru.py "lokasi/Cell 1 Measurements - Fourteenth Load - Discharge Pulses.csv"
```

Perintah ini melatih GRU dan mengganti `model_soc.keras` dengan hasil training terbaru. Data sesi dibagi menurut waktu menjadi 70% latih, 15% validasi, dan 15% uji. Normalisasi dihitung dari bagian latih.

## Hasil dan batas pengujian

Model ini dilatih dan diuji pada satu sesi pengukuran sel OPzS dari bank baterai 1200 Ah. MAE pada bagian uji adalah **27,38 poin persentase SoC**. Program menampilkan keluaran asli model dan menandai estimasi di luar 0–100%. Pengujian pada baterai VRLA atau tegangan bank 12 V dan 24 V masih diperlukan.

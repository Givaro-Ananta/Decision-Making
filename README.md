# Decision Making — Transisi State Hotspot Sumatra (ITERA)

Proyek riset Decision Making (SD25-41302), Sains Data, Institut Teknologi Sumatera.

**Judul kerja:** *Analisis Retrospektif Transisi State Intensitas Deteksi Hotspot di Sumatra Menggunakan Model Markov Bergantung Kovariat Per Grid.*

Penelitian mengestimasi `P(S[g,t+1] | S[g,t], X[g,t])` secara independen pada setiap grid daratan utama Sumatra. Output utama adalah state hari berikutnya beserta probabilitasnya. Scope final tidak mencakup prioritas patroli, ranking, Recall@K, alokasi sumber daya, fitur tetangga atau training pooled.

## Status hasil dan audit validitas

Modeling dan evaluasi frozen selesai. **Validitas operasional dan cakupan pengamatan target belum terverifikasi.** Hasil saat ini merupakan evaluasi retrospektif terhadap label panel yang dibangun; jangan menafsirkannya sebagai sistem siap memprediksi besok menggunakan informasi hari ini.

Audit 11 Oktober 2026 mengonfirmasi:

- Cuaca bertanggal t berasal dari produk yang diterbitkan belakangan. ERA5-Land memiliki keterlambatan nominal sekitar lima hari; CHIRPS preliminary mengikuti akhir pentad dan produk final terbit pada bulan berikutnya. Daily `rnl` membagi total pentadal menggunakan ERA5. Split kronologis belum memeriksa historical `available_at`.
- Ada 84 tanggal tanpa record di dua CSV FIRMS lokal. Panel mengisi grid-day tanpa record menjadi nol tanpa sertifikat coverage pengamatan. Pada 46 grid final, 736 transisi training, 1.288 validation dan 2.392 test menyentuh tanggal tersebut pada t atau t+1. Jumlah ini bukan jumlah label yang terbukti salah.
- Inventory publik NASA mencantumkan seluruh Mei 2026 sebagai missing untuk produk **standard S-NPP**. Ini mengonfirmasi gap database produk, bukan gangguan sensor sepanjang bulan. Pemulihan produk dan QA per grid masih diperlukan.
- Argmax menangkap 804 dari 3.917 target berlabel aktif (20,53%). Accuracy tinggi didominasi No Activity. AUC pooled tidak membuktikan keberhasilan ranking antargrid per hari.

Laporan lengkap dan solusi: [audit validitas final](docs/final_validity_review.md). Bukti JSON/CSV dan provenance disimpan di `outputs/reports/final_validity_review/`; audit harian serta dampak per grid di `outputs/reports/firms_observation_coverage/`. Bukti audit JSON/CSV/Markdown yang kecil ikut dilacak Git; data satelit besar dan fitted models tetap diabaikan. Laporan utama tersedia di `docs/`.

Sumber primer: [ECMWF ERA5-Land](https://www.ecmwf.int/en/era5-land), [UCSB CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3), [NASA outages](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html), [NASA missing-data](https://firms.modaps.eosdis.nasa.gov/api/missing_data).

## Desain frozen

| Komponen | Keputusan |
|---|---|
| Domain | Daratan utama Sumatra; polygon delapan provinsi, komponen daratan terbesar |
| Grid | Reguler 0.5°; 187 grid studi, 46 eligible |
| Temporal | Harian; panel 2016-09-22–2026-08-31 |
| State 0 | Tidak ada deteksi retained (`hotspot_count=0`); observabilitas masih perlu diverifikasi |
| State 1 | Active/non-high: ada deteksi dan FRP sum <16.84 MW |
| State 2 | High: ada deteksi dan FRP sum ≥16.84 MW |
| Boundary | Q75 FRP sum aktif training, sebelum eligibility filtering |
| Prediktor | Temperatur maksimum sampel, RH rata-rata, angin rata-rata, rainfall; same-grid pada t |
| Model | 138 multinomial transition models unweighted, independen per grid × current state |
| Training | Target sampai 2022-12-31 |
| Validation | Target 2023–2024 |
| Test | Target 2025-01-01–2026-08-31; evaluasi frozen sekali |
| Primary rule | Standard argmax |
| Sensitivity | Guarded 0.20 dan two-stage 0.20/0.50; tidak diadopsi sebagai primary |

WorldCover diaudit sebagai konteks preprocessing dan tidak masuk empat prediktor final. State merupakan kategori intensitas deteksi relatif, bukan severity kebakaran yang dikalibrasi atau energi harian.

## Hasil frozen

27.968 observasi test dari 46 grid. Angka berikut conditional terhadap label panel; belum merupakan skor setelah koreksi coverage.

| Metrik primary | Nilai |
|---|---:|
| Accuracy | 0.860912 |
| Balanced Accuracy | 0.432223 |
| Macro F1 | 0.443943 |
| Weighted F1 | 0.821760 |
| Log Loss | 0.411651 |

Recall Active/non-high 0.045015 dan High 0.269325. Hasil tidak mendukung klaim active-state recovery yang kuat. Figur/tabel asli berada di [outputs/final_artifacts](outputs/final_artifacts/final_artifacts_summary.md). Artefak frozen tidak ditimpa dalam audit validitas.

## Solusi dan reproduksibilitas audit

Koreksi dokumentasi sudah dilakukan. Langkah data berikutnya adalah memperoleh provenance standard/NRT dan grid-day coverage FIRMS, termasuk gap Mei 2026. Jika perlu koreksi, tulis panel dan eksperimen versi baru; jangan mengubah data/model frozen. Menggunakan lag atau forecast meteorologi merupakan eksperimen operasional terpisah dengan pemeriksaan release/issuance. Periode test yang sudah dilihat tidak boleh disebut untouched untuk eksperimen revisi.

Audit tidak melakukan refit, retuning, rescreening, recalibration, perubahan threshold/state atau test inference ulang.

```bash
# Requires duckdb, pandas, shapely and pyarrow; use a fresh output directory.
python scripts/audit_firms_observation_coverage.py --output /tmp/firms-coverage-review

# Saved NASA JSON and frozen result tables; standard library only.
python scripts/build_final_validity_review.py --nasa-missing-response outputs/reports/final_validity_review/nasa_missing_dates_response.json --output /tmp/final-validity-review
```

## Struktur dan sejarah penelitian

- `configs/`: keputusan frozen dan konfigurasi eligibility/model.
- `src/preprocessing/`, `scripts/`: preprocessing, audit dan implementasi penelitian. Script fitting/evaluasi lama bukan perintah untuk dijalankan ulang.
- `tests/`: pemeriksaan implementasi; tidak menggantikan verifikasi ilmiah coverage.
- `data/`: data lokal, diabaikan Git; provenance unduhan di [data/README.md](data/README.md) masih perlu dilengkapi.
- `outputs/final_artifacts/`: figur/tabel frozen yang dilacak Git.
- `docs/final_validity_review.md`: temuan, solusi dan batas klaim terkini.
- `literature/`: literature sprint historis; bukan bukti semua gap/novelty telah divalidasi ulang.

README sebelum koreksi disimpan byte-identik di [dokumen historis Week 4](docs/history/README_week4_before_validity_review.md). Tujuan patroli dan novelty Recall@K di sana adalah konteks proposal lama, bukan kontrak implementasi final.

## Aturan kontribusi

Setiap anggota mencatat kontribusi dengan identitasnya sendiri. Setiap dataset/angka harus memiliki provenance yang dapat ditelusuri. Dual submission dilarang. Penggunaan AI harus diungkapkan dan hasil diverifikasi manusia sebelum masuk manuskrip. PDF penerbit tidak disimpan di repository.

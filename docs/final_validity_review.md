# Hasil akhir audit validitas penelitian — 11 Oktober 2026

## Keputusan

Ketiga laporan teman diperiksa. Masalah ketersediaan cuaca dan observabilitas target terkonfirmasi. Kritik tujuan patroli benar terhadap README lama, tetapi bukan persyaratan desain final: scope yang disetujui adalah prediksi state per grid, tanpa ranking/patroli. Klaim dan README diperbaiki; hasil frozen dipertahankan.

**Status hasil sekarang: hasil retrospektif eksploratif terhadap label panel yang dibangun; belum merupakan validasi forecast operasional dan belum tervalidasi terhadap cakupan pengamatan target.** Keberhasilan numerik fitting dan pemisahan kronologis tidak mengatasi dua masalah tersebut. Ini tidak membuktikan semua hasil salah, tetapi juga tidak bisa diselesaikan seluruhnya hanya dengan mengganti judul.

## Desain yang tetap utuh

Grid reguler 0.5°, daratan utama Sumatra, 187 grid studi dan 46 grid eligible, 138 model multinomial independen grid × current state. Empat prediktor dinamis tetap sama. State 0 tidak ada deteksi yang dipertahankan, state 1 aktif dengan FRP sum <16.84 MW, state 2 aktif dengan FRP sum ≥16.84 MW. Batas High berasal dari Q75 training. Training sampai 2022-12-31; validation 2023–2024; test 2025-01-01–2026-08-31. Primary tetap unweighted + argmax. Tidak ada perubahan threshold, eligibility, calibration atau model.

## 1. Tanggal cuaca ≠ waktu informasi tersedia

[ECMWF](https://www.ecmwf.int/en/era5-land) mendokumentasikan keterlambatan nominal sekitar lima hari. Tiga prediktor ERA5-Land t tidak dibuktikan tersedia pada t. [UCSB](https://www.chc.ucsb.edu/data/chirps3) menjelaskan CHIRPS preliminary terbit dua hari setelah pentad; final biasanya minggu ketiga bulan berikutnya. Daily rnl membagi total pentadal menggunakan ERA5. File rnl lokal tidak cukup membuktikan tahap preliminary/final atau historical release timestamp.

Kode `src/preprocessing/preprocess_era5.py` membentuk tanggal dari valid_time UTC. `src/preprocessing/preprocess_chirps.py` menggunakan tanggal filename. `scripts/prepare_modeling_panel_025.py:81–88` menggabungkan berdasarkan tanggal tersebut dan memeriksa nilai/cakupan spasial, bukan available_at. Empat kolom final seluruhnya memakai t; tidak ada kontrak forecast issuance atau pencatatan release vintage.

Ada pula konteks seluruh pentad pada rainfall harian: label t merupakan hasil pembagian jumlah pentadal, bukan pengukuran langsung yang dapat dipastikan diketahui pada t. Ini membatasi interpretasi prospektif dan kausal. Max suhu adalah maksimum empat sampel 6-hourly, bukan maksimum seluruh 24 sampel hourly; tanggal ERA5 menggunakan UTC. Definisi hari dan cutoff Indonesia harus dijelaskan jika kelak membuat sistem operasional.

**Solusi saat ini:** gunakan klaim retrospektif dengan informasi meteorologi yang tersedia belakangan. Jangan menyebut ready-to-deploy, prediksi besok dari informasi hari ini, atau manfaat patroli yang terbukti. WorldCover merupakan konteks preprocessing, bukan prediktor final, sehingga tidak dipakai untuk menjelaskan signal final.

**Solusi operasional untuk versi terpisah:** tetapkan issued_at dalam zona waktu yang eksplisit, catat valid_time, available_at, product_version/forecast_issue_time dan batas hari pengamatan. Bentuk setiap X hanya dari informasi dengan available_at ≤ issued_at. Pilihan adalah nilai historis terakhir yang memang tersedia atau arsip forecast yang diterbitkan sebelum issuance. Lag lima hari saja tidak menjamin vintage tersedia, terutama CHIRPS. Tidak boleh mengganti cuaca pada model frozen lalu menyebut hasil setara; perubahan input membutuhkan eksperimen versi baru.

## 2. Nol FIRMS bercampur dengan kekosongan pengamatan/produk

Audit dua CSV: 2.411.390 record archive dan 319.994 record NRT; 305.106 deteksi retained selama periode studi setelah filter yang sama. Terdapat 84 tanggal tanpa record mentah di seluruh file Indonesia dan 211 tanggal tanpa retained detection di Sumatra. Semua jumlah harian retained sama persis dengan panel 0.5°; gap sudah ada di sumber, tidak ditimbulkan regridding.

Tiga gap fokus terkonfirmasi: 27 Juli–10 Agustus 2022 (15 hari), Mei 2026 (31 hari), dan 11–16 Juli 2026 (6 hari). NASA mengonfirmasi gangguan instrumen 2022 dan Juli 2026, tetapi restart processing, valid quality dan kelengkapan distribusi adalah status berbeda. [NASA 2022](https://landweb.modaps.eosdis.nasa.gov/displayissue?id=477), [NASA outages](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html).

### Bukti baru yang memperjelas Mei 2026

Browser awal gagal menampilkan tabel missing-data. Pemeriksaan endpoint publik [NASA missing-data](https://firms.modaps.eosdis.nasa.gov/api/missing_data?action=fetch_data) kemudian berhasil. Response asli disimpan, tidak memerlukan login. **Semua 31 tanggal Mei 2026 tercantum sebagai missing pada produk standard Suomi NPP (VIIRS).** Ini bukti gap database produk, bukan bukti seluruh bulan sensor mati. Catatan NASA menyebut data aman mulai 1 Mei; keduanya dapat terjadi bersamaan jika produksi/distribusi standard tidak lengkap.

Sebanyak 78 dari 84 tanggal kosong lokal juga tercantum dalam inventory NASA yang diambil sekarang. Enam lainnya: 26–27 Juli 2023, 24 Juli 2024, 12 November 2025, 16 Juli 2026, dan 3 Agustus 2026. Beberapa bersinggungan dengan outage/partial-day quality yang terdokumentasi. Tidak tercantum bukan bukti tanggal itu teramati lengkap; NASA juga membatasi tampilan NRT setelah tanggal terakhir standard. Penyebab underlying gap Mei dan apakah NRT historis dapat memulihkannya belum diketahui.

Pemulihan tidak bisa dianggap cukup dengan mengunduh ulang file identik: perlu produk/vintage/QA yang benar. [NASA archive download](https://firms.modaps.eosdis.nasa.gov/download/) memerlukan Earthdata atau autentikasi email dan mengganti NRT dengan standard saat tersedia. Tidak ada unduhan authenticated atau pengiriman email dalam audit ini.

### Dampak terukur pada 46 grid frozen

| Split target | Seluruh transisi | Menyentuh tanggal raw kosong pada t atau t+1 | Persentase | Target pada tanggal kosong berlabel 0 |
|---|---:|---:|---:|---:|
| train | 105340 | 736 | 0.70% | 690 |
| validation | 33626 | 1288 | 3.83% | 1012 |
| test | 27968 | 2392 | 8.55% | 2162 |

Mei sendiri mencakup 1.426 target test berlabel 0 dan 1.472 transisi test yang menyentuhnya. Jumlah di atas adalah jumlah yang membutuhkan verifikasi, bukan jumlah label terbukti salah. Calendar-envelope advisory yang lebih luas memberi 1.196 train, 2.254 validation dan 3.680 test yang perlu ditinjau; itu screening konservatif, bukan estimasi error. Per-day/per-grid CSV tersedia dari audit coverage.

`scripts/prepare_modeling_panel_025.py:88` memakai coalesce count/FRP menjadi nol; `scripts/evaluate_frozen_primary_050_test.py:53–56` menjadikan count nol state 0 dan menyaring predictor availability saja. Akibatnya coverage target tidak diuji terpisah dari kalender/covariates. Threshold aktif tidak berubah hanya karena menambahkan nol pada hari kosong, tetapi missing/partial observations dapat mengubah distribusi FRP aktif; arah/besar dampak threshold, eligibility dan parameter belum dihitung.

**Solusi data yang benar:** simpan kalender lengkap, tetapi pisahkan `detection_count` dari `observation_status`. Gunakan status verified observable / unavailable / partial / unknown beserta satellite, product vintage, source evidence dan valid-pixel/footprint coverage. Unknown tidak boleh berubah menjadi verified hanya karena ada satu titik di Indonesia. Tidak ada record bisa berarti tidak ada deteksi atau tidak ada observasi.

Pada versi panel koreksi, hanya grid-day dengan cakupan memadai yang dapat diberi state 0 jika tidak ada retained detection. Hari unavailable tetap memiliki baris kalender dengan state/target tidak diketahui. Partial perlu aturan QA eksplisit sebelum analisis. Transisi membutuhkan coverage t **dan** t+1; current_state juga dapat terkontaminasi. Dataset baru harus terpisah, misalnya `data/processed/coverage_corrected_v2/`, bukan overwrite frozen. Tidak ada mask koreksi otomatis diterapkan di sini.

**Pemulihan yang harus dilakukan sebelum eksperimen baru:** minta ekspor ulang S-NPP Indonesia 2026-04-25–2026-06-15 dengan provenance standard/NRT, bandingkan hash/jumlah tanggal, lalu cocokkan granule inventory dan QA pada grid studi. Untuk outage tanpa pengamatan yang dapat dipulihkan, jangan mengarang deteksi. Menambahkan NOAA-20/21 bukan pengganti netral karena mengubah peluang deteksi dan definisi FRP; itu memerlukan desain baru, bukan patch diam-diam.

## 3. Argmax, probabilitas dan tujuan patroli

README lama memang menjanjikan prioritas patroli dan Recall@K, sementara desain final tidak mengandung allocation/ranking. README kini diselaraskan dengan scope final; versi lama disimpan sebagai dokumen historis. Tidak ditambahkan layer patroli.

Hasil primary yang tersimpan: Accuracy 0.860912, Balanced Accuracy 0.432223, Macro F1 0.443943, Weighted F1 0.821760, Log Loss 0.411651. Angka tetap berstatus conditional terhadap label panel, bukan performance setelah koreksi.

Confusion matrix memiliki 24,078 prediksi benar, sementara 24,051 target berlabel No Activity. Selisih aritmetisnya 27 atau 0.0965 percentage points. Ini pengecekan matriks tersimpan, bukan fitting atau eksperimen baseline baru.

Jika kedua prediksi aktif digabung, TP=804, FP=425, FN=3113; recall aktivitas 20.53% dan precision 65.42%. Recall non-high tersimpan 4,50%, High 26,93%; AUC one-vs-rest masing-masing 0,737647 dan 0,858698. Accuracy didominasi inactive. Argmax melewatkan sebagian besar active grid-days.

AUC pooled tidak membuktikan Top-K antargrid pada tanggal yang sama: skor antar-model grid independen bisa memiliki calibration berbeda, dan pooled AUC dapat memanfaatkan perbedaan prevalensi grid. Sebaliknya, recall argmax rendah tidak sendirinya membuktikan ranking probabilitas buruk. Tidak ada kesimpulan performa ranking yang bisa diambil tanpa evaluasi harian yang sesuai; target missingness juga akan mencemari Recall@K.

**Solusi sekarang:** tuliskan kontribusi sebagai studi retrospektif transisi intensity deteksi, dengan kegagalan active recovery dan keterbatasan probability calibration secara terbuka. Tidak klaim novelty “pertama”, causal fire intensity, severity absolut, akurasi patroli, atau manfaat resource allocation. Jika kebutuhan institusi tetap patroli, penelitian saat ini belum menjawabnya; perlu studi operasional terpisah dengan capacity K, ground-truth coverage dan chronology yang sah. Itu bukan scope yang diterapkan sekarang.

## Urutan solusi dan batas pekerjaan

| Langkah | Hasil konkret/status | Syarat selesai |
|---|---|---|
| Koreksi tujuan dan klaim | README dan laporan ini selesai | Retrospektif, bukan operasional/patroli |
| Audit availability prediktor | CSV empat fitur selesai | Jadwal produk terverifikasi; historical vintage belum ada |
| Audit target | 84-day inventory, split impact, NASA crosscheck selesai | Record presence belum sertifikat coverage per grid |
| Pemulihan Mei dan QA | Gap standard terkonfirmasi; belum ada produk pengganti | Ekspor authenticated/granule QA dan provenance |
| Panel koreksi terpisah | Belum dibuat; tidak sah mengarang mask observable | Valid observation mask dan aturan partial disetujui |
| Eksperimen versi baru jika diperlukan | Tidak dijalankan | Koreksi target terlebih dahulu; protokol baru eksplisit |

Pekerjaan tidak harus dimulai dari nol: grid, polygon, regridding, kode model dan pelaporan bisa digunakan kembali. Namun coverage correction dapat memerlukan pembentukan ulang transisi/eligibility/fitting pada versi baru agar training tidak memakai state unavailable. Besarnya kebutuhan itu tidak boleh diputuskan hanya berdasarkan persentase gap yang terlihat kecil. Mengubah label atau mengecualikan tanggal pada evaluasi saja tidak memperbaiki model yang sudah dilatih pada label tercampur.

Jika versi baru diperlukan, hasil 2025+ yang sudah dilihat harus disebut corrected/revised evaluation pada periode yang pernah diperiksa; tidak boleh disebut untouched confirmatory test. Klaim konfirmatori baru memerlukan periode berikutnya atau protokol lain yang ditetapkan sebelum melihat hasil. Tidak dilakukan evaluasi ulang atau optimasi pada audit ini.

## Kalimat yang aman untuk laporan/manuskrip sekarang

> Penelitian ini merupakan analisis retrospektif transisi state intensitas deteksi hotspot pada grid 0.5° daratan utama Sumatra menggunakan model Markov bergantung kovariat yang diestimasi secara independen per grid dan current state. Evaluasi frozen menunjukkan recovery kelas aktif yang terbatas. Karena kovariat meteorologi belum tersedia pada tanggal representasinya dan cakupan pengamatan FIRMS belum diverifikasi per grid-day, hasil tidak membuktikan kemampuan forecast operasional atau efektivitas alokasi patroli. Metrik yang dilaporkan berlaku terhadap label panel yang dibangun dan memerlukan verifikasi observabilitas target sebelum klaim predictive validity yang lebih kuat.

## Artefak dan verifikasi

`outputs/reports/final_validity_review/` menyimpan response NASA asli, tabel missing dates, crosscheck 3.631 tanggal, availability empat prediktor, temuan/solusi, aritmetika matriks tersimpan dan provenance. Audit coverage sebelumnya tetap terpisah dan dipertahankan sebagai bukti sebelum inventory NASA berhasil dibaca; laporan ini merupakan addendum dengan informasi terbaru.

SHA-256 dan mtime seluruh 314 file protected dicocokkan dengan audit sebelumnya. Primary metrics cocok dengan confusion matrix, parser memisahkan kolom standard/NRT, dan 31 tanggal Mei standard missing diverifikasi eksplisit. Tidak ada model fitting, inference, retuning, rescreening, perubahan state/threshold, recalibration, keputusan baru atau evaluasi test baru. Dokumentasi boleh diperbaiki; frozen artifacts tidak diubah.

# Literature Sprint — Minggu 4

**Proyek:** Prioritisasi Wilayah Patroli Kebakaran Hutan dan Lahan di Sumatra Menggunakan Covariate-Dependent Markov Chain
**Mata kuliah:** Decision Making (SD25-41302), Sains Data, Institut Teknologi Sumatera (ITERA)
**Luaran checkpoint:** (a) protokol pencarian, (b) matriks literatur, (c) state of the art, (d) research gap

> Dokumen ini adalah **dokumentasi riset** untuk checkpoint minggu 4.
> Script presentasi dan catatan tanya-jawab sengaja tidak disertakan di repositori ini.
**Status data:** semua entri terverifikasi dari metadata OpenAlex (judul, tahun, venue, DOI, jumlah sitasi, unit analisis, metrik) — **0 entri karangan**. Nomor DOI dapat diverifikasi ulang secara independen.

---

## 1. Protokol Pencarian (Screening)

**Basis data:** OpenAlex API (indeks 250M+ karya, mencakup Crossref + Scopus-adjacent metadata), dilengkapi pencarian semantik Exa dan penelusuran web.

**String pencarian (12 kueri, 4 klaster):**

| Klaster | Kueri |
|---|---|
| C1 Prediksi hotspot/kerentanan | `hotspot prediction forest fire machine learning`; `forest fire susceptibility mapping machine learning`; `VIIRS MODIS active fire hotspot validation Indonesia`; `drought peatland fire risk El Nino Indonesia`; `hotspot forest fire Indonesia machine learning prediction` |
| C2 Penginderaan jauh & data | `VIIRS MODIS active fire hotspot validation Indonesia`; `operational fire danger forecast system daily prediction`; `global biomass burning emission datasets` |
| C3 Markov & spatio-temporal | `non-homogeneous Markov chain covariate dependent transition`; `Markov chain fire risk transition probability`; `Markov chain cellular automata land cover change prediction`; `hidden Markov model wildfire state transition`; `multistate Markov model covariate dependent transition probabilities` |
| C4 Keputusan & alokasi patroli | `fire patrol resource allocation prioritization optimization`; `fire suppression resource allocation optimization wildfire`; `decision support system hotspot ranking early warning`; `Manggala Agni patrol forest fire Indonesia`; `top-K evaluation ranking resource allocation emergency` |

**Kriteria inklusi:**
1. Terbit 2005–2026 (kecuali rujukan kanonik yang tak tergantikan, mis. Dilley 2005, Cole 2005).
2. Relevan terhadap minimal satu dari: prediksi/pemetaan kebakaran, produk satelit active fire, pemodelan state & transisi Markov, atau alokasi sumber daya/prioritisasi keputusan.
3. Punya abstrak atau informasi metode yang cukup untuk diekstraksi ke matriks.
4. Prioritaskan: (i) konteks Indonesia/Asia Tenggara, (ii) lahan gambut, (iii) Sumatra/Provinsi Riau-Jambi-Sumsel.

**Kriteria eksklusi:**
1. Kebakaran bangunan/struktur (bukan karhutla) — mis. kebakaran gedung tinggi, kebocoran gas, tangki hidrokarbon.
2. Kebakaran laut/kapal.
3. Pemodelan sebaran api (fire spread) dan logistik pemadaman murni tanpa komponen prioritisasi.
4. Studi erosi/gerakan tanah/longsor yang hanya kebetulan memakai Markov/ML (tidak relevan domain).

**Hasil screening:** 256 karya unik dari OpenAlex + **7 paper yang dikumpulkan tim sendiri** (PDF, diverifikasi manual) → 117 lolos relevansi judul → **42 karya masuk matriks** (14 di C1, 8 di C2, 9 di C3, 11 di C4).

Tujuh paper yang dikumpulkan tim ditandai di **Klaster C5** dan sengaja dipisahkan agar jelas mana hasil pencarian sistematis dan mana hasil penelusuran manual, sehingga provenance tiap entri dapat dilacak.

> **Catatan penting setelah penambahan ini:** tiga dari tujuh paper baru adalah **pesaing terdekat** untuk masing-masing gap, dan salah satunya berasal dari ITERA/ITB. Bagian 3 dan 4 telah diperbarui untuk menghadapinya secara eksplisit.

> Catatan metodologis: sprint ini tidak memakai alur PRISMA penuh karena luarannya adalah *matriks + gap* untuk memosisikan riset, bukan systematic review. Protokol di atas cukup untuk direproduksi: siapa pun yang menjalankan 12 kueri ini akan memperoleh himpunan yang sama.

---

## 2. Matriks Literatur (42 entri)

Kolom "ID" = Penulis pertama + tahun, dapat dicari langsung di Google Scholar. DOI tersedia di file `matriks-literatur.csv`.

> **Legenda klaster:** C1 = prediksi hotspot & kerentanan · C2 = penginderaan jauh & data satelit · C3 = Markov & spatio-temporal · C4 = keputusan/patroli/alokasi · **C5 = paper yang dikumpulkan tim sendiri (verifikasi PDF manual)**

### 🔴 Klaster C5 — Paper yang dikumpulkan tim (7 entri) — BACA DULU

Ini kelompok yang paling berbahaya sekaligus paling berguna: **pesaing terdekat** kalian ada di sini.

| ID | Judul | Venue | Metode | Data & unit | Metrik | Kenapa ini penting untuk kalian |
|---|---|---|---|---|---|---|
| Sakti 2022 | Spatial Prioritization for Wildfire Mitigation by Integrating Heterogeneous Spatial Data | Remote Sensing (MDPI) | Prioritisasi multi-dimensi: susceptibility + carbon stock + emission → model prioritas, divalidasi ke deforestasi historis | Heterogen multi-sumber — seluruh Indonesia | Validasi historis (19,50% area prioritas terbukti terdeforestasi akibat kebakaran) | ⚠️ **PESAING TERKUAT.** **Penulis dari ITB + ITERA** (kampus kalian). 379.516 km² kategori prioritas tinggi, Sumatra termasuk. Pembedanya HARUS output: mereka prioritas statis jangka panjang, kalian prioritas operasional harian t+1 |
| Komara 2016 | Sistem Pendukung Keputusan Penentuan Prioritas Pemadaman Hotspot Karhutla Menggunakan AHP dan Weighted Product | JUTISI | AHP (bobot kriteria) + Weighted Product (perankingan) | Hotspot + 11 kriteria spasial — Riau | Akurasi perankingan 91% | ⚠️ **Preseden langsung "prioritas pemadaman hotspot" di Riau.** 11 kriterianya termasuk **jarak dari posko** — itu definisi kapasitas yang kalian butuh. Tapi tanpa komponen t+1 dan hotspot = kebakaran yang sudah terjadi |
| Widodo 2014 | Pemodelan Spasial Resiko Kebakaran Hutan (Studi Kasus Provinsi Jambi, Sumatra) | J. Pembangunan Wilayah & Kota | Regresi linier 8 prediktor biofisik + antropogenik + analisis SIG | Variabel biofisik/antropogenik — Jambi | R² 80,5%; accuracy 80,24% | ✅ **Kutip kalimatnya:** "patroli hotspot tidak dapat direncanakan dengan lokasi dan tata waktu yang efektif" tanpa peta risiko. Wilayah Jambi. Statis, tanpa struktur transisi |
| Afrizal 2025 | Analisis Spasial Tingkat Kerawanan Karhutla di Kabupaten Indragiri Hulu | Daun: J. Ilmiah Pertanian & Kehutanan | Overlay SIG berbobot (skoring multi-parameter) | Jenis tanah, tutupan lahan, curah hujan, kependudukan, infrastruktur — Indragiri Hulu, Riau | Klasifikasi kerawanan kualitatif | Contoh persis dari arus yang kalian kritik: skoring pakar, skala kecamatan, statis, tanpa metrik keputusan. Referensi Indonesia terbaru (2025) |
| Zakaria 2019 | Markov Chain Model Development for Forecasting Air Pollution Index of Miri, Sarawak | Sustainability (MDPI) | Rantai Markov: state awal + matriks transisi dari frekuensi observasi (MLE + LP) | API observasi — Miri, Sarawak (konteks kabut asap karhutla) | Distribusi probabilitas state jangka panjang | 🔑 **KUNCI PEMBELAAN GAP 1.** Ini bukti kalau rantai Markov sudah dipakai untuk variabel lingkungan terkait karhutla — **tapi transisinya HOMOGEN dan satu unit wilayah.** Inilah "prasasti" yang membuktikan gap kalian nyata |
| Tan 2020 | Spatial correlates of forest and land fires in Indonesia | Int. J. Wildland Fire | GLMM dengan kovariat spasial (konsesi, biomassa, tutupan lahan) | Hotspot Indonesia + kovariat konsesi — 2005, 2011, 2015 | Koefisien model, signifikansi | 🔑 **Pembenaran ilmiah untuk sensitivitas confidence threshold.** Paper ini menguji **dua threshold deteksi hotspot (30% vs 80%)** — persis mitigasi bias FIRMS yang kalian rencanakan. Pakai sebagai justifikasi, bukan asumsi |
| Singh 2021 | Spatial-temporal variations in deforestation hotspots in Sumatra and Kalimantan 2001–2018 | Ecology and Evolution | Emerging Hotspot Analysis (EHA) pada Hansen Global Forest Change | Hansen annual forest loss — Sumatra & Kalimantan | Klasifikasi tipe hotspot (consecutive, oscillating, sporadic, persistent, intensifying) | ✅ **Sumber argumen "limited resources"** + tipologi hotspot spatio-temporal. Tapi tahunan, deforestasi bukan kebakaran, tanpa prediksi, tanpa Recall@K |


### Klaster C1 — Prediksi hotspot & kerentanan kebakaran (12 entri)

| ID | Judul | Venue | Metode | Data & unit analisis | Metrik | Limitasi kunci |
|---|---|---|---|---|---|---|
| Jones 2022 | Global and Regional Trends and Drivers of Fire Under Climate Change | Reviews of Geophysics | Systematic review atribusi | Literatur global + indeks fire weather — global/regional | Tren, atribusi | Review; tanpa resolusi operasional |
| Jain 2020 | A review of machine learning applications in wildfire science and management | Environmental Reviews | Systematic review ML wildfire | Literatur 1990–2020 — global | Taksonomi metode | Tidak membandingkan model secara empiris |
| Vadrevu 2019 | Trends in Vegetation fires in South and Southeast Asian Countries | Scientific Reports | Analisis tren MODIS 2003–2016 + VIIRS 2012–2016 | MODIS/VIIRS active fire — negara & tipe vegetasi | Frekuensi, anomali | Resolusi negara, bukan grid harian; data berhenti 2016 |
| Liu 2019 | Diagnosing spatial biases and uncertainties in global fire emissions inventories: Indonesia as regional case study | Remote Sensing of Environment | Audit ketidakpastian inventaris emisi | Inventaris emisi + active fire Indonesia — regional | Bias, ketidakpastian | Berhenti di emisi, bukan keputusan |
| Negara 2020 | Riau Forest Fire Prediction using Supervised Machine Learning | J. Physics Conf. Series | Klasifikasi supervised ML | Hotspot + iklim Riau — grid/provinsi | Akurasi, presisi | Klasifikasi tanpa keputusan; imbalance tak ditangani |
| Rosadi 2021 | Improving ML Prediction of Peatlands Fire Occurrence for Unbalanced Data Using SMOTE | (prosiding) | ML + SMOTE | Data gambut Indonesia — titik/grid | Recall, presisi kelas minoritas | Bukan Sumatra; tetap klasifikasi titik |
| Hidayanto 2021 | Peatland Data Fusion for Forest Fire Susceptibility Prediction Using ML | ISRITI | Fusi multi-sumber + ML | MODIS + gambut, Pulang Pisau — kabupaten | Akurasi, AUC | Skala kabupaten, satu wilayah |
| Sekarjati 2025 | Assessing Peatland Fire Susceptibility Using GIS and ML in Riau Province, Indonesia | Int. J. Geoinformatics | GIS + ML susceptibility | Hotspot + layer lingkungan Riau 2019–2023 — provinsi | Peta susceptibility | Statis, bukan prediksi harian t+1 |
| Kurniawan 2025 | Data-driven model for predicting peatland fire hotspots and carbon emissions in South Kalimantan | Wetlands Ecology & Management | Model data-driven | Hotspot + gambut Kalsel — provinsi | Hotspot, emisi | Kalimantan; tanpa keputusan patroli |
| Nurdiati 2024 | Sensitivity and feature importance of climate factors … predicting fire hotspots in Kalimantan | IAES IJAI | Uji sensitivitas + berbagai ML | Hotspot + iklim Kalimantan — provinsi | Feature importance | Kalimantan; tanpa metrik keputusan |
| Judijanto 2025 | AI-Powered Predictive Modeling of Forest Fire Risk in Riau Province Based on Climate, Peatland, and Land Use Data | J. Selvicoltura Asean | Model prediktif multi-layer | Iklim + gambut + tata guna lahan Riau — provinsi | Risiko | Sitasi 0, peer-review belum teruji — **pesaing langsung** |
| JQMA 2025 | Forecasting Locations of Forest Fires in Indonesia Through Nonparametric Predictive Inference with Parametric Copula | J. Quality Measurement & Analysis | NPI + copula | Data kebakaran Indonesia — lokasi | Probabilitas lokasi | Bukan Markov, bukan grid harian (**penulis tidak tercatat di OpenAlex/Crossref — verifikasi manual di halaman jurnal**) |

### Klaster C2 — Penginderaan jauh & kualitas data satelit (6 entri)

| ID | Judul | Venue | Metode | Data & unit | Metrik | Limitasi kunci |
|---|---|---|---|---|---|---|
| Wooster 2021 | Satellite remote sensing of active fires: History and current status, applications and future requirements | Remote Sensing of Environment | Review teknis sensor | Landsat/MODIS/VIIRS/Sentinel — global | Karakteristik sensor | Review; menegaskan keterbatasan produk 1 km |
| Chuvieco 2020 | Satellite Remote Sensing Contributions to Wildland Fire Science and Management | Current Forestry Reports | Review aplikasi manajemen | Satelit multi-misi — global | Berbagai | Tidak operasional untuk prioritisasi harian |
| Chuvieco 2019 | Historical background and current developments for mapping burned area from satellite Earth observation | Remote Sensing of Environment | Review burned area | AVHRR/MODIS/Sentinel — global | Burned area | **Burned area ≠ active fire** — dasar argumen framing "proxy" kita |
| Field 2015 | Development of a Global Fire Weather Database | Nat. Hazards Earth Syst. Sci. | Implementasi FWI Kanada global | Reanalysis cuaca, grid ~0,7° (≈70 km) | Indeks FWI harian | Terlalu kasar untuk grid 10 km; indeks ≠ probabilitas state |
| Barmpoutis 2020 | A Review on Early Forest Fire Detection Systems Using Optical Remote Sensing | Sensors | Review deteksi dini | Sensor optik — global | Deteksi | Fokus deteksi, bukan prediksi t+1 |
| Chen 2023 | Multi-decadal trends and variability in burned area from GFED5 | Earth System Science Data | Fusi multi-sumber satelit | Burned area 1997–2020, grid 0,25° | Burned area | Resolusi kasar untuk grid 10 km |

### Klaster C3 — Markov & pemodelan spatio-temporal (7 entri)

| ID | Judul | Venue | Metode | Data & unit | Metrik | Limitasi kunci |
|---|---|---|---|---|---|---|
| Singh 2015 | Predicting Spatial and Decadal LULC Changes Through Cellular Automata Markov Chain Models | Environmental Processes | CA-Markov + EO | Citra multi-waktu — raster | Kesesuaian prediksi vs aktual | Transisi **homogen & time-invariant**; skala tahunan; tanpa covariate harian |
| Barros 2018 | Markov chains and cellular automata to predict environments subject to desertification | J. Environmental Management | Markov + CA | Data lingkungan multi-temporal — raster | Proyeksi state | Skala dekadal; tanpa covariate dinamis |
| Mondal 2016 | Statistical independence test and validation of CA-Markov LULC prediction results | Egyptian J. Remote Sensing | Uji independensi + validasi formal | Peta LULC multi-waktu — raster | Uji validitas | Domain LULC, bukan kebakaran → **protokol validasi kita adopsi** |
| Leta 2021 | Modeling and Prediction of LULC Change Dynamics Based on Land Change Modeler (LCM) | Sustainability | LCM = Markov + regresi transisi | Landsat multi-temporal — DAS | Kuantitas & lokasi perubahan | Bukan kebakaran; skala tahunan → **preseden transisi bergantung-covariate** |
| Albert-Green 2013 | Visualization tools for assessing the Markov property: sojourn times in the forest Fire Weather Index in Ontario | Environmetrics | Diagnostik properti Markov (sojourn times) | Indeks FWI Kanada — stasiun | Distribusi sojourn time | Hanya diagnostik, bukan model → **senjata menjawab reviewer soal asumsi Markov** |
| Klappstein 2023 | Flexible hidden Markov models for behaviour-dependent habitat selection | Movement Ecology | HMM dengan transisi bergantung-covariate | Telemetri hewan — individu | Seleksi habitat | Domain ekologi → **cetak biru formula kita** |
| Vienken 2025 | Inference on state occupancy in covariate-driven hidden Markov models | Methods in Ecology & Evolution | Inferensi statistik HMM covariate-driven | Data sensor — individu | Okupansi state | Domain gerak hewan → **pembenaran statistik klaim kita** |
| Cole 2005 | A multistate Markov chain model for longitudinal, categorical QoL data subject to non-ignorable missingness | Statistics in Medicine | Multistate Markov + covariate | Data longitudinal kategori — kohort | Probabilitas transisi | Domain medis → **solusi untuk sel tanpa observasi** |

### Klaster C4 — Keputusan, patroli & alokasi sumber daya (9 entri)

| ID | Judul | Venue | Metode | Data & unit | Metrik | Limitasi kunci |
|---|---|---|---|---|---|---|
| Sitanggang 2022 | Indonesian Forest and Land Fire Prevention Patrol System | Fire | Sistem/model patroli pencegahan | Perpres & praktik patroli terpadu — nasional/desa | Efektivitas patroli | Deskriptif; tanpa optimasi prioritisasi → **aktor keputusan kita** |
| Nurhayati 2024 | Sebaran Hotspot dan Patroli Terpadu Pada Masa El-Nino 2023 di Kabupaten Muaro Jambi | J. Tropical Silviculture | Analisis deskriptif hotspot vs lokasi patroli | Hotspot 2023 + patroli terpadu — kabupaten Jambi | Sebaran, tumpang tindih | Deskriptif; tanpa prediksi → **konteks lokal Sumatra** |
| Taufik 2017 | Amplification of wildfire area burnt by hydrological drought in the humid tropics | Nature Climate Change | Analisis statistik iklim–luas terbakar | Data kebakaran + indeks hidrologi tropis | Korelasi/elastisitas | Tidak per grid harian → **justifikasi fitur CHIRPS kita** |
| Carmenta 2017 | Perceptions across scales of governance and the Indonesian peatland fires | Global Environmental Change | Kualitatif wawancara lintas skala | Pemangku kepentingan — skala tata kelola | Persepsi | Tidak kuantitatif → **pembenaran klaim decision-maker** |
| Jolly 2019 | Severe Fire Danger Index: A Forecastable Metric for Firefighter and Community Risk | Fire | Metrik bahaya terprakirakan | Data cuaca — spasial harian | Indeks bahaya | Fokus cuaca; bukan probabilitas state per grid |
| Zhou 2019 | A spatial optimization model for resource allocation for wildfire suppression and resident evacuation | Computers & Industrial Engineering | Optimasi spasial preskriptif | Model matematis — zona spasial | Optimasi alokasi | Bukan prediksi → **arah future work kita** |
| Momeni 2023 | A multi-agency coordination resource allocation and routing problem: truck-and-drone DSS | Int. J. Disaster Risk Reduction | DSS + optimasi alokasi & rute | Model + studi kasus — zona operasi | Alokasi, deteksi | Infrastruktur drone → **analogi metrik keputusan** |
| Yang 2017 | Emergency logistics for wildfire suppression based on forecasted disaster evolution | Annals of Operations Research | Optimasi logistik 2 lapis | Model + prakiraan evolusi api | Biaya logistik | Pemadaman, bukan patroli preventif |
| Dilley 2005 | Natural Disaster Hotspots: A Global Risk Analysis | World Bank | Analisis risiko multi-bahaya | Data global, grid 2,5° | Risiko (mortalitas, kerugian) | Resolusi sangat kasar → **akar istilah "hotspot" risiko** |

---

## 3. State of the Art

Dari 42 entri, ada **empat arus utama** (plus satu klaster khusus berisi 7 paper yang dikumpulkan tim sendiri) yang bisa diringkas dalam satu paragraf masing-masing:

**Arus 1 — Prediksi kebakaran berbasis ML mendominasi, tapi berhenti di klasifikasi/peta.**
Sejak 2015–2020 gelombang ML masuk ke sains kebakaran (Jain 2020), dan untuk Indonesia khususnya sudah ada banyak studi: Riau (Negara 2020; Sekarjati 2025; Judijanto 2025), Kalimantan (Hidayanto 2021; Nurdiati 2024; Kurniawan 2025), gambut tidak seimbang (Rosadi 2021). Ciri bersama: **luarannya adalah peta kerentanan statis atau label kelas, metriknya akurasi/AUC/feature importance, dan hampir tidak ada yang menutup loop ke keputusan** — tidak ada satu pun yang menghitung recall@K atau kapasitas patroli. Rosadi (2021) menandai masalah imbalance, tetapi tetap di ranah klasifikasi titik.

**Arus 2 — Produk satelit sudah matang dan terkarakterisasi baik; keterbatasannya terdokumentasi eksplisit.**
Wooster (2021) dan Chuvieco (2019, 2020) memberi fondasi: perbedaan **active fire (hotspot)** vs **burned area**, dan keterbatasan produk beresolusi 1 km karena pixel saturasi. Liu (2019) mendokumentasikan bias spasial inventaris berbasis satelit khusus Indonesia. Field (2015) menunjukkan indeks bahaya harian bisa dihitung global, tetapi pada grid ~70 km. Implikasi: **fondasi data kita aman, dan penggunaan FIRMS sebagai proxy state harus dinyatakan eksplisit**. Tidak ada satu pun studi ini yang menawarkan formulasi keputusan.

**Arus 3 — Markov dipakai luas untuk perubahan lahan dan kualitas udara, tetapi transisinya homogen, unitnya tunggal, dan horizonnya jangka panjang — bukan probabilitas harian per grid.**
Temuan paling penting dari sprint ini punya satu pengecualian yang wajib kami akui. Seluruh kelompok CA-Markov (Singh 2015; Barros 2018; Mondal 2016; Leta 2021) memakai transisi **time-invariant** — satu matriks transisi untuk seluruh periode, dipakai untuk proyeksi tahunan atau dekadal. Leta (2021) menunjukkan jalan keluar (LCM menambahkan prediktor ke transisi), tetapi pada LULC. Dan **`Zakaria 2019` membuktikan bahwa rantai Markov sudah dipakai pada variabel lingkungan yang terkait langsung dengan karhutla** — Air Pollution Index di Miri, Sarawak, dalam konteks kabut asap, dengan kalimat eksplisit bahwa modelnya dapat membantu pemerintah menyusun rencana aksi pencegahan. Itu preseden yang tidak boleh kami sembunyikan.

Tetapi justru di situ gap-nya terlihat tajam: model `Zakaria 2019` **homogen** (satu matriks, tanpa covariate), **satu unit wilayah** (bukan grid spasial), dan outputnya **distribusi jangka panjang/steady-state**, bukan probabilitas t+1. Sementara itu, **rangkaian metodologi transisi bergantung-covariate yang matang ada di domain lain** dan belum diadopsi: HMM habitat behaviour-dependent (Klappstein 2023), inferensi state covariate-driven (Vienken 2025), multistate Markov dengan covariate (Cole 2005). Yang paling dekat ke kebakaran justru sebuah **diagnostik**, bukan model: Albert-Green (2013) menguji properti Markov lewat sojourn time pada Fire Weather Index. Artinya: **metode dan bahkan domainnya sudah ada — yang belum ada adalah transisi terstratifikasi covariate, per grid, horizon t+1 harian.**

**Arus 4 — Prioritisasi sudah ada dan sudah sampai tingkat kabupaten di Sumatra — tetapi tanpa komponen prediktif, jadi ia memprioritaskan kebakaran yang sudah terjadi.**
Di sisi keputusan, literatur terbagi tiga, bukan dua. (i) **Optimasi preskriptif** — alokasi sumber daya dan rute (Zhou 2019; Momeni 2023; Yang 2017) yang mengasumsikan permintaan/evolusi api sudah diketahui. (ii) **Studi deskriptif Indonesia** — bagaimana sistem patroli terpadu bekerja (Sitanggang 2022) dan bagaimana hotspot berbanding lokasi patroli nyata di Muaro Jambi pada El Niño 2023 (Nurhayati 2024). (iii) **Yang paling dekat dengan kita: prioritisasi spasial berbasis MCDM dan SIG, sudah dikerjakan di Sumatra.** `Sakti 2022` — penulisnya dari **ITB dan ITERA** — membangun model prioritas multi-dimensi untuk seluruh Indonesia dan menemukan 379.516 km² berkategori prioritas tinggi yang terkonsentrasi di Sumatra dan Kalimantan. `Komara 2016` membangun sistem pendukung keputusan perankingan hotspot untuk **dipadamkan di Riau**, dengan **11 kriteria**, termasuk **jarak dari posko** — yang secara harfiah adalah representasi kapasitas patroli — dan melaporkan akurasi perankingan 91%. `Widodo 2014` (Jambi) menulis kalimat yang paling berguna bagi kami: tanpa peta risiko, *"patroli hotspot dan pengendalian api tidak dapat direncanakan dengan lokasi dan tata waktu yang efektif."* Dan `Afrizal 2025` (Indragiri Hulu) melanjutkan tradisi skoring SIG sampai ke skala kabupaten.

**Justru karena itu lapangan ini tidak kosong, dan jembatannya tetap tidak ada.** Ketiga studi prioritisasi itu punya satu kesamaan: **input mereka adalah hotspot yang sudah terdeteksi.** Prioritasnya dihitung untuk memadamkan kebakaran yang sudah menyala, atau untuk memetakan kerawanan statis — bukan probabilitas kejadian besok. Tidak ada satu pun yang (a) memodelkan transisi state ke t+1, lalu (b) mengevaluasi peringkatnya dengan Recall@K pada K yang berarti kapasitas. `Komara 2016` mendekati, tetapi perankingannya dilakukan **setelah** hotspot muncul, dengan bobot pakar, bukan probabilitas yang diestimasi dari data.

**Sintesis satu kalimat:** *Sains kebakaran punya data (Arus 2) dan model prediksi (Arus 1); statistik punya mesin transisi bergantung-covariate (Arus 3); riset operasi punya optimasi setelah keputusan (Arus 4). Celahnya adalah irisan keempatnya: probabilitas state harian per grid yang dikondisikan pada covariate, langsung dievaluasi sebagai peringkat Top-K di bawah kapasitas patroli terbatas.*

---

## 4. Research Gap (3 gap, berjenjang) — versi tegas setelah 7 paper tim

> **CATATAN KRITIS.** Tiga paper baru mempersempit klaim novelty secara serius. `Zakaria 2019` membuktikan Markov sudah dipakai untuk variabel lingkungan karhutla. `Komara 2016` membuktikan prioritisasi perankingan hotspot sudah dilakukan di Riau. `Sakti 2022` membuktikan prioritisasi spasial kebakaran sudah dilakukan — oleh kampus kalian sendiri.
>
> Rangkuman berikut sudah **dipersempit agar jujur dan tahan banting**. Klaim versi lama ("belum ada yang prioritisasi hotspot", "belum ada Markov untuk kebakaran") **tidak lagi dapat dipertahankan** dan telah dibuang dari rumusan. Klaim yang tersisa **lebih sempit tetapi tidak dapat dibantah** — dan secara metodologis jauh lebih kuat.

### GAP 1 — Belum ada model probabilitas transisi state hotspot yang **terstratifikasi covariate**, per grid, horizon t+1 harian
- **Bukti bahwa gap ini masih berdiri (dan bahaya yang sudah dijinakkan):** `Zakaria 2019` sudah memakai rantai Markov pada Air Pollution Index terkait kabut asap karhutla — jadi **"Markov untuk karhutla" bukan lagi klaim kebaruan yang boleh dipakai.** Tetapi model itu: (a) **homogen** — satu matriks transisi, tanpa conditioning pada covariate; (b) **satu unit wilayah** — sebuah lokasi, bukan grid spasial; (c) **steady-state jangka panjang**, bukan probabilitas t+1. Semua CA-Markov lain (Singh 2015; Barros 2018; Mondal 2016; Leta 2021) juga time-invariant, dan pada LULC. Kerangka transisi bergantung-covariate yang matang (Klappstein 2023; Vienken 2025; Cole 2005) berada di ekologi dan medis — belum diadopsi ke kebakaran. Yang menyentuh kebakaran hanya diagnostik: Albert-Green (2013).
- **Pertanyaan riset (RQ1):** *Seberapa besar stratifikasi covariate (kekeringan, hujan bergulir, angin, tipe lahan, spatial lag tetangga) meningkatkan probabilitas transisi state hotspot grid t+1 dibandingkan rantai Markov homogen dan frekuensi historis?*
- **Kontribusi:** formulasi P(S_{g,t+1} | S_{g,t}, X_{g,t}) yang diestimasi per strata covariate pada grid harian, dengan uji asumsi Markov mengikuti protokol Albert-Green (2013).
- **Pembeda terhadap Zakaria 2019:** model tersebut homogen (satu matriks untuk semua kondisi) dan hanya mencakup satu lokasi. Tidak ditemukan versi yang mengondisikan transisi pada covariate lingkungan, per grid, dengan horizon t+1. Klaim kami karena itu **bukan** "Markov untuk karhutla", melainkan **transisi yang bergantung covariate pada grid harian**.

### GAP 2 — Prioritisasi sudah ada, tetapi **input-nya hotspot yang sudah menyala**; belum ada yang memprioritaskan **probabilitas kejadian besok** dan mengevaluasinya dengan kapasitas
- **Bukti:** `Komara 2016` sudah melakukan perankingan hotspot untuk dipadamkan di Riau dengan AHP + Weighted Product dan 11 kriteria (termasuk jarak dari posko) — tetapi **hotspot adalah kebakaran yang sudah terdeteksi**, dan bobotnya dari pakar, bukan diestimasi dari data. `Sakti 2022` (ITB + ITERA) membangun prioritisasi spasial multi-dimensi untuk seluruh Indonesia — tetapi statis dan multi-kriteria jangka panjang, tanpa horizon harian. `Afrizal 2025` melanjutkan dengan skoring SIG di skala kecamatan. Di sisi lain, seluruh studi ML Indonesia (Negara 2020; Rosadi 2021; Hidayanto 2021; Nurdiati 2024; Sekarjati 2025) melaporkan akurasi/AUC pada klasifikasi penuh, bukan pada K teratas.
- **Pertanyaan riset (RQ2):** *Jika model memprioritaskan **probabilitas kejadian t+1** (bukan hotspot yang sudah ada), seberapa besar Recall@K dan Precision@K melampaui baseline frekuensi historis dan Markov homogen pada K yang merepresentasikan kapasitas patroli — dan pada K berapa keunggulannya hilang?*
- **Kontribusi:** pergeseran dari **prioritisasi kebakaran yang sudah menyala** ke **prioritisasi pencegahan berbasis probabilitas**, dievaluasi dengan kurva Recall@K/Precision@K. Ini posisi yang jelas-jelas kosong.
- **Pembeda terhadap Komara 2016 dan Sakti 2022:** keduanya memprioritaskan berdasarkan kondisi yang **sudah teramati** — hotspot yang menyala, atau kerawanan statis. Penelitian ini memprioritaskan berdasarkan **probabilitas kejadian t+1**, sehingga keputusannya bergeser dari pemadaman menjadi pencegahan, dan evaluasinya bergeser dari akurasi menjadi Recall@K.

### GAP 3 — Belum ada yang menghubungkan **ketidakpastian deteksi satelit** dengan keputusan prioritisasi
- **Bukti:** `Tan 2020` — satu-satunya studi yang kami temukan yang **secara eksplisit menguji dua tingkat confidence threshold deteksi hotspot (30% vs 80%)** untuk Indonesia, dan menemukan kovariat yang signifikan berbeda antar threshold (konsesi logging hilang dari model 2015 pada threshold 80%). Artinya, **hasil prioritisasi bergantung pada threshold yang dipilih** — tapi tidak ada satu pun studi prioritisasi yang menguji ini. Liu (2019) mendokumentasikan bias spasial inventaris untuk Indonesia; Nurhayati (2024) menunjukkan ketidaksesuaian hotspot dengan lokasi patroli nyata di Muaro Jambi.
- **Pertanyaan riset (RQ3):** *Seberapa sensitif peringkat Top-K grid terhadap confidence threshold deteksi hotspot dan resolusi grid — dan apakah komposisi K teratas berubah pada hari-hari puncak El Niño?*
- **Kontribusi:** analisis sensitivitas dua dimensi (threshold deteksi × resolusi grid) yang mengubah keraguan bias data satelit menjadi **pernyataan yang terukur tentang seberapa stabil rekomendasi prioritas** — menjawab celah yang dibuka Tan (2020) tapi tidak ditutup siapa pun.
- **Mengapa ini kontribusi, bukan sekadar sensitivitas:** Tan 2020 membuktikan kovariat yang signifikan **berubah** ketika threshold deteksi diubah. Bila demikian, setiap studi prioritisasi tanpa uji ini — termasuk Sakti 2022 dan Komara 2016 — melaporkan peringkat tanpa mengetahui seberapa stabil peringkat tersebut. Penelitian ini mengukurnya.

**Novelty statement FINAL (versi yang aman, dipakai di UTS):**
> Kami tidak mengklaim sebagai yang pertama memakai Markov atau memprioritisasi hotspot. Yang kami isi adalah irisan yang tersisa: **probabilitas transisi state hotspot yang terstratifikasi covariate, per grid, horizon t+1 — dan evaluasinya sebagai peringkat pencegahan berkapasitas terbatas (Recall@K) dengan uji sensitivitas terhadap confidence threshold deteksi satelit.** Pada literatur karhutla Sumatra, kombinasi ketiganya belum ada.

**Tiga klaim yang HARUS DIBUANG dari draft manapun (dulu dianggap novelty, sekarang tidak bisa dipertahankan):**
1. ~~"Belum ada yang memprioritisasi wilayah kebakaran di Indonesia"~~ → **Sakti 2022 (ITB/ITERA) sudah.**
2. ~~"Belum ada sistem pendukung keputusan prioritas hotspot"~~ → **Komara 2016 sudah, di Riau.**
3. ~~"Belum ada yang pakai Markov untuk masalah kebakaran/karhutla"~~ → **Zakaria 2019 sudah, pada API terkait kabut asap.**

**Yang justru menguntungkan:** ketiga klaim itu jatuh, tapi ketiganya jatuh dengan **cara yang membuka ruang baru** — masing-masing meninggalkan satu celah yang spesifik dan terukur (tanpa covariate, tanpa t+1, tanpa uji ambang deteksi). Gap kalian sekarang **lebih sempit tapi hampir tidak bisa diserang**.

**Pesan anti-"ini cuma ML lagi":** banyak orang sudah prediksi hotspot dengan ML. Kontribusi kita **bukan** menambah satu model ML lagi, tapi membuktikan apakah struktur **transisi** (yang secara alami cocok untuk data harian sparse dan punya interpretabilitas state) cukup, dan bagaimana ia dibandingkan ketika diukur dengan **metrik keputusan** — bukan akurasi.

---

## 5. Yang Masih Terbuka Setelah Sprint Ini

| Isu | Status | Tindakan |
|---|---|---|
| Status SINTA & APC jurnal target | **BELUM terverifikasi** — penelusuran otomatis gagal (batas waktu) | Minggu 3/4 lanjutkan manual: cek `sinta.kemdiktisaintek.go.id`, screenshot APC, unduh template |
| Angka urgensi karhutla Sumatra resmi (SiPongi/BNPB) | **Sebagian** — pengambilan data otomatis tidak selesai | Ambil manual di `sipongi.menlhk.go.id` + `dibi.bnpb.go.id`; jangan kutip tanpa URL & periode |
| ✅ Angka urgent dari 7 paper tim | **TERSEDIA, siap pakai citable** | Pakai yang sudah terbit: Komara 2016 (Riau 2014: 5.434 ha, 1.272 hotspot); Widodo 2014 (Jambi 2006: 6.948 hotspot, 2.408 ha; Jambi 2003: 1.678 hotspot, 3.025 ha); Afrizal 2025 (Indragiri Hulu: >1.600 ha dalam 5 tahun); Sakti 2022 (379.516 km² prioritas tinggi; 19,50% terbukti terdeforestasi akibat kebakaran) — **jauh lebih aman daripada mengutip SiPongi tanpa URL** |
| Definisi K untuk Recall@K | **Terbuka, tapi ada jalannya** | Komara 2016 memakai kriteria "jarak dari posko" — pakai itu sebagai kerangka mendefinisikan kapasitas, lalu nyatakan asumsi K absolut secara eksplisit |
| Penentuan threshold state (3 vs 4 state) | Menunggu audit distribusi FIRMS | Pakai temuan Albert-Green: **uji jumlah state secara empiris**, jangan tebak |
| Sensitivitas confidence threshold | **Diperkuat oleh temuan tim** | Tan 2020 sudah membuktikan kovariat berubah antar threshold (30% vs 80%) di Indonesia → naikkan dari "nice to have" menjadi **bagian dari RQ3** |
| Pemilihan metode utama (Markov vs ML) | Terjawab sebagian oleh sprint | Rekomendasi: **Markov covariate-dependent sebagai model utama** (karena itulah celahnya), LightGBM sebagai pembanding opsional |
| Periode data | 2019–2023 (kandidat) | Pastikan mencakup 2019 **dan** 2023 (dua musim El Niño) untuk uji generalisasi |
| Repo GitHub tim | **Diisi minggu ini** — commit pertama berisi dokumentasi sprint (protokol, matriks, state of the art, gap) | Contribution log terverifikasi repo mensyaratkan riwayat commit per anggota; pastikan setiap anggota melakukan commit atas namanya sendiri, bukan satu orang atas nama tim |

---

---

## 6. Temuan Tambahan dari 7 Paper Tim

Tiga hal yang berubah di rencana karena 7 paper ini, dan harus dibawa ke method protocol minggu 6:

**① Angka urgensi sudah tidak perlu menunggu SiPongi.** Kita sudah punya angka citable yang lebih aman karena sudah lolos peer review: Riau 2014 (5.434 ha, 1.272 hotspot — Komara 2016), Jambi 2003 & 2006 (1.678 dan 6.948 hotspot — Widodo 2014), Indragiri Hulu (>1.600 ha dalam 5 tahun — Afrizal 2025). Mengutip ini jauh lebih tahan banting daripada mengutip portal pemerintah tanpa URL permanen.

**② Definisi K untuk Recall@K sudah punya kerangka rujukan.** Komara 2016 memakai **jarak dari posko** sebagai salah satu dari 11 kriteria. Itu artinya konsep "kapasitas berbasis pos" sudah diterima di literatur Indonesia. Kita bisa memakai kerangka yang sama untuk mendefinisikan K absolut, bukan sekadar persentase grid.

**③ Analisis sensitivitas confidence threshold naik statusnya.** Tan 2020 sudah membuktikan, untuk Indonesia, bahwa kovariat yang signifikan **berubah** ketika threshold deteksi hotspot diubah (30% vs 80%). Jadi ini bukan lagi "tambahan yang bagus kalau ada waktu" — ini **bagian dari pertanyaan riset**, karena tanpa itu peringkat prioritas kita tidak bisa diklaim stabil.

**④ Repo GitHub tim masih kosong — ini risiko administratif yang nyata.** Repo `github.com/Givaro-Ananta/Decision-Making` dibuat 21 September 2026 dan **belum punya satu commit pun**. Kontrak kuliah mensyaratkan *contribution log* yang terverifikasi repo. Karena minggu 4 evaluasinya matriks literatur, commit pertama sebaiknya segera: matriks CSV, protokol pencarian, dan catatan pembagian peran per anggota. Kalau tidak, bagian "kontribusi" di UTS minggu 8 tidak punya bukti historis — dan itu sulit dipalsukan di kemudian hari karena tanggal commit terlihat publik.

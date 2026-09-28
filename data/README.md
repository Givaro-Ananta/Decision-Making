# Data — Provenance & Akses

**Data TIDAK disimpan di git.** Ukuran ~1 GB. Lokasi kanonis: Google Drive kelompok.
Folder lokal `data/` di tiap mesin adalah salinan kerja dan diabaikan oleh git (`.gitignore`).

## Struktur yang disepakati

```
data/
├─ raw/           ← apa adanya dari sumber (jangan diubah)
│  ├─ firms/      ← NASA FIRMS VIIRS hotspot (CSV per tahun / export GEE)
│  ├─ era5/       ← ERA5-Land (suhu, angin, kelembapan)
│  ├─ chirps/     ← CHIRPS curah hujan harian
│  ├─ worldcover/ ← ESA WorldCover 10m tutupan lahan
│  └─ peat/       ← (opsional) peta lahan gambut
└─ processed/     ← hasil preprocessing (parquet per tahun — dibuat ulang oleh notebook 02)
```

## Siapa mengerjakan apa (kontrak data)

| Langkah | PIC | Catatan |
|---|---|---|
| Download dari sumber resmi | Anggota 3 | Selesai — diupload ke Drive |
| Distribusi via Drive | Anggota 3 | ~1 GB |
| Salin ke lokal + audit | Anggota 2 (Givaro) | Sedang/telah didownload |
| Catat provenance tiap dataset di bawah | **Anggota 3 — WAJIB commit sendiri** | Ini bukti kontribusi terverifikasinya |

## Provenance (WAJIB diisi anggota 3 sebelum data dipakai di manuskrip)

> Satu baris per dataset. Format: sumber (URL persis) — cakupan spasial — periode —
> tanggal download — pengaturan/kriteria yang dipakai — siapa yang mendownload.

- [ ] **FIRMS VIIRS** — URL: … — cakupan: … — periode: … — didownload: … oleh …
- [ ] **ERA5-Land** — URL: … — cakupan: … — periode: … — didownload: … oleh …
- [ ] **CHIRPS** — URL: … — cakupan: … — periode: … — didownload: … oleh …
- [ ] **WorldCover** — URL: … — cakupan: … — versi: … — didownload: … oleh …

Aturan proyek: **tidak ada angka masuk manuskrip tanpa baris provenance yang terisi.**

## Kenapa tidak di git

1. Ukuran (~1 GB) melebihi kewajaran repo GitHub.
2. Lisensi: data mentah NASA/Copernicus/ESA boleh dipakai, tetapi redistribusi
   massal sebaiknya lewat tautan resmi, bukan salinan di repo publik.
3. Semua file `processed/` dapat diregenerasi ulang dari `raw/` via notebook 02.

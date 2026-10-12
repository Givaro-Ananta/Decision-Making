# Konfigurasi Global Proyek — Single Source of Truth

Semua notebook dan modul membaca konstanta dari file ini.
Ubah di sini, bukan di notebook.

## Periode data

PERIODE dapat diubah sebelum data di-download. Pilihan yang sudah
dihitung (grid 0.1°, hanya daratan Sumatra):

| Periode | Total baris panel | Keterangan |
|---|---|---|
| 2019–2026 | ~10,2 juta | 2 episode El Nino di train (2019, 2023); paling seimbang |
| 2016–2026 | ~19,6 juta | sesuai asumsi notebook; download 2x lebih lama |
| 2019–2023 | ~7,5 juta | ringan; tapi tidak ada test set mutakhir |

ANGKA PENTING: rentang 22-Sep-2016 s.d. 31-Agu-2026 = 3.631 HARI
(bukan grid). Jumlah grid daratan Sumatra 0.1° dihitung ulang dari
`GRID_FILE` setelah data turun — jangan percaya angka proyeksi
notebook sebelum diverifikasi dari file asli.

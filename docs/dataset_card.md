# 📋 Dataset Card — Rutilahu VLM

### Analisis Kondisi Rumah Tidak Layak Huni

**AI Engine 1 | Tim 2 - VLM | Diskominfo Pemprov Jawa Timur**

---

## 1. Identitas Dataset

| Field                 | Keterangan                                                                            |
| --------------------- | ------------------------------------------------------------------------------------- |
| **Nama Dataset**      | Rutilahu-VLM-ID                                                                       |
| **Versi**             | v0.1 (Draft — In Development)                                                         |
| **Tanggal Pembuatan** | Maret 2026                                                                            |
| **Tim Pengembang**    | Farhan Vier Syarif Hilmi, Aulia Haq, Muhammad Lutfi Aziz                              |
| **Instansi**          | Diskominfo Pemerintah Provinsi Jawa Timur                                             |
| **Lisensi**           | ⚠️ RESTRICTED — Hanya untuk keperluan internal pemerintah, tidak boleh dipublikasikan |
| **Bahasa**            | Bahasa Indonesia                                                                      |
| **Modalitas**         | Multimodal (Citra + Teks)                                                             |
| **Task**              | Visual Question Answering (VQA) — Analisis Kondisi Hunian                             |

---

## 2. Deskripsi Dataset

Dataset ini dikembangkan untuk melatih model Vision Language Model (VLM) dalam menganalisis kondisi fisik **Rumah Tidak Layak Huni (Rutilahu)** di Indonesia. Dataset terdiri dari pasangan citra rumah beserta anotasi tekstual yang mendeskripsikan kondisi komponen struktural bangunan — atap, dinding, dan lantai — serta klasifikasi kelayakan hunian secara keseluruhan.

Dataset digunakan dalam proses **Supervised Fine-Tuning (SFT)** pada model `Qwen3-VL-8B-Instruct` sebagai bagian dari AI Engine 1 pada sistem pemetaan kemiskinan berbasis AI untuk mendukung program Rutilahu Pemprov Jawa Timur.

---

## 3. Sumber Data

### 3.1 Citra Rumah

Citra rumah diperoleh dari dua sumber utama:

- **Data existing dari Dinas Sosial** — foto kondisi rumah yang dikumpulkan melalui survei lapangan. Bersifat RESTRICTED, tidak boleh dipublikasikan.
- **Dataset publik** — foto rumah yang dikumpulkan melalui crawling dari Google Images dan Bing Images menggunakan kata kunci yang relevan dengan kondisi hunian di Indonesia.

### 3.2 Cakupan Wilayah

- **Negara:** Indonesia
- **Lingkup geografis:** Seluruh wilayah Indonesia
- **Catatan:** Data dari dinas sosial berfokus pada wilayah Jawa Timur

---

## 4. Statistik Dataset

| Atribut              | Nilai                                   |
| -------------------- | --------------------------------------- |
| **Total Sampel**     | 100 – 500 foto _(dalam pengembangan)_   |
| **Format Citra**     | JPG / PNG                               |
| **Format Dataset**   | JSONL (JSON Lines)                      |
| **Split Dataset**    | Train 70% \| Validation 15% \| Test 15% |
| **Annotator**        | Tim internal magang (Label Studio)      |
| **Storage Citra**    | AWS S3                                  |
| **Storage Metadata** | MongoDB Atlas                           |

---

## 5. Label dan Skema Anotasi

### 5.1 Label Kondisi Komponen

Setiap citra dianotasi dengan label kondisi untuk tiga komponen struktural utama:

| Komponen          | Label                                                  | Keterangan                  |
| ----------------- | ------------------------------------------------------ | --------------------------- |
| `roof_condition`  | `good` / `minor_damage` / `major_damage` / `collapsed` | Kondisi fisik atap rumah    |
| `wall_condition`  | `good` / `minor_damage` / `major_damage` / `collapsed` | Kondisi fisik dinding rumah |
| `floor_condition` | `good` / `wooden` / `dirt_floor` / `unknown`           | Jenis dan kondisi lantai    |

### 5.2 Severity Score & Kategori Kelayakan Hunian

| Severity Score | Kategori (User-Facing)    | Keterangan                                             |
| :------------: | ------------------------- | ------------------------------------------------------ |
|      `1`       | ✅ Layak Huni             | Kondisi baik, tidak ada kerusakan signifikan           |
|      `2`       | 🟡 Perlu Perbaikan Ringan | Kerusakan minor, dapat diperbaiki tanpa renovasi besar |
|      `3`       | 🟠 Perlu Perbaikan Berat  | Kerusakan signifikan pada satu atau lebih komponen     |
|      `4`       | 🔴 Tidak Layak Huni       | Kerusakan parah, membahayakan penghuni                 |

> **Catatan:** `severity_score` hanya digunakan secara internal sistem dan sebagai input ke AI Engine 2. Kategori (label teks) yang ditampilkan ke petugas lapangan.

### 5.3 Alat Anotasi

- **Platform:** Label Studio
- **Annotator:** Tim internal magang
- **Pipeline:** Raw annotation → ETL cleaning → Metadata tersimpan di MongoDB Atlas

---

## 6. Format Dataset

### 6.1 Struktur Metadata (MongoDB Atlas)

Setiap entri metadata mengacu pada satu citra rumah:

```json
{
  "image": "mkn_img_0018.jpg",
  "roof_condition": "collapsed",
  "wall_condition": "major_damage",
  "floor_condition": "unknown",
  "severity_score": 4,
  "severity_label": "tidak_layak"
}
```

### 6.2 Format Training Dataset (JSONL — ChatML Qwen3-VL)

Dataset training menggunakan format chat conversation sesuai arsitektur `Qwen3-VL-8B-Instruct`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "raw_img/mkn_img_0068.jpg" },
        { "type": "text", "text": "Analisis kondisi rumah ini." }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"Status Hunian\": \"Tidak Layak Huni\", \"explanation\": \"Atap bagian kanan mengalami kerusakan berat...\"}"
        }
      ]
    }
  ]
}
```

### 6.3 Format Output Model (Inference)

**Output user-facing** (ditampilkan ke petugas lapangan):

```json
{
  "Status Hunian": "Tidak Layak Huni",
  "explanation": "Atap bagian kanan mengalami kerusakan berat dengan beberapa genteng hilang. Dinding menunjukkan retakan kecil di sudut kiri. Lantai masih berupa tanah."
}
```

**Output internal sistem** (tidak ditampilkan ke petugas):

```json
{
  "status_hunian": "Tidak Layak Huni",
  "severity_score": 4,
  "confidence": 0.82,
  "perlu_verifikasi": true
}
```

**Output error** (foto tidak valid):

```json
{
  "error": "not_a_house",
  "kategori": null,
  "message": "Foto tidak menampilkan rumah atau bangunan, harap masukkan foto rumah!"
}
```

---

## 7. Arsitektur Penyimpanan Data

```
Foto Rumah
    │
    ▼
AWS S3 ──────────────────────────────────── Storage citra (.jpg/.png)
    │
    ▼
Label Studio ────── Annotator (Tim Magang) ─ Raw annotation
    │
    ▼ (Webhook)
FastAPI Middleware
    │
    ▼ (ETL Cleaning)
MongoDB Atlas ───────────────────────────── Metadata anotasi (post-ETL)
    │                                        Dataset JSONL (siap training)
    ▼
AI Ready Dataset (JSONL)
```

| Komponen         | Teknologi     | Isi                                   |
| ---------------- | ------------- | ------------------------------------- |
| Citra Rumah      | AWS S3        | File gambar (.jpg/.png)               |
| Metadata Anotasi | MongoDB Atlas | Hasil anotasi Label Studio (post-ETL) |
| Dataset Training | MongoDB Atlas | Format JSONL siap fine-tuning         |
| Raw Annotation   | Label Studio  | Anotasi mentah sebelum ETL cleaning   |

---

## 8. Quality Gate & Validasi Data

Setiap foto melewati proses quality gate **sebelum** diproses model. Urutan pengecekan dari yang paling ringan ke paling berat (fail fast):

```
Foto masuk via API
       │
       ▼
1. low_resolution  → Cek dimensi piksel (min. 640×480)
       │ lolos
       ▼
2. too_dark        → HSV brightness histogram
       │ lolos
       ▼
3. blur            → Laplacian Variance
       │ lolos
       ▼
4. not_a_house     → CLIP similarity score
       │ lolos
       ▼
  Preprocessing & VLM
```

| Jenis Validasi          | Metode                            | Library                  | Error Code           |
| ----------------------- | --------------------------------- | ------------------------ | -------------------- |
| Resolusi terlalu rendah | Cek dimensi piksel (min. 640×480) | Pillow                   | `low_resolution`     |
| Foto terlalu gelap      | HSV brightness histogram          | OpenCV                   | `too_dark`           |
| Foto blur               | Laplacian Variance                | OpenCV                   | `blur`               |
| Bukan foto rumah        | CLIP similarity score             | HuggingFace Transformers | `not_a_house`        |
| Batas retry terlampaui  | Max 3x percobaan                  | —                        | `max_retry_exceeded` |

---

## 9. Keterbatasan dan Potensi Bias

### 9.1 Keterbatasan Dataset

- Jumlah sampel saat ini masih terbatas (100–500 foto), berpotensi menyebabkan model overfit pada kondisi tertentu.
- Distribusi kelas severity belum diverifikasi — kemungkinan terdapat **class imbalance** (data severity 4 lebih sedikit dari severity 1).
- Foto crawling dari internet belum tentu merepresentasikan kondisi rumah di Indonesia secara akurat.
- Anotasi dilakukan oleh tim internal magang — **inter-annotator agreement** belum diukur secara formal.

### 9.2 Potensi Bias

- **Geographic bias:** Data dari dinas sosial lebih banyak dari wilayah Jawa Timur, sehingga model mungkin kurang akurat untuk wilayah lain di Indonesia.
- **Annotation bias:** Definisi operasional label (misal: batas antara `minor_damage` dan `major_damage`) perlu didefinisikan lebih ketat dalam annotation guideline.
- **Visual bias:** Foto crawling dari internet cenderung menampilkan kondisi ekstrem (sangat rusak atau sangat bagus), sehingga kondisi menengah (severity 2–3) mungkin kurang terwakili.

---

## 10. Penggunaan Dataset

### 10.1 Tujuan Penggunaan

- Fine-tuning model `Qwen3-VL-8B-Instruct` untuk klasifikasi kondisi Rutilahu.
- Evaluasi performa model (validation dan test split).
- Input ke AI Engine 2 untuk analisis dan klasifikasi kondisi kemiskinan.

### 10.2 Batasan Penggunaan

> ⚠️ **RESTRICTED**
>
> - Dataset **TIDAK boleh** dipublikasikan atau dibagikan ke pihak luar tanpa izin Diskominfo Pemprov Jawa Timur.
> - Data dari dinas sosial bersifat confidential — hanya untuk keperluan internal pemerintah.
> - Dataset tidak boleh digunakan untuk keperluan komersial.

---

## 11. Riwayat Versi

| Versi  | Tanggal    | Perubahan                                                                          |
| ------ | ---------- | ---------------------------------------------------------------------------------- |
| `v0.1` | Maret 2026 | Inisiasi dataset — pengumpulan data awal, setup Label Studio, pipeline ETL         |
| `v1.0` | TBD        | Dataset final untuk training — jumlah sampel mencukupi, annotation guideline final |

---

_Dokumen ini dibuat oleh Tim 2 - VLM | Diskominfo Pemprov Jawa Timur | Maret 2026_

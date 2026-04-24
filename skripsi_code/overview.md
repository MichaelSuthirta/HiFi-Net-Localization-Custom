# Skripsi: Deteksi Lokalisasi Mayoritas *Deepfake* (HiFi-IFDL)
Dokumentasi ini memberikan tinjauan tingkat tinggi (High-Level Overview) tentang arsitektur, alur data, dan fungsionalitas setiap komponen di dalam *pipeline* `skripsi_code`. *Project* ini merupakan inkarnasi termodularisasi dari struktur *paper* HiFi-IFDL khusus untuk kebutuhan lokalisasi level piksel (*pixel-level localization*) gambar yang termanipulasi.

---

## 1. Topologi Arsitektur (Code Flow)

Alur kerja model utamanya dibagi menjadi dua sayap: Pengekstrakan Fitur (oleh HRNet) dan Prediksi Lokalisasi (oleh NLCDetection). Data pelatihan diberi umpan balik koreksi melalui kombinasi *Binary Cross Entropy (BCE)* dan *Deep Metric Isolating Loss*.

```mermaid
flowchart TD
    A["dataset.py\n(Image & Target Mask)"] --> B{"train.py\n(Entry Point)"}
    
    subgraph Arsitektur Model (models/)
        B -->|Forward Image| C["seg_hrnet.py\n(FENet / Feature Extractor)"]
        C -->|Feature Maps| D["NLCDetection_loc.py\n(SegNet / Localization)"]
        D -->|Raw Binary Mask | O1(("Mask Binary\n(Heatmap 0~1)"))
        D -->|Deep Features| O2(("Feature Maps\n(Ruang Multidimensi)"))
    end

    subgraph Perhitungan Loss & Evaluasi
        O1 -->|BCE Loss| E["Kalkulasi BCE Loss"]
        O2 -->|Isolating Loss| F["custom_loss.py\n(Mengisolasi Jati Diri Pixels)"]
        
        K["center_loc/radius_center.pth\n(Titik Piksel Asli)"] -.->|Referenced by| F
        
        E & F -->|Total Network Loss| G["Backward Pass\n& Optimizer Step"]
    end
    
    subgraph Utilitas Uji (Evaluasi)
        G --> M["weights/\n(FENet_latest, SegNet_latest)"]
        M -.->|Loaded by| H["test.py\n(Simulasi Prediksi)"]
        H --> I["results/\n(Gambar Perbandingan\nGT vs Raw vs Threshold)"]
        
        G -.->|Saves CSV| L["plot_metrics.py"]
        L --> J["logs/\n(Grafik Curve Loss)"]
    end

```

---

## 2. Peran File Utama

Masing-masing berkas di dalam lingkungan `skripsi_code` memiliki kegunaan struktural yang terpisah dan sudah kami rancang agar bersih dari *script* yang tidak beroperasi (*dead-code*):

### `train.py` (Mesin Penggerak Training)
Merupakan *entry point* (titik awal) untuk melatih model dari _scratch_ maupun sekedar menyambung _training_ sebelumnya. Tugasnya:
- Mengidentifikasi sistem (Mac `mps`, NVIDIA `cuda`, atau `cpu`).
- Menggabungkan *Dataloader*, *Model* dan *Loss*.
- Melakukan iterasi per-*epoch* untuk menghitung `Total Loss = loss_bce + loss_metric + loss_cls_ringan`.
- Mengekspor rekaman nilai rata-rata tiap akhir *epoch* ke sebuah file `CSV`.
- Menyimpan hasil belajar berupa `.pth` ke dalam folder `/weights`.

### `dataset.py` (Kuli Data / *Data Loader*)
Mengotomatisasi penyandingan file **Fake Image** di dalam suatu folder dengan **Ground Truth Mask** miliknya. Di sinilah gambar secara dinamis dinormalisasi ukurannya menjadi *tensor* `256 x 256` dengan _mean/std_ setara syarat pra-syarat arsitektur umum ImageNet agar HRNet dapat bekerja.

### `utils/custom_loss.py` (Jantung Matematis Deteksi)
Menyimpan *class* algoritma gahar bernama **IsolatingLossFunction**. Ia menyedot `radius` dan `center` dari pra-kalkulasi (*center_loc/radius_center.pth*) untuk memaksa model merenggut fitur yang dimanipulasi dengan melemparnya jauh keluar radius geometris 2.5× lipat, sekaligus memeluk piksel asli dengan memadatkannya sebesar 0.15× lipat ke *center*.

### `models/seg_hrnet.py` (Ekstraktor Mata Tajam)
Arsitektur ***High-Resolution Net* (HRNet)**. Jika ResNet bersifat vertikal, HRNet membedah gambar dari sudut multi-resolusi secara paralel dan melebarkannya untuk mempertahankan tekstur mikro resolusi tinggi di gambar. Modul ini diinisialisasikan bersama *pre-trained weights* khusus HRNet dari ImageNet (`hrnet_w18_small_v2.pth`) demi mempersingkat durasi _learning_.

### `models/NLCDetection_loc.py` (Detektif Penentu Mask)
Jaringan konvolusi final yang menerima ekstraksi matang dari HRNet. Model ini yang berurusan dengan penentuan vonis *(classification branch)* dan memproduksi probabilitas `mask_binary` *(segmentation branch)* dalam wujud *Heatmap Soft Mask*.

### Script Pendamping Harian Khusus Evaluasi
- **`plot_metrics.py`**: Mengambil `.csv` dari *training loop* setiap selang iterasi, lalu merender grafis (matplot) fluktuasi _metrics_ Loss seiring perjalanan *epoch*. Menghasilkan `logs/metrics_plot.png`.
- **`test.py`**: Ujung tombak demonstrasi laporan skripsi. Mengambil model hasil pelatihan terakhir, disimulasikan ke sekumpulan sampel, lalu merender 4 sumbu layar (*Original*, *Ground Truth*, *Heatmap Model Segmen*, *Threshold Model*). Hasil per gambar ditumpahkan ke `/results/`.

---

## 3. Konfigurasi Modifikasi Terbaru (*Cross-Platform*)
Proyek ini mengadopsi 100% *codebase* paper *HiFi-IFDL*, namun menambal cacat aslinya, yaitu:
1. Menghilangkan ketergantungan absolut pada OS Linux/Nvidia CUDA (kami memodifikasi inisialisasi `.cuda()` yang *hardcoded* agar ramah Mac / *Apple Silicone* `mps`).
2. Isolasi *Python Dependencies* menggunakan lingkungan `uv` dengan resolusi *sub-second* super kilat, melepaskan dari keterkaitan _conda_ di repositori lama.

> [!NOTE]
> *Project environment* lengkap ini memampukanmu memulai _Deep Learning_ tingkat lanjut cukup dengan `uv run python train.py` dalam seketika!

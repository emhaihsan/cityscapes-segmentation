# CityScapes Segmentation
Project 3: Indonesia AI Computer Vision Bootcamp

Cityscapes adalah dataset yang digunakan untuk keperluan segmentasi semantik dan pemahaman lingkungan perkotaan. Dataset ini fokus pada citra jalan, trotoar, gedung, kendaraan, dan objek-objek lain yang ditemukan di lingkungan perkotaan. Cityscapes terdiri dari lebih dari 5.000 citra kota dari berbagai kota di Eropa, dengan anotasi yang mencakup label semantik piksel-per-piksel yang mengidentifikasi objek-objek dalam citra. Pada tugas ini, data yang digunakan hanya menggunakan 367 data latih dan 101 data validasi dan label segmentasi yang dibatasi ke dalam 11 label. Arsitektur yang digunakan adalah FCN (32,16,8) dan U-Net

1. **Arsitektur FCN**

![FCN](https://velog.velcdn.com/images%2Fcha-suyeon%2Fpost%2F48abdfa6-b98b-42ec-a262-4252e3f37a03%2Fimage.png)

2. **Arsitektur U-Net**
   
![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

3. **Konfigurasi Pelatihan**

* Data : 367 Training, 101 Validasi
* Augmentasi : flip horizontal
* Ukuran : 192 x 256 px 
* Loss : Categorical Cross Entropy Loss
* Batch Size : 4 
* Optimizer : Adam (lr=1e-3 + weight_decay max 1e-5)
* Epoch : +- 130 (Early Stopping)
 
5. **Tabel Perbandingan pada Data Validasi**
   
|Model|Loss|IoU|
|-|-|-|
|[FCN-32](https://github.com/mhihsan/cityscapes-segmentation/blob/main/trainfcn32notebook.ipynb)|0.7554|0.3013|
|[FCN-16](https://github.com/mhihsan/cityscapes-segmentation/blob/main/trainfcn16notebook.ipynb)|0.6441|0.3363|
|[FCN-8](https://github.com/mhihsan/cityscapes-segmentation/blob/main/trainfcn8notebook.ipynb)|0.6204|0.3658|
|[U-Net](https://github.com/mhihsan/cityscapes-segmentation/blob/main/unetnotebook.ipynb)|0.5429|0.4693|

5. **Ilustrasi dari hasil segmentasi**
![gambar distribusi](https://github.com/mhihsan/cityscapes-segmentation/blob/main/images/komparasi.png)


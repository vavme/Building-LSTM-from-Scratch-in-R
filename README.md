# Building-LSTM-from-Scratch-in-R
Building simple LSTM from scratch without using Keras or Pytorch or any Neural Network packages in R. <br/>
This model try to do binary text classification. 
All the reference I used is in Python so I try to create it in R. <br/>
This project still contains some bugs on the backpropation that causes the model cant learn properly. <br/>
This project was built in order to understand how LSTM works when I was completing my Thesis.
The progess is 90% completed. <br/>

Reference :<br/>
https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb <br/>
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial <br/>
https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch <br/>
https://github.com/keras-team/keras/blob/v2.6.0/keras/layers/recurrent_v2.py#L943-L1275 <br/>
<br/>


Basic Principle of LSTM
=======================
![image](https://user-images.githubusercontent.com/67742339/144520823-b96df5b9-1c26-42bf-aa83-b524bcba777e.png)
(In Indonesian Language)<br/>
<br/>
Akan dibentuk
permodelan bahasa akan menggunakan LSTM. Pembahasan akan dilanjutkan dengan melihat pada salah satu cell LSTM dari model tersebut. Cell LSTM akan digunakan adalah cell yang menerima input berupa kata “horse” dan mengeluakan output berupa hasilprediksi kata “ran”. Berikut merupakan ilustrasi model yang akan digunakan. <br/>
<br/>
![image](https://user-images.githubusercontent.com/67742339/144520977-1983f465-ce54-403b-b01d-1e08a9a2f959.png)<br/>
<br/>
Di dalam cell LSTM tersebut, matrix input (Xt) akan digabung dengan matriks aktivasi cell sebelumnya (ht-1). Kemudian matriks gabungan tersebut akan dilewatkan melalui keempat layer yang mengatur jalannya informasi sepert gambar di bawah ini. <br/>
![image](https://user-images.githubusercontent.com/67742339/144521014-316d861d-900b-42d3-89f2-fcb4fd60755d.png) <br/>
<br/>
Langkah pertama dalam cell LSTM adalah memutuskan informasi apa yang akan dibuang. Matriks gabungan yang sebelumnya telah dibentuk akan dilewatkan melalui forget gate. Pada forget gate, matriks gabungan tersebut akan dikalikan dengan bobot forget gate dan hasil perkalian tersebut akan dilewatkan melalui lapisan sigmoid. <br/>
![image](https://user-images.githubusercontent.com/67742339/144521095-01c1def7-3f79-42b8-b44a-5da0b38dbc51.png) <br/>
<br/>
Langkah selanjutnya adalah matriks gabungan matrix input (Xt) dan matriks aktivasi cell sebelumnya (ht-1) akan dilewatkan melalui input gate. Selain itu, matriks gabungan juga akan dilewatkan melalui gate gate yang merupakan gate yang menghasilkan kandidat baru untuk membantu input gate dalam menyaring informasi. Pada input gate, matriks gabungan akan dikalikan dengan bobot input gate dan dilewatkan melalui lapisan sigmoid. Pada gate gate, matriks gabungan juga akan dikalikan dengan bobot gate gate dan dilewatkan melalui lapisan tanh. Persamaan yang digunakan dalam input gate dan gate gate adalah sebagai berikut. <br/>
![image](https://user-images.githubusercontent.com/67742339/144521133-6cd0e6ed-8b0a-4bfc-9c72-614f3ac3fa0c.png) <br/>
<br/>
Langkah selanjutnya adalah membentuk matriks cell state saat ini (Ct). Pertama, matriks cell state sebelumnya (Ct-1) akan dikalikan dengan matriks hasil dari forget gate. Kemudian, hasil perkalian tersebut akan ditambahkan dengan matriks hasil perkalian antara hasil input gate dan hasil gate gate. Hasil yang diperoleh merupakan matriks cell state saat ini (Ct). Berikut merupakan persamaan yang digunakan dalam membentuk matriks cell state. <br/>
![image](https://user-images.githubusercontent.com/67742339/144521158-b6e2ba79-2ad9-430a-bcb4-2421178736ad.png) <br/>
<br/>
Selanjutnya, gabungan matrix input (Xt) dan matriks cell sebelumnya (ht-1) juga akan dilewatkan melalui output gate. Pada gerbang ini, matriks akan dikalikan dengan bobot matriks output gate dan dilewatkan melalui lapisan sigmoid. Selanjutnya, hasil output gate akan dikalikan dengan matriks cell state saat ini (Ct) yang telah dilewatkan melalui lapisan tanh. Perkalian kedua matriks tersebut akan menghasilkan matriks aktivasi cell saat ini (ht). Berikut merupakan persamaan yang digunakan dalam menghasilkan ouput cell dan matriks aktivasi cell. <br/>
<br/>
Langkah selanjutnya adalah menghasilkan output dari cell saat ini. Matriks aktivasi cell saat ini (ht) akan dikalikan dengan matriks bobot hidden output. Matriks hasil perkalian tersebut akan dilewatkan melalui lapisan softmax untuk mendapatkan hasil dengan probabilitas antara 0 hingga 1 dan jumlah semua probabilitas akan sama dengan satu. <br/>
![image](https://user-images.githubusercontent.com/67742339/144521212-90c90477-e3ce-4fd8-adc0-248d6483a648.png) <br/>

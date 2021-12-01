library(dplyr)
df = read.csv("data.csv")
df_pos = df %>%
  filter(klasifikasi == "positif")

df_neg = df %>%
  filter(klasifikasi == "negatif")

df_train= merge(x=df_pos, y=df_neg, all=TRUE)
df_train = df_train[sample(nrow(df_train)),]
row.names(df_train) = NULL

df_train

df_train = df_train[c("teks","klasifikasi")]
df_train$klasifikasi = factor(df_train$klasifikasi)
df_train$sentimen = ifelse(df_train$klasifikasi=="negatif", 0, 1)

library(tokenizers)
library(hash)
library(stringr)

df_train = df_train[c("teks","sentimen")]

#Mencari ukuran terpanjang dari kalimat yang ada pada data
max_length = 0
for (i in 1:nrow(df_train)) {
  word_len = lengths(gregexpr("\\W+", df_train[i,1]))+ 1
  max_length = max(max_length, word_len)
}
print(max_length)

#Melakukan padding dengan ### pada kalimat yang panjangnya kurang dari max_length
trans_data = df_train
for (i in 1:nrow(df_train)){
  word_len = lengths(gregexpr("\\W+", df_train[i,1]))+ 1
  len = max_length - word_len
  string = paste(replicate(len, "###"), collapse=" ")
  trans_data[i,1] = paste(df_train[i,1], string, collapse="")
}

head(trans_data)


#Membuat corpus atau kamus yang berisikan kata-kata yang ada pada data
corpus = tokenize_words(trans_data$teks)
corpus = unlist(corpus)
corpus = unique(corpus)

corpus = append(corpus,"###")

corpus_size = length(corpus)
corpus_size
word_id = hash()
id_word = hash()

#Memetakan index ke setiap kata yang ada pada corpus
for (i in 1:length(corpus)) {
  word_id[[corpus[i]]] = as.character(i)
  id_word[[as.character(i)]] = corpus[i]
}
corpus_size
x_raw = trans_data$teks
y_raw = trans_data$sentimen
y_raw

x_raw = strsplit(x_raw,"[[:space:]]")
x_raw

#Membagi data menjadi 5 batches
# batch = 5
batch = 2
x_train = c()
batch_size = length(x_raw)/batch
batch_size
for (i in 0:(batch-1)) {
  start = i*batch_size+1
  end = start+batch_size-1
  cat(start, end, "\n")
  #batch_data
  batch_data = x_raw[start:end]
  
  #convert setiap kata pada setiap kalimat pada setiap batch menjadi 
  #one hot encoding
  word_list = c()
  batch_dataset=c()
  for (k in (1:batch_size)){
    batch_dataset = matrix(0.0, length(x_raw[[1]]), corpus_size)
    print(batch_dataset)
    for (j in (1:length(x_raw[[1]]))) {
      sentence = batch_data[[k]]
      word = sentence[j]
      cat(k,j,word,"\n")
      word_index = word_id[[word]]
      batch_dataset[j,as.integer(word_index)] = 1.0
    }
    
    #Menyimpan kata ke-i hasil one-hot-encoding ke dalam setiap kalimat dalam
    #batch data
    word_list = append(word_list,list(batch_dataset))
  }
  
  #Menyimpan setiap kata dari setiap kalimat dalam batch dataset ke dalam
  #train dataset
  x_train = append(x_train, list(word_list))
}
x_train

length(corpus)
length(x_train)
dim(x_train[[1]][[1]])
#Mempersipakan y train
y_train = c()
for (i in 0:(batch-1)){
  start = i*batch_size+1
  end = start+batch_size-1
  cat(start,end,"\n")
  
  batch_label = y_raw[start:end]
  batch_label = matrix(batch_label)
  y_train = append(y_train, list(batch_label))
}

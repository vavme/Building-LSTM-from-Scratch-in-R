library(hash)
library(ramify)
library(matrixStats)

#Jumlah Neuron untuk input/embedding
input_units = 20

#Jumlah Neuron hidden layer
hidden_units = 4

#Jumlah neuron output 
output_units = 1

#learning rate
learning_rate = 0.005

#beta1 untuk parameter V pada Adam Optimizer
beta1 = 0.90

#beta2 untuk parameter S pada Adam Optimizer
beta2 = 0.99

#Fungsi Aktivasi
#Sigmoid
sigmoid <- function(x) {
  y = 1.0 / (1.0+exp(-x))
  return(y)
}

aktivasi_tanh <- function(x) {
  y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  return(y)
}

softmax <- function(x) {
  sum_exp_x = sum(exp(x))
  y = exp(x) / sum_exp_x
  return(y)
}

turunan_tanh <- function(x) {
  y = 1-(x**2)
  return(y)
}

#Melakukan inisiasi parameter
inisiasi_parameter <- function() {
  mean = 0
  std = 0.01
  
  #Cell LSTM
  #Inisiasi
  N = input_units+hidden_units
  M = hidden_units
  forget_gate_weights = matrix(rnorm((N*M),mean=mean,sd=std), N, M) 
  input_gate_weights = matrix(rnorm((N*M),mean=mean,sd=std), N, M)
  gate_gate_weights = matrix(rnorm((N*M),mean=mean,sd=std), N, M)
  output_gate_weights = matrix(rnorm((N*M),mean=mean,sd=std), N, M)
  
  N = hidden_units
  M = output_units
  hidden_output_weights = matrix(rnorm((N*M),mean=mean,sd=std), N, M) 
  
  parameters = hash()
  parameters[['fgw']] = forget_gate_weights 
  parameters[['igw']] = input_gate_weights
  parameters[['ogw']] = output_gate_weights
  parameters[['ggw']] = gate_gate_weights
  parameters[['how']] = hidden_output_weights
  
  return(parameters)
}

lstm_cell <- function(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters) {
  fgw = parameters[['fgw']]
  igw = parameters[['igw']]
  ogw = parameters[['ogw']]
  ggw = parameters[['ggw']]
  how = parameters[['how']]

  concat_dataset = cbind(batch_dataset,prev_activation_matrix)
  fa = concat_dataset %*% fgw
  fa = sigmoid(fa)
  
  #aktivasi input gate
  ia = concat_dataset %*% igw
  ia = sigmoid(ia)
  
  #aktivasi output gate
  oa = concat_dataset %*% ogw
  oa = sigmoid(oa)
  
  #aktivasi gate gate
  ga = concat_dataset %*% ggw
  ga = aktivasi_tanh(ga)
  
  #memory cell baru dalam bentuk matriks
  memory_cell_matriks = (fa * prev_cell_matrix) + (ia * ga)
  
  #aktivasi matriks pada cell ini
  activation_matriks = oa * aktivasi_tanh(memory_cell_matriks)
  
  #Menyimpan aktivasi untuk back propagation
  lstm_activations = hash()
  
  lstm_activations[['fa']] = fa
  lstm_activations[['ia']] = ia
  lstm_activations[['oa']] = oa
  lstm_activations[['ga']] = ga
  
  result = c(list(lstm_activations), list(memory_cell_matriks), list(activation_matriks))
  return(result)
}

#Output Cell
output_cell <- function(activation_matriks, parameters) {
  #hidden ke output
  how = parameters[['how']]
  
  #Output
  output_matriks = activation_matriks %*% how
  return(output_matriks)
}

get_embeddings <- function(batch_dataset, embeddings) {
  embedding_dataset = batch_dataset %*% embeddings
  return(embedding_dataset)
}

forward_propagation <- function(batches, parameters, embeddings) {
  batch_size = dim(batches[[1]])[1]
  
  #Menyimpan hasil fungsi aktivasi
  lstm_cache = hash()
  activation_cache = hash()
  cell_cache = hash()
  output_cache = hash()
  embedding_cache = hash()
  
  #inisasi matriks aktivasi a1 dan matriks cell (c1)
  a1 = matrix(0.0, batch_size, hidden_units)
  c1 = matrix(0.0, batch_size, hidden_units)
  
  activation_cache[['a1']] = a1
  cell_cache[['c1']] = c1
  
  #Membuka kalimat-kalimat pada data
  for (i in 1:(length(batches))) {
    batch_dataset = batches[[i]]
    
    #get embedding
    batch_dataset = get_embeddings(batch_dataset, embeddings)
    embedding_cache[[paste('emb',as.character(i), sep="")]] = batch_dataset
    
    #Cell LSTM
    
    result_lstm_cell = lstm_cell(batch_dataset, a1, c1, parameters)
    lstm_activations = result_lstm_cell[[1]]
    ct = result_lstm_cell[[2]]
    at = result_lstm_cell[[3]]
    
    #Cell Output
    ot = output_cell(at, parameters)
    
    #Menyimpan aktivasi waktu t pada cache
    lstm_cache[[paste('lstm',as.character(i+1), sep="")]] = lstm_activations
    activation_cache[[paste('a',as.character(i+1), sep="")]] = at
    cell_cache[[paste('c',as.character(i+1), sep="")]] = ct
    output_cache[[paste('o',as.character(i+1), sep="")]] = ot
    
    #Update a1 dan c1 ke at dan ct yang baru untuk cell lstm selanjutnya
    a1 = at
    c1 = ct
  }
  result = c(list(embedding_cache), list(lstm_cache), list(activation_cache), 
             list(cell_cache), list(output_cache))
  return(result)
}

cal_acc <- function(batch_labels, output_cache){
  acc = 0
  
  batch_size = length(batch_labels)
  
  #Loop untuk setiap langkah waktu
  for (i in 2:(length(output_cache)+1)){
    labels = batch_labels[i-1,]
    output = output_cache[[paste('o',as.character(i), sep="")]]
    pred = output[4]
    acc = acc + (labels==round(pred))
  }
  
  #print(acc)
  acc = acc/length(batch_size)
  acc = acc/length(output_cache)
  return(acc)
}

calculate_output_cell_error <- function(batch_labels, output_cache, parameters) {
  #Menyimpan error untuk setiap langkah waktu
  output_error_cache = hash()
  activation_error_cache = hash()
  how = parameters[['how']]
  
  for (i in 2:(length(output_cache)+1)) {
    labels = batch_labels[i-1,]
    output = output_cache[[paste('o',as.character(i), sep="")]]
    pred = output
    error_output = pred - labels
    error_activation = error_output %*% t(how)
    
    output_error_cache[[paste('eo',as.character(i), sep="")]] = error_output
    activation_error_cache[[paste('ea',as.character(i), sep="")]] = error_activation
  }
  y = c(list(output_error_cache), list(activation_error_cache))
  return(y)
}

#Menghitung error untuk satu cell LSTM
calculate_single_lstm_cell_error <- function(activation_output_error,
                                             next_activation_error,
                                             next_cell_error,parameters,
                                             lstm_activation,cell_activation,
                                             prev_cell_activation) {
  
  activation_error = activation_output_error + next_activation_error
  
  #Ouput Gate Error
  oa = lstm_activation[['oa']]
  eo = activation_error %*% aktivasi_tanh(cell_activation)
  eo = (eo %*% oa) %*% (1 - oa)
  
  #Aktivasi cell error
  cell_error = activation_error %*% oa
  cell_error = cell_error %*% turunan_tanh(aktivasi_tanh(cell_activation))
  cell_error = cell_error + next_cell_error
  
  #input gate error
  ia = lstm_activation[['ia']]
  ga = lstm_activation[['ga']]
  ei = cell_error %*% ga
  ei = (ei%*%ia) %*% (1-ia)
  
  #gate error
  eg = cell_error %*% ia
  eg = eg %*% turunan_tanh(ga)
  
  #forget gate error
  fa = lstm_activation[['fa']]
  ef = cell_error %*% prev_cell_activation
  #print(dim(ef))
  ef = (ef %*% fa) %*% (1-fa)
  #print(dim(ef))
  prev_cell_error = cell_error %*% fa
  
  #Parameters
  fgw = parameters[['fgw']]
  igw = parameters[['igw']]
  ggw = parameters[['ggw']]
  ogw = parameters[['ogw']]
  
  #Embedding + aktivasi hidden error
  embed_activation_error = ef %*% t(fgw)
  embed_activation_error = embed_activation_error + (ei %*% t(igw))
  embed_activation_error = embed_activation_error + (eo %*% t(ogw))
  embed_activation_error = embed_activation_error + (eg %*% t(ggw))
  
  input_hidden_units = dim(fgw)[1]
  hidden_units = dim(fgw)[2]
  input_units = input_hidden_units - hidden_units

  prev_activation_error = embed_activation_error[, -(1:(input_units))]
  embed_error = embed_activation_error[,1:input_units]
  
  #Menyimpan error lstm
  lstm_error = hash()
  lstm_error[['ef']] = ef
  lstm_error[['ei']] = ei
  lstm_error[['eo']] = eo
  lstm_error[['eg']] = eg
  
  y = c(list(prev_activation_error), list(prev_cell_error), list(embed_error), 
        list(lstm_error))
  return(y)
}

calculate_output_cell_derivatives <- function(output_error_cache,
                                              activation_cache,parameters) {
  #Total turunan dar setiap langkah waktu

  N=dim(parameters[['how']])[1]
  M=dim(parameters[['how']])[2]
  
  dhow = matrix(0.0, N, M)
  
  batch_size = dim(activation_cache[['a2']])[1]
  
  for (i in 2:(length(output_error_cache)+1)) {
    output_error = output_error_cache[[paste('eo',as.character(i), sep="")]]
    activation = activation_cache[[paste('a',as.character(i), sep="")]]

    dhow = dhow + (t(activation) %*% output_error)/batch_size
  }
  return(dhow)
}

calculate_single_lstm_cell_derivatives <- function(lstm_error,embedding_matrix,
                                                   activation_matrix) {
  #Error untuk setiap langkah waktu
  
  ef = lstm_error[['ef']]
  ei = lstm_error[['ei']]
  eo = lstm_error[['eo']]
  eg = lstm_error[['eg']]
  
  concat_matrix = cbind(embedding_matrix, activation_matrix)
  batch_size = dim(embedding_matrix)[1]
  #Turunan untuk setiap langkah waktu
  dfgw = (t(concat_matrix) %*% ef)/batch_size
  digw = (t(concat_matrix) %*% ei)/batch_size
  dogw = (t(concat_matrix) %*% eo)/batch_size
  dggw = (t(concat_matrix) %*% eg)/batch_size

  #Menyimpan turunan untuk setiap langkah waktu
  derivatives = hash()
  
  derivatives[['dfgw']] = dfgw
  derivatives[['digw']] = digw
  derivatives[['dogw']] = dogw
  derivatives[['dggw']] = dggw
  
  return(derivatives)
}

backward_propagation <- function(batch_labels,embedding_cache,lstm_cache,
                                 activation_cache,cell_cache,output_cache,
                                 parameters) {
  
  result = calculate_output_cell_error(batch_labels,output_cache,parameters)
  output_error_cache = result[[1]]
  activation_error_cache = result[[2]]

  lstm_error_cache = hash()
  embedding_error_cache = hash()

  N=dim(activation_error_cache[['ea2']])[1]
  M=dim(activation_error_cache[['ea2']])[2]
  
  eat = matrix(0,N,M)
  ect = matrix(0,N,M)
  
  for (i in ((length(lstm_cache)+1):2)) {
    result = calculate_single_lstm_cell_error(activation_error_cache[[paste('ea',as.character(i), sep="")]],
                                              eat,ect,parameters,
                                              lstm_cache[[paste('lstm',as.character(i), sep="")]],
                                              cell_cache[[paste('c',as.character(i), sep="")]],
                                              cell_cache[[paste('c',as.character(i-1), sep="")]])

    pae = result[[1]]
    pce = result[[2]]
    ee = result[[3]]
    le = result[[4]]
    
    #lstm error
    lstm_error_cache[[paste('elstm',as.character(i), sep="")]] = le
    #embedding error
    embedding_error_cache[[paste('eemb',as.character(i-1), sep="")]] = ee
    
    #Meng-update aktivasi error dan cell error selanjutnya
    eat = pae
    ect = pce
  }
  
  #Turunan dari output cell
  derivatives = hash()
  derivatives[['dhow']] = calculate_output_cell_derivatives(output_error_cache,
                                                            activation_cache,
                                                            parameters)
  
  #Menghitung turunan cell lstm untuk setiap langkah waktu
  lstm_derivatives = hash()

  for (i in 2:(length(lstm_error_cache)+1)){ 
    result = calculate_single_lstm_cell_derivatives(lstm_error_cache[[paste('elstm',as.character(i), sep="")]],
                                                    embedding_cache[[paste('emb',as.character(i-1), sep="")]],
                                                    activation_cache[[paste('a',as.character(i-1), sep="")]])
    lstm_derivatives[[paste('dlstm',as.character(i), sep="")]] = result 
  }
  #Inisiasi matriks untuk turunan
  dim1=dim(parameters[['fgw']])
  dim2=dim(parameters[['igw']])
  dim3=dim(parameters[['ogw']])
  dim4=dim(parameters[['ggw']])
  
  derivatives[['dfgw']] = matrix(0, dim1[1], dim1[2])
  derivatives[['digw']] = matrix(0, dim2[1], dim2[2])
  derivatives[['dogw']] = matrix(0, dim3[1], dim3[2])
  derivatives[['dggw']] = matrix(0, dim4[1], dim4[2])
  
  #Total turunan untuk setiap langkah waktu
  for (i in 2:(length(lstm_error_cache)+1)) {
    derivatives[['dfgw']] = derivatives[['dfgw']] + lstm_derivatives[[paste('dlstm',as.character(i),sep="")]][['dfgw']]
    derivatives[['digw']] = derivatives[['digw']] + lstm_derivatives[[paste('dlstm',as.character(i),sep="")]][['digw']]
    derivatives[['dogw']] = derivatives[['dogw']] + lstm_derivatives[[paste('dlstm',as.character(i),sep="")]][['dogw']]
    derivatives[['dggw']] = derivatives[['dggw']] + lstm_derivatives[[paste('dlstm',as.character(i),sep="")]][['dggw']]
  }
  y = c(list(derivatives), list(embedding_error_cache))
  return(y)
}

#Meng update parameter menggunakan adam optimizer
update_parameters <- function(parameters, derivatives, V, S, t){
  #Turunan
  dfgw = derivatives[['dfgw']]
  digw = derivatives[['digw']]
  dogw = derivatives[['dogw']]
  dggw = derivatives[['dggw']]
  dhow = derivatives[['dhow']]
  
  #print(dim(dfgw))
  #print(dim(digw))
  #print(dim(dogw))
  #print(dim(dggw))
  #print(dim(dhow))
  #Parameter gate
  fgw = parameters[['fgw']]
  igw = parameters[['igw']]
  ogw = parameters[['ogw']]
  ggw = parameters[['ggw']]
  how = parameters[['how']]
  
  #print(dim(fgw))  
  #Parameter V
  vfgw = V[['vfgw']]
  vigw = V[['vigw']]
  vogw = V[['vogw']]
  vggw = V[['vggw']]
  vhow = V[['vhow']]
  
  #print(dim(vfgw))  
  #Parameter S
  sfgw = S[['sfgw']]
  sigw = S[['sigw']]
  sogw = S[['sogw']]
  sggw = S[['sggw']]
  show = S[['show']]
  
  #print(dim(sfgw))  
  #Menghitung parameter V
  vfgw = (beta1*vfgw + (1-beta1)*dfgw)
  vigw = (beta1*vigw + (1-beta1)*digw)
  vogw = (beta1*vogw + (1-beta1)*dogw)
  vggw = (beta1*vggw + (1-beta1)*dggw)
  vhow = (beta1*vhow + (1-beta1)*dhow)
  
  #Menghitung parameter S
  sfgw = (beta2*sfgw + (1-beta2)*(dfgw**2))
  sigw = (beta2*sigw + (1-beta2)*(digw**2))
  sogw = (beta2*sogw + (1-beta2)*(dogw**2))
  sggw = (beta2*sggw + (1-beta2)*(dggw**2))
  show = (beta2*show + (1-beta2)*(dhow**2))
  
  #Meng update parameter dengan learning rate = 1e-6
  
  fgw = fgw - learning_rate*((vfgw)/(sqrt(sfgw) + 1e-6))
  igw = igw - learning_rate*((vigw)/(sqrt(sigw) + 1e-6))
  ogw = ogw - learning_rate*((vogw)/(sqrt(sogw) + 1e-6))
  ggw = ggw - learning_rate*((vggw)/(sqrt(sggw) + 1e-6))
  how = how - learning_rate*((vhow)/(sqrt(show) + 1e-6))
  #print("error34")
  #print("=====================================")
  #print(dim(fgw))
  
  #Menyimpan bobot baru
  parameters[['fgw']] = fgw
  parameters[['igw']] = igw
  parameters[['ogw']] = ogw
  parameters[['ggw']] = ggw
  parameters[['how']] = how
  
  #Parameter V
  V[['vfgw']] = vfgw 
  V[['vigw']] = vigw 
  V[['vogw']] = vogw 
  V[['vggw']] = vggw
  V[['vhow']] = vhow
  
  #Parameter S
  S[['sfgw']] = sfgw 
  S[['sigw']] = sigw 
  S[['sogw']] = sogw 
  S[['sggw']] = sggw
  S[['show']] = show
  
  y = c(list(parameters), list(V), list(S))
  return(y)
}

update_embeddings <- function(embeddings,embedding_error_cache,batch_labels) {
  #inisiasi
  dim=dim(embeddings)
  
  embedding_derivatives = matrix(0, dim[1], dim[2])
  
  batch_size = length(batch_labels[[1]][,1])
  #Total turunan embedding untuk setiap langkah waktu
  for (i in 1:length(embedding_error_cache)) {
    embedding_error_cache_current = embedding_error_cache[[paste('eemb',as.character(i),sep="")]]
    embedding_derivatives =  embedding_derivatives + ((t(batch_labels[[i]]) %*% 
                                                         embedding_error_cache_current)/batch_size)
  }
  embeddings = embeddings - learning_rate * embedding_derivatives
  return(embeddings)
}

#INisiasi parameter V
inisiasi_V <- function(parameters){
  dim1=dim(parameters[['fgw']])
  dim2=dim(parameters[['igw']])
  dim3=dim(parameters[['ogw']])
  dim4=dim(parameters[['ggw']])
  dim5=dim(parameters[['how']])
  
  Vfgw = matrix(0, dim1[1], dim1[2])
  Vigw = matrix(0, dim2[1], dim2[2])
  Vogw = matrix(0, dim3[1], dim3[2]) 
  Vggw = matrix(0, dim4[1], dim4[2])
  Vhow = matrix(0, dim5[1], dim5[2])
  
  V = hash()  
  V[['vfgw']] = Vfgw
  V[['vigw']] = Vigw
  V[['vogw']] = Vogw
  V[['vggw']] = Vggw
  V[['vhow']] = Vhow
  
  return(V)
}

#Inisiasi parameter S
inisiasi_S <- function(parameters){
  dim1=dim(parameters[['fgw']])
  dim2=dim(parameters[['igw']])
  dim3=dim(parameters[['ogw']])
  dim4=dim(parameters[['ggw']])
  dim5=dim(parameters[['how']])
  
  Sfgw = matrix(0, dim1[1], dim1[2])
  Sigw = matrix(0, dim2[1], dim2[2])
  Sogw = matrix(0, dim3[1], dim3[2]) 
  Sggw = matrix(0, dim4[1], dim4[2])
  Show = matrix(0, dim5[1], dim5[2])
  
  S = hash()
  S[['sfgw']] = Sfgw
  S[['sigw']] = Sigw
  S[['sogw']] = Sogw
  S[['sggw']] = Sggw
  S[['show']] = Show
  
  return(S)
}

#Fungsi Train
train <- function(train_dataset, labels, iters=1000){
  #Batch_size
  batch_size = dim(train_dataset[[1]][[1]])[1]
  #inisiasi parameter
  parameters = inisiasi_parameter()
  
  #inisiasi parameter V dan S
  V = inisiasi_V(parameters)
  S = inisiasi_S(parameters)
  
  #inisiasi random embbeding
  N=length(corpus)
  M=input_units
  embeddings = matrix(rnorm((N*M), mean=0, sd=0.01), N, M)
  
  #Inisiasi acc
  A = c()
  
  for (step in 1:(iters+1)){
    #cat("iter : ", step,"\n")
    index = (step %% length(train_dataset))+1########
    batches = train_dataset[[index]]
    label = labels[[index]]
    
    #forward propagation
    forward_prop = forward_propagation(batches,parameters,embeddings)
    embedding_cache = forward_prop[[1]]
    lstm_cache = forward_prop[[2]]
    activation_cache = forward_prop[[3]]
    cell_cache = forward_prop[[4]]
    output_cache = forward_prop[[5]]
    
    #acc
    acc = cal_acc(label,output_cache)
    
    #Backward propagation
    back_prop = backward_propagation(label,embedding_cache,lstm_cache,activation_cache,
                                     cell_cache,output_cache,parameters)
    derivatives = back_prop[[1]]
    embedding_error_cache = back_prop[[2]]
    
    #Update Parameter
    updated_params = update_parameters(parameters,derivatives, V, S,step)
    parameters = updated_params[[1]]
    V = updated_params[[2]]
    S = updated_params[[3]]
    
    #update embeddings
    embeddings = update_embeddings(embeddings, embedding_error_cache ,batches)
    
    A = append(A, acc)
    
    if(step%%100 == 0){
      cat("Single Bacth, Iteration : ", step)
      cat(', Accuracy   = ', round(acc*100,2),"%")
      cat('\n')
    }
  }
  
  y = c(list(embeddings), list(parameters), list(A))
  return(y)
}

#Melatih model
training_result = train(x_train, y_train,iters=1001)

#Referensi
#https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb
#https://medium.com/x8-the-ai-community/building-a-recurrent-neural-network-from-scratch-f7e94251cc80
#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial
#https://github.com/keras-team/keras/blob/v2.6.0/keras/layers/recurrent_v2.py#L943-L1275


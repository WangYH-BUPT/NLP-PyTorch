# Seq2seq

和 RNN、LSTM、CNN 不同，在 `Data.Dataset` 的构造上、计算 `loss` 值区别较大。关于每一步的矩阵维数变化，在 `1_layer_Seq2seq.pdf` 中体现。（稍后画完上传，先睡觉，命要紧）

##

**1.1 作用：** 实现 sequence to sequence

##

**1.2 网络结构：** Encoder 层和 Decoder 层。

***1.2.1 Encoder 层：*** 输入矩阵 `input_data` 通过 `nn.Embedding` 进行词嵌入：

	class Seq2seq(nn.Module):
    	def __init__(self):
        	...
        	self.encoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
   			...

    	def forward(self, encoder_input, encoder_hidden, decoder_input):
        	_, h_t = self.encoder(encoder_input, encoder_hidden)  # RNN 两输入两输出
        	...
 
***1.2.2 Decoder 层：*** 

	class Seq2seq(nn.Module):
	    def __init__(self):
	        ...
	        self.decoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
	        ...
	
	    def forward(self, encoder_input, encoder_hidden, decoder_input):
	        ...
	        output, _ = self.decoder(decoder_input, h_t)  # 不能反
	        ...

##

**1.3 代码实现：**

***1.3.1 导入包：***

	import torch
	import numpy as np
	import torch.nn as nn
	import torch.optim as optim
	import torch.utils.data as Data

***1.3.2 在 GPU 上运行：***

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.FloatTensor

***1.3.3 构建语料库：***

	# Corpus
	# S: Symbol that shows starting of decoding input (<SOS>)
	# E: Symbol that shows starting of encoding and decoding output (<EOS>)
	# ?: Symbol that will fill in blank sequence if current batch data size is short than n_step (<pad>)
	letter = [let for let in 'SE?abcdefghijklmnopqrstuvwxyz']
	letter2idx = {letter: idx for idx, letter in enumerate(letter)}
	seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
	# translation: man --> women;  black --> white;  king --> queen;  ...

***1.3.4 参数设置：***

	# Seq2seq parameters
	n_step = max([max(len(trans_before), len(trans_later)) for trans_before, trans_later in seq_data])  # max_len(= 5)
	n_hidden = 128
	n_class = len(letter2idx)  # classification problem
	batch_size = 3

***1.3.5 构建 dataset 和 dataloader：*** 由于三个变量无法直接用 Data.Dataset，因此继承 Data，构建三个变量的 dataset。需要\_\_init\_\_, \_\_len\_\_, \_\_getitem\_\_ 三个最基础的函数

	# dataset, dataloader
	def make_data(seq_data):
    	encoder_input_all, decoder_input_all, decoder_output_all = [], [], []

    	for seq in seq_data:
        	for i in range(2):
            	seq[i] = seq[i] + '?' * (n_step - len(seq[i]))  # 补齐不足max_len长度的单词，例如: "man" --> "man??"

        	encoder_input = [letter2idx[n] for n in (seq[0] + 'E')]  # ['m', 'a', 'n', '?', '?', 'E(<EOS>)']
        	decoder_input = [letter2idx[n] for n in ('S' + seq[1])]  # ['S(<SOS>)', 'w', 'o', 'm', 'e', 'n']
        	decoder_output = [letter2idx[n] for n in (seq[1] + 'E')]  # ['w', 'o', 'm', 'e', 'n', 'E(<EOS>)']
        	"""
        	encoder_input_all: [6(seq_data的句子个数), n_step+1 (because of 'E'), n_class]
        	decoder_input_all: [6, n_step+1 (because of 'S'), n_class]
        	decoder_output_all: [6, n_step+1 (because of 'E')]
        	"""
        	encoder_input_all.append(np.eye(n_class)[encoder_input])
        	decoder_input_all.append(np.eye(n_class)[decoder_input])
        	decoder_output_all.append(decoder_output)
    	return torch.Tensor(encoder_input_all), torch.Tensor(decoder_input_all), torch.LongTensor(decoder_output_all)

	encoder_input_all, decoder_input_all, decoder_output_all = make_data(seq_data)

	class Seq2seq_dataset(Data.Dataset):
    	def __init__(self, encoder_input_all, decoder_input_all, decoder_output_all):
        	self.encoder_input_all = encoder_input_all
        	self.decoder_input_all = decoder_input_all
        	self.decoder_output_all = decoder_output_all

    	def __len__(self):
        	return len(self.encoder_input_all)

    	def __getitem__(self, idx):
        	return self.encoder_input_all[idx], self.decoder_input_all[idx], self.decoder_output_all[idx]

	loader = Data.DataLoader(dataset=Seq2seq_dataset(encoder_input_all, decoder_input_all, decoder_output_all), batch_size=batch_size, shuffle=True)


***1.3.6 利用 RNN 构建一层 Seq2seq 网络结构：***

	# Seq2seq model (use RNN)
	class Seq2seq(nn.Module):
    	def __init__(self):
        	super(Seq2seq, self).__init__()
        	self.encoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
        	self.decoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
        	self.fc = nn.Linear(in_features=n_hidden, out_features=n_class)

    	def forward(self, encoder_input, encoder_hidden, decoder_input):
        	# encoder_input = encoder_input.transpose(0, 1)  # --> [n_step+1, batch_size, n_class]
        	# decoder_input = decoder_input.transpose(0, 1)  # --> [n_step+1, batch_size, n_class]
        	_, h_t = self.encoder(encoder_input, encoder_hidden)  # RNN 两输入两输出
        	output, _ = self.decoder(decoder_input, h_t)  # 不能反
        	model = self.fc(output)
        return model

	model = Seq2seq()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

***1.3.7 迭代训练：***

	# Training
	for epoch in range(5000):
    	for encoder_input_batch, decoder_input_batch, decoder_output_batch in loader:
        	h_0 = torch.zeros(1, batch_size, n_hidden).to(device)  # [num_layers*num_directions, batch_size, n_hidden]
        	encoder_input_batch = encoder_input_batch.to(device)  # [batch_size, n_step+1, n_class]
        	decoder_input_batch = decoder_input_batch.to(device)  # [batch_size, n_step+1, n_class]
        	decoder_output_batch = decoder_output_batch.to(device)  # [batch_size, n_step+1], 不是one-hot编码
        	predict_data = model(encoder_input_batch, h_0, decoder_input_batch)  # [n_step+1, batch_size, n_class]
        	# predict_data = predict_data.transpose(0, 1)  # [batch_size, n_step+1, n_class]
        	loss = 0
        	for i in range(len(decoder_output_batch)):
            	loss += loss_fn(predict_data[i], decoder_output_batch[i])
        	if (epoch + 1) % 1000 == 0:
            	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        	optimizer.zero_grad()
        	loss.backward()
        	optimizer.step()

***1.3.8 预测结果：***

	# Test
	def translate(word):
    	encoder_input, decoder_input, _ = make_data([[word, '?' * n_step]])
    	encoder_input, decoder_input = encoder_input.to(device), decoder_input.to(device)
    	# make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    	hidden = torch.zeros(1, 1, n_hidden).to(device)
    	output = model(encoder_input, hidden, decoder_input)
    	output = output.transpose(0, 1)  # [n_step+1, batch_size, n_class]
    	predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
    	decoded = [letter[i] for i in predict]
    	translated = ''.join(decoded[: decoded.index('E')])
    	return translated.replace('?', '')

	print('test')
	print('man -->', translate('man'))
	print('mans -->', translate('mans'))
	print('king -->', translate('king'))
	print('black -->', translate('black'))
	print('up -->', translate('up'))

---

# Tips

**1. RNN 中 `batch_first=True\False` 的代码改动:**

**1.1** `batch_first=True`

	# Seq2seq model (use RNN)
	class Seq2seq(nn.Module):
    	def __init__(self):
        	super(Seq2seq, self).__init__()
        	self.encoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
        	self.decoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
			...

    	def forward(self, encoder_input, encoder_hidden, decoder_input):
        	_, h_t = self.encoder(encoder_input, encoder_hidden)  # RNN 两输入两输出
			...
			return model


	# Training
	for epoch in range(5000):
		for encoder_input_batch, decoder_input_batch, decoder_output_batch in loader:
			...
			predict_data = model(encoder_input_batch, h_0, decoder_input_batch)
			loss = 0
			...


	# Test
	def translate(word):
		...
		output = model(encoder_input, hidden, decoder_input)
    	output = output.transpose(0, 1)  # [n_step+1, batch_size, n_class]
		...

**1.2** `batch_first=False`	

	# Seq2seq model (use RNN)
	class Seq2seq(nn.Module):
    	def __init__(self):
        	super(Seq2seq, self).__init__()
        	self.encoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
        	self.decoder = nn.RNN(batch_first=True, input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
			...

    	def forward(self, encoder_input, encoder_hidden, decoder_input):
			encoder_input = encoder_input.transpose(0, 1)  # --> [n_step+1, batch_size, n_class]
        	decoder_input = decoder_input.transpose(0, 1)  # --> [n_step+1, batch_size, n_class]
        	_, h_t = self.encoder(encoder_input, encoder_hidden)  # RNN 两输入两输出
			...
			return model


	# Training
	for epoch in range(5000):
		for encoder_input_batch, decoder_input_batch, decoder_output_batch in loader:
			...
			predict_data = model(encoder_input_batch, h_0, decoder_input_batch)
			predict_data = predict_data.transpose(0, 1)
			loss = 0
			...


	# Test
	def translate(word):
		...
		output = model(encoder_input, hidden, decoder_input)
		...

##

**2. 三输入的 Dataset 构建：**

	encoder_input_all, decoder_input_all, decoder_output_all = make_data(seq_data)
	
	class Seq2seq_dataset(Data.Dataset):
	    # 由于三个变量无法直接用Data.Dataset，因此继承Data，构建三个变量的dataset
	    # 需要__init__, __len__, __getitem__三个最基础的函数
	    def __init__(self, encoder_input_all, decoder_input_all, decoder_output_all):
	        self.encoder_input_all = encoder_input_all
	        self.decoder_input_all = decoder_input_all
	        self.decoder_output_all = decoder_output_all
	
	    def __len__(self):
	        return len(self.encoder_input_all)
	
	    def __getitem__(self, idx):
	        return self.encoder_input_all[idx], self.decoder_input_all[idx], self.decoder_output_all[idx]
	
	loader = Data.DataLoader(dataset=Seq2seq_dataset(encoder_input_all, decoder_input_all, decoder_output_all), batch_size=batch_size, shuffle=True)

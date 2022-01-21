# Text_BiLSTM

参考Text_LSTM

##

**1.1 作用：** 利用上文预测下文词。

##

**1.2 网络结构：** 三层神经网络，输入层、隐藏层、输出层。

***1.2.1 输入层：*** 输入矩阵 `input_data` 是one-hot编码，因此每一行的维度是词典去重后的长度 `[n_step, n_class]`。

	input_data: [batch_size, n_step, n_class]
和`torch.rnn`相同，应 `torch.LSTM` 的 `input_data: [seq_len, batch_size, word2vec]` 要求，进行变形：

	input_lstm = input_data.transpose(0, 1)  # [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
 
***1.2.2 隐藏层：*** 

	hidden_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
    cell_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
	outputs, (_, _) = self.lstm(input_data, (hidden_state, cell_state))  # [max_len, batch_size, n_hidden * 2]

***1.2.3 输出层：*** 

	outputs = outputs[-1]  # [batch_size, n_hidden * 2]
    model = self.fc(outputs)  # [batch_size, n_class]

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
	sentence = (
    	'GitHub Actions makes it easy to automate all your software workflows '
    	'from continuous integration and delivery to issue triage and more'
	)
	words = sentence.split()
	word2idx = {word: idx for idx, word in enumerate(set(words))}
	idx2word = {idx: word for idx, word in enumerate(set(words))}

***1.3.4 参数设置：***

	# Bi-LSTM Parameters
	n_class = len(word2idx)  # classification problem, 不同词的个数
	max_len = len(sentence.split())  # 21
	batch_size = 16
	n_hidden = 5

***1.3.5 构建 dataset 和 dataloader：***

	# dataset, dataloader
	def make_data(sentence):
    input_batch = []
    target_batch = []
    for i in range(max_len - 1):
        input = [word2idx[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        # i = 0: input = [idx(Github), 0, 0, ..., 0] (21-1个0)
        # i = 1: input = [idx(Github), idx(Actions), 0, 0, ..., 0] (21-2个0)
        # 多个0只是用来占位，也可以用其他的表示，例如: 'UNK'_idx = max_len + 1。这里可以理解成transformer中的mask
        target = word2idx[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])  # 每一行是Github、Actions、...对应的one-hot编码
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)

	input_batch, target_batch = make_data(sentence)
	# input_batch: [max_len - 1, max_len, n_class]; target_batch = [1, max_len - 1]
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

***1.3.6 构建一层Bi-LSTM网络结构：***

	# BiLSTM
	class BiLSTM(nn.Module):
    	def __init__(self):
        	super(BiLSTM, self).__init__()
        	self.lstm = nn.LSTM(batch_first=False, input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        	self.fc = nn.Linear(n_hidden * 2, n_class)
    
    	def forward(self, input_data):
        	batch_size = input_data.shape[0]
        	input_data = input_data.transpose(0, 1)  # [batch_size, max_len, n_class] --> [max_len, batch_size, n_class]
        	hidden_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
        	cell_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]

        	outputs, (_, _) = self.lstm(input_data, (hidden_state, cell_state))  # [max_len, batch_size, n_hidden * 2]
        	outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        	model = self.fc(outputs)  # [batch_size, n_class]
        return model

	model = BiLSTM()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

***1.3.7 迭代训练：***

	# Training
	for epoch in range(5000):
    	for input_data, target_data in loader:
        	predict_data = model(input_data)
        	loss = loss_fn(predict_data, target_data)
        	if (epoch + 1) % 1000 == 0:
            	print('Epoch:', '%04d' % (epoch + 1), 'Loss =', '{:.6f}'.format(loss))

        	optimizer.zero_grad()
        	loss.backward()
        	optimizer.step()

***1.3.8 预测结果：***

	# Predict
	predict = model(input_batch).data.max(1, keepdim=True)[1]
	print(sentence)
	print([idx2word[n.item()] for n in predict.squeeze()])

---

# Tips

**1.batch_first=True/False**

代码第69行中： `input_data = input_data.transpose(0, 1)` 是为了符合 `nn.LSTM` 要求的维数设置，但如果在定义 `nn.LSTM` 的时候加一个参数 `batch_first=True`，就不用改变 `batch_size` 和 `max_len` 的位置了。并且输出也是 `batch_size` 在第一位。所以需要注意维数的变化。因此，以下两种在效果上等价：

***1.1 batch_first=True***

	class BiLSTM(nn.Module):
    	def __init__(self):
        	super(BiLSTM, self).__init__()
        	self.lstm = nn.LSTM(batch_first=False, input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        	self.fc = nn.Linear(n_hidden * 2, n_class)
    
    	def forward(self, input_data):
        	batch_size = input_data.shape[0]
        	input_data = input_data.transpose(0, 1)  # [batch_size, max_len, n_class] --> [max_len, batch_size, n_class]
        	hidden_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
        	cell_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]

        	outputs, (_, _) = self.lstm(input_data, (hidden_state, cell_state))  # [max_len, batch_size, n_hidden * 2]
        	outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        	model = self.fc(outputs)  # [batch_size, n_class]
        	return model

***1.2 batch_first=False***

	class BiLSTM(nn.Module):
    	def __init__(self):
        	super(BiLSTM, self).__init__()
        	self.lstm = nn.LSTM(batch_first=True, input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        	self.fc = nn.Linear(n_hidden * 2, n_class)
    
    	def forward(self, input_data):
        	batch_size = input_data.shape[0]
        	hidden_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
        	cell_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]

        	outputs, (_, _) = self.lstm(input_data, (hidden_state, cell_state))  # [batch_size, max_len, n_hidden * 2]
        	outputs = outputs[:, -1]  # [batch_size, n_hidden * 2]
        	model = self.fc(outputs)  # [batch_size, n_class]
        	return model


经过对比，`batch_first = True` 比 `batch_first = False` **处理速度要快**，同时**网上很多地方**都在说 `batch_first = False` **性能更好**。 ***1.1*** 和 ***1.2***：

	model = BiLSTM()
	Epoch = 5000
    True:  >>> times_cost: 127.26299452781677; Loss = 0.656417, 0.672440
	False: >>> times_cost: 132.71624612808228; Loss = 0.704323, 0.586287

用下面代码作为一个小demo，比较直观的能看出来处理速度的对比：

	import torch
	import time

	x_1 = torch.randn(100, 200, 512)
	x_2 = x_1.transpose(0, 1)

	model_1 = torch.nn.LSTM(batch_first=True, hidden_size=1024, input_size=512)
	model_2 = torch.nn.LSTM(batch_first=False, hidden_size=1024, input_size=512)
	start_time_1 = time.time()

	result_1 = model_1(x_1)
	end_time_1 = time.time()

	result_2 = model_2(x_2)
	end_time_2 = time.time()

	print(end_time_1 - start_time_1, end_time_2 - end_time_1)

	>>> 2.47576642036438 2.530911684036255

##

**2. optim.Adam():**

源代码：

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):

也就是说 `lr` 缺省值是：1e-3，因此调用时，`lr=1e-3` 可以不用写：

	optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
	# 等价于
	optimizer = optim.Adam(params=model.parameters()）
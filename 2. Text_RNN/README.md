# TextRNN

一个背景：有 n 句话，每句话都由 3 个单词组成。要做的的是，将每句话的前两个单词作为输入，最后一个词作为输出，训练一个 RNN 模型。类似于一个**多分类问题**。

##

**1.1 作用：** 利用上文预测下文词。

##

**1.2 网络结构：** 三层神经网络，输入层、隐藏层、输出层。

***1.2.1 输入层：*** 输入矩阵 `X` 是one-hot编码，因此每一行的维度是词典去重后的长度 `[n_step, n_class]`。

	X: [batch_size, n_step, n_class]
应 `torch.rnn` 的 `X: [seq_len, batch_size, word2vec]` 要求，进行变形：

	X = X.transpose(0, 1)  # X: [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
 
***1.2.2 隐藏层：*** 

	out, hidden = self.rnn(X, hidden)
	# hidden: [num_of_layers(1层RNN) * num_directions(=1), batch_size, hidden_size]
    # out: [seq_len, batch_size, hidden_size]

***1.2.3 输出层：*** 

	out = out[-1]  # 最后一维是预测词
    model = self.fc(out)

##

**1.3 代码实现：**

***1.3.1 导入包：***

	import torch
	import torch.nn as nn
	import numpy as np
	import torch.optim as optimizer
	import torch.utils.data as Data

***1.3.2 在 GPU 上运行：***

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.FloatTensor

***1.3.3 构建语料库：***

	# Corpus
	sentences = ["i like dog", "i love coffee", "i hate milk"]
	sentences_list = " ".join(sentences).split()
	vocab = set(sentences_list)
	word2idx = {word: idx for idx, word in enumerate(vocab)}
	idx2word = {idx: word for idx, word in enumerate(vocab)}
	n_class = len(vocab)

***1.3.4 参数设置：***

	# Parameters
	batch_size = 2
	n_step = 2  # number of input words (= number of cells)
	n_hidden = 5  # number of hidden units in one cell (word_embedding_dim --> hidden_dim)

***1.3.5 将 input\_data 和 output\_data 赋值，并构建 dataset 和 loader：***

	# Construct datasets and dataloader
	def make_data(sentences):
    	input_batch = []
    	target_batch = []

    	for sen in sentences:
        	word = sen.split()  # Take each sentence as a unit, extract words
        	input = [word2idx[n] for n in word[: -1]]  # Traverse the first word to the penultimate word of each sentence
        	target = word2idx[word[-1]]  # the last word of each sentence

        	input_batch.append(np.eye(n_class)[input])
        	target_batch.append(target)
        	"""
        	print(np.eye(5)[[0, 2]])  # 单位矩阵的第0行、第2行取出来
        	[[1. 0. 0. 0. 0.]
         	[0. 0. 1. 0. 0.]]
        	"""
    	return input_batch, target_batch

	input_batch, target_batch = make_data(sentences)  # input_batch: [num(sentences), n_step, n_class] (here: [3, 2, 7])
	input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)  # !! target_batch: LongTensor !!
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

***1.3.6 构建网络结构：***

	# Construct RNN
	class TextRNN(nn.Module):
    	def __init__(self):
        	super(TextRNN, self).__init__()
        	self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)  # 7-dim(n_class) --> 5-dim(n_hidden)
        	# !input_size and hidden_size are essential parameters in nn.RNN
        	# !input_size is n-dim vector encoding of each word. (Here is one-hot, so 7-dim vector is represented each word)
        	# !hidden_size is the hidden dimension through RNN
        	self.fc = nn.Linear(n_hidden, n_class)

    	def forward(self, hidden, X):
        	# torch要求的X是: [seq_len, batch_size, word2vec]
        	# 我们的X是: [batch_size, n_step, n_class]
        	X = X.transpose(0, 1)  # X: [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
        	out, hidden = self.rnn(X, hidden)  # out, _ = self.rnn(X, hidden) 
        	# hidden: [num_of_layers(1层RNN) * num_directions(=1), batch_size, hidden_size]
        	# out: [seq_len, batch_size, hidden_size]
        	out = out[-1]
        	model = self.fc(out)
        	return model

	model = TextRNN().to(device)
	loss_fn = nn.CrossEntropyLoss().to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

***1.3.7 迭代训练：***

	# Training
	for epoch in range(5000):
    	for input_data, output_data in loader:  # input_data: [batch_size, n_step, n_class]; output_data: [batch_size]
        	input_data = input_data.to(device)
        	output_data = output_data.to(device)
        	hidden = torch.zeros(1, input_data.shape[0], n_hidden).to(device)
        	predict = model(hidden, input_data)  # predict: [batch_size, n_class]
        	loss = loss_fn(predict, output_data)

        	if (epoch + 1) % 1000 == 0:
            	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        	optimizer.zero_grad()
        	loss.backward()
        	optimizer.step()

***1.3.8 预测结果：***

	# Predict
	input = [sen.split()[: 2] for sen in sentences]
	hidden = torch.zeros(1, len(input), n_hidden)
	predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
	print([sen.split()[: 2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])

---

# 注意
1.

	input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)  # !! target_batch: LongTensor !!
中的 `target_batch` 是 **LongTensor**，如果是 Tensor，会报以下错误：

	Traceback (most recent call last):
  	File "C:/Users/demo_Text_RNN.py", line 93, in <module>
    	loss = loss_fn(predict, output_data)
  	File "D:\PyCharm\lib\site-packages\torch\nn\modules\module.py", line 727, in _call_impl
    	result = self.forward(*input, **kwargs)
  	File "D:\PyCharm\lib\site-packages\torch\nn\modules\loss.py", line 961, in forward
    	return F.cross_entropy(input, target, weight=self.weight,
  	File "D:\PyCharm\lib\site-packages\torch\nn\functional.py", line 2468, in cross_entropy
    	return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  	File "D:\PyCharm\lib\site-packages\torch\nn\functional.py", line 2264, in nll_loss
    	ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
	RuntimeError: expected scalar type Long but found Float

2.每层的矩阵大小需要注意，我根据自己的理解花了一个两层RNN的网络结构图，如 `2_layers_TextRNN.pdf` 所示，有错误请指正。

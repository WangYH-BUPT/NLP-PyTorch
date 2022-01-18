# Word2Vec

Word2Vec是2013年由Tomas Mikolov提出的，其核心思想是用一个词的上下文来刻画这个词。Word2Vec可以分为两种不同的模型：**Skip-gram** 和 **CBOW** 模型。

---

### 1. Skip-gram

**1.1 作用：** 利用中心词预测上下文词，这里的上下文词是以中心词为中心，某个窗口内的词。窗口大小：`window_size`。

##

**1.2 网络结构：** 三层神经网络，输入层、中间层、输出层，如图3-24所示。

***1.2.1 输入层：*** 输入矩阵 `training_input` 的每一行是中心词的one-hot编码，因此每一行的维度是词典去重后的长度 `vocab_size`。

	size(training_input) = [batch_size, vocab_size]
 
***1.2.2 中间层：*** 权重矩阵 `W` 和输入矩阵 `training_input` 相乘，得到每一个单词的词嵌入向量（为了前后统一，这里叫 `hidden` ）。

	hidden = training_input * W
	size(W) = [vocab_size, word_embedding_dim]
	size(hidden) = [batch_size, word_embedding_dim]

***1.2.3 输出层：*** 词嵌入向量 `hidden` 和权重矩阵 `V` 相乘，得到输出上下文词的类别 `output`，也就是在 vocab 中的索引，因此，每一行的维度是词典的维度，预测的是哪个词，就产生一个对应 `vocab_size` 大小的 one-hot 行矩阵。

	output = hidden * V
	size(V) = [word_embedding_dim, vocab_size]
	size(output) = [batch_size, vocab_size]

***1.2.4 损失函数：*** 在 Skip-gram 中，我们的目的是由中心词 $w_t$ 去预测窗口内的上下文词 $w_{t-1}$ 和 $w_{t+1}$，此时，可以建模为：

$$ P(w_{t-1}, w_{t+1}|w_t) = P(w_{t-1}|w_t)P(w_{t+1}|w_t)$$

利用交叉熵误差函数。我们希望损失函数值最小：

$$ L=-{\rm log}P(w_{t-1}, w_{t+1}|w_t)=-{\rm log}P(w_{t-1}|w_t)P(w_{t+1}|w_t)=-({\rm log}P(w_{t-1}|w_t)+{\rm log}P(w_{t+1}|w_t)) $$

扩展到整个语料库，则损失函数可以表示为：

$$L_{skipgram}=-\frac{1}{T}\Sigma^T_{t=1}({\rm log}P(w_{t-1}|w_t)+{\rm log}P(w_{t+1}|w_t))$$

	import torch.nn as nn
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	loss_fn = nn.CrossEntropyLoss().to(device)

##

**1.3 代码实现：**

***1.3.1 导入包：***

	import torch
	import torch.nn as nn
	import numpy as np
	import torch.optim as optimizer
	import torch.utils.data as Data
	import matplotlib.pyplot as plt

***1.3.2 在 GPU 上运行：***

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.FloatTensor

***1.3.3 构建语料库：***

	sentences = ["jack like dog", "jack like cat", "jack like animal",
             	 "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             	 "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             	 "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
	sentences_list = " ".join(sentences).split()  # 分割句子为单词
	vocab = set(sentences_list)  # 构建语料库set
	word2idx = {word: idx for idx, word in enumerate(vocab)}  # word到idx的映射：{'jack': 0, 'like': 1, ...}
	idx2word = {idx: word for idx, word in enumerate(vocab)}  # idx到word的映射： {0: 'jack', 1: 'like', ...}
	vocab_size = len(vocab)

***1.3.4 参数设置：***

	# model parameters
	window_size = 2  # 窗口大小
	batch_size = 8
	word_embedding_dim = 2  # 词嵌入向量的维数

***1.3.5 构建中心词和上下文词的对应关系：***

	skip_grams = []
	for center_idx in range(window_size, len(sentences_list)-window_size):  # idx = 2, 3, ..., len-2
    	center_word2idx = word2idx[sentences_list[center_idx]]  # center_word2idx is the unique index of center in word2idx
    	context_idx = list(range(center_idx-window_size, center_idx)) + list(range(center_idx+1, center_idx+window_size+1))
    	context_word2idx = [word2idx[sentences_list[i]] for i in context_idx]

    	for w in context_word2idx:
        	skip_grams.append([center_word2idx, w])
			# len(skip_gram): 168 = (len(sentences_list) - window_size*2) * window_size*2 = (46 - 2*2) * 2*2

***1.3.6 将 input\_data 和 output\_data 赋值，并构建 dataset 和 loader：***

	def make_data(skip_grams):
    	input_data = []  # input is one-hot code
    	output_data = []  # output is a class

    	for center_one_hot, context_class in skip_grams:
        	input_data.append(np.eye(vocab_size)[center_one_hot])
        	output_data.append(context_class)
    	return input_data, output_data

	input_data, output_data = make_data(skip_grams)  # instantiate
	input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
	dataset = Data.TensorDataset(input_data, output_data)
	loader = Data.DataLoader(dataset, batch_size, True)

***1.3.6 构建网络结构：***

	class Word2Vec(nn.Module):
    	def __init__(self):
        	super(Word2Vec, self).__init__()
        	self.W = nn.Parameter(torch.randn(vocab_size, word_embedding_dim).type(dtype))
        	self.V = nn.Parameter(torch.randn(word_embedding_dim, vocab_size).type(dtype))

    	def forward(self, training_input):  # training_input: [batch_size, vocab_size], each line is one-hot code
        	hidden = torch.mm(training_input, self.W)  # [batch_size, word_embedding_dim]
        	output = torch.mm(hidden, self.V)  # [batch_size, vocab_size], class_num = vocab_size
        	return output

	model = Word2Vec().to(device)
	loss_fn = nn.CrossEntropyLoss().to(device)
	optim = optimizer.Adam(model.parameters(), lr=1e-3)

***1.3.7 迭代训练：***

	for epoch in range(2000):
    	for i, (batch_x, batch_y) in enumerate(loader):
        	batch_x = batch_x.to(device)
        	batch_y = batch_y.to(device)
        	predict = model(batch_x)
        	loss = loss_fn(predict, batch_y)

        	if (epoch + 1) % 1000 == 0:
            	print("epoch =", epoch + 1, i, "loss =", loss.item())

        	optim.zero_grad()
        	loss.backward()
        	optim.step()

***1.3.8 可视化（画图）：***

	for i, label in enumerate(vocab):
    	W, WT = model.parameters()
    	x, y = float(W[i][0]), float(W[i][1])
    	plt.scatter(x, y)
    	plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
	plt.show()

---

### 2. CBOW

**2.1 作用：** 利用上下文词预测中心词。损失值为：

$$ L_{CBOW} = -\frac{1}{T}\Sigma^T_{t=1}({\rm log}P(w_t|w_{t-1}, w_{t+1})$$

通过对比$L_{skipgram}$和$L_{CBOW}$，我们应该使用 skip-gram 模型这是因为，从单词的分布式表示的准确度来看，在大多数情况下，skip-gram 模型的结果更好。特别是随着语料库规模的增大，在低频词和类推问题的性能方面，skip-gram模型往往会有更好的表现。此外，就学习速度而言，CBOW 模型比 skip-gram 模型要快。这是因为 skip-gram 模型需要根据上下文数量计算相应个数的损失，计算成本变大。

CBOW 实现不再阐述，网络结构和反向传播模型如图3-9和图3-20所示。

---

# 扩展

### 改进：（根据《深度学习进阶自然语言处理——斋藤康毅》P152-P193总结）

不管是 skip-gram 还是 CBOW 模型，在处理大规模语料库时（例如：100万个词汇），这些模型就存在一些问题。例如在 skip-gram 模型中，输入层的大小是 `[batch_size, vocab_size]` ，那么 size(W) = [1,000,000, vocab\_size]，size(V) = [vocab\_size, 1,000,000]。在如此多神经元的情况下，中间计算过程需要很长时间。具体来说，以下两个地方的计算会出现瓶颈:

1. 输入层的 one-hot 矩阵 `training_input` 和权重矩阵 `W` 的乘积
2. 中间层 `hidden` 和权重矩阵 `V` 的乘积以及 Softmax 层的计算

##

**第一个问题：** `torch.mm` 使得 one-hot 行矩阵 [0 1 0 ... 0] 众多的零都进行了计算，但真正有意义的计算只有 one-hot矩阵中为 1 对应的 `W` 那一行原样输出了，如图4-3所示。若先不考虑batch_size，则 `torch.mm` 进行的运算是：
		
	[1, vocab_size] × [vocab_size, word_embedding_dim] = [1, word_embedding_dim]

此时 `hidden: [1, word_embedding_dim]` 其实就是 one-hot 矩阵中值为 1 对应的权重矩阵 `W` 中的那一行原样输出。 因此，直觉上将单词转化为 one-hot 向量的处理和 `torch.mm` 层中的矩阵乘法似乎没有必要。

只用选择 one-hot 向量值为 1 的列号就能实现 `torch.mm` 的作用，这里称这一层为 `Embedding层`。 

##

**第二个问题：** 为了将 `[batch_size, word_embedding_dim]` 变成 `[batch_size, vocab_size]`，因此，这里的权重矩阵 `V: [word_embedding_dim, vocab_size]` 依然是一个很大的矩阵。

并且，根据 Softmax 层的计算公式: $y=\frac{exp(s_k)}{\Sigma^{1000000}_{i=1}exp(s_i)}$ 可知，分母需要计算100万次的exp，这个计算也和词汇量成正比。因此，需要一个可以替代 Softmax 的轻量计算。

在此，权重矩阵 `V` 可以只计算目标词的得分（也就是内积）。输出侧的权重 `V` 中保存了各个单词 ID 对应的单词向量。此处，我们提取 `output` 这个单词向量，再求这个向量和中间层神经元的内积，这就是最终的得分。然后，使用 Sigmoid 函数将其转化为概率。

图见《深度学习进阶自然语言处理——斋藤康毅》 P167、P168 图4-11、4-12。

# Word2Vec

Word2Vec是2013年由Tomas Mikolov提出的，其核心思想是用一个词的上下文来刻画这个词。Word2Vec可以分为两种不同的模型：**Skip-gram**和**CBOW**模型。

---

### 1. Skip-gram

**1.1 作用：** 利用中心词预测上下文词，这里的上下文词是以中心词为中心，某个窗口内的词。窗口大小：`window_size`。

**1.2 网络结构：** 三层神经网络，输入层、中间层、输出层。

*1.2.1 输入层：* 输入矩阵`training\_input`的每一行是中心词的one-hot编码，因此每一行的维度是词典去重后的长度`vocab\_size`。

	size(training_input) = [batch_size, vocab_size]

*1.2.2 中间层：* 权重矩阵`W`和输入矩阵`training\_input`相乘，得到每一个单词的词嵌入向量（为了前后统一，这里叫`hidden`）。

	hidden = training_input * W
	size(W) = [vocab_size, word_embedding_dim]
	size(hidden) = [batch_size, word_embedding_dim]

*1.2.3 输出层：* 词嵌入向量`hidden`和权重矩阵`V`相乘，得到输出上下文词的类别`output`，也就是在vocab中的索引，因此，每一行的维度是词典的维度，预测的是哪个词，就产生一个对应`vocab_size`大小的one-hot行矩阵。

	output = hidden * V
	size(V) = [word_embedding_dim, vocab_size]
	size(output) = [batch_size, vocab_size]

*1.2.4 损失函数：* 在Skip-gram中，我们的目的是由中心词$w_t$去预测窗口内的上下文词w_{t-1}和w_{t+1}，此时，可以建模为：

$$ P(w_{t-1}, w_{t+1}|w_t) = P(w_{t-1}|w_t)P(w_{t+1}|w_t)$$

利用交叉熵误差函数。我们希望损失函数值最小：

$$ L=-{\rm log}P(w_{t-1}, w_{t+1}|w_t)=-{\rm log}P(w_{t-1}|w_t)P(w_{t+1}|w_t)=-({\rm log}P(w_{t-1}|w_t)+{\rm log}P(w_{t+1}|w_t)) $$

扩展到整个语料库，则损失函数可以表示为：

$$L=-\frac{1}{T}\Sigma^T_{t=1}({\rm log}P(w_{t-1}|w_t)+{\rm log}P(w_{t+1}|w_t))$$

	import torch.nn as nn
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	loss_fn = nn.CrossEntropyLoss().to(device)





---

### 2. CBOW

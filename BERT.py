import os
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
from d2l import torch as d2l
from tqdm.auto import tqdm
from datasets import list_datasets, load_dataset
from pprint import pprint
import h5py
from torch.optim import Adam
os.environ["http_proxy"] = "http://192.168.104.50:7890"
os.environ["https_proxy"] = "http://192.168.104.50:7890"


class Wiki_book_Dataset(torch.utils.data.Dataset): 
    def __init__(self, train_path): 
        self.train_path=train_path
    
    def __len__(self): 
        # return the number of samples 
        return 3353*10_000
 
    def __getitem__(self, i):
        file_idx,content_idx = i/100_000,i%100_000
        inputm_path = self.train_path+'/inputm/inputm_%d.h5'%(file_idx)
        attm_path = self.train_path+'/attm/attm_%d.h5'%(file_idx)
        pred_path = self.train_path+'/pred/pred_%d.h5'%(file_idx)
        labels_path = self.train_path+'/label/label_%d.h5'%(file_idx)
        weight_path = self.train_path+'/weight/weight_%d.h5'%(file_idx)
        with h5py.File(inputm_path, 'r') as f:
            inputm =f['data'][content_idx]
        with h5py.File(attm_path, 'r') as f:
            attm =f['data'][content_idx]
        with h5py.File(pred_path, 'r') as f:
            pred =f['data'][content_idx]
        with h5py.File(labels_path, 'r') as f:
            label =f['data'][content_idx]
        with h5py.File(weight_path, 'r') as f:
            weight =f['data'][content_idx]
        inputm_t = torch.tensor(inputm)
        attm_t = torch.tensor(attm)
        pred_t = torch.tensor(pred)
        label_t = torch.tensor(label)
        weight_t = torch.tensor(weight)

        #return 1,2
        return inputm_t,attm_t,pred_t,label_t,weight_t

#@save

#@save
class Vocab:
    def __init__(self,file):
        with open(file, 'r') as file:
            content = file.read()
        self.idx_to_token = content.split()
        self.token_to_idx = {token: idx
            for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    def unk(self):  # 未知词元的索引为0
        return 100
# 使用空格分割字符串，得到单词列表

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        # print(type(X))
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, models,vocab_size, num_hiddens=768, hid_in_features=768, mlm_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = models
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)

    def forward(self, tokens, attention_mask,pred_positions=None):
        encoded_X = self.encoder(tokens, attention_mask=attention_mask)['last_hidden_state']
        mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        return encoded_X, mlm_Y_hat

#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         attention_mask,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y):
    # 前向传播
    _, mlm_Y_hat = net(tokens_X, attention_mask, pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    return mlm_l



def train(net, train_loader, learning_rate, epochs,vocab_size):
  # 通过Dataset类获取训练和验证集
    #train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate)
    # step, timer = 0, d2l.Timer()
    # animator = d2l.Animator(xlabel='step', ylabel='loss',
    #                         xlim=[1, num_steps], legend=['mlm', 'nsp'])

    if use_cuda:
            net = net.cuda()
            loss = loss.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
      # 进度条函数tqdm
            for inputm,attm,pred,label,weight in tqdm(train_loader):
                inputm = inputm.to(device)
                attm = attm.to(device)
                pred = pred.to(device)
                label = label.to(device)
                weight = weight.to(device)
                optimizer.zero_grad()
                l = _get_batch_loss_bert(net, loss, vocab_size, inputm, attm, pred, weight, label)
                l.backward()
                optimizer.step()


model = AutoModel.from_pretrained('bert-base-uncased')
train_path='/home/ubutnu/nvmessd/DeepLearning/wiki_book/'
vocab=Vocab('./vocab.txt')
net=BERTModel(models=model,vocab_size=len(vocab), num_hiddens=768, hid_in_features=768, mlm_in_features=768)
train_dataset=Wiki_book_Dataset(train_path)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=12,num_workers=12,shuffle=True)
train(net,train_loader,learning_rate=0.0001,epochs=1,vocab_size=len(vocab))

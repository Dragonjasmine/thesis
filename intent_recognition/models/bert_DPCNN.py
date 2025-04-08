# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer
#from pytorch_pretrained import BertModel, BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert_DPCNN'
        self.train_path = dataset + '/data_intents/train.txt'                                # 训练集
        self.dev_path = dataset + '/data_intents/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data_intents/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data_intents/class.txt').readlines()]                                # 类别名单
        
        self.save_path = dataset + '/saved_dictsss/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 100
        self.num_classes = len(self.class_list)
        
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 2e-5
        
        
        self.bert_path = "bert-base-uncased"#'./bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_filters = 250
        
        self.dropout = 0.2
        
        self.rnn_hidden = 768
        self.num_layers = 1
        self.lstm_dropout_value = 0.2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)
        
        
        self.lstm = nn.GRU(config.hidden_size, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        
        self.max_len = config.pad_size
        self.hidden_size = config.hidden_size
        self.att_weight = nn.Parameter(torch.randn(1, config.hidden_size, 1))
        self.dropout_rnn = nn.Dropout(config.lstm_dropout_value)
        self.fc_rnn = nn.Linear(config.hidden_size, config.num_classes)
        
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(2 * config.num_classes, config.num_classes)
        self.sig = nn.Sigmoid()
        
    def rnn_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1).cpu()
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)
        return h
    
    def atten_layer(self, h , mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)
        att_score = torch.bmm(self.tanh(h), att_weight)
        mask = mask.unsqueeze(dim=-1)
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)
        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)
        reps = self.tanh(reps)
        return reps
    
    
    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        x = encoder_out.unsqueeze(1)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()
        x = self.fc(x)

        out = self.rnn_layer(encoder_out, mask)
        out = self.dropout_rnn(out)
        out = self.atten_layer(out, mask)
        out = self.fc_rnn(out)
        
        x = x + out
    
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px  # short cut
        return x

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.drop_prob = 0.2
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=self.num_layers,dropout = self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)
     
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        input = torch.cat((features.unsqueeze(1), embeddings), 1)
        out,hidden= self.lstm(input)
        out = self.dropout(out)
        output = self.linear(out)
        #output = self.softmax(out)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        for i in range(max_len):
            if states is None:
                _input = inputs
            else:
                embedding = self.embed(states)
                _input = torch.cat((inputs,embedding),1)
            
            out,hidden = self.lstm(_input)
            out = self.linear(out)
            #out = self.softmax(out)
            _,states = out.max(2)
        output = states.tolist()[0]
        return output
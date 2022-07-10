import torch
import torch.nn as nn

class LSTMLinear(nn.Module): #The main model (NO PLAYER DATA)

    def __init__(self, device, info = 1):
        super().__init__()

        self.device = device
        self.info = info
        
        self.LSTM1 = nn.LSTM(185, 500, bias = True, batch_first = False)
        self.lDrop1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(500, 100, bias = True)
        self.drop1 = nn.Dropout(0.1)
        self.linAct1 = nn.ReLU()
        self.LSTM2 = nn.LSTM(100, 300, bias = True, batch_first = False)
        self.lDrop2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(info * 300, 60, bias = True)
        self.drop2 = nn.Dropout(0.1)
        self.linAct2 = nn.ReLU()
        self.linear3 = nn.Linear(60, 2, bias = True)

        self.hidden_cell1 = (torch.zeros(1, 1, 500).to(device), torch.zeros(1, 1, 500).to(device)) #Empty hidden state
        self.hidden_cell2 = (torch.zeros(1, 1, 300).to(device), torch.zeros(1, 1, 300).to(device))

        self.init_weights()

    
    def init_weights(self):
        RANGE = 0.5
        self.linear1.weight.data.uniform_(-RANGE, RANGE)
        self.linear1.bias.data.uniform_(-RANGE, RANGE)
        self.linear2.weight.data.uniform_(-RANGE, RANGE)
        self.linear2.bias.data.uniform_(-RANGE, RANGE)
        self.linear3.weight.data.uniform_(-RANGE, RANGE)
        self.linear3.bias.data.uniform_(-RANGE, RANGE)

    
    def forward(self, input, collecting = False): #The network is not always making a prediction, it may be "collecting" information first
        
        z, self.hidden_cell1 = self.LSTM1(input, self.hidden_cell1)
        z = self.lDrop1(z)
        z = self.linAct1(self.drop1(self.linear1(z)))
        z, self.hidden_cell2 = self.LSTM2(z, self.hidden_cell2)

        if (not collecting):
            z = z.view(1, len(z[0]), self.info * 300)
            z = self.lDrop2(z)
            z = self.linAct2(self.drop2(self.linear2(z)))
            z = self.linear3(z)

        return z


    def reset_cells(self, batch_size):
        self.hidden_cell1 = (torch.zeros(1, batch_size, 500).to(self.device), torch.zeros(1, batch_size, 500).to(self.device))
        self.hidden_cell2 = (torch.zeros(1, batch_size, 300).to(self.device), torch.zeros(1, batch_size, 300).to(self.device))

    
    def cuda(self, device):
        self.LSTM1.cuda(device)
        self.lDrop1.cuda(device)
        self.linear1.cuda(device)
        self.drop1.cuda(device)
        self.linAct1.cuda(device)
        self.LSTM2.cuda(device)
        self.lDrop2.cuda(device)
        self.linear2.cuda(device)
        self.drop2.cuda(device)
        self.linAct2.cuda(device)
        self.linear3.cuda(device)


    def train(self, mode = True):
        self.LSTM1.train(mode)
        self.lDrop1.train(mode)
        self.linear1.train(mode)
        self.drop1.train(mode)
        self.linAct1.train(mode)
        self.LSTM2.train(mode)
        self.lDrop2.train(mode)
        self.linear2.train(mode)
        self.drop2.train(mode)
        self.linAct2.train(mode)
        self.linear3.train(mode)
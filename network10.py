from types import NoneType
import torch.nn as nn
import torch
import requests #For communicating with javascript
import json
import random



class GameDataset(torch.utils.data.Dataset):

    def __init__(self, connection, device = "cpu"): #Retrieves the size from a connection - a string containing the http ip and port
        self.connection = "http://" + connection
        self.length = int(requests.get(self.connection + "/size_main", headers = {}).text)
        self.device = device

    
    def __getitem__(self, ind):
        result = requests.post(url = self.connection, headers = {}, json = {"index": ind})
        if (result.status_code == 400):
          raise Exception("400 Bad Request: " + result.text)

        #print(result.text)
        result = json.loads(result.text)
        #for k, game in enumerate(result[0][0][0]):
        #    for i, val in enumerate(game):
        #        if (type(val) == NoneType):
        #            print(i)
        #            print(game)
        #            print(val)
        #            print(k)
        data = [torch.FloatTensor(result[0][0]).to(self.device)[None, :].swapdims(0, 1), torch.FloatTensor(result[0][1]).to(self.device)[None, :].swapdims(0, 1)]
        return data

    
    def __len__(self):
        return self.length



class EvaluationalGameDataset(torch.utils.data.Dataset):

    def __init__(self, connection, device = "cpu"): #Retrieves the size from a connection - a string containing the http ip and port
        self.connection = "http://" + connection
        self.length = int(requests.get(self.connection + "/size_eval", headers = {}).text)
        self.device = device

    
    def __getitem__(self, ind, extra = False): #"Extra" denotes if the model is to recieve information about the games requested
        result = requests.post(url = self.connection, headers = {}, json = {"index": ind, "type": "eval", "extra": extra})
        if (result.status_code == 400):
          raise Exception("400 Bad Request: " + result.text)

        if (not extra):
            result = json.loads(result.text)
            data = [torch.FloatTensor(result[0][0]).to(self.device)[None, :].swapdims(0, 1), torch.FloatTensor(result[0][1]).to(self.device)[None, :].swapdims(0, 1)]
            return data
        else: #Result will have a length of 3
            result = json.loads(result.text)
            data = [torch.FloatTensor(result[0][0]).to(self.device)[None, :].swapdims(0, 1), torch.FloatTensor(result[0][1]).to(self.device)[None, :].swapdims(0, 1)]
            gameInfo = result[0][2]
            return (data, gameInfo)

    
    def __len__(self):
        return self.length



class FeedForwardAttention(nn.Module): #The main model

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

    
    def forward(self, input, collecting = False): #Input will be made up of two parts
        
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



def collate(batch):
    data = batch[0]

    for i in range(1, len(batch)):
        data[0] = torch.cat((data[0], batch[i][0]), 1)
        data[1] = torch.cat((data[1], batch[i][1]), 1)

    return data



BATCH_SIZE = 64

device = torch.device("cuda:0")
INFO = 1
ffa = FeedForwardAttention(device, INFO) #The model
ffa.cuda(device)

lossFn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ffa.parameters())

dataset = GameDataset("127.0.0.1:3000", device)
loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True,  collate_fn = collate)

evalDataset = EvaluationalGameDataset("127.0.0.1:3000", device)
#evalLoader = torch.utils.data.DataLoader(evalDataset, batch_size = 10, shuffle = True, collate_fn = collate)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)

#data = dataset[0]
#input = (data[0][0], data[0][1]) #Adds dummy batch dimension
#output = ffa(input)
#print(output)
#exit()


def trainEpoch(ffa, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE):
    totalLoss = 0
    previousLoss = 0

    for index, data in enumerate(loader):

        input = data[0]
        target = data[1]

        optimizer.zero_grad()
        ffa.reset_cells(len(input[0]))
        loss = None
        output = None
        ind = 0

        while (ind <= len(input) - INFO):
            if (ind == len(input) - INFO):
                output = ffa(input[ind:(ind + INFO)], False)
            else:
                output = ffa(input[ind:(ind + INFO)], True)

            if (ind == len(input) - INFO): #Evaluate at the last value
                loss = lossFn(output, target[len(input) - 1][None, :])
            ind += 1

        loss.backward()
        optimizer.step() #The learning

        totalLoss += loss.item()
        if (index % 50 == 49): #Every 50 batches/steps
            previousLoss = totalLoss / 50
            print(f"\tBatch {(index + 1)}; Loss: {previousLoss}. {(BATCH_SIZE * (index + 1)) / len(dataset)}% Complete")
            totalLoss = 0

            #if (index % 2000 == 1999): #Views the results every 2000 batches
            #    ffa.train(False)
            #    evaluate(ffa, evalDataset, lossFn, float("inf"), False)
            #    ffa.train(True)

    return previousLoss


def evaluate(ffa, evalDataset, lossFn, previousPercentCorrect, save = True):

    totalEvalLoss = 0
    numCorrect = 0
    numCorrectRandom = 0
    percentCorrect = 0
    modelNum = 10.5
    softmax = nn.Softmax(dim = 1)

    for i in range(len(evalDataset)):
        data = evalDataset[i]
        input = data[0]
        target = data[1]

        ffa.reset_cells(1)
        loss = None
        output = None
        ind = 0

        while (ind <= len(input) - INFO):
            if (ind == len(input) - INFO):
                output = ffa(input[ind:(ind + INFO)], False)
            else:
                output = ffa(input[ind:(ind + INFO)], True)

            if (ind == len(input) - INFO): #Evaluate at the last value
                loss = lossFn(output, target[len(input) - 1][None, :])
            ind += 1

        totalEvalLoss += loss

        output = output[0]
        target = target[len(input) - 1]
        output = softmax(output)

        predOutcome = 0
        if (output[0][0] > output[0][1]): #Team 1 won
            predOutcome = 1
        
        actualOutcome = 0
        if (target[0][0] > target[0][1]):
            actualOutcome = 1

        numCorrect += (predOutcome == actualOutcome)
        numCorrectRandom += (random.randint(0, 1) == actualOutcome)

    newLoss = (totalEvalLoss / len(evalDataset))
    percentCorrect = (numCorrect / len(evalDataset))
    percentCorrectRandom = (numCorrectRandom / len(evalDataset))
    print(f"Evaluational Loss: {newLoss}")
    print(f"Percent Correct: {percentCorrect}")
    print(f"Random Choice Correct: {percentCorrectRandom}")

    if (previousPercentCorrect < percentCorrect):
        previousPercentCorrect = percentCorrect
        if (save):
            torch.save(ffa.state_dict(), f"model{modelNum}")
    
    return previousPercentCorrect


def train(ffa, EPOCHS, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE, previousEvalLoss):

    for epoch in range(EPOCHS):
        
        print(f"EPOCH {epoch + 1}")

        ffa.train(True)
        previousLoss = trainEpoch(ffa, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE)

        ffa.train(False) #Evaluate
        previousEvalLoss = evaluate(ffa, evalDataset, lossFn, previousEvalLoss, True)



print(ffa.load_state_dict(torch.load("model10.5")))
ffa.train(False)
previousPercentCorrect = evaluate(ffa, evalDataset, lossFn, float('inf'), save = False)
EPOCHS = 5
train(ffa, EPOCHS, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE, previousPercentCorrect)
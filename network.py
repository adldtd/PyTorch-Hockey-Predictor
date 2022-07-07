from types import NoneType
import torch.nn as nn
import torch
import requests #For communicating with javascript
import json
import random
from LSTMLinear_2 import LSTMLinear



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



def collate(batch):
    data = batch[0]

    for i in range(1, len(batch)):
        data[0] = torch.cat((data[0], batch[i][0]), 1)
        data[1] = torch.cat((data[1], batch[i][1]), 1)

    return data



BATCH_SIZE = 64

device = torch.device("cuda:0")
INFO = 1
ffa = LSTMLinear(device, INFO) #The model
ffa.cuda(device)

lossFn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ffa.parameters())

dataset = GameDataset("127.0.0.1:3000", device)
loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True,  collate_fn = collate)

evalDataset = EvaluationalGameDataset("127.0.0.1:3000", device)


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

    return previousLoss


def evaluate(ffa, evalDataset, lossFn, previousPercentCorrect, modelName, save = True):

    totalEvalLoss = 0
    numCorrect = 0
    numCorrectRandom = 0
    percentCorrect = 0
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
            torch.save(ffa.state_dict(), modelName)
    
    return previousPercentCorrect


def train(ffa, EPOCHS, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE, previousEvalLoss, modelName):

    for epoch in range(EPOCHS):
        
        print(f"EPOCH {epoch + 1}")

        ffa.train(True)
        previousLoss = trainEpoch(ffa, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE)

        ffa.train(False) #Evaluate
        previousEvalLoss = evaluate(ffa, evalDataset, lossFn, previousEvalLoss, modelName, True)



MODEL_NAME = "model10.5"
print(ffa.load_state_dict(torch.load(MODEL_NAME)))
ffa.train(False)
previousPercentCorrect = evaluate(ffa, evalDataset, lossFn, float('inf'), MODEL_NAME, save = False)
EPOCHS = 5
train(ffa, EPOCHS, dataset, loader, evalDataset, lossFn, optimizer, BATCH_SIZE, previousPercentCorrect, MODEL_NAME)
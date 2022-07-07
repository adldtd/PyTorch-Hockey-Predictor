import requests
import json
from network10_5model import FeedForwardAttention
import torch
import torch.nn as nn

MODEL_NAME = "model10.5"
INFO = 1 #How many games are fed into the model at once
softmax = nn.Softmax(dim = 0)
order = json.load(open("hockey-scraper/order.json", encoding = "utf-8"))
teams = json.load(open("hockey-scraper/teams.json", encoding = "utf-8"))

print("Loading model...")
device = torch.device("cuda:0")
model = FeedForwardAttention(device)
model.cuda(device)
model.train(False)
print(model.load_state_dict(torch.load(MODEL_NAME)))

while (True):
    print("\nEnter a team (3 letter string): ", end = "")
    team = input()
    if (team == "EXIT"): break

    print("Enter the opponent: ", end = "")
    opp = input()
    if (opp == "EXIT"): break

    print("Enter the date of game (yyyy-mm-dd): ", end = "")
    date = input()
    if (date == "EXIT"): break

    jsn = {"team": team, "opp": opp, "date": date, "number_games": "1"}
    print("\nFetching data...")
    result = requests.post(url = "http://127.0.0.1:3000/fetch", headers = {}, json = jsn)

    if (result.status_code != 201):
        print(f"{result.status_code} {result.reason}: {result.text}")
        continue

    print("Retrieving data...")
    inp = requests.get(url = "http://127.0.0.1:3000/crunch", headers = {}, json = {"noUsePlayers": ""})
    inp = json.loads(inp.text)

    inp = torch.FloatTensor(inp).to(device)[None, :].swapdims(0, 1)

    print("Generating prediction...")
    out = None
    ind = 0
    model.reset_cells(1)

    while (ind <= len(inp) - INFO):
        if (ind == len(inp) - INFO):
            out = model(inp[ind:(ind + INFO)], False)
        else:
            out = model(inp[ind:(ind + INFO)], True)
        ind += 1

    out = softmax(out[0][0])

    pred = team
    for i in range(35):
        if (inp[len(inp) - 1][0][i] == 1):
            if (order[pred] != i):
                pred = opp
            break

    print(f"\n{team} ({teams[team][0]}) VS. {opp} ({teams[opp][0]}), on {date}")
    print("And the winner is...")

    if (pred == team):
        if (out[0] > out[1]):
            print(teams[team][0])
            print(f"With a certainty of {out[0] * 100}%")
        else:
            print(teams[opp][0])
            print(f"With a certainty of {out[1] * 100}%")
    else:
        if (out[0] > out[1]):
            print(teams[opp][0])
            print(f"With a certainty of {out[0] * 100}%")
        else:
            print(teams[team][0])
            print(f"With a certainty of {out[1] * 100}%")

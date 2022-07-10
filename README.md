#PyTorch-Hockey-Predictor

A machine learning project using both NodeJS and PyTorch to scrape, process, and train on hockey data.

##Contents

LSTM_Linear.py: The PyTorch neural network. Used by network.py and predicter.py to train and output predictions.
network.py: The neural network trainer. Loads a certain model, and trains for a certain amount of epochs (both modifiable in the last six lines of the program); uplinks with database.js.
predicter.py: Uses a pre-trained neural network to predict hockey data; uplinks with retriever.js.

teams.js: Scrapes all hockey teams from a certain start date (default: 2000) from https://www.hockey-reference.com, and saves them in a data folder as teams.json.
games.js: Scrapes hockey game data for the teams contained in teams.json from a certain start date (default: 2000) from hockey-reference; saves each as "games{three letter team indicator}.json" in the data folder.
database.js: Retrieves, crunches, and formats the game data downloaded into trainable info, and stores it all in a javascript array. The program runs as a server (default: at 127.0.0.1 at port 3000), from which network.py can communicate with through HTTP during runtime. In short, database.js "feeds" python data as it trains the neural net.
retriever.js: A sort-of "combination" of the functionality of games.js and teams.js; retrieves data from https://www.hockey-reference.com, and sends it as a json string upon request. Like database.js, the program also runs as a server (default: 127.0.0.1 at port 3000), which predicter.py uses to fetch data to make predictions on.

NOTE: Scraping scripts like teams.js, games.js are for educational purposes only. https://www.hockey-reference.com, as it states in the Terms and Conditions, forbids the "...use (of) any automated means to access or use the Site, including scripts, bots, scrapers, data miners, or similar software, in a manner that adversely impacts site performance or access..." Thus, please excercise caution and care when editing or using the former two programs.

##The Model

As the name suggests, the ML Model is made up of LSTM and Linear layers. It takes in the previous games played by each team to predict the results of a next one. By default, it is meant to take in only one game per team, though due to its structure, it can take in an indefinite amount of games before making a prediction. As the model takes in games, the two LSTM layers "choose" what information to retain in a storage array.

##Results

This project started off as a simple idea. I believed that I could use previous hockey game results in order to predict a future game. The intuition was that the model could come up with accurate predictions based on strings of games. For example, if Team A beat Team B, and Team B beat Team C, it could be reasonable to assume that Team A will beat Team C. Thus, I jumped to data collecting and model building for a binary classification task - using team statistics over time to predict the winner. Throughout the project, I experimented with a multitude of models, algorithms, and stats to include. I would go from transformer based architecture, to recurrent neural networks, to LSTMs. Eventually, I got down to an evaluational accuracy of 72%.

Overall, my initial hypothesis seems to be incorrect. In the end, I ended up only using 1 previous game per team, whereas I planned for something like 20 or 15. In fact, by adding more previous games, the model seemed to lose accuracy. Larger model sizes were tried and implemented, but they either barely helped accuracy or trained for much too long. What ended up really helping was the inclusion of "average" data (for instance, the average amount of wins, losses, OTL for each team). These attributes provide a way to wrap some of the previous game stats into one number. Though, I still regard this project to be a success. In the future, I may add even more "average" stats to improve performance.

##Download

The python and NodeJS files use versions 3.10.0 and 16.15.0, respectively. Python requires PyTorch, preferrably CUDA (if your graphics card is compatible), and Requests. NodeJS requires Axios, Express, and Seedrandom. From there, you can make adjustments to the model, if needed, and use predicter.py with a pre-trained model.
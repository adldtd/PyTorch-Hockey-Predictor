const fs = require("fs");
const express = require("express");
const seedrandom = require("seedrandom");
const gen = seedrandom("Cake");
const app = express();
const functions = require("./functions");
app.use(express.json()); //For recieving index data

  /******************************************************************************************/
 /* The database "wrapper," which formalizes game data and sends it to Python through HTTP */
/******************************************************************************************/

var teams = JSON.parse(fs.readFileSync("data/teams.json"));
var allTeams = {};
var order = {};
var borders = [];

var main_array = []; //Used for ordering all of the data; contains all trainable games
var eval_array = []; //Used for testing evaluation purposes; contains data from March of 2022 to beyond
const GAME_SIZE = 1; //Extremely important constant; how many previous games the program collects for predictions
const USE_PLAYERS = false; //Whether to include individual player data; false by default

var avgWeight = 0; //Averages are used when height/weight data is unavailable
var weights = 0;
var avgHeight = 0;
var heights = 0;

var iterations = 0;
var shoots = {};
var positions = {};
var shotsNum = 0;
var posNum = 0;
var maxNumGoalies = 0;
var minNumGoalies = Number.MAX_SAFE_INTEGER;
var maxNumFielders = 0;
var minNumFielders = Number.MAX_SAFE_INTEGER;


//***********************************************************************************
//Orders the games into an indexable array for data retrieval
//***********************************************************************************
function orderGames(main_array, eval_array, GAME_SIZE, allTeams, teams) {

  let eval_used = {};
  for (team in allTeams)
    eval_used[team] = {};
  
  for (team in allTeams) {

    for (gm in allTeams[team]) {

      gm = parseInt(gm);

      if (gm >= allTeams[team].length - 1) //Means the next game result is not available
        continue;

      let game = allTeams[team][gm]; //Contains the next game to be predicted
  
      let teamDate = new Date(game["nextGame"]["date"]);
      let opp = game["nextGame"]["opp"];
  
      while (teams[opp].length > 2)
        opp = teams[opp][2]; //The opp name might be under a previous alias - like the Mighty Ducks of Anaheim
  
      let l = 0; let h = allTeams[opp].length - 1; //Binary search the opponent's games for the corresponding "next game"
      let oppIndex = 0;
  
      while (l <= h) {
  
        let m = Math.floor((l + h) / 2);
        let oppDate = new Date(allTeams[opp][m]["nextGame"]["date"]);
        
        if (teamDate > oppDate)
          l = m + 1;
        else if (teamDate < oppDate)
          h = m - 1;
        else { //Equal; assumes that a team will not play more than one game on a single date
          oppIndex = m;
          break;
        }
      }
  
      if (oppIndex >= allTeams[opp].length - 1) //Unlikely but possible
        continue;

      let oppGame = allTeams[opp][oppIndex]; //Transfer of information

      game["nextGame"]["oppWins"] = oppGame["nextGame"]["teamWins"]; oppGame["nextGame"]["oppWins"] = game["nextGame"]["teamWins"];
      game["nextGame"]["oppLosses"] = oppGame["nextGame"]["teamLosses"]; oppGame["nextGame"]["oppLosses"] = game["nextGame"]["teamLosses"];
      game["nextGame"]["oppTies"] = oppGame["nextGame"]["teamTies"]; oppGame["nextGame"]["oppTies"] = game["nextGame"]["teamTies"];
      game["nextGame"]["oppOTL"] = oppGame["nextGame"]["teamOTL"]; oppGame["nextGame"]["oppOTL"] = game["nextGame"]["teamOTL"];
      game["nextGame"]["oppStreak"] = oppGame["nextGame"]["teamStreak"]; oppGame["nextGame"]["oppStreak"] = game["nextGame"]["teamStreak"];
      game["nextGame"]["oppStreakType"] = oppGame["nextGame"]["teamStreakType"]; oppGame["nextGame"]["oppStreakType"] = game["nextGame"]["teamStreakType"];

      game["nextGame"]["oppHomeWins"] = oppGame["nextGame"]["teamHomeWins"]; oppGame["nextGame"]["oppHomeWins"] = game["nextGame"]["teamHomeWins"];
      game["nextGame"]["oppHomeLosses"] = oppGame["nextGame"]["teamHomeLosses"]; oppGame["nextGame"]["oppHomeLosses"] = game["nextGame"]["teamHomeLosses"];
      game["nextGame"]["oppAwayWins"] = oppGame["nextGame"]["teamAwayWins"]; oppGame["nextGame"]["oppAwayWins"] = game["nextGame"]["teamAwayWins"];
      game["nextGame"]["oppAwayLosses"] = oppGame["nextGame"]["teamAwayLosses"]; oppGame["nextGame"]["oppAwayLosses"] = game["nextGame"]["teamAwayLosses"];

      game["nextGame"]["playoffOppWins"] = oppGame["nextGame"]["playoffTeamWins"]; oppGame["nextGame"]["playoffOppWins"] = game["nextGame"]["playoffTeamWins"];
      game["nextGame"]["playoffOppLosses"] = oppGame["nextGame"]["playoffTeamLosses"]; oppGame["nextGame"]["playoffOppLosses"] = game["nextGame"]["playoffTeamLosses"];

      game["nextGame"]["oppPointsPercent"] = oppGame["nextGame"]["teamPointsPercent"]; oppGame["nextGame"]["oppPointsPercent"] = game["nextGame"]["teamPointsPercent"];
      game["nextGame"]["playoffOppPointsPercent"] = oppGame["nextGame"]["playoffTeamPointsPercent"]; oppGame["nextGame"]["playoffOppPointsPercent"] = game["nextGame"]["playoffTeamPointsPercent"];

      game["nextGame"]["oppAvgGoalsFor"] = oppGame["nextGame"]["teamAvgGoalsFor"]; oppGame["nextGame"]["oppAvgGoalsFor"] = game["nextGame"]["teamAvgGoalsFor"];
      game["nextGame"]["oppAvgGoalsAgainst"] = oppGame["nextGame"]["teamAvgGoalsAgainst"]; oppGame["nextGame"]["oppAvgGoalsAgainst"] = game["nextGame"]["teamAvgGoalsAgainst"];

      game["nextGame"]["oppGames"] = oppGame["nextGame"]["teamGames"]; oppGame["nextGame"]["oppGames"] = game["nextGame"]["teamGames"];
      game["nextGame"]["playoffOppGames"] = oppGame["nextGame"]["playoffTeamGames"]; oppGame["nextGame"]["playoffOppGames"] = game["nextGame"]["playoffTeamGames"];

      if (gm < GAME_SIZE - 1) //Not enough previous data for the game to be indexable
        continue;
      
      if (oppIndex < GAME_SIZE - 1)
        continue;

      let gameInformation = {};
      gameInformation["team"] = team;
      gameInformation["teamIndex"] = gm;
      gameInformation["opp"] = opp;
      gameInformation["oppIndex"] = oppIndex;

      let roll = Math.floor(gen() * 100);

      //if (roll !== 1 && !(gm in eval_used[team]) && !(gm in eval_used[opp]))
      if (roll !== 1 && !(gm in eval_used[team]) && !(gm in eval_used[opp])) { //Random selection of evaluation data
        if (teamDate > new Date("2008-08-01")) //Cutoff date; attempt to "modernize" data
          main_array.push(gameInformation);
      }
      else {
        if (teamDate > new Date("2008-08-01")) { //NOTE: 2005-07-22 is when ties were removed from the NHL
          eval_array.push(gameInformation);
          eval_used[team][gm] = 0;
          eval_used[opp][oppIndex] = 0;
        }
      }
    }
  }

}



//***********************************************************************************
//Converts collected and loaded data into trainable information for faster indexing
//***********************************************************************************
function crunchData(default_array) {

  for (ind in default_array) {

    ind = parseInt(ind);
    data = []

    let team = default_array[ind]["team"];
    let teamIndex = default_array[ind]["teamIndex"];
    let opp = default_array[ind]["opp"];
    let oppIndex = default_array[ind]["oppIndex"];

    let game = allTeams[team][teamIndex]; //Contains the next game to be predicted
    let thatNextGame = allTeams[team][teamIndex + 1]; //Contains the match data for that next game; specifically the result
    let oppGame = allTeams[opp][oppIndex];
    let thatNextOppGame = allTeams[opp][oppIndex + 1]; //Due to how scraping was preformed on different dates, this MIGHT not always be available

    let previousTeamGames = allTeams[team].slice(teamIndex + 1 - GAME_SIZE, teamIndex + 1)
    let previousOppGames = allTeams[opp].slice(oppIndex + 1 - GAME_SIZE, oppIndex + 1);

    for (let q = 0; q < previousTeamGames.length - 1; q++) {
      previousTeamGames[q]["nextGame"]["result"] = previousTeamGames[q + 1]["previousGame"]["result"];
    }
    previousTeamGames[previousTeamGames.length - 1]["nextGame"]["result"] = thatNextGame["previousGame"]["result"];

    for (let q = 0; q < previousOppGames.length - 1; q++) {
      previousOppGames[q]["nextGame"]["result"] = previousOppGames[q + 1]["previousGame"]["result"];
    }
    previousOppGames[previousOppGames.length - 1]["nextGame"]["result"] = thatNextOppGame["previousGame"]["result"];

    
    chunk = functions.crunchGames(previousTeamGames, previousOppGames, order, positions, true, USE_PLAYERS);
    data.push(chunk);
    default_array[ind] = data;
    
  }
}



for (team in teams) { 
  console.log(team);
  loadedTeam = JSON.parse(fs.readFileSync("data/games" + team + ".json"));

  let gamesPlayed = 0;
  let playoffGamesPlayed = 0; //Reset at the end of each season
  let lastNormalGame = 0; //Index of the last non-playoff game

  for (game in loadedTeam) {

    game = parseInt(game);

    let numTeamGoalies = 0;
    for (player in loadedTeam[game]["previousGame"]["teamPlayers"]) {

      let shot = loadedTeam[game]["previousGame"]["teamPlayers"][player]["shoots"];
      if (!(shot in shoots)) {
        shoots[shot] = shotsNum;
        shotsNum++;
      }

      let pos = loadedTeam[game]["previousGame"]["teamPlayers"][player]["POS"];
      if (!(pos in positions)) {
        positions[pos] = posNum;
        posNum++;
      }

      if ("GA" in loadedTeam[game]["previousGame"]["teamPlayers"][player])
        numTeamGoalies++;

      let height = loadedTeam[game]["previousGame"]["teamPlayers"][player]["height"];
      if (!(height === "" || height === undefined || height === null)) { //Do not count otherwise
        avgHeight += height;
        heights++;
      }

      let weight = loadedTeam[game]["previousGame"]["teamPlayers"][player]["weight"];
      if (!(weight === "" || weight === undefined || weight === null)) {
        avgWeight += weight;
        weights++;
      }
    }

    let numOppGoalies = 0;
    for (player in loadedTeam[game]["previousGame"]["oppPlayers"]) {

      let shot = loadedTeam[game]["previousGame"]["oppPlayers"][player]["shoots"];
      if (!(shot in shoots)) {
        shoots[shot] = shotsNum;
        shotsNum++;
      }

      let pos = loadedTeam[game]["previousGame"]["oppPlayers"][player]["POS"];
      if (!(pos in positions)) {
        positions[pos] = posNum;
        posNum++;
      }

      if ("GA" in loadedTeam[game]["previousGame"]["oppPlayers"][player])
        numOppGoalies++;

      let height = loadedTeam[game]["previousGame"]["oppPlayers"][player]["height"];
      if (!(height === "" || height === undefined || height === null)) {
        avgHeight += height;
        heights++;
      }

      let weight = loadedTeam[game]["previousGame"]["oppPlayers"][player]["weight"];
      if (!(weight === "" || weight === undefined || weight === null)) {
        avgWeight += weight;
        weights++;
      }
    }

    maxNumGoalies = Math.max(numTeamGoalies, numOppGoalies, maxNumGoalies);
    minNumGoalies = Math.min(numTeamGoalies, numOppGoalies, minNumGoalies);

    let numTeamFielders = loadedTeam[game]["previousGame"]["teamPlayers"].length - numTeamGoalies;
    let numOppFielders = loadedTeam[game]["previousGame"]["oppPlayers"].length - numOppGoalies;

    maxNumFielders = Math.max(numTeamFielders, numOppFielders, maxNumFielders);
    minNumFielders = Math.min(numTeamFielders, numOppFielders, minNumFielders);

    if (loadedTeam[game]["previousGame"]["type"] === "playoffs" && game > 0) {
      loadedTeam[game]["nextGame"]["teamWins"] += loadedTeam[game - 1]["nextGame"]["teamWins"];
      loadedTeam[game]["nextGame"]["teamLosses"] += loadedTeam[game - 1]["nextGame"]["teamLosses"];
      loadedTeam[game]["nextGame"]["teamTies"] += loadedTeam[game - 1]["nextGame"]["teamTies"];
      loadedTeam[game]["nextGame"]["teamOTL"] += loadedTeam[game - 1]["nextGame"]["teamOTL"];
    }


    loadedTeam[game]["nextGame"]["teamHomeWins"] = 0; //Maybe divide this into non-playoff data and playoff data again?
    loadedTeam[game]["nextGame"]["teamHomeLosses"] = 0;
    loadedTeam[game]["nextGame"]["teamAwayWins"] = 0;
    loadedTeam[game]["nextGame"]["teamAwayLosses"] = 0;
    loadedTeam[game]["nextGame"]["oppHomeWins"] = null;
    loadedTeam[game]["nextGame"]["oppHomeLosses"] = null;
    loadedTeam[game]["nextGame"]["oppAwayWins"] = null;
    loadedTeam[game]["nextGame"]["oppAwayLosses"] = null;

    loadedTeam[game]["nextGame"]["playoffTeamWins"] = 0;
    loadedTeam[game]["nextGame"]["playoffTeamLosses"] = 0;
    loadedTeam[game]["nextGame"]["playoffOppWins"] = null;
    loadedTeam[game]["nextGame"]["playoffOppLosses"] = null;

    loadedTeam[game]["nextGame"]["teamPointsPercent"] = 0; //Points are calculated by multiplying wins by 2, losses by 0, ties by 1, and OTL by 1, adding all and dividing by num. games * 2
    loadedTeam[game]["nextGame"]["playoffTeamPointsPercent"] = 0;
    loadedTeam[game]["nextGame"]["oppPointsPercent"] = null;
    loadedTeam[game]["nextGame"]["playoffOppPointsPercent"] = null;

    loadedTeam[game]["nextGame"]["teamGoalsFor"] = 0;
    loadedTeam[game]["nextGame"]["teamGoalsAgainst"] = 0;

    loadedTeam[game]["nextGame"]["teamAvgGoalsFor"] = 0;
    loadedTeam[game]["nextGame"]["teamAvgGoalsAgainst"] = 0;
    loadedTeam[game]["nextGame"]["oppAvgGoalsFor"] = null;
    loadedTeam[game]["nextGame"]["oppAvgGoalsAgainst"] = null;

    loadedTeam[game]["nextGame"]["teamGames"] = 0;
    loadedTeam[game]["nextGame"]["playoffTeamGames"] = 0;
    loadedTeam[game]["nextGame"]["oppGames"] = null;
    loadedTeam[game]["nextGame"]["playoffOppGames"] = null;


    if (game !== 0 && (loadedTeam[game - 1]["previousGame"]["season"] === loadedTeam[game]["previousGame"]["season"])) {
      loadedTeam[game]["nextGame"]["teamHomeWins"] = loadedTeam[game - 1]["nextGame"]["teamHomeWins"];
      loadedTeam[game]["nextGame"]["teamHomeLosses"] = loadedTeam[game - 1]["nextGame"]["teamHomeLosses"];
      loadedTeam[game]["nextGame"]["teamAwayWins"] = loadedTeam[game - 1]["nextGame"]["teamAwayWins"];
      loadedTeam[game]["nextGame"]["teamAwayLosses"] = loadedTeam[game - 1]["nextGame"]["teamAwayLosses"];

      loadedTeam[game]["nextGame"]["teamGoalsFor"] = loadedTeam[game - 1]["nextGame"]["teamGoalsFor"];
      loadedTeam[game]["nextGame"]["teamGoalsAgainst"] = loadedTeam[game - 1]["nextGame"]["teamGoalsAgainst"];
    } else {
      gamesPlayed = 0;
      overtimeGamesPlayed = 0;
    }

    let result = loadedTeam[game]["previousGame"]["result"];
    let field = loadedTeam[game]["previousGame"]["field"];

    if (result === 1 && field === "home") //Ties are not counted here
      loadedTeam[game]["nextGame"]["teamHomeWins"] += 1;
    else if (result === 0 && field === "home")
      loadedTeam[game]["nextGame"]["teamHomeLosses"] += 1;
    else if (result === 1 && field === "away")
      loadedTeam[game]["nextGame"]["teamAwayWins"] += 1;
    else if (result === 0 && field === "away")
      loadedTeam[game]["nextGame"]["teamAwayLosses"] += 1;

    let teamScr = loadedTeam[game]["previousGame"]["teamScore"];
    let oppScr = loadedTeam[game]["previousGame"]["oppScore"];

    loadedTeam[game]["nextGame"]["teamGoalsFor"] += teamScr;
    loadedTeam[game]["nextGame"]["teamGoalsAgainst"] += oppScr;


    gamesPlayed++;

    loadedTeam[game]["nextGame"]["teamAvgGoalsFor"] = loadedTeam[game]["nextGame"]["teamGoalsFor"] / gamesPlayed;
    loadedTeam[game]["nextGame"]["teamAvgGoalsAgainst"] = loadedTeam[game]["nextGame"]["teamGoalsAgainst"] / gamesPlayed;

    if (loadedTeam[game]["previousGame"]["type"] === "normal") {
      lastNormalGame = game;
    } else {
      playoffGamesPlayed++;
      
      loadedTeam[game]["nextGame"]["playoffTeamWins"] = loadedTeam[game]["nextGame"]["teamWins"];
      loadedTeam[game]["nextGame"]["playoffTeamLosses"] = loadedTeam[game]["nextGame"]["teamLosses"];
      //Keeps track of team stats throughout the normal season, as well as solely the playoffs
      loadedTeam[game]["nextGame"]["teamWins"] += loadedTeam[lastNormalGame]["nextGame"]["teamWins"];
      loadedTeam[game]["nextGame"]["teamLosses"] += loadedTeam[lastNormalGame]["nextGame"]["teamLosses"];
      loadedTeam[game]["nextGame"]["teamTies"] += loadedTeam[lastNormalGame]["nextGame"]["teamTies"];
      loadedTeam[game]["nextGame"]["teamOTL"] += loadedTeam[lastNormalGame]["nextGame"]["teamOTL"];
    }

    
    let points = (2 * loadedTeam[game]["nextGame"]["teamWins"]) + loadedTeam[game]["nextGame"]["teamTies"] + loadedTeam[game]["nextGame"]["teamOTL"];
    let playoffPoints = (2 * loadedTeam[game]["nextGame"]["playoffTeamWins"]);

    loadedTeam[game]["nextGame"]["teamPointsPercent"] = (points / (gamesPlayed * 2));
    if (playoffGamesPlayed > 0)
      loadedTeam[game]["nextGame"]["playoffTeamPointsPercent"] = (playoffPoints / (playoffGamesPlayed * 2));

    
    loadedTeam[game]["nextGame"]["teamGames"] = gamesPlayed;
    loadedTeam[game]["nextGame"]["playoffTeamGames"] = playoffGamesPlayed;

  }

  allTeams[team] = loadedTeam;
  order[team] = iterations;
  iterations++;
}

avgHeight = Math.floor(avgHeight / heights);
avgWeight = Math.floor(avgWeight / weights);

for (team in teams) { //Combines teams' previous games when they had a different name; however, it keeps that name
  if (!(team in allTeams))
    continue;

  let aliasTeam = "";
  let prevTeam = team;
  while (teams[prevTeam].length > 2) { //For teams that have changed their names
    aliasTeam = teams[prevTeam][2];
    allTeams[aliasTeam] = allTeams[prevTeam].concat(allTeams[aliasTeam]);
    allTeams[prevTeam] = undefined;
    prevTeam = aliasTeam;
  }

  if (borders.length === 0)
    borders.push([allTeams[prevTeam].length - 1, prevTeam]);
  else
    borders.push([allTeams[prevTeam].length - 1 + borders[borders.length - 1][0], prevTeam]);
}

console.log("Data loaded");
orderGames(main_array, eval_array, GAME_SIZE, allTeams, teams);
crunchData(main_array);
crunchData(eval_array);
console.log("Data formatted\n");
//fs.writeFileSync("data/positions.json", JSON.stringify(positions, null, 2));
console.log("Dataset size: " + main_array.length); //All indexable games
console.log("Evaluation size: " + eval_array.length); //All evaluational games



//***********************************************************************************
//Returns a certain game from a specified index
//***********************************************************************************
app.post("/", function (req, res) {

  let default_array = main_array;
  if ("type" in req.body && req.body["type"] === "eval") { //Evaluational data requested
    default_array = eval_array;
  }

  let ind = parseInt(req.body["index"]);

  if (isNaN(ind)) {
    res.status(400).send("NaN");
    console.log("\nError: Value is NaN.");
    return;
  }
  if (ind < 0 || ind > default_array.length) {
    res.status(400).send("Invalid");
    console.log("\nError: Value is invalid.")
    return;
  }

  data = default_array[ind];

  res.status(200).send(JSON.stringify(data));

});



//***********************************************************************************
//Returns the size of the dataset, for epoch training
//***********************************************************************************
app.get("/size_main", function (req, res) {
  res.status(200).send("" + main_array.length);
});



//***********************************************************************************
//Returns the size of the evaluation dataset
//***********************************************************************************
app.get("/size_eval", function (req, res) {
  res.status(200).send("" + eval_array.length);
});

/*

Past: (TOTAL 1389); WITHOUT PLAYER STATS, DIMENSIONALITY IS JUST 77
Team: 35 features
Opp: 35
23 team players (20 fielders, 3 goalies), and 2 for each team: (TOTAL 1312)
  Fielders: (TOTAL 31)
    TOI: 1
    POS: 17
    Shifts: 1
    PlusMinus: 1
    Goals: 1
    Assists: 1
    Shots: 1
    PIM: 1
    Age: 1
    Exp: 1
    Height: 1
    Weight: 1
    Shoots: 3
  Goalies: (TOTAL 12)
    TOI: 1
    GA: 1
    SA: 1
    Shutouts: 1
    PIM: 1
    Age: 1
    Exp: 1
    Height: 1
    Weight: 1
    Catches: 3
TeamScore: 1
OppScore: 1
Result: 1
Field: 1
Type: 1
Distance to Current Game: 1
Seasonal Distance: 1

Future: (TOTAL 84); WITH "AVERAGE" STATS, DIM IS 108
Team: 35
Opp: 35
Team Stats (2 for each): (TOTAL 12)
  Wins: 1
  Losses: 1
  Ties: 1
  OTL: 1
  Streak: 1
  StreakType: 1
Field: 1
Type: 1

*/

app.listen(3000, '127.0.0.1', function()
{
	console.log('Database online!');
});
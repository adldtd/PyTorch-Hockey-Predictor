const fs = require("fs");
const express = require("express");
const seedrandom = require("seedrandom");
const gen = seedrandom("Cake");
const app = express();
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

var avgWeight = 0; //Averages are used when height/weight data is unavailable
var weights = 0;
var avgHeight = 0;
var heights = 0;

var iterations = 0;
/*
var totalGames = 0;
var averageNumPlayers = 0;
var maxNumPlayers = 0;
var minNumPlayers = Number.MAX_SAFE_INTEGER;
*/
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

  console.log(main_array.length); //All indexable games
  console.log(eval_array.length); //All evaluational games
}


for (team in teams) { 
  console.log(team);
  loadedTeam = JSON.parse(fs.readFileSync("data/games" + team + ".json"));

  let gamesPlayed = 0;
  let playoffGamesPlayed = 0; //Reset at the end of each season
  //let inPlayoffs = false; //Set to true when entering a string of playoff games
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
    //totalGames++;


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
//fs.writeFileSync("data/order.json", JSON.stringify(order, null, 2));
var iter = 1;



//***********************************************************************************
//Returns a certain amount of games from a specified index
//***********************************************************************************
app.post("/", function (req, res) {

  let time_start = new Date();

  let data = [];
  const SIZE = 1; //How much training data to crunch and retrieve
  
  console.log("\n");
  console.log(req.body);

  let default_array = main_array;
  let extra = false;
  if ("type" in req.body && req.body["type"] === "eval") { //Evaluational data requested
    default_array = eval_array;
    extra = req.body["extra"];
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

  for (let i = 0; i < SIZE; i++) {

    console.log(default_array[ind]);
    let team = default_array[ind]["team"];
    let teamIndex = default_array[ind]["teamIndex"];
    let opp = default_array[ind]["opp"];
    let oppIndex = default_array[ind]["oppIndex"];

    let game = allTeams[team][teamIndex]; //Contains the next game to be predicted
    let thatNextGame = allTeams[team][teamIndex + 1]; //Contains the match data for that next game; specifically the result
    console.log(thatNextGame["previousGame"]["date"]);
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

    let k = 0; let j = 0; //Combine previous team and opp games
    let previousGames = [];

    while (k < previousTeamGames.length || j < previousOppGames.length) { //Sort games by date

      if (k >= previousTeamGames.length) {
        previousGames.push(previousOppGames[j]);
        j++;
      } else if (j >= previousOppGames.length) {
        previousGames.push(previousTeamGames[k]);
        k++;
      } else { //Compate dates

        let date1 = new Date(previousTeamGames[k]["previousGame"]["date"]);
        let date2 = new Date(previousOppGames[j]["previousGame"]["date"]);

        if (date1 < date2) {
          previousGames.push(previousTeamGames[k]);
          k++;
        } else {
          previousGames.push(previousOppGames[j]);
          j++;
        }
      }
    }


    let nextGameDate = new Date(game["nextGame"]["date"]);
    let nextGameSeason = game["nextGame"]["season"];

    let chunk = [];

    let past = []; //Made up of all past games
    let output = [];

    for (gm in previousGames) {

      let single = []; //Represents a single game

      /*
      let teamFocused = (new Array(35)).fill(0); //The classifier should know which teams are being predicted
      teamFocused[order[game["nextGame"]["team"]]] = 1;
      single = single.concat(teamFocused);

      let oppFocused = (new Array(35)).fill(0);
      oppFocused[order[game["nextGame"]["opp"]]] = 1;
      single = single.concat(oppFocused);
      */

      let team = (new Array(35)).fill(0);
      team[order[previousGames[gm]["previousGame"]["team"]]] = 1; //Equivalent to one-hot encoding
      single = single.concat(team);
      
      let teamFielders = [];
      let teamGoalies = [];
      let numTeamFielders = 0;
      let numTeamGoalies = 0;

      /*//M
      for (player in previousGames[gm]["previousGame"]["teamPlayers"]) {

        let playerStats = previousGames[gm]["previousGame"]["teamPlayers"][player];
        if (!("GA" in playerStats)) {

          let playerInfo = (new Array(31)).fill(0);

          playerInfo[0] = playerStats["TOI"] / 60;
          playerInfo[positions[playerStats["POS"]] + 1] = 1;
          playerInfo[18] = playerStats["shifts"] / 50;
          
          let plusMinus = playerStats["plusMinus"];
          if (plusMinus === "" || plusMinus === undefined || plusMinus === null)
            playerInfo[19] = 0;
          else
            playerInfo[19] = plusMinus;

          playerInfo[20] = playerStats["goals"];
          playerInfo[21] = playerStats["assists"];
          playerInfo[22] = playerStats["shots"];
          playerInfo[23] = playerStats["PIM"] / 60;
          playerInfo[24] = playerStats["age"] / 50;
          playerInfo[25] = playerStats["EXP"] / 50;

          let height = playerStats["height"];
          if (height === "" || height === undefined || height === null)
            playerInfo[26] = 0; //Distance from average height and weight
          else
            playerInfo[26] = (height - avgHeight) / 100;
          let weight = playerStats["weight"];
          if (weight === "" || weight === undefined || weight === null)
            playerInfo[27] = 0;
          else
            playerInfo[27] = (weight - avgWeight) / 100;
          
          let index = 0;
          if ("shoots" in playerStats) {
            if (playerStats["shoots"] === "Left")
              index = 0;
            else if (playerStats["shoots"] === "Right")
              index = 1;
            else //Non existent
              index = 2;
          } else
            index = 2;
          playerInfo[index + 28] = 1;

          teamFielders = teamFielders.concat(playerInfo);
          numTeamFielders++;
        } else {

          let playerInfo = (new Array(12)).fill(0);

          playerInfo[0] = playerStats["TOI"] / 60;
          playerInfo[1] = playerStats["GA"];
          playerInfo[2] = playerStats["SA"];
          playerInfo[3] = playerStats["shutouts"];
          playerInfo[4] = playerStats["PIM"] / 60;
          playerInfo[5] = playerStats["age"] / 50;
          playerInfo[6] = playerStats["EXP"] / 50;
          
          let height = playerStats["height"];
          if (height === "" || height === undefined || height === null)
            playerInfo[7] = 0;
          else
            playerInfo[7] = (height - avgHeight) / 100;
          let weight = playerStats["weight"];
          if (weight === "" || weight === undefined || weight === null)
            playerInfo[8] = 0;
          else
            playerInfo[8] = (weight - avgWeight) / 100;

          let index = 0;
          if ("shoots" in playerStats) {
            if (playerStats["shoots"] === "Left")
              index = 0;
            else if (playerStats["shoots"] === "Right")
              index = 1;
            else
              index = 2;
          } else
            index = 2;
          playerInfo[index + 9] = 1;

          teamGoalies = teamGoalies.concat(playerInfo);
          numTeamGoalies++;
        }
      }
      

      teamFielders = teamFielders.concat((new Array(31 * (20 - numTeamFielders))).fill(0)); //Padding
      teamGoalies = teamGoalies.concat((new Array(12 * (3 - numTeamGoalies))).fill(0));

      single = single.concat(teamFielders);
      single = single.concat(teamGoalies);
      *///M

      let opp = (new Array(35)).fill(0);
      opp[order[previousGames[gm]["previousGame"]["opp"]]] = 1;
      single = single.concat(opp);
      
      let oppFielders = [];
      let oppGoalies = [];
      let numOppFielders = 0;
      let numOppGoalies = 0;
      
      /*//M
      for (player in previousGames[gm]["previousGame"]["oppPlayers"]) {

        let playerStats = previousGames[gm]["previousGame"]["oppPlayers"][player];
        if (!("GA" in playerStats)) {

          let playerInfo = (new Array(31)).fill(0);

          playerInfo[0] = playerStats["TOI"] / 60;
          playerInfo[positions[playerStats["POS"]] + 1] = 1;
          playerInfo[18] = playerStats["shifts"] / 50;
          
          let plusMinus = playerStats["plusMinus"]; //Extremely rare but possible for plus minus to be somehow missing
          if (plusMinus === "" || plusMinus === undefined || plusMinus === null)
            playerInfo[19] = 0;
          else
            playerInfo[19] = plusMinus;
          
          playerInfo[20] = playerStats["goals"];
          playerInfo[21] = playerStats["assists"];
          playerInfo[22] = playerStats["shots"];
          playerInfo[23] = playerStats["PIM"] / 60;
          playerInfo[24] = playerStats["age"] / 50;
          playerInfo[25] = playerStats["EXP"] / 50;
          
          let height = playerStats["height"];
          if (height === "" || height === undefined || height === null)
            playerInfo[26] = 0;
          else
            playerInfo[26] = (height - avgHeight) / 60;
          let weight = playerStats["weight"];
          if (weight === "" || weight === undefined || weight === null)
            playerInfo[27] = 0;
          else
            playerInfo[27] = (weight - avgWeight) / 60;
          
          let index = 0;
          if ("shoots" in playerStats) {
            if (playerStats["shoots"] === "Left")
              index = 0;
            else if (playerStats["shoots"] === "Right")
              index = 1;
            else //Non existent
              index = 2;
          } else
            index = 2;
          playerInfo[index + 28] = 1;

          oppFielders = oppFielders.concat(playerInfo);
          numOppFielders++;
        } else {

          let playerInfo = (new Array(12)).fill(0);

          playerInfo[0] = playerStats["TOI"] / 60; //In minutes
          playerInfo[1] = playerStats["GA"];
          playerInfo[2] = playerStats["SA"];
          playerInfo[3] = playerStats["shutouts"];
          playerInfo[4] = playerStats["PIM"] / 60;
          playerInfo[5] = playerStats["age"] / 50;
          playerInfo[6] = playerStats["EXP"] / 50;
          
          let height = playerStats["height"];
          if (height === "" || height === undefined || height === null)
            playerInfo[7] = 0;
          else
            playerInfo[7] = (height - avgHeight) / 100;
          let weight = playerStats["weight"];
          if (weight === "" || weight === undefined || weight === null)
            playerInfo[8] = 0;
          else
            playerInfo[8] = (weight - avgWeight) / 100;

          let index = 0;
          if ("shoots" in playerStats) {
            if (playerStats["shoots"] === "Left")
              index = 0;
            else if (playerStats["shoots"] === "Right")
              index = 1;
            else
              index = 2;
          } else
            index = 2;
          playerInfo[index + 9] = 1;

          oppGoalies = oppGoalies.concat(playerInfo);
          numOppGoalies++;
        }
      }

      oppFielders = oppFielders.concat((new Array(31 * (20 - numOppFielders))).fill(0));
      oppGoalies = oppGoalies.concat((new Array(12 * (3 - numOppGoalies))).fill(0));

      single = single.concat(oppFielders);
      single = single.concat(oppGoalies);
      *///M

      let totalPoints = previousGames[gm]["previousGame"]["teamScore"] + previousGames[gm]["previousGame"]["oppScore"];
      if (totalPoints !== 0) {
        single.push(previousGames[gm]["previousGame"]["teamScore"] / totalPoints);
        single.push(previousGames[gm]["previousGame"]["oppScore"] / totalPoints);
      } else {
        single.push(0);
        single.push(0);
      }
      single.push(previousGames[gm]["previousGame"]["result"]);
      
      let field = previousGames[gm]["previousGame"]["field"];
      if (field === "away")
        single.push(0);
      else
        single.push(1);
      let type = previousGames[gm]["previousGame"]["type"];
      if (type === "normal")
        single.push(0);
      else
        single.push(1);

      let date = new Date(previousGames[gm]["previousGame"]["date"]);
      single.push((nextGameDate - date) / (1000 * 60 * 60 * 24 * 182.5)); //Distance in HALF years
      let season = previousGames[gm]["previousGame"]["season"];
      single.push(nextGameSeason - season);


      let tm = (new Array(35)).fill(0);
      tm[order[previousGames[gm]["nextGame"]["team"]]] = 1;
      single = single.concat(tm);

      single.push(previousGames[gm]["nextGame"]["teamWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamTies"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamOTL"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamStreak"] / 18);

      let streakType = previousGames[gm]["nextGame"]["teamStreakType"];
      if (streakType === "L")
        single.push(0);
      else if (streakType === "T")
        single.push(0.5);
      else
        single.push(1);

      single.push(previousGames[gm]["nextGame"]["playoffTeamWins"] / 36);
      single.push(previousGames[gm]["nextGame"]["playoffTeamLosses"] / 36);
      single.push(previousGames[gm]["nextGame"]["teamHomeWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamHomeLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamAwayWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamAwayLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["teamPointsPercent"]);
      single.push(previousGames[gm]["nextGame"]["playoffTeamPointsPercent"]);
      single.push(previousGames[gm]["nextGame"]["teamAvgGoalsFor"] / 10);
      single.push(previousGames[gm]["nextGame"]["teamAvgGoalsAgainst"] / 10);
      single.push(previousGames[gm]["nextGame"]["teamGames"] / 106);
      single.push(previousGames[gm]["nextGame"]["playoffTeamGames"] / 36);
      

      let op = (new Array(35)).fill(0);
      op[order[previousGames[gm]["nextGame"]["opp"]]] = 1;
      single = single.concat(op);

      single.push(previousGames[gm]["nextGame"]["oppWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppTies"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppOTL"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppStreak"] / 18);

      streakType = previousGames[gm]["nextGame"]["oppStreakType"];
      if (streakType === "L")
        single.push(0);
      else if (streakType === "T")
        single.push(0.5);
      else
        single.push(1);

      single.push(previousGames[gm]["nextGame"]["playoffOppWins"] / 36);
      single.push(previousGames[gm]["nextGame"]["playoffOppLosses"] / 36);
      single.push(previousGames[gm]["nextGame"]["oppHomeWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppHomeLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppAwayWins"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppAwayLosses"] / 82);
      single.push(previousGames[gm]["nextGame"]["oppPointsPercent"]);
      single.push(previousGames[gm]["nextGame"]["playoffOppPointsPercent"]);
      single.push(previousGames[gm]["nextGame"]["oppAvgGoalsFor"] / 10);
      single.push(previousGames[gm]["nextGame"]["oppAvgGoalsAgainst"] / 10);
      single.push(previousGames[gm]["nextGame"]["oppGames"] / 106);
      single.push(previousGames[gm]["nextGame"]["playoffOppGames"] / 36);


      let fld = previousGames[gm]["nextGame"]["field"];
      if (fld === "away")
        single.push(0);
      else
        single.push(1);
      let typ = previousGames[gm]["nextGame"]["type"];
      if (typ === "normal")
        single.push(0);
      else
        single.push(1);

      //single.push(season / 100);
      //single.push(0);
      past.push(single);


      let target = previousGames[gm]["nextGame"]["result"];
      if (target === 1)
        output.push([1, 0]); //Target probability of the first team winning vs the second
      else if (target === 0)
        output.push([0, 1]);
      else
        output.push([0.5, 0.5]); //Tie game

    }

    chunk.push(past);
    chunk.push(output);

    if (extra)
      chunk.push("" + team + " vs. " + opp + " on " + thatNextGame["previousGame"]["date"] + ", with a score of " + thatNextGame["previousGame"]["teamScore"] + "-" + thatNextGame["previousGame"]["oppScore"]);

    data.push(chunk);
    
    //console.log(chunk);
    //console.log(thatNextGame["previousGame"]["team"] + " vs " + thatNextGame["previousGame"]["opp"]);
    //console.log("Date: " + thatNextGame["previousGame"]["date"]);

    //fs.writeFileSync("data/trainingData.json", JSON.stringify(data));
    //process.exit(0);
  }

  console.log(data.length);
  console.log("Called: " + iter);
  iter++;
  //fs.writeFileSync("data/trainingData.json", JSON.stringify(data));

  res.status(200).send(JSON.stringify(data));
  console.log((new Date() - time_start) + " ms");

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

Past: (TOTAL 1389 + 1 padding, * 15 * 2 = 41700); WITHOUT TEAM STATS, DIMENSIONALITY IS JUST 77 (w/o padding)
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

Future: (TOTAL 84)
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
	console.log('Server running!');
});
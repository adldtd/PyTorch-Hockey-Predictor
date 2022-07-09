const axios = require("axios").default;
const fs = require("fs");
const express = require("express");
const app = express();
const functions = require("./functions");
app.use(express.json()); //For recieving index data


var teams = JSON.parse(fs.readFileSync("data/teams.json"));
var playerReference = JSON.parse(fs.readFileSync("data/playerReference.json"));
var order = JSON.parse(fs.readFileSync("data/order.json"));
var data = []; //For the prediction
const TIME_OUT = 500; //ms to wait before sending a request
var GAME_SIZE = 1;

var config =
{
  headers:
  {
    accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36",
    cookie: "is_live=true; sr_note_box_countdown=0; srcssfull=yes"
  },
  timeout: 25000,
  responseType: "html",
  responseEncoding: "utf-8",
  validateStatus: null, //Accept all statuses; 404 might occur
};



//********************************************************************
//Retrieves appropriate input data from the hockey reference site
//********************************************************************
app.post("/fetch", async function (req, res) { //Request will have the team name, opp name, date, and games to collect

  if (data.length > 0) {
    res.status(409).send("Data remains unsent")
    return;
  }

  let team = req.body["team"];
  if (!(team in teams)) {
    res.status(404).send("Team not found");
    return;
  }

  let opp = req.body["opp"];
  if (!(opp in teams)) {
    res.status(404).send("Opponent not found");
    return;
  }

  let date = req.body["date"];
  let year = 0; //The year determines the season to start scraping in
  let numDate = 0;
  
  if (date !== "") { //If so, predict the most recent game instead

    numDate = Date.parse(date);

    if (isNaN(numDate) || (numDate > (Date.now() + 31356000000)) || (numDate < new Date("2000-01-01"))) { //Either larger than about a year from now or less than 2000
      res.status(400).send("Invalid date");
      return;
    }

    try {
      year = parseInt(date.split("-", 1)[0]) + 1; //The next year season might contain some games in the year before (October - December)
      if (isNaN(year)) {
        res.status(400).send("Invalid date");
        return;
      }
    } catch (exp) {
      res.status(400).send("Invalid date");
      return;
    }

  } else {
    year = parseInt((new Date()).getFullYear()) + 1;
  }

  let number_games = parseInt(req.body["number_games"]);
  if (isNaN(number_games) || number_games < 1) {
    res.status(400).send("Invalid number of games");
    return;
  }
  GAME_SIZE = number_games;

  let gamesRecorded = {}; //Useful for streak data later on
  gamesRecorded[team] = {}; gamesRecorded[opp] = {};
  let seasonsRecorded = {};
  seasonsRecorded[team] = {}; seasonsRecorded[opp] = {};
  let allGames = [];

  clans = [team, opp];
  for (indexe in clans) {
    
    let clan = clans[indexe];
    let seasonInfo = "";
    let pivotYear = year;

    let seasonLink = "https://www.hockey-reference.com/teams/" + clan + "/" + pivotYear + "_games.html";
    await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
    seasonInfo = await axios.get(seasonLink, config);
    
    while (seasonInfo.status !== 200) {
      pivotYear--;
      seasonLink = "https://www.hockey-reference.com/teams/" + clan + "/" + pivotYear + "_games.html";
      await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
      seasonInfo = await axios.get(seasonLink, config);
    }

    let newAllGames = seasonInfo.data.split('date_game" >');
    newAllGames = newAllGames.slice(1, newAllGames.length);
    newAllGames = newAllGames.reverse(); //Start with the most recent games
    allGames.push(newAllGames); //Stores previous seasons
    
    seasonsRecorded[clan][pivotYear] = newAllGames;

    let finding = true; //If false, means that the program is "collecting" the previous games
    let indGames = 0; //How many previous games were recorded so far
    let previousGame = {};
    let nextGame = {};

    //console.log("big");

    while (indGames < number_games) {
      
      let allGame = allGames[allGames.length - 1];
      for (let i = 0; i < allGame.length && indGames < number_games; i++) {
        
        let game = allGame[i];

        if (finding) {
        
          let gameDate = "";
          if (game.substring(0, 8) === "<a href=")
            gameDate = game.split('html">', 2)[1].split("<", 1)[0];
          else
            gameDate = game.split("<", 1)[0];

          if ((new Date(gameDate)).getTime() > numDate) //This is not the date you're looking for
            continue;
          else if ((new Date(gameDate)).getTime() == numDate) { //Found the date

            nextGame["team"] = clan;
            nextGame["date"] = gameDate;
            nextGame["season"] = pivotYear - 2000 + 1;
            
            let opponent = game.split('data-stat="opp_name" csk="', 2)[1].substring(0, 3);
            if ((clan === team && opponent !== opp) || (clan === opp && opponent !== team)) {
              res.status(404).send("Game not found");
              data = [];
              return;
            }
            nextGame["opp"] = opponent;

            let gametype = "normal";
            if (!game.includes("losses_ot"))
              gametype = "playoffs";
            nextGame["type"] = gametype;

            let field = game.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
            if (field.includes("@"))
              field = "away";
            else
              field = "home";
            nextGame["field"] = field;

            nextGame["teamWins"] = null;
            nextGame["teamLosses"] = null;
            nextGame["teamTies"] = null;
            nextGame["teamOTL"] = null;
            nextGame["teamStreak"] = null;
            nextGame["teamStreakType"] = null;

            nextGame["oppWins"] = null;
            nextGame["oppLosses"] = null;
            nextGame["oppTies"] = null;
            nextGame["oppOTL"] = null;
            nextGame["oppStreak"] = null;
            nextGame["oppStreakType"] = null;

            finding = false;

          } else { //No such game exists
            res.status(404).send("Game not found");
            data = [];
            return;
          }

        } else { //Collect game info

          if (game.substring(0, 8) !== "<a href=") //Game has NOT occurred
            continue;

          previousGame["team"] = clan;

          let opponent = game.split('data-stat="opp_name" csk="', 2)[1].substring(0, 3);
          previousGame["opp"] = opponent;

          let gametype = "normal";
          if (!game.includes("losses_ot"))
            gametype = "playoffs";
          previousGame["type"] = gametype;
  
          let gameDate = "";
          if (game.substring(0, 8) === "<a href=")
            gameDate = game.split('html">', 2)[1].split("<", 1)[0];
          else
            gameDate = game.split("<", 1)[0];
          previousGame["date"] = gameDate;

          let field = game.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
          if (field.includes("@"))
            field = "away";
          else
            field = "home";
          previousGame["field"] = field;

          previousGame["teamScore"] = parseInt(game.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
          previousGame["oppScore"] = parseInt(game.split('data-stat="opp_goals" >', 2)[1].split("<", 1)[0]);

          let result = 0; //Streaks will be calculated after data collection
          if (previousGame["teamScore"] > previousGame["oppScore"]) {
            result = 1;
          } else if (previousGame["teamScore"] < previousGame["oppScore"]) {
            result = 0;
          } else {
            result = 0.5;
          }
          previousGame["result"] = result;

          nextGame["teamWins"] = parseInt(game.split('data-stat="wins" >', 2)[1].split("<", 1)[0]);
          nextGame["teamLosses"] = parseInt(game.split('data-stat="losses" >', 2)[1].split("<", 1)[0]);

          let ties = 0;
          if (pivotYear < 2006) //When ties were removed
            ties = parseInt(game.split('data-stat="ties" >', 2)[1].split("<", 1)[0]);
          nextGame["teamTies"] = ties;

          let overtimeLosses = 0;
          if (game.includes("losses_ot")) //Playoffs do not include overtime losses
            overtimeLosses = parseInt(game.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);
          nextGame["teamOTL"] = overtimeLosses;
          
          let gameLink = game.split('<a href="', 2)[1].split('"', 1)[0];


          await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
          let gameinfo = await axios.get("https://www.hockey-reference.com" + gameLink, config);
          while (gameinfo.status !== 200) {
            await new Promise((resolve) => setTimeout(resolve, 10000)); //10 seconds
            gameinfo = await axios.get("https://www.hockey-reference.com" + gameLink, config);
          }

          let players = gameinfo.data.split("header_scoring"); //Splits the players up, into two teams
          players = players.slice(1, 3);

          if (pivotYear > 2007) //When advanced stats become available; screws up some of the scraping
            players[1] = players[1].split("All Situations", 1)[0];

          if (field === "home") //In home games, the opposing team's stats are listed first
            players = players.reverse();

          let teamPlayers = [];
          let oppPlayers = [];

          for (let j = 0; j < 2; j++) { //Iterate through the player groups; add their stats

            let playerSplit = players[j].split("data-append-csv");
            playerSplit = playerSplit.slice(1, playerSplit.length);

            let modifier = 0; //Used if a player found is "invalid"
            
            for (let k = 0; k < playerSplit.length; k++) {

              let onePlayer = playerSplit[k];

              let goalie = false;
              if (onePlayer.includes('data-stat="decision" >') && !onePlayer.includes("TOTAL"))
                goalie = true;

              let name = onePlayer.split('.html">', 2)[1].split("<", 1)[0];
              let penaltyMin = parseInt(onePlayer.split('data-stat="pen_min" >', 2)[1].split("<", 1)[0]);

              //console.log((k + 1) + " " + name);
              
              let toi = onePlayer.split('data-stat="time_on_ice"', 2)[1].split(">", 2)[1].split("<", 1)[0].split(":", 2);
              toi = parseInt(toi[0]) + (parseInt(toi[1]) / 60); //Convert seconds to minutes; combine the two

              let shifts, plusMinus, goals, assists, shots = 0; //Specifically player data
              let goalsAgainst, shotsAgainst, shutouts = 0;

              if (!goalie) {
                shifts = parseInt(onePlayer.split('data-stat="shifts" >', 2)[1].split("<", 1)[0]);

                if (isNaN(shifts) || shifts === undefined) //Goalie, actually; invalid player, so skip
                  continue;

                plusMinus = parseInt(onePlayer.split('data-stat="plus_minus" >', 2)[1].split("<", 1)[0]);

                goals = parseInt(onePlayer.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
                assists = parseInt(onePlayer.split('data-stat="assists" >', 2)[1].split("<", 1)[0]);
                shots = parseInt(onePlayer.split('data-stat="shots" >', 2)[1].split("<", 1)[0]);

              } else {

                goalsAgainst = parseInt(onePlayer.split('data-stat="goals_against" >', 2)[1].split("<", 1)[0]);
                shotsAgainst = parseInt(onePlayer.split('data-stat="shots_against" >', 2)[1].split("<", 1)[0]);
                if (previousGame["oppScore"] === 0)
                  shutouts = 1;
              }

              let id = onePlayer.split('="', 2)[1].split('"', 1)[0];

              if (!(id in playerReference)) { //Record the players identification, in order to minimize the amount of get requests needed

                let playerInfoLink = "https://hockey-reference.com/players/" + id.substring(0, 1) + "/" + id + ".html";
                
                await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
                let playerInfo = await axios.get(playerInfoLink, config);
                while (playerInfo.status !== 200) {
                  await new Promise((resolve) => setTimeout(resolve, 10000)); //10 seconds
                  playerInfo = await axios.get(playerInfoLink, config);
                }
                
                let position = playerInfo.data.split("<strong>Position</strong>: ", 2)[1].split("<", 1)[0].split("\n", 1)[0].split("&", 1)[0];
                
                let shootsCatches = "";
                let shoots = playerInfo.data.split("<strong>Shoots</strong>: ", 2);
                let catches = playerInfo.data.split("<strong>Catches</strong>: ", 2);
                if (shoots.length > 1) //Some goalies have "shoots" data instead of catches
                    shootsCatches = shoots[1].split(' "', 1)[0].split("\n<", 1)[0];
                else if (catches.length > 1)
                    shootsCatches = catches[1].split(' "', 1)[0].split("\n<", 1)[0];

                physicalInfo = playerInfo.data.split("Born:", 1)[0].split("&nbsp;");

                let height = "";
                let weight = "";
                if (physicalInfo.length > 1) {
                    
                  height = physicalInfo[physicalInfo.length - 2].split("cm,", 2);
                  if (height.length > 1) //Sometimes both height and weight data is not listed
                    height = parseInt(height[0].split("(", 2)[1]);
                  else
                    height = "";

                  weight = physicalInfo[physicalInfo.length - 1].split("kg", 2);
                  if (weight.length > 1)
                    weight = parseInt(weight[0]);
                  else
                    weight = "";
                }

                let firstSeasonLink = playerInfo.data.split("Game-by-game stat line for the player.", 2)[1].split('<a href="', 2)[1].split('"', 1)[0];
                
                await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
                let firstSeason = await axios.get("https://hockey-reference.com" + firstSeasonLink, config); //Used to get the date of the first game, which is used to determine age and experience
                while (firstSeason.status !== 200) {
                  await new Promise((resolve) => setTimeout(resolve, 10000)); //10 seconds
                  firstSeason = await axios.get("https://hockey-reference.com" + firstSeasonLink, config);
                }
                
                firstSeason = firstSeason.data.split('data-stat="date_game" >', 2)[1];

                let firstGameDate = firstSeason.split(">", 2)[1].split("<", 1)[0];
                let firstGameAge = firstSeason.split('data-stat="age" >', 2)[1].split("<", 1)[0].split("-");
                firstGameAge = parseInt(firstGameAge[0]) + ((parseInt(firstGameAge[1]) / 365));


                playerReference[id] = {} //Add everything
                playerReference[id]["name"] = name;
                playerReference[id]["height"] = height;
                playerReference[id]["weight"] = weight;
                playerReference[id]["position"] = position;
                playerReference[id]["shootsCatches"] = shootsCatches;
                playerReference[id]["firstGameDate"] = firstGameDate;
                playerReference[id]["firstGameAge"] = firstGameAge;

              }
                
              let experience = (new Date(gameDate) - new Date(playerReference[id]["firstGameDate"])) / (1000 * 60 * 60 * 24 * 365); //Years from first NHL game played
              let age = playerReference[id]["firstGameAge"] + experience;

              playersAdd = teamPlayers;
              if (j === 1) //Opp players data
                playersAdd = oppPlayers;


              let ind = playersAdd.length; //End
              playersAdd.push({});
              playersAdd[ind]["name"] = name;
              playersAdd[ind]["TOI"] = toi;
              playersAdd[ind]["POS"] = playerReference[id]["position"];

              if (!goalie) {
                playersAdd[ind]["shifts"] = shifts;
                playersAdd[ind]["plusMinus"] = plusMinus;
                playersAdd[ind]["goals"] = goals;
                playersAdd[ind]["assists"] = assists;
                playersAdd[ind]["shots"] = shots;
              } else {
                playersAdd[ind]["GA"] = goalsAgainst;
                playersAdd[ind]["SA"] = shotsAgainst;
                playersAdd[ind]["shutouts"] = shutouts;
              }

              playersAdd[ind]["PIM"] = penaltyMin;
              playersAdd[ind]["age"] = age;
              playersAdd[ind]["EXP"] = experience;
              playersAdd[ind]["height"] = playerReference[id]["height"];
              playersAdd[ind]["weight"] = playerReference[id]["weight"];
              !goalie ? playersAdd[ind]["shoots"] = playerReference[id]["shootsCatches"] : playersAdd[ind]["catches"] = playerReference[id]["shootsCatches"];

            }
          }


          previousGame["teamPlayers"] = teamPlayers;
          previousGame["oppPlayers"] = oppPlayers;
          previousGame["season"] = pivotYear - 2000 + 1;

          data.push({});
          data[data.length - 1]["previousGame"] = previousGame;
          data[data.length - 1]["nextGame"] = nextGame;

          gamesRecorded[clan][gameDate] = data[data.length - 1]; //Record game for easier retrieval in the future

          previousGame = {}; //Clear both for more data
          nextGame = {};

          if (indGames < number_games - 1) { //Fill next game with data

            nextGame["team"] = clan;
            nextGame["date"] = gameDate;
            nextGame["season"] = pivotYear - 2000 + 1;

            nextGame["opp"] = opponent;
            nextGame["field"] = field;
            nextGame["type"] = gametype;

            nextGame["teamWins"] = null;
            nextGame["teamLosses"] = null;
            nextGame["teamTies"] = null;
            nextGame["teamOTL"] = null;
            nextGame["teamStreak"] = null;
            nextGame["teamStreakType"] = null;

            nextGame["oppWins"] = null;
            nextGame["oppLosses"] = null;
            nextGame["oppTies"] = null;
            nextGame["oppOTL"] = null;
            nextGame["oppStreak"] = null;
            nextGame["oppStreakType"] = null;
          }

          indGames++;

        }

      }


      if (indGames < number_games) {

        pivotYear--;
        if (pivotYear === 2005)
          pivotYear--;

        seasonLink = "https://www.hockey-reference.com/teams/" + clan + "/" + pivotYear + "_games.html";
        await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
        seasonInfo = await axios.get(seasonLink, config);

        if (seasonInfo.status !== 200) {
          res.status(404).send("Games not found")
          data = [];
          return;
        }

        let newAllGames = seasonInfo.data.split('date_game" >');
        newAllGames = newAllGames.slice(1, newAllGames.length);
        newAllGames = newAllGames.reverse();
        allGames.push(newAllGames);

        seasonsRecorded[clan][pivotYear] = newAllGames;
      }

    }
  }

  //console.log(gamesRecorded);
  //console.log(seasonsRecorded);

  //console.log(Object.keys(allGames[0]));

  for (indexe in clans) { //Retrieve next game opponent information, as well as streaks

    let clan = clans[indexe];
    for (y in seasonsRecorded[clan]) { //Years are placed in ascending integer order in a JS object
      
      let allGame = seasonsRecorded[clan][y];
      let streak = 0;
      let streakType = "";

      let gamesPlayed = 0;
      let playoffGamesPlayed = 0;

      let lastTeamHomeWins = 0;
      let lastTeamHomeLosses = 0;
      let lastTeamAwayWins = 0;
      let lastTeamAwayLosses = 0;

      let lastTeamWins = 0;
      let lastTeamLosses = 0;
      let lastTeamTies = 0;
      let lastTeamOTL = 0;

      let lastTeamGoalsFor = 0;
      let lastTeamGoalsAgainst = 0;

      for (let k = allGame.length - 1; k >= 0; k--) { //Start from the least recent games

        let game = allGame[k];

        let gameDate = "";
        if (game.substring(0, 8) === "<a href=")
          gameDate = game.split('html">', 2)[1].split("<", 1)[0];
        else
          gameDate = game.split("<", 1)[0];

        if (gameDate === date) //Reached the last game; no need to go on
          break;

        let teamScore = parseInt(game.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
        let oppScore = parseInt(game.split('data-stat="opp_goals" >', 2)[1].split("<", 1)[0]);

        let rslt = "T";
        if (teamScore > oppScore) {
          rslt = "W";
          lastTeamWins++;
        } else if (oppScore > teamScore) {
          rslt = "L";
          let OTLs = 0;
          if (game.includes("losses_ot")) {
            OTLs = parseInt(game.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);
            if (OTLs > lastTeamOTL)
              lastTeamOTL = OTLs;
            else
              lastTeamLosses++;
          } else
            lastTeamLosses++;
        }

        if (rslt === "T")
          lastTeamTies++;

        if (rslt !== streakType) {
          streakType = rslt;
          streak = 1;
        } else
          streak++;


        let field = game.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
        if (field.includes("@"))
          field = "away";
        else
          field = "home";

        let result = 0.5;
        if (rslt === "W")
          result = 1;
        else if (rslt === "L")
          result = 0;
    
        if (result === 1 && field === "home")
          lastTeamHomeWins += 1;
        else if (result === 0 && field === "home")
          lastTeamHomeLosses += 1;
        else if (result === 1 && field === "away")
          lastTeamAwayWins += 1;
        else if (result === 0 && field === "away")
          lastTeamAwayLosses += 1;

        lastTeamGoalsFor += teamScore;
        lastTeamGoalsAgainst += oppScore;


        gamesPlayed++;
        if (!game.includes("losses_ot"))
            playoffGamesPlayed++;

        if (gameDate in gamesRecorded[clan]) { //Means the game info was recorded

          gamesRecorded[clan][gameDate].nextGame.teamStreak = streak;
          gamesRecorded[clan][gameDate].nextGame.teamStreakType = streakType;

          gamesRecorded[clan][gameDate]["nextGame"]["teamHomeWins"] = lastTeamHomeWins;
          gamesRecorded[clan][gameDate]["nextGame"]["teamHomeLosses"] = lastTeamHomeLosses;
          gamesRecorded[clan][gameDate]["nextGame"]["teamAwayWins"] = lastTeamAwayWins;
          gamesRecorded[clan][gameDate]["nextGame"]["teamAwayLosses"] = lastTeamAwayLosses;

          gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamWins"] = 0;
          gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamLosses"] = 0;
          if (gamesRecorded[clan][gameDate]["previousGame"]["type"] === "playoffs") {
            gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamWins"] = gamesRecorded[clan][gameDate]["nextGame"]["teamWins"];
            gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamLosses"] = gamesRecorded[clan][gameDate]["nextGame"]["teamLosses"];

            gamesRecorded[clan][gameDate]["nextGame"]["teamWins"] = lastTeamWins;
            gamesRecorded[clan][gameDate]["nextGame"]["teamLosses"] = lastTeamLosses;
            gamesRecorded[clan][gameDate]["nextGame"]["teamTies"] = lastTeamTies;
            gamesRecorded[clan][gameDate]["nextGame"]["teamOTL"] = lastTeamOTL;
          }

          let points = (2 * gamesRecorded[clan][gameDate]["nextGame"]["teamWins"]) + gamesRecorded[clan][gameDate]["nextGame"]["teamTies"] + gamesRecorded[clan][gameDate]["nextGame"]["teamOTL"];
          let playoffPoints = (2 * gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamWins"]);
          gamesRecorded[clan][gameDate]["nextGame"]["teamPointsPercent"] = points / (gamesPlayed * 2);
          gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamPointsPercent"] = 0;
          if (playoffGamesPlayed > 0)
            gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamPointsPercent"] = playoffPoints / (playoffGamesPlayed * 2);

          gamesRecorded[clan][gameDate]["nextGame"]["teamAvgGoalsFor"] = lastTeamGoalsFor / gamesPlayed;
          gamesRecorded[clan][gameDate]["nextGame"]["teamAvgGoalsAgainst"] = lastTeamGoalsAgainst / gamesPlayed;

          gamesRecorded[clan][gameDate]["nextGame"]["teamGames"] = gamesPlayed;
          gamesRecorded[clan][gameDate]["nextGame"]["playoffTeamGames"] = playoffGamesPlayed;

          let opponent = gamesRecorded[clan][gameDate].nextGame.opp; //Retrieve opponent data

          //The team's next game (with the opponent) will either be in the current season (year), or
          //in the next one (if it is the end of the season.) The opponent's previous game, before the
          //next game we are trying to find, may be in that current season, the next one, or the season
          //before the current one. If the next game starts at the next season, fetch the season data
          //there, else, fetch the current season data. In both of these cases, if the next game we
          //are trying to find is the first one in the season, this means the previous game data (with
          //stuff we need like streak data and WLOTL) is in the previous season; fetch that data

          let prevSeason = gamesRecorded[clan][gameDate].previousGame.season;
          let nextSeason = gamesRecorded[clan][gameDate].nextGame.season;

          let startYear = 0;
          if (nextSeason - prevSeason > 0)
            startYear = nextSeason + 2000 - 1;
          else
            startYear = prevSeason + 2000 - 1;

          if (!(opponent in seasonsRecorded)) {

            seasonLink = "https://www.hockey-reference.com/teams/" + opponent + "/" + startYear + "_games.html";
            await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
            seasonInfo = await axios.get(seasonLink, config);

            if (seasonInfo.status !== 200) {
              res.status(404).send("Games not found")
              data = [];
              return;
            }

            let newAllGames = seasonInfo.data.split('date_game" >');
            newAllGames = newAllGames.slice(1, newAllGames.length);
            newAllGames = newAllGames.reverse();
            seasonsRecorded[opponent] = {};
            seasonsRecorded[opponent][startYear] = newAllGames;

          }

          let previousGameFound = false;
          let oppStreak = 0;
          let oppStreakType = "";
          let oppPreviousGame = "";

          let oppGamesPlayed = 0;
          let oppPlayoffGamesPlayed = 0;

          let lastOppHomeWins = 0;
          let lastOppHomeLosses = 0;
          let lastOppAwayWins = 0;
          let lastOppAwayLosses = 0;

          let lastOppWins = 0;
          let lastOppLosses = 0;
          let lastOppTies = 0;
          let lastOppOTL = 0;

          let lastOppGoalsFor = 0;
          let lastOppGoalsAgainst = 0;


          while (!previousGameFound) {

            if (!(startYear in seasonsRecorded[opponent])) {
              seasonLink = "https://www.hockey-reference.com/teams/" + opponent + "/" + startYear + "_games.html";
              await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
              seasonInfo = await axios.get(seasonLink, config);

              if (seasonInfo.status !== 200 || startYear === 1997) { //Cutoff date
                res.status(404).send("Games not found")
                data = [];
                return;
              }

              let newAllGames = seasonInfo.data.split('date_game" >');
              newAllGames = newAllGames.slice(1, newAllGames.length);
              newAllGames = newAllGames.reverse();
              seasonsRecorded[opponent][startYear] = newAllGames;
            }

            let probeSeason = seasonsRecorded[opponent][startYear];

            for (let n = probeSeason.length - 1; n >= 0; n--) { //Again, least recent game

              let oppGame = probeSeason[n];

              let oppGameDate = "";
              if (oppGame.substring(0, 8) === "<a href=")
                oppGameDate = oppGame.split('html">', 2)[1].split("<", 1)[0];
              else
                oppGameDate = oppGame.split("<", 1)[0];


              if (oppGameDate === gamesRecorded[clan][gameDate].nextGame.date) { //Found

                if (oppStreakType === "") { //Our worst nightmare; get the previous season

                  startYear--;
                  if (startYear === 2005)
                    startYear--;

                  seasonLink = "https://www.hockey-reference.com/teams/" + opponent + "/" + startYear + "_games.html";
                  await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
                  seasonInfo = await axios.get(seasonLink, config);

                  if (seasonInfo.status !== 200) {
                    res.status(404).send("Games not found")
                    data = [];
                    return;
                  }

                  let newAllGames = seasonInfo.data.split('date_game" >');
                  newAllGames = newAllGames.slice(1, newAllGames.length);
                  newAllGames = newAllGames.reverse();
                  seasonsRecorded[opponent][startYear] = newAllGames;

                  for (let m = newAllGames.length - 1; m >= 0; m--) { //Go from the least recent game to the most recent game - the one we want

                    let yetAnotherGame = newAllGames[m];

                    let tmScr = parseInt(yetAnotherGame.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
                    let opScr = parseInt(yetAnotherGame.split('data-stat="opp_goals" >', 2)[1].split("<", 1)[0]);

                    let oppRslt = "T";
                    if (tmScr > opScr) {
                      oppRslt = "W";
                      lastOppWins++;
                    } else if (opScr > tmScr) {
                      oppRslt = "L";
                      let OTLs = 0;
                      if (yetAnotherGame.includes("losses_ot")) {
                        OTLs = parseInt(yetAnotherGame.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);
                        if (OTLs > lastOppOTL)
                          lastOppOTL = OTLs;
                        else
                          lastOppLosses++;
                      } else
                        lastOppLosses++;
                    }

                    if (oppRslt === "T")
                      lastOppTies++;

                    if (oppRslt !== oppStreakType) {
                      oppStreakType = oppRslt;
                      oppStreak = 1;
                    } else
                      oppStreak++;

                    
                    let fild = yetAnotherGame.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
                    if (fild.includes("@"))
                      fild = "away";
                    else
                      fild = "home";

                    let oppResult = 0.5;
                    if (rslt === "W")
                      oppResult = 1;
                    else if (rslt === "L")
                      oppResult = 0;
                
                    if (oppResult === 1 && fild === "home")
                      lastOppHomeWins += 1;
                    else if (oppResult === 0 && fild === "home")
                      lastOppHomeLosses += 1;
                    else if (oppResult === 1 && fild === "away")
                      lastOppAwayWins += 1;
                    else if (oppResult === 0 && fild === "away")
                      lastOppAwayLosses += 1;
            
                    lastOppGoalsFor += tmScr;
                    lastOppGoalsAgainst += opScr;
            
            
                    oppGamesPlayed++;
                    if (!yetAnotherGame.includes("losses_ot"))
                        oppPlayoffGamesPlayed++;

                    oppPreviousGame = yetAnotherGame;
                  }
                }


                gamesRecorded[clan][gameDate].nextGame.oppStreak = oppStreak;
                gamesRecorded[clan][gameDate].nextGame.oppStreakType = oppStreakType;

                gamesRecorded[clan][gameDate].nextGame.oppWins = parseInt(oppPreviousGame.split('data-stat="wins" >', 2)[1].split("<", 1)[0]);
                gamesRecorded[clan][gameDate].nextGame.oppLosses = parseInt(oppPreviousGame.split('data-stat="losses" >', 2)[1].split("<", 1)[0]);

                let ties = 0;
                if (startYear < 2006) //When ties were removed
                  ties = parseInt(oppPreviousGame.split('data-stat="ties" >', 2)[1].split("<", 1)[0]);
                gamesRecorded[clan][gameDate].nextGame.oppTies = ties;

                let overtimeLosses = 0;
                if (oppPreviousGame.includes("losses_ot")) //Playoffs do not include overtime losses
                  overtimeLosses = parseInt(oppPreviousGame.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);
                gamesRecorded[clan][gameDate].nextGame.oppOTL = overtimeLosses;


                gamesRecorded[clan][gameDate]["nextGame"]["oppHomeWins"] = lastOppHomeWins;
                gamesRecorded[clan][gameDate]["nextGame"]["oppHomeLosses"] = lastOppHomeLosses;
                gamesRecorded[clan][gameDate]["nextGame"]["oppAwayWins"] = lastOppAwayWins;
                gamesRecorded[clan][gameDate]["nextGame"]["oppAwayLosses"] = lastOppAwayLosses;

                gamesRecorded[clan][gameDate]["nextGame"]["playoffOppWins"] = 0;
                gamesRecorded[clan][gameDate]["nextGame"]["playoffOppLosses"] = 0;
                if (gamesRecorded[clan][gameDate]["previousGame"]["type"] === "playoffs") {
                  gamesRecorded[clan][gameDate]["nextGame"]["playoffOppWins"] = gamesRecorded[clan][gameDate]["nextGame"]["oppWins"];
                  gamesRecorded[clan][gameDate]["nextGame"]["playoffOppLosses"] = gamesRecorded[clan][gameDate]["nextGame"]["oppLosses"];

                  gamesRecorded[clan][gameDate]["nextGame"]["oppWins"] = lastOppWins;
                  gamesRecorded[clan][gameDate]["nextGame"]["oppLosses"] = lastOppLosses;
                  gamesRecorded[clan][gameDate]["nextGame"]["oppTies"] = lastOppTies;
                  gamesRecorded[clan][gameDate]["nextGame"]["oppOTL"] = lastOppOTL;
                }

                let pts = (2 * gamesRecorded[clan][gameDate]["nextGame"]["oppWins"]) + gamesRecorded[clan][gameDate]["nextGame"]["oppTies"] + gamesRecorded[clan][gameDate]["nextGame"]["oppOTL"];
                let playoffPts = (2 * gamesRecorded[clan][gameDate]["nextGame"]["playoffOppWins"]);
                gamesRecorded[clan][gameDate]["nextGame"]["oppPointsPercent"] = pts / (oppGamesPlayed * 2);
                gamesRecorded[clan][gameDate]["nextGame"]["playoffOppPointsPercent"] = 0;
                if (playoffGamesPlayed > 0)
                  gamesRecorded[clan][gameDate]["nextGame"]["playoffOppPointsPercent"] = playoffPts / (oppPlayoffGamesPlayed * 2);

                gamesRecorded[clan][gameDate]["nextGame"]["oppAvgGoalsFor"] = lastOppGoalsFor / oppGamesPlayed;
                gamesRecorded[clan][gameDate]["nextGame"]["oppAvgGoalsAgainst"] = lastOppGoalsAgainst / oppGamesPlayed;

                gamesRecorded[clan][gameDate]["nextGame"]["oppGames"] = oppGamesPlayed;
                gamesRecorded[clan][gameDate]["nextGame"]["playoffOppGames"] = oppPlayoffGamesPlayed;

                previousGameFound = true;
                break;

              } else { //Not found; keep counting

                if (oppGame.substring(0, 8) !== "<a href=") //No data to collect (yet)
                  continue;

                let tmScr = parseInt(oppGame.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
                let opScr = parseInt(oppGame.split('data-stat="opp_goals" >', 2)[1].split("<", 1)[0]);

                let oppRslt = "T";
                if (tmScr > opScr) {
                  oppRslt = "W";
                  lastOppWins++;
                } else if (opScr > tmScr) {
                  oppRslt = "L";
                  let OTLs = 0;
                  if (oppGame.includes("losses_ot")) {
                    OTLs = parseInt(oppGame.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);
                    if (OTLs > lastOppOTL)
                      lastOppOTL = OTLs;
                    else
                      lastOppLosses++;
                  } else
                    lastOppLosses++;
                }

                if (oppRslt === "T")
                  lastOppTies++;

                if (oppRslt !== oppStreakType) {
                  oppStreakType = oppRslt;
                  oppStreak = 1;
                } else
                  oppStreak++;

                
                let fild = oppGame.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
                if (fild.includes("@"))
                  fild = "away";
                else
                  fild = "home";

                let oppResult = 0.5;
                if (rslt === "W")
                  oppResult = 1;
                else if (rslt === "L")
                  oppResult = 0;
            
                if (oppResult === 1 && fild === "home")
                  lastOppHomeWins += 1;
                else if (oppResult === 0 && fild === "home")
                  lastOppHomeLosses += 1;
                else if (oppResult === 1 && fild === "away")
                  lastOppAwayWins += 1;
                else if (oppResult === 0 && fild === "away")
                  lastOppAwayLosses += 1;
        
                lastOppGoalsFor += tmScr;
                lastOppGoalsAgainst += opScr;
        
        
                oppGamesPlayed++;
                if (!oppGame.includes("losses_ot"))
                    oppPlayoffGamesPlayed++;

                oppPreviousGame = oppGame;
              }

            }

            startYear--;
            if (startYear === 2005)
              startYear--;

          }

        }



      }
    }
  }

  /* //For debugging purposes
  for (gm in data) {
    console.log("\n" + gm)
    for (attrib in data[gm]["previousGame"])
      console.log(attrib + ": " + data[gm]["previousGame"][attrib]);
    console.log("\n");
    for (attrib in data[gm]["nextGame"])
      console.log(attrib + ": " + data[gm]["nextGame"][attrib]);
  }
  */

  res.status(201).send("Data retrieved");
  return;

});



//********************************************************************
//Converts the data recieved into model data and sends it
//********************************************************************
app.get("/crunch", function (req, res) {

  if (data.length == 0) {
    res.status(409).send("No data to send")
    return;
  }

  let usePlayers = true;
  if ("noUsePlayers" in req.body)
    usePlayers = false;

  let previousTeamGames = data.slice(0, GAME_SIZE).reverse();
  let previousOppGames = data.slice(GAME_SIZE, data.length).reverse();

  past = functions.crunchGames(previousTeamGames, previousOppGames, order, false, false)[0];

  data = []; //Clear for further requests
  res.status(200).send(JSON.stringify(past));

});



app.listen(3000, '127.0.0.1', function()
{
	console.log('Server running!');
});
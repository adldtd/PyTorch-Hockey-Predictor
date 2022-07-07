const axios = require("axios").default;
const fs = require("fs");

  /****************************************************************************/
 /* Scrapes all of the games from the different teams in the teams json file */
/****************************************************************************/

var games = [];
/*STRUCTURE:
  PreviousGame: object:
    Team: 3 letter str
    Players: list of lists, the latter which includes:
    Name: string (will probably be removed during training)
    TOI: float, in minutes
    POS: string
    Shifts: integer
    PlusMinus: integer
    Goals: integer
    Assists: integer
    Shots: integer
    PenaltyMin: integer
    Age: float, in years
    Experience: float; calculated by (current age) - (age at first game)
    Weight: integer
    Shoots: string
    Goalies: list of lists, the latter which includes:
    Name: string (will probably be removed during training)
    TOI: float, in minutes
    GA: integer
    SA: integer
    Shutouts: integer; cumulative from the start of the season
    PenaltyMin: integer
    Age: float, in years
    Weight: integer
    Catches: string
    Opp: 3 letter str
    Players: same as before
    Goalies: same as before
    TeamScore: integer
    OppScore: integer
    Result: float, with 1 being victory, 0 being loss, and 0.5 being tie
    Field: string, either home or away
    Type: string, either normal, playoff, or final
    Date: string
    Season: integer, with 1 being the season in 2000
  NextGame: object:
    Team: 3 letter str, same as before
    Wins: integer
    Losses: integer
    Ties: integer
    OTL: integer
    Streak: integer
    Opp: 3 letter str, might be different
    Wins: integer
    Losses: integer
    Ties: integer
    OTL: integer
    Streak: integer
    Field: string, either home or away
    Type: string, either normal, playoff, or final
    Date: string
*/

var playerReference = {}; //Uses player id as key, and an object as the contents, containing name, weight, position, age at first game, and date of first game
var listOfTeams = {};
const TIME_OUT = 100; //How many ms to wait before pinging the server again

//games = JSON.parse(fs.readFileSync("data/games" + "CAR" + ".json"));
playerReference = JSON.parse(fs.readFileSync("data/playerReference.json"));
listOfTeams = JSON.parse(fs.readFileSync("data/teams.json"));

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


(async () => { //Main

  const START = 2000; //Year to start
  let ignoreList = {"PHI":"", "TBL":"", "COL":"", "MDA":"", "NYR":"", "OTT":"", "ANA":"", "ATL":"", "CAR":"", "WPG":"", "NJD":"", "PIT":"", "TOR":"", "NYI":"", "MTL":"", "BUF":"", "FLA":"", "BOS":"", "WSH":"", "DET":"", "STL":"", "CHI":"", "NSH":"", "VAN":"", "SJS":"", "PHX":"", "LAK":"", "EDM":"", "DAL":"", "CGY":"", "CBJ":"", "MIN":"", "ARI":"", "VEG":"", "SEA":""};
  let special = []; //Teams that need to be worked with manually

  for (team in listOfTeams) {

    if (team in ignoreList)
      continue;

    console.log(team);
    await new Promise((resolve) => setTimeout(resolve, 5000));

    let year = START;

    switch (team) { //Special cases
      case "ANA": //Anaheim Ducks
        year = 2007;
        break;
      case "WPG": //Winnipeg Jets
        year = 2012;
        break;
      case "CBJ": //Columbus Blue Jackets
        year = 2001;
        break;
      case "MIN": //Minnesota Wild
        year = 2001;
        break;
      case "ARI": //Arizona Coyotes
        year = 2015;
        break;
      case "VEG": //Las Vegas Golden Knights
        year = 2018;
        break;
      case "SEA": //Seattle Kraken
        year = 2022;
        break;
      default:
        year = START;
        break;
    }

    let link = "https://www.hockey-reference.com/teams/" + team + "/" + year + "_games.html";
    let season = year - 2000 + 1;

    let resp = await axios.get(link, config);
    
    while (resp.status === 200) {

      let allGames = resp.data.split('date_game" >');
      allGames = allGames.slice(1, allGames.length);

      let streakType = "";
      let streak = 0; //Changes as wins, losses, or ties rack up

      for (let i = 0; i < allGames.length; i++) { //Iterate through all of the games

        let game = allGames[i];
        //console.log(game);

        let opp = game.split('data-stat="opp_name" csk="', 2)[1].substring(0, 3);
        let gametype = "normal";
        if (!game.includes("losses_ot"))
          gametype = "playoffs";

        let gameDate = "";
        if (game.substring(0, 8) === "<a href=")
          gameDate = game.split('html">', 2)[1].split("<", 1)[0];
        else
          gameDate = game.split("<", 1)[0];
        let field = game.split('data-stat="game_location" >', 2)[1].split("/", 1)[0];
        if (field.includes("@"))
          field = "away";
        else
          field = "home";

        console.log(gameDate);


        if (game.substring(0, 8) === "<a href=") { //This element indicates that the game has already occurred

          let teamscore = parseInt(game.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
          let oppscore = parseInt(game.split('data-stat="opp_goals" >', 2)[1].split("<", 1)[0]);
          
          let result = 0;
          let newStreakType = "";
          if (teamscore > oppscore) {
            result = 1;
            newStreakType = "W"
          } else if (teamscore < oppscore) {
            result = 0;
            newStreakType = "L";
          } else {
            result = 0.5;
            newStreakType = "T";
          }

          if (newStreakType !== streakType) {
            streakType = newStreakType;
            streak = 1; //Reset streak
          } else
            streak++;

          let wins = parseInt(game.split('data-stat="wins" >', 2)[1].split("<", 1)[0]);
          let losses = parseInt(game.split('data-stat="losses" >', 2)[1].split("<", 1)[0]);
          let ties = 0;
          if (year < 2006) //When ties were removed
            ties = parseInt(game.split('data-stat="ties" >', 2)[1].split("<", 1)[0]);
          let overtimeLosses = 0;
          if (game.includes("losses_ot"))
            overtimeLosses = parseInt(game.split('data-stat="losses_ot" >', 2)[1].split("<", 1)[0]);

          let gameLink = game.split('<a href="', 2)[1].split('"', 1)[0];

          await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
          let gameinfo = await axios.get("https://www.hockey-reference.com" + gameLink, config);
          while (gameinfo.status !== 200) {
            console.log("\n" + gameinfo.data + "\n");
            await new Promise((resolve) => setTimeout(resolve, 10000)); //10 seconds
            gameinfo = await axios.get("https://www.hockey-reference.com" + gameLink, config);
          }

          let players = gameinfo.data.split("header_scoring"); //Splits the players up, into two teams
          players = players.slice(1, 3);

          if (year > 2007) //When advanced stats become available; screw up some of the scraping
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

              //if (playerSplit[k].split(">", 2)[1].substring(0, 8) === "<a href="); //In case the player does not have their own page
              let name = onePlayer.split('.html">', 2)[1].split("<", 1)[0];
              //console.log(onePlayer);
              //console.log(onePlayer.split('pen', 2));
              let penaltyMin = parseInt(onePlayer.split('data-stat="pen_min" >', 2)[1].split("<", 1)[0]);

              console.log((k + 1) + " " + name);
              
              let toi = onePlayer.split('data-stat="time_on_ice"', 2)[1].split(">", 2)[1].split("<", 1)[0].split(":", 2);
              toi = parseInt(toi[0]) + (parseInt(toi[1]) / 60); //Convert seconds to minutes; combine the two

              let shifts, plusMinus, goals, assists, shots = 0; //Specifically player data
              let goalsAgainst, shotsAgainst, shutouts = 0;

              if (!goalie) {
                shifts = parseInt(onePlayer.split('data-stat="shifts" >', 2)[1].split("<", 1)[0]);

                if (isNaN(shifts) || shifts === undefined) //Goalie, actually; invalid player, skip
                  continue;

                plusMinus = parseInt(onePlayer.split('data-stat="plus_minus" >', 2)[1].split("<", 1)[0]);

                goals = parseInt(onePlayer.split('data-stat="goals" >', 2)[1].split("<", 1)[0]);
                assists = parseInt(onePlayer.split('data-stat="assists" >', 2)[1].split("<", 1)[0]);
                shots = parseInt(onePlayer.split('data-stat="shots" >', 2)[1].split("<", 1)[0]);

              } else {

                goalsAgainst = parseInt(onePlayer.split('data-stat="goals_against" >', 2)[1].split("<", 1)[0]);
                shotsAgainst = parseInt(onePlayer.split('data-stat="shots_against" >', 2)[1].split("<", 1)[0]);
                if (oppscore === 0)
                  shutouts = 1;
              }

              let id = onePlayer.split('="', 2)[1].split('"', 1)[0];

              if (!(id in playerReference)) { //Record the players identification, in order to minimize the amount of get requests needed

                let playerInfoLink = "https://hockey-reference.com/players/" + id.substring(0, 1) + "/" + id + ".html";
                
                await new Promise((resolve) => setTimeout(resolve, TIME_OUT));
                let playerInfo = await axios.get(playerInfoLink, config);
                while (playerInfo.status !== 200) {
                  console.log("\n" + playerInfo.data + "\n");
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
                  console.log("\n" + firstSeason.data + "\n");
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

            /*
            console.log("\n");
            for (ind in teamPlayers) {
              for (key in teamPlayers[ind])
                console.log(key + ": " + teamPlayers[ind][key]);
              console.log("\n");
            }
            */

          }


          let indi = games.length;
          games.push({});

          games[indi]["previousGame"] = {};
          games[indi]["previousGame"]["team"] = team;
          games[indi]["previousGame"]["teamPlayers"] = teamPlayers;
          games[indi]["previousGame"]["opp"] = opp;
          games[indi]["previousGame"]["oppPlayers"] = oppPlayers;
          games[indi]["previousGame"]["teamScore"] = teamscore;
          games[indi]["previousGame"]["oppScore"] = oppscore;
          games[indi]["previousGame"]["result"] = result;
          games[indi]["previousGame"]["field"] = field;
          games[indi]["previousGame"]["type"] = gametype;
          games[indi]["previousGame"]["date"] = gameDate;
          games[indi]["previousGame"]["season"] = season;

          games[indi]["nextGame"] = {};
          games[indi]["nextGame"]["team"] = team;
          games[indi]["nextGame"]["teamWins"] = wins;
          games[indi]["nextGame"]["teamLosses"] = losses;
          games[indi]["nextGame"]["teamTies"] = ties;
          games[indi]["nextGame"]["teamOTL"] = overtimeLosses;
          games[indi]["nextGame"]["teamStreak"] = streak;
          games[indi]["nextGame"]["teamStreakType"] = streakType;
          games[indi]["nextGame"]["opp"] = null; //These values are not known, until the program iterates once more OR merges two games together
          games[indi]["nextGame"]["oppWins"] = null;
          games[indi]["nextGame"]["oppLosses"] = null;
          games[indi]["nextGame"]["oppTies"] = null;
          games[indi]["nextGame"]["oppOTL"] = null;
          games[indi]["nextGame"]["oppStreak"] = null;
          games[indi]["nextGame"]["oppStreakType"] = null;
          games[indi]["nextGame"]["field"] = null;
          games[indi]["nextGame"]["type"] = null;
          games[indi]["nextGame"]["date"] = null;
          games[indi]["nextGame"]["season"] = null;

          if (games.length > 1) { //Add data to the last next game

            games[indi - 1]["nextGame"]["opp"] = opp;
            games[indi - 1]["nextGame"]["field"] = field;
            games[indi - 1]["nextGame"]["type"] = gametype;
            games[indi - 1]["nextGame"]["date"] = gameDate;
            games[indi - 1]["nextGame"]["season"] = season;
          }
          
        } else { //Game has not occurred yet; stop iterating

          let indi = games.length;

          games[indi - 1]["nextGame"]["opp"] = opp;
          games[indi - 1]["nextGame"]["field"] = field;
          games[indi - 1]["nextGame"]["type"] = gametype;
          games[indi - 1]["nextGame"]["date"] = gameDate;
          games[indi - 1]["nextGame"]["season"] = season;
          break;
        }

        //process.exit(0);
      }

      year++;
      season++;

      if (year === 2005) {
        year++;
        season++;
      }

      link = "https://www.hockey-reference.com/teams/" + team + "/" + year + "_games.html";
      resp = await axios.get(link, config);

      fs.writeFileSync("data/games" + team + ".json", JSON.stringify(games, null, 2));
      fs.writeFileSync("data/playerReference.json", JSON.stringify(playerReference, null, 2)); //Save before continuing

    }

    console.log(resp.headers);
    console.log(resp.data);
    console.log(resp.status);

    if (year !== 2023)
      special.push(team);

    games = [];
  }

  for (let i = 0; i < special.length; i++)
    console.log(special[i]);

})();
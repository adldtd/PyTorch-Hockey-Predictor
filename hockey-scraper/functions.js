
module.exports = {

    crunchGames: function (previousTeamGames, previousOppGames, order, positions, returnResults = true, usePlayers = false) {

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


        let game = previousTeamGames[previousTeamGames.length - 1]; //Contains the next game to be predicted
        let nextGameDate = new Date(game["nextGame"]["date"]);
        let nextGameSeason = game["nextGame"]["season"];

        let chunk = [];
        let past = []; //Made up of all past games
        let output = [];

        for (gm in previousGames) {

        let single = []; //Represents a single game

        let team = (new Array(35)).fill(0);
        team[order[previousGames[gm]["previousGame"]["team"]]] = 1; //Equivalent to one-hot encoding
        single = single.concat(team);
        
        let teamFielders = [];
        let teamGoalies = [];
        let numTeamFielders = 0;
        let numTeamGoalies = 0;


        if (usePlayers) { //*********************************************************************************

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
                    playerInfo[26] = (height - 185) / 100;
                let weight = playerStats["weight"];
                if (weight === "" || weight === undefined || weight === null)
                    playerInfo[27] = 0;
                else
                    playerInfo[27] = (weight - 91) / 100;
                
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
                    playerInfo[7] = (height - 185) / 100;
                let weight = playerStats["weight"];
                if (weight === "" || weight === undefined || weight === null)
                    playerInfo[8] = 0;
                else
                    playerInfo[8] = (weight - 91) / 100;

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
        } //*********************************************************************************


        let opp = (new Array(35)).fill(0);
        opp[order[previousGames[gm]["previousGame"]["opp"]]] = 1;
        single = single.concat(opp);
        
        let oppFielders = [];
        let oppGoalies = [];
        let numOppFielders = 0;
        let numOppGoalies = 0;
        

        if (usePlayers) { //*********************************************************************************

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
                    playerInfo[26] = (height - 185) / 60;
                let weight = playerStats["weight"];
                if (weight === "" || weight === undefined || weight === null)
                    playerInfo[27] = 0;
                else
                    playerInfo[27] = (weight - 91) / 60;
                
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
                    playerInfo[7] = (height - 185) / 100;
                let weight = playerStats["weight"];
                if (weight === "" || weight === undefined || weight === null)
                    playerInfo[8] = 0;
                else
                    playerInfo[8] = (weight - 91) / 100;

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
        } //*********************************************************************************


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

        past.push(single);

        if (returnResults) {
            let target = previousGames[gm]["nextGame"]["result"];
            if (target === 1)
                output.push([1, 0]); //Target probability of the first team winning vs the second
            else if (target === 0)
                output.push([0, 1]);
            else
                output.push([0.5, 0.5]); //Tie game
        }

        }

        chunk.push(past); //Array of objects
        if (returnResults)
        chunk.push(output); //Array of arrays

        return chunk;
    }

}
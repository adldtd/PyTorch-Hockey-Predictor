const axios = require("axios").default;
const fs = require("fs");
const {exit} = require("process");
  
  /***************************************************************************************/
 /* Creates a list of NHL teams existing from the 2000 season to the current one (2022) */
/***************************************************************************************/

const START = 2000; //Start in the 2000 season
var year = START;
var teams = {}; //Object; the 3 letter team indicator is the key, and the team name (and/or a related 3 letter team indicator) corresponds

var config =
{
  headers:
  {},
  timeout: 25000,
  responseType: "html",
  responseEncoding: "utf-8",
  validateStatus: null, //Accept all statuses; 404 might occur
};


(async () => { //Main

  res = await axios.get("https://www.hockey-reference.com/leagues/NHL_" + year + ".html", config);
  
  while (res.status !== 404) {

    console.log(year);

    data = res.data.split('data-stat="team_name" csk="');
    data = data.slice(1, data.length);
    for (let i = 0; i < data.length; i++) {
    
      let elem = data[i];
      if (elem.substring(0, 3) !== "Zzz") {
        
        elem = elem.split('href=', 2)[1].split("</a>", 2)[0];
        let indicator = elem.split("/teams/", 2)[1].split("/" + year, 2)[0];
        let name = elem.split(">", 2)[1];
        
        if (!(indicator in teams)) {

          teams[indicator] = [name, []]; //Team name; arenas; future indicator
          
          resSeason = await axios.get("https://www.hockey-reference.com/teams/" + indicator + "/" + year + ".html", config);

          let hasNextSeason = true;
          let link = "/teams/" + indicator + "/" + year + ".html"; //Keeps track of the last link 

          while (hasNextSeason) { //Meant to resolve name changes; make sure the program recognizes the same team

            console.log(resSeason.request["path"]);
            if (resSeason.data === undefined || resSeason.status > 200) {
              console.log(resSeason.status);
              console.log(resSeason.data);
              await new Promise((resolve) => setTimeout(resolve, 30000)); //Safety wait; 30 seconds
              resSeason = await axios.get("https://www.hockey-reference.com" + link, config);
              continue;
            } else
              await new Promise((resolve) => setTimeout(resolve, 400));

            if (resSeason.data.includes("Primary Arena")) { //Sometimes the arena is unlisted
              let arena = resSeason.data.split("Primary Arena:</strong>", 2)[1].split('>', 2)[1].split("</a", 2)[0];
              if (!teams[indicator][1].includes(arena)) {
                teams[indicator][1].push(arena);
                console.log(arena);
              }
            }

            let seasonData = resSeason.data.split("prevnext", 2)[1].split("</div>", 2)[0];

            if (seasonData.includes("Previous Season"))
              seasonData = seasonData.split("</a>")[1];

            if (seasonData.includes("Next Season")) { //Has not reached the current season
              
              link = seasonData.split('href="', 2)[1].split('"', 2)[0];
              let nextIndicator = link.split("/teams/", 2)[1].split("/", 2)[0];

              if (nextIndicator !== indicator) {
                teams[indicator].push(nextIndicator); //Creates a "chain" of names, starting from the oldest indicator to the newest
                hasNextSeason = false; //Can stop here
              } else
                resSeason = await axios.get("https://www.hockey-reference.com" + link, config);
            } else
              hasNextSeason = false;
            
          }
          
        }
      }
    }

    year++;

    if (year === 2005)
      year++;

    res = await axios.get("https://www.hockey-reference.com/leagues/NHL_" + year + ".html", config);
  }

  teams = JSON.stringify(teams, null, 2);
  console.log(teams);
  fs.writeFileSync("data/teams.json", teams);

})();
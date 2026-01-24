const express = require("express");
const os = require("os");

const app = express();
const PORT = 5000;


function sleep(ms){
    return new Promise (resolve => setTimeout(resolve,ms));
}


app.get("/api",async(req,res)=>{
    
    const start = Date.now();

    let sum = 0;
    for ( let i = 0 ; i < 3e9; i ++) sum += 1;

    await sleep(50 + Math.random() * 150);

    
    console.log(`[${new Date().toISOString()}] ${os.hostname()} handled request`);
    
    
    const responseTime = Date.now() - start;
    res.json({
        message: "request received",
        hostname: os.hostname(),
        sum: sum,
        pid: process.pid,
        responseTime: responseTime,
        timeStamp: new Date().toISOString()

    });

});



app.get("/health", (req,res)=>{
    res.send("OK");
})



app.listen(PORT, ()=>{
    console.log("Server is listening at PORT", PORT);
});
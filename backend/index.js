const { timeStamp } = require("console");
const express = require("express");
const os = require("os");

const app = express();
const PORT = 5000;


function sleep(ms){
    return new Promise (resolve => setTimeout(resolve,ms));
}


app.get("/home",async(req,res)=>{

    const start = Date.now();
    await sleep(50 + Math.random() * 150);

    const responseTime = Date.now() - start;

    console.log(`[${new Date().toISOString()}] ${os.hostname()} handled request`);


    res.json({
        message: "request received",
        hostname: os.hostname(),
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
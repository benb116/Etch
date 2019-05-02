var express     = require('express');
var compression = require('compression');
var helmet      = require('helmet');
var bodyParser  = require('body-parser');
var path        = require('path');

var ben = require('ben-jsutils');
ben.logTS();

// Set express settings
var app = express();
app.use(compression());
app.use(helmet()); // Hides certain HTTP headers, supposedly more secure?
app.use(helmet.noCache());
app.use(bodyParser.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded
app.use(express.static(path.join(__dirname, 'public')));

// Start the server
var listenPort = 4000;
app.listen(listenPort, function() {
    console.log("Node app is running on port "+listenPort+". Better go catch it.");
});

app.get('/', function(req, res) { res.sendFile(path.join(__dirname+'/public/screen.html')); });
    

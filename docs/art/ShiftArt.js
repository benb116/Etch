const ben = require('ben-jsutils');
const fs = require('fs');

const filename = './Phillies.json';
const data = require(filename);

const canvasSize = [1280, 700];
const border = 100;

let [x, y] = ben.arr.transpose(data.points);
let maxwidth = ben.arr.max(x)[0] - ben.arr.min(x)[0];
let maxheight = ben.arr.max(y)[0] - ben.arr.min(y)[0];
console.log(maxwidth, maxheight );
let widthScale = 1;
let heightScale = 1;
if (maxwidth > canvasSize[0]-2*border) {
    widthScale = (canvasSize[0]-2*border) / maxwidth;
}
if (maxheight > canvasSize[1]-2*border) {
    heightScale = (canvasSize[1]-2*border) / maxheight;
}
console.log(widthScale, heightScale);
let scale = Math.min(widthScale, heightScale);
console.log(x[0]);
x = x.map((xc) => xc * scale);
y = y.map((yc) => yc * scale);
console.log(x[0]);
maxwidth = ben.arr.max(x)[0] - ben.arr.min(x)[0];
maxheight = ben.arr.max(y)[0] - ben.arr.min(y)[0];

let offCenterX = ben.arr.min(x)[0] + (maxwidth / 2) - canvasSize[0]/2;
let offCenterY = ben.arr.min(y)[0] + (maxheight / 2) - canvasSize[1]/2;

let xS = x.map((xc) => Math.round(xc - offCenterX));
let yS = y.map((yc) => Math.round(yc - offCenterY));
console.log(xS[0]);
let pts = ben.arr.transpose([xS, yS]);
ben.logJSON(pts);
UpdateFile(pts);
console.log(ben.arr.sum(diff(data.points)));

function diff(a) {
    var a2 = a.slice();
    var b = a.slice();
    a2.shift();
    b.pop();
    return a2.map(function(e, i) {
        return Math.sqrt(Math.pow(e[0] - b[i][0], 2) + Math.pow(e[1] - b[i][1], 2));
    });
}

function UpdateFile(pts) {
    data.points = pts;
    return new Promise(function(resolve, reject) {
        fs.writeFile(filename, JSON.stringify(data), 'utf8', function() {
            resolve('Success');
        });
    }).catch(function(e) {
        resolve('Error updating config file:', e);
    });
}
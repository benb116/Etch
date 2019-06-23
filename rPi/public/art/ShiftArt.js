const ben = require('ben-jsutils');

const filename = './apple.json';
const data = require(filename);

const canvasSize = [1280, 700];

let [x, y] = ben.arr.transpose(data.points);
console.log(data.points);
let maxwidth = ben.arr.max(x)[0] - ben.arr.min(x)[0];
let maxheight = ben.arr.max(y)[0] - ben.arr.min(y)[0];

let offCenterX = ben.arr.min(x)[0] + (maxwidth / 2) - canvasSize[0]/2;
let offCenterY = ben.arr.min(y)[0] + (maxheight / 2) - canvasSize[1]/2;

let xS = x.map((xc) => Math.round(xc - offCenterX));
let yS = y.map((yc) => Math.round(yc - offCenterY));

let pts = ben.arr.transpose([xS, yS])
ben.logJSON(pts);
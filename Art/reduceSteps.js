// A simple JS file that does the following to a points json file:
// "simplifies" the path (removes collinear steps)
// Scales the points to fit on the screen

const ben = require('ben-jsutils');
const simplify = require('simplify-path');

const filename = process.argv[2] || 'David';
const filepath = '../rPi/public/art/'+filename+'.json';
let data = require(filepath);

console.log(data.points.length);
const path = simplify(data.points, 0);
data.points = path;
console.log(data.points.length);

const screensize = [1500, 865];
const screenmid = [Math.round(screensize[0]/2), Math.round(screensize[1]/2)];

function getMax(arr) {
    let len = arr.length;
    let max = -Infinity;

    while (len--) {
        max = arr[len] > max ? arr[len] : max;
    }
    return max;
}

function getMin(arr) {
    let len = arr.length;
    let min = Infinity;

    while (len--) {
        min = arr[len] < min ? arr[len] : min;
    }
    return min;
}

const xy = ben.arr.transpose(data.points);
let maxX = getMax(xy[0]);
let maxY = getMax(xy[1]);
let minX = getMin(xy[0]);
let minY = getMin(xy[1]);

console.log('maxes');
console.log(maxX, minX);
console.log(maxY, minY);

const bigX = (maxX - minX) / (screensize[0]-80);
const bigY = (maxY - minY) / (screensize[1]-80);
console.log(bigX, bigY);
const big = Math.max(bigX, bigY, 0);
console.log('big', big);
if (big > 0) {
    xy[0] = xy[0].map(e => (e / big));
    xy[1] = xy[1].map(e => (e / big));
}
maxX = getMax(xy[0]);
maxY = getMax(xy[1]);
minX = getMin(xy[0]);
minY = getMin(xy[1]);
const avgX = Math.round((maxX + minX) / 2);
const avgY = Math.round((maxY + minY) / 2);
const difX = avgX - screenmid[0];
const difY = avgY - screenmid[1];
console.log(maxX, minX, avgX, difX);
console.log(maxY, minY, avgY, difY);

const newX = xy[0].map(e => ((e - difX)));
const newY = xy[1].map(e => ((e - difY)));
const newpts = ben.arr.transpose([newX, newY]);
data.points = newpts;

ben.fs.writeFile(filepath, JSON.stringify(data));
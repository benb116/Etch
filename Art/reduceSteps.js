const ben = require('ben-jsutils');
const simplify = require('simplify-path');

const filename = process.argv[2] || 'David';
const filepath = '../rPi/public/art/'+filename+'.json';
let data = require(filepath);

console.log(data.points.length);
const path = simplify(data.points, 1);
data.points = path;
console.log(data.points.length);

const screensize = [1000, 700];
const screenmid = [Math.round(screensize[0]/2), Math.round(screensize[1]/2)];

const xy = ben.arr.transpose(data.points);
let maxX = ben.arr.max(xy[0])[0];
let maxY = ben.arr.max(xy[1])[0];
let minX = ben.arr.min(xy[0])[0];
let minY = ben.arr.min(xy[1])[0];

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
maxX = ben.arr.max(xy[0])[0];
maxY = ben.arr.max(xy[1])[0];
minX = ben.arr.min(xy[0])[0];
minY = ben.arr.min(xy[1])[0];
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
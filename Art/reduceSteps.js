const ben = require('ben-jsutils');
const simplify = require('simplify-path');

const filename = process.argv[2] || 'Sweet2';
const filepath = '../rPi/public/art/'+filename+'.json';
let data = require(filepath);

console.log(data.points.length);
const path = simplify(data.points, 1);
data.points = path;
console.log(data.points.length);

const screensize = [1280, 700];
const screenmid = [Math.round(screensize[0]/2), Math.round(screensize[1]/2)];

const xy = ben.arr.transpose(data.points);
const maxX = ben.arr.max(xy[0])[0];
const maxY = ben.arr.max(xy[1])[0];
const minX = ben.arr.min(xy[0])[0];
const minY = ben.arr.min(xy[1])[0];
const avgX = Math.round((maxX + minX) / 2);
const avgY = Math.round((maxY + minY) / 2);
const difX = avgX - screenmid[0];
const difY = avgY - screenmid[1];
console.log(maxY, minY, avgY, difY);

const newX = xy[0].map(e => e - difX);
const newY = xy[1].map(e => e - difY);
const newpts = ben.arr.transpose([newX, newY]);
data.points = newpts;

ben.fs.writeFile(filepath, JSON.stringify(data));
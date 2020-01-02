const ben = require('ben-jsutils');
const simplify = require('simplify-path');

const filename = 'Knot';
const filepath = '../rPi/public/art/'+filename+'.json';
let data = require(filepath);
console.log(data.points.length);
const path = simplify(data.points, 1);
data.points = path;
console.log(data.points.length);
ben.fs.writeFile(filepath, JSON.stringify(data))
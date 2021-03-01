import math from 'mathjs';

export function randn_bm() {
  var u = 0, v = 0;
  while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function initializeThetas(size1 = 2, size2 = 2, totalNum = 2) {
  let final = [];
  for (var x = 0; x < size1; x++) {
    let temp = [];
    for (var y = 0; y < size2; y++) {
      let num = (randn_bm() * (Math.random() * (totalNum - 1))) * Math.sqrt(2 / (totalNum - 1));

      //randn_bm
      temp.push(num);
    }
    final.push(temp);
  }
  return final;
}

export function initializeThetasVector(size1 = 2, size2 = 2, totalNum = 2) {
  let final = [];
  for (var x = 0; x < size1; x++) {
    for (var y = 0; y < size2; y++) {
      let num = (randn_bm() * (Math.random() * (totalNum - 1))) * Math.sqrt(2 / (totalNum - 1));
      final.push(num);
    }
  }
  return final;
}

export function sigmoid(t) {
  return 1 / (1 + Math.pow(Math.E, -t));
}

export function convertYandVector(matrix) {

  let data = matrix;
  let newArr = [];

  for (let i = 0; i < data.length; i++) {
    switch (data[i].join('')) {
      case '1000000000':
        newArr.push(0);
        break;
      case '0100000000':
        newArr.push(1);
        break;
      case '0010000000':
        newArr.push(2);
        break;
      case '0001000000':
        newArr.push(3);
        break;
      case '0000100000':
        newArr.push(4);
        break;
      case '0000010000':
        newArr.push(5);
        break;
      case '0000001000':
        newArr.push(6);
        break;
      case '0000000100':
        newArr.push(7);
        break;
      case '0000000010':
        newArr.push(8);
        break;
      case '0000000001':
        newArr.push(9);
        break;
      default:
        console.log(data[i].join(''));
        break;
    }
  }
  if (newArr.length !== data.length) {
    console.log('mismatched');
    return 'Error';

  }

  return newArr;
}
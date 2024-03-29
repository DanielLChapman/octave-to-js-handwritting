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
  let x = Math.E;
  let y = Math.pow(x, -t);
  return 1 / (1 + y);
}

export function sigmoid_x(variable_matrix) {
  let data = math.matrix(variable_matrix)._data;
  
  for (var x = 0; x < data.length; x++) {

      if(!data[x].length) {
        data[x] = sigmoid(data[x]);
      } else {
        for (var i = 0; i < data[x].length; i++) {
            data[x][i] = sigmoid(data[x][i]);
        }
      }
      
  }

  return data;
}

export function sigmoidGradient(z) {
  let data = math.matrix(z);
  let x = sigmoid_x(data);
  let subtracted = math.subtract(1, x);

  return math.dotMultiply(x, subtracted);
}

export function yVectored(matrix) {
  return math.transpose(math.matrix(data));
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

export function clearInfinity(matrix, minVal, maxVal) {
  
  let ntest2 = matrix.map(function(value, index, matrix) {
    
    if (value.toString() === 'Infinity' ) {
        let temp = Math.random() * maxVal;
        return parseFloat(temp);
    }
    else if (value.toString() === '-Infinity') {
        let temp = Math.random() * minVal;
        return parseFloat(temp);
    } else {
        return value;
    }
  });

  return ntest2;
}

export function convertToVector(matrix1) {
  let output = [];
  for (var x = 0; x < matrix1.length; x++) {
    output = output.concat(matrix1[x]);
  }

  return output;
}

export function combineTwoVectors(matrix1, matrix2) {
  let output = [];
  let matrix_one = math.matrix(matrix1);
  let matrix_two = math.matrix(matrix2);

  let matrix_one_data = matrix_one._data;
  let matrix_two_data = matrix_two._data;

  output = matrix_one_data.concat(matrix_two_data);

  return math.matrix(output);
}

export function grabVectorColumn(matrix1, column_num) {
  let output = [];
  let matrix = math.matrix(matrix1)._data;
  for (let x = 0; x < matrix.length; x++) {
    output = output.concat(matrix[x][column_num]);
  }

  return math.matrix(output);
}

export function vectorMultiplication(vector1, vector2) {
  let output = [];
  let vec_1 = math.matrix(vector1)._data;
  let vec_2 = math.matrix(vector2)._data;

  for (let i = 0; i < vec_1.length; i++) {
    let temp = [];
    for (let j = 0;j < vec_2.length; j++) {
      temp.push(vec_1[i] * vec_2[j]);
    }

    output.push(temp);
  }

  return output;
}

export function maxRow(matrix) {
  let output = [];
  for (let x = 0; x < matrix.length; x++) {
      let max = {
          index: 0,
          value: matrix[x][0],
      }
      for (let y = 0; y < matrix.length; y++) {
          if (matrix[x][y] > max.value) {
              max.index = y;
              max.value = matrix[x][y] 
          }
      }
      output.push(max);
  }

  return output;
}


export function maxRowIndexOnly(matrix) {
  let output = [];
  for (let x = 0; x < matrix.length; x++) {
      let max = 0;
      let value = matrix[x][0]
      for (let y = 0; y < matrix.length; y++) {
          if (matrix[x][y] > value) {
              max = y;
              value = matrix[x][y] 
          }
      }
      output.push(max);
  }

  return output;
}

export function singleMaxRow(matrix) {
  let output = 0;
  let value = matrix[0];
  for (let i = 0; i < matrix.length; i++) {
    if (matrix[i] > value) {
      output = i;
      value = matrix[i];
    }
  }
  return output;
}

export function testPrediction(prediction, testData) {
  if (prediction.length !== testData.length) {

    return 'Error, mismatched lengths'
  }

  let total = 0;
  for (let i = 0; i < prediction.length; i++) {
    if (prediction[i] === testData[i]) {
      total += 1;
    }
  }

  return (total/prediction.length)*100;
}


export function top3(matrix) {
  let best = {
    a: {
      index: 0, 
      value: 0
    },
    b: {
      index: 0, 
      value: 0
    },
    c: {
      index: 0, 
      value: 0
    },
  };

  for (let i = 0; i < matrix.length; i++) {
    if (matrix[i] > best['a']['value']) {
      best['c'] = best['b'];
      best['b'] = best['a'];
      best['a'] = {
        index: i,
        value: matrix[i]
      };
    } else if (matrix[i] > best['b']['value']) {
      best['c'] = best['b'];
      best['b'] = {
        index: i,
        value: matrix[i]
      };
    } else if (matrix[i] > best['c']['value']) {
      best['c'] = {
        index: i,
        value: matrix[i]
      };
    }
  };

  document.querySelector('.prediction-1').innerHTML = best['a'].index;
  document.querySelector('.prediction-2').innerHTML = best['b'].index;
  document.querySelector('.prediction-3').innerHTML = best['c'].index;

  console.log(best);

}
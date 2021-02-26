import math from 'mathjs';
import { sigmoid } from './util';

function combiner(value1, value2) {
    console.log(value1);
    console.log(value2);
    if (value1.length !== value2.length) {
        console.log('mismatch');
        return 'Error'
    }

    let final = [];
    for(let i = 0; i < value1.length; i++) {
        let temp = [value1[i]].concat(value2[i]);
        final.push(temp);
    }

    return final;

}

function sigmoid_x(variable_matrix) {
    let data = variable_matrix._data;

    for (var x = 0; x < data.length; x++) {
        for (var i = 0; i < data[x].length; i++) {
            data[x][i] = sigmoid(data[x][i]);
        }
    }

    return data;
}

export function nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) {
    let m = X.length;
    let Theta1 = nn_params[0];
    let Theta2 = nn_params[1];
    let J = 0;
    let Theta1Grad = math.zeros(Theta1.length);
    let Theta2Grad = math.zeros(Theta2.length);


    let ones = math.ones(m);

    let newX = (math.matrix(combiner(ones._data, X)));

    let z2 = math.multiply(Theta1, math.transpose(newX));

    let a2 = sigmoid_x(z2);

    let newA2 = math.matrix(combiner(ones._data, math.transpose(a2)));

    let z3 = math.multiply(Theta2, math.transpose(newA2));

    let h_of_x = sigmoid_x(z3);

    console.log(h_of_x);
    
}
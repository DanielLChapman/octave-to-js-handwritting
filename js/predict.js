import math from 'mathjs';
import {sigmoid_x, maxRow, maxRowIndexOnly} from './util';


export function predict (Theta1, Theta2, X) {
    let m = math.matrix(X)._data[0].length;
    let num_labels = math.matrix(Theta2)._data[0].length;

    let p = math.zeros(m);

    let h1 = sigmoid_x(math.multiply(X, math.transpose(Theta1)));
    let h2 = sigmoid_x(math.multiply(h1, math.transpose(Theta2)))
    


    console.log({
        h1, 
        h2,
    })

    return maxRowIndexOnly(h2);
}
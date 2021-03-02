import math from 'mathjs';
import { sigmoid, convertToVector, clearInfinity} from './util';

function combiner(value1, value2) {
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
    let temp_theta1 = math.sum(math.dotMultiply(Theta1, Theta1));
    //temp_theta1 = math.sum(temp_theta1);

    let Theta2 = nn_params[1];
    
    let temp_theta2 = math.sum(math.dotMultiply(Theta2, Theta2));
    //temp_theta2 = math.sum(temp_theta2);
    let reg =  lambda / (2*m)  * (temp_theta1 + temp_theta2);


    let J = 0;
    let Theta1Grad = math.zeros(Theta1.length);
    let Theta2Grad = math.zeros(Theta2.length);


    let ones = math.ones(m);

    let newX = (math.matrix(combiner(ones._data, X)));

    let z2 = math.multiply(Theta1, math.transpose(newX));

    let a2 = sigmoid_x(z2);

    let newA2 = math.matrix(combiner(ones._data, math.transpose(a2)));

    let z3 = math.multiply(Theta2, math.transpose(newA2));

    let h_of_x = math.matrix(sigmoid_x(z3));

    let onem = 1 / m;

    let negativeY = math.multiply(y, -1);//math.multiply(y, -1), math.log(h_of_x))

    //Check for infinities
    


    /*let test = math.sum(
                math.sum(
                    math.multiply(-y, math.log(h_of_x)) - math.multiply(math.subtract(1, y), math.log(math.subtract(1, h_of_x)))
                )
    )
*/

    let log_h_of_x = (math.log10(h_of_x));

    log_h_of_x = clearInfinity(log_h_of_x, -550, 550);

    let math_multiple_one = (math.dotMultiply(negativeY, log_h_of_x));

    let subtract_one = math.subtract(1, y);

    let one_minus_log_h_of_x = math.subtract(1, h_of_x);
    one_minus_log_h_of_x = math.log10(one_minus_log_h_of_x);
    one_minus_log_h_of_x = clearInfinity(one_minus_log_h_of_x, -550, 550);

    let math_multiple_two = math.dotMultiply(subtract_one, one_minus_log_h_of_x);

    let total = math.subtract(math_multiple_one, math_multiple_two);

    //New sum formula
    console.log(math.sum(math.sum(total)));

    J = onem * math.sum(math.sum(total));

    console.log(J);

    J = J + reg;

    console.log(J);

    
}
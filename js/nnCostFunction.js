import math from 'mathjs';
import { sigmoid, convertToVector, clearInfinity, sigmoid_x, grabVectorColumn, sigmoidGradient, vectorMultiplication } from './util';

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
    let Theta1Grad = math.zeros(Theta1.length, Theta1[0].length);
    let Theta2Grad = math.zeros(Theta2.length, Theta2[0].length);


    let ones = math.ones(m);

    //let newX = (math.matrix(combiner(ones._data, X)));


    let z2 = math.multiply(Theta1, math.transpose(X));

    let a2 = sigmoid_x(z2);

    let newA2 = math.matrix( math.transpose(a2));

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

    J = onem * math.sum(math.sum(total));

    J = J + reg;

    console.log(J);

    //back propoagation

    for(let i = 0; i < 40; i++) {
        console.log(i);
        let a1 = X[i];
        a1 = math.transpose(a1);

       z2 = math.multiply(Theta1, a1);

       a2 = sigmoid_x(z2);

       z3 = math.multiply(Theta2, a2);

       let a3 = sigmoid_x(z3);

       let delta3 = math.subtract(a3, grabVectorColumn(y, i));

      let delta2 = math.transpose(Theta2);
      delta2 = math.multiply(delta2, delta3);
      delta2 = math.dotMultiply(delta2, sigmoidGradient(z2));

      Theta2Grad = math.add(Theta2Grad, math.matrix(vectorMultiplication(delta3, a2)));
      Theta1Grad = math.add(Theta1Grad, vectorMultiplication(delta2, a1));

    }

    Theta2Grad = math.add(math.dotMultiply(onem, Theta2Grad), math.multiply((lambda/m), Theta2));
    Theta1Grad = math.add(math.dotMultiply(onem, Theta1Grad), math.multiply((lambda/m), Theta1));

    return [Theta1Grad, Theta2Grad];

/*
    a1 = X(t, :);
    a1 = a1';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1 ; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    delta3 =  a3 - y_vectored(:, t);

    z2 = [1; z2];
    delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);

    delta2 = delta2(2:end);

    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta1_grad = Theta1_grad + delta2 * a1';
*/


    
}
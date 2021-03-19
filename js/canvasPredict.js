import 'regenerator-runtime/runtime'
import math from 'mathjs';
import {initializeThetas, initializeThetasVector, convertYandVector, sigmoidGradient, combineTwoVectors, convertToVector, testPrediction} from './util';
import {iter400data} from './data/iter400data';
import {predict} from './predict';


function prepareImage() {
    //find canvas
    let canvas = document.querySelector('#main_drawing');
    //create new canvas and redraw current canvas to 1/10th size
    var dataURL = canvas.toDataURL('image/png', '1.0');
    var img = new Image();

    img.src=dataURL;
    let newData;
    img.addEventListener("load", function () {

        newData = imageToDataUri(img, 28, 28);
        let dataSet = newData;
        let p = predict(newTheta1, newTheta2, dataSet);

        console.log(p);
    

    });
}


function imageToDataUri(img, width, height) {

    // create an off-screen canvas
    var canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d');

    // set its dimension to target size
    canvas.width = width;
    canvas.height = height;

    // draw source image into the off-screen canvas:
    ctx.drawImage(img, 0, 0, width, height);

    // encode image to data-uri with base64 version of compressed image
    const imageData = ctx.getImageData(0, 0, width, height);
    let datasetBytesView = [];
    for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels hold an equal value since the image is grayscale, so
        // just read the red channel.
        datasetBytesView[j] = imageData.data[j * 4] / 255;
    }

    return datasetBytesView;

}
let nn_params_for_validation;
let newTheta1;
let newTheta2;


const input_layer_size = 28*28;
const hidden_layer_size = 25;
const num_layers = 10;

(() => {
    let prediction_button = document.querySelector('.prediction_button');
    prediction_button.onclick = function() {

      let p = prepareImage();
    }

    let nn_params = math.matrix(iter400data[0].data);
    //verify right data is being submitted
    console.log(nn_params);

    //copy nn_params for new thetas for prediction
    nn_params_for_validation = [...nn_params._data];

    //convert nn_params into new theta objects
    newTheta1 = nn_params_for_validation.splice(0, input_layer_size*hidden_layer_size);
    newTheta1 = math.reshape(math.matrix(newTheta1), [hidden_layer_size, input_layer_size]);
    newTheta2 = math.reshape(math.matrix(nn_params_for_validation), [num_layers, hidden_layer_size]);

})()

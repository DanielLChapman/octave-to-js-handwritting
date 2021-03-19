import 'regenerator-runtime/runtime'
import math from 'mathjs';
import {initializeThetas, initializeThetasVector, convertYandVector, sigmoidGradient, combineTwoVectors, convertToVector, testPrediction} from './util';
import {nnCostFunction} from './nnCostFunction';
import {fmincg} from './Plugin/fmincg';
import {iter400data} from './data/iter400data';
import {predict} from './predict';

const input_layer_size = 28*28;
const hidden_layer_size = 25;
const num_layers = 10;
//theta1 = 25   401
//theta2 = 10   26

let batchData;

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);

    
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);



    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());


    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    

  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}
let examples = [];


let pos = {
  x: 0,
  y:0,
}
let canvas;
let ctx;

function setPosition(e) {
  var rect = canvas.getBoundingClientRect();
  pos.x = e.clientX - rect.left;
  pos.y = e.clientY - rect.top;
}

function draw(e) {
  // mouse left button must be pressed
  if (e.buttons !== 1) return;

  ctx.beginPath(); // begin

  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#FFF';

  ctx.moveTo(pos.x, pos.y); // from
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to

  ctx.stroke(); // draw it!
}

async function showExamples(data) {
    // Create a container in the visor
    const surface =
      tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
  
    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    batchData = examples;

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
      
      const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });
      

      examples[i] = await (tf.browser.toPixels(imageTensor));
     
      
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = 'margin: 4px;';
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);
  
      imageTensor.dispose();
    }

}


(async  () => {

  /*
    MNIST LOADING, NO NEEDED AT MOMENT

    const data = new MnistData();
    await data.load();

    

    await showExamples(data);

    let temp = convertYandVector(  batchData.labels.arraySync());
    console.log(temp);
    let testX = batchData.xs.arraySync();
    
    */
    

    //old original thetas
    //let Theta1 = initializeThetas(25, input_layer_size, NUM_TRAIN_ELEMENTS/100);
    //let Theta2 = initializeThetas(10, hidden_layer_size, NUM_TRAIN_ELEMENTS/100);

    //lambda value

    //let lambda = 3;

    //training data
    /*
    const examples = data.nextTrainBatch(NUM_TRAIN_ELEMENTS);
    let X = examples.xs.arraySync();

    let y = examples.labels.arraySync();

    let newY = math.transpose(math.matrix(y))
    */

    //testing data
    /*
    let testExamples = data.nextTestBatch(NUM_TEST_ELEMENTS);
    let testingX = testExamples.xs.arraySync();
    let testingY = convertYandVector(testExamples.labels.arraySync());
*/
    //X = image data
    //y = labels

    //fmincg options data pass
    /*
    let options = {
      maxIterations: 5
    };
*/
    //vectored thetas for nn_params
    /*
    let VTheta1 = convertToVector(Theta1);
    let VTheta2 = convertToVector(Theta2);
    let nn_params = math.matrix(VTheta1.concat(VTheta2));
    //nn_params = math.reshape(nn_params, [nn_params._size, 1]);
    */

    //grab thetas from iter data sheet
    let nn_params = math.matrix(iter400data[0].data);
    //verify right data is being submitted
    console.log(nn_params);

    //actual training program fmincg running function and output
    //let [nn_params, cost] 
    //let output = fmincg(nnCostFunction, nn_params, options, input_layer_size, hidden_layer_size, num_layers, X, newY._data, lambda);
    
    //console.log([...output]);

    //cost function testing
    //let J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_layers, X, newY._data, lambda);
    //console.log(J);

    //copy nn_params for new thetas for prediction
    let nn_params_for_validation = [...nn_params._data];

    //convert nn_params into new theta objects
    let newTheta1 = nn_params_for_validation.splice(0, input_layer_size*hidden_layer_size);
    newTheta1 = math.reshape(math.matrix(newTheta1), [hidden_layer_size, input_layer_size]);
    let newTheta2 = math.reshape(math.matrix(nn_params_for_validation), [num_layers, hidden_layer_size]);

    //prediction of test batch data
    /*
    let p = predict(newTheta1, newTheta2, testX);

    console.log(p);

    console.log(testPrediction(p, temp))

    let testP = predict(newTheta1, newTheta2, testingX);
    console.log(testPrediction(testP, testingY));
    93.5-95%;
    */

    document.addEventListener('mousemove', draw);
    document.addEventListener('mousedown', setPosition);
    document.addEventListener('mouseenter', setPosition);

    
    canvas = document.querySelector("#main_drawing");
    console.log(canvas);
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 280, 280);

})();




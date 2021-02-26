import math from 'mathjs';

export function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  }

export function initializeThetas(size1 = 2, size2 = 2, totalNum = 2) {
    let final = [];
    for (var x = 0; x < size1; x++) {
      let temp = [];
      for (var y = 0; y < size2; y++) {
        let num = (randn_bm()*(Math.random() * (totalNum - 1))) * Math.sqrt(2/(totalNum - 1));
        
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
        let num = (randn_bm()*(Math.random() * (totalNum - 1))) * Math.sqrt(2/(totalNum - 1));
        final.push(num);
      }
    }
    return final;
  }

  export function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}
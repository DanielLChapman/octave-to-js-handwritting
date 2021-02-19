import math from 'mathjs';

const identity = (n = 1, m = n) => {
    let a = [];

    if (n < 2 || m < 2) {
        return math.matrix([1], [1]);
    }

    for (var x = 0; x < n; x++ ) {
        let t = [];
        for (var y = 0; y < m; y++) {
            if (x === y) {
                t.push(1);
            } else {
                t.push(0);
            }
        }
        a.push(t);
    }

    return math.matrix(a);
}

(function() {

    //matrix math test

    const matrix = math.matrix([[7,1], [-2, 3]]);

    console.log(matrix);
    console.log(math.square(matrix));
    console.log(math.add(matrix, math.square(matrix)));

    console.log(math.matrix([[0, 1], [2, 3], [4, 5]]) );

    console.log(identity(5, 5));
    console.log(identity(10));

    console.log(math.zeros(3) );

    console.log(math.multiply(identity(10), math.square(matrix).resize([10, 10])));

    const a = [0, 1, 2, 3]
    const b = [[0, 1], [2, 3]]
    const c = math.zeros(2, 2)
    const d = math.matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    const e = math.matrix()

    console.log(math.subset(a, math.index(1))  )               // 1
    console.log(math.subset(a, math.index([2, 3]))  )          // Array, [2, 3]
    console.log(math.subset(a, math.index(math.range(0,4))) )  // Array, [0, 1, 2, 3]
    console.log(math.subset(b, math.index(1, 0))         )     // 2
    console.log(math.subset(b, math.index(1, [0, 1]))    )     // Array, [2, 3]
    console.log(math.subset(b, math.index([0, 1], 0))    )     // Matrix, [[0], [2]]

}());
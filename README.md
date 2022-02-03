Converting Octave code to JS

Experimenting with html5-boilerplate code and parcel.

Using mathjs (https://mathjs.org/) and ml-util (https://github.com/gulfaraz/ml-util) to assist.

Issues with parcel and hosting so clone this repo and run 'npm install' and 'npm run dev' to experiment!

Data is being console logged if you were ever curious about some values

Errors within this application:

This was all built with Javascript with no training from octave or with using tensor flow. I did every calculation with Javascript, which for those who know is a terrible idea with very small digits (parseInt(0.000005, 10) === 0, parseInt(0.0000005, 10) === 5). But this was just more of an experiment than anything serious, there are much more efficient ways to do this with tensorflow or training data with python and importing that. 
const harness = require("../harness/harness.js");

class Classifier {
  constructor(knn) {
    this.trainingData = [];
    this.trainingLabels = [];
    this.knn = knn;
  }
  fit(trainingData, trainingLabels) {
    this.trainingData = trainingData;
    this.trainingLabels = trainingLabels;
  }
  predict(testingData) {
    const predictions = testingData.map((testRow, testIdx) => {
      const distances = this.trainingData.map((trainRow, trainIdx) => {
        return {
          distance: calculateEuclideanDistance(testRow, trainRow),
          index: trainIdx,
          outcome: this.trainingLabels[trainIdx]
        };
      });
      distances.sort(sortDistances);
      console.log(distances);
      const knn = distances.splice(0, this.knn);
      return getMode(knn);
    });
    console.log("PREDICTIONS", predictions);
    return predictions;
  }
}

const getMode = (arr, result) => {
  result = result || {};

  if (arr.length === 0) return result;

  let head = arr.shift().outcome;

  if (result[head]) result[head]++;
  else result[head] = 1;

  return getMode(arr, result);
};

const calculateEuclideanDistance = (arr1, arr2) => {
  const sum = arr1.reduce(
    (acc, data, idx, arr) => acc + Math.pow(data - arr2[idx], 2)
  );
  return Math.sqrt(sum);
};

const sortDistances = function(a, b) {
  if (a.distance < b.distance) {
    return -1;
  }
  if (a.distance > b.distance) {
    return 1;
  }
  // a must be equal to b
  return 0;
};

const algo = new Classifier(3);
const result = harness.evaluator("../diabetes.csv", algo);

console.log("RESULT", result);

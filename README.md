# Machine Learning Workshop
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/fresh8/mlworkshop/harness?status.svg)](https://godoc.org/github.com/fresh8/mlworkshop/harness) 
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp)

Training and Evaluation harness for machine learning algorithms in Go and JavaScript.

## Usage

### Go

Go get the evaluation harness:

```
go get -u github.com/fresh8/mlworkshop/harness
```

To use the harness simply call the `Evaluate()` function passing in the path to the dataset csv file and your model:

``` go
package main

import (
	"log"

	"github.com/fresh8/mlworkshop/harness"
	"gonum.org/v1/gonum/mat"
)

func main() {

	log.Println("Evaluating model")

	model := MyClassifier{
                // model parameters
        }

	result, err := harness.Evaluate("diabetes.csv", &model)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Result = %f", result)
}
```

Where the model of type `MyClassifier` implements the `harness.Predictor` interface:

``` go
// Predictor implementations define specific machine learning algorithms
// to classify data.  The interface contains Fit and Predict methods to
// Fit the model to training data and Predict classes of previously
// unseen observations respectively.
type Predictor interface {
	// Fit the model to the training data.  This trains the model learning
	// associations between each feature vector (row vector) within matrix
	// X and it's associated ground truth class label in slice Y.
	Fit(X mat.Matrix, Y []string)
	// Predict will classify the feature vectors (row vectors) within
	// matrix X, predicting the correct class for each based upon what
	// what the model learned during training.
	Predict(X mat.Matrix) []string
}
``` 

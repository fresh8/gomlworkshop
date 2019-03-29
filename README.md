# Machine Learning Workshop
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/fresh8/mlworkshop/harness?status.svg)](https://godoc.org/github.com/fresh8/mlworkshop/harness) 
[![Go Report Card](https://goreportcard.com/badge/github.com/fresh8/mlworkshop/harness)](https://goreportcard.com/report/github.com/fresh8/mlworkshop/harness)

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

Where the model of type `MyClassifier` implements the `harness.Predictor` interface.  Please refer to the [Godoc](https://godoc.org/github.com/fresh8/mlworkshop/harness) for more details. 

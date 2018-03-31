# Translation Quality Estimation

## Setup

### Clone this repository
```
$ git clone https://github.com/nisargjhaveri/tqe
```

### Python dependencies
The dependencies are listed in `requirements.txt`.

To install all the dependencies, run `pip` as followed.
```
$ pip install -U -r requirements.txt
```

### Setup Stanford CoreNLP (used in TQE feature extraction)
https://stanfordnlp.github.io/CoreNLP/index.html#download

Get and setup Stanford CoreNLP.
Set `CORENLP_HOST` with the address to the CoreNLP server.

### Setup KenLM (used in TQE feature extraction)
https://github.com/kpu/kenlm

Setup KenLM and set an environment variable `KENLM_BIN` with the path to directory containing `lmplz` binary.

### Setup tercom (used in TQE data preparation)
http://www.cs.umd.edu/~snover/tercom/

Setup TERCOM and set environment variable `TERCOM_JAR` with the path to `tercom.7.25.jar`.


## Datasets

### WMT17 en-de dataset
The WMT17 en-de dataset can be downloaded from the official website of the shared task.
http://www.statmt.org/wmt17/quality-estimation-task.html

### news.gu
This dataset can be obtained by contacting the authors.

### ICLI datasets
The parallel corpus can be obtained from http://tdil-dc.in.
The automatic translations and quality labels can be obtained by contacting the authors.

## Run

These example assumes that the dataset files (`.src`, `.mt`, `.ref` and `.hter` file for each dataset) is present in a directory called `workspace`.
You can change the directory name accordingly.

### Common arguments
```
workspace_dir         Directory containing prepared files
data_name             Identifier for prepared files

# Help
-h, --help            Show help message and details about possible arguments.


# If you have train/dev/test splits in different files
--dev-file-suffix DEV_FILE_SUFFIX
                      Suffix for dev files
--test-file-suffix TEST_FILE_SUFFIX
                      Suffix for test files

# To save model
--save-model SAVE_MODEL
                      Save the model with this basename

```


### SVR baseline models
```
$ python tqe.py baseline workspace/ news.gu --tune
```

### POSTECH models

#### Multi-task learning

```
$ python tqe.py postech workspace/ news.gu
```

#### Two-step learning
```
$ python tqe.py postech --two-step workspace/ news.gu
```

#### Learning with stack propagation
```
$ python tqe.py postech --stack-prop workspace/ news.gu
```

### SHEF/CNN model

```
$ python tqe.py shef workspace/ news.gu
```

### RNN models

```
$ python tqe.py rnn --with-attention workspace/ news.gu
```
or
```
$ python tqe.py rnn --with-attention -summary-attention workspace/ news.gu
```

### CNN based models
#### CNN.Siamese model
```
$ python tqe.py siamese workspace/ news.gu
```

#### CNN.Combined models

##### Without features
```
$ python tqe.py siamese-shef workspace/ news.gu --no-features
```

##### With features
```
$ python tqe.py siamese-shef workspace/ news.gu --normalize
```

#### +fastText
Append `--source-embeddings wiki.en --target-embeddings wiki.gu` to any CNN based model.
This assumes fastText embeddings models in `wordspace/fastText`.

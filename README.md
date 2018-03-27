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

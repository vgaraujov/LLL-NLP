# Episodic Memory in Lifelong Language Learning

This repository contains a PyTorch implementation of MbPA++ from the paper [Episodic Memory in Lifelong Language Learning](https://arxiv.org/abs/1906.01076).

## Dependencies
```
torch                1.10.0
transformers         3.0.2
```

## Run the Experiments
For Text Classification:
```python
bash run_tc.sh
```

For Question Answering:
```python
bash run_qa.sh
```

In both cases, `DATA_DIR` and `ORDER` variables must be set to run the experiment correctly.

## References
* [Episodic Memory in Lifelong Language Learning](https://arxiv.org/abs/1906.01076)
* Part of the code is borrowed from [https://github.com/Daikon-Sun/EM-in-LLL](https://github.com/Daikon-Sun/EM-in-LLL)

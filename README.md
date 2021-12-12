# iannwtf_hw7

## Model Parameter:
  - data_samples: 96000: 64000/16000/16000
  - data_seq_length: 25
  - batch_size: 32
  - learning_rate = 0.001
  - optimizer: Adam
  - epochs = 2

## Model Architechture
### LSTM Model (with 2 layers)
  ![image](https://user-images.githubusercontent.com/93341845/145704957-ece79ffb-d57f-41a8-b70c-2cfe8ac8587f.png)
### LSTM Cell
  ![image](https://user-images.githubusercontent.com/93341845/145704583-9f63d377-782d-4229-84bb-006cd47af13a.png)
  ![image](https://user-images.githubusercontent.com/93341845/145704114-983bc81e-0347-425f-adcc-afbb291faa6c.png)
### Model Results

## Outstanding Questions
### Can / should you use truncated BPTT here?
- BPTT can be computationally expensive as the number of timesteps increases.
- TBPTT cuts down computation and memory requirements (but the truncation-length has to be chosen carefully in order to work well)
- To use TBPTT we would need to implement backpropagation on a different level or way, because we would have to optimize our model for each individual timestep, not at the end for all timesteps together.
-  We could theoretically use TBPTT to reduce computation and memory while training our model.


### Should you rather take this as a regression, or a classification problem?
- In our we problem our input consists of 25 numbers and our target is either 1 or 0, depending on the sum of all inputs.
- This is a function with the dimensions R^25 -> {0,1}, and is a **classification** problem.
- Non the less, in the LSTM layers we perform regression tasks with the function R^25 -> R (for the last hidden_output).



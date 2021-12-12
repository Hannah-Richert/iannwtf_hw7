# iannwtf_hw7
## To-Dos
- reviewing the code together and optimizing it
- Pipeline Structure (as discussed last time)
- Commenting all files
- Answering qutstanding questions
## Model Parameter:
  - data_samples: 96000: 64000/16000/16000
  - data_seq_length: 25
  - batch_size: 32
  - learning_rate = 0.001
  - optimizer: Adam
  - epochs = 2

## Model Architechture
### LSTM Model
  ![image](https://user-images.githubusercontent.com/93341845/145704957-ece79ffb-d57f-41a8-b70c-2cfe8ac8587f.png)
### LSTM Cell
  ![image](https://user-images.githubusercontent.com/93341845/145704583-9f63d377-782d-4229-84bb-006cd47af13a.png)
  ![image](https://user-images.githubusercontent.com/93341845/145704114-983bc81e-0347-425f-adcc-afbb291faa6c.png)
### Model Results

## Outstanding Questions
### Can / should you use truncated BPTT here?
- BPTT can be computationally expensive as the number of timesteps increases
### Should you rather take this as a regression, or a classification problem?


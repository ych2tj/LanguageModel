from fastapi import FastAPI
from pydantic import BaseModel
import json
from train import model_run

app = FastAPI()

# set input parameters
class model_parameters(BaseModel):
    inputs : str
    mode: str
    batch_size: int
    block_size: int # considering GPU abilities
    n_embd: int
    n_layer: int
    n_head: int
    max_iters: int
    eval_iters: int
    dropout: float
    lr: float

# build language model
modelrun = model_run()

@app.post('/predict_language')
def pred_language(input_parameters: model_parameters):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    inputs=input_dictionary['inputs']
    mode = input_dictionary['mode']
    block_size = input_dictionary['block_size'] # considering GPU abilities
    n_embd = input_dictionary['n_embd']
    n_layer = input_dictionary['n_layer']
    n_head = input_dictionary['n_head']
    max_iters = input_dictionary['max_iters']
    eval_iters = input_dictionary['eval_iters']
    dropout = input_dictionary['dropout']
    batch_size = input_dictionary['batch_size']
    lr = input_dictionary['lr']

    results = modelrun.train_test(inputs=inputs,
                                  mode = mode,
                                  block_size = block_size, # considering GPU abilities
                                  n_embd = n_embd,
                                  n_layer = n_layer,
                                  n_head = n_head,
                                  max_iters = max_iters,
                                  eval_iters = eval_iters,
                                  dropout = dropout,
                                  batch_size = batch_size,
                                  lr = lr)
    
    return results





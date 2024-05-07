import torch
import pickle


# Check the device: GPU or CPU. 
# Attention: do not give list brackets "[]" here, because tensor.to() doesn't recieve it.
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print("device is", device)
batch_size = 4
block_size = 8 # considering GPU abilities
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2

chars = ""
with open('trading_strategies.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode and decode the characters
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# load model
model = pickle.load(open('model_01.pkl', 'rb'))
model.eval()
m = model.to(device)

# outputs
prompt = 'Range Tr'
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generate_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=200, block_size=block_size)[0].tolist())
print(generate_chars)





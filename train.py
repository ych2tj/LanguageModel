import torch
import pickle
from transformer import largelanguagemodel


# Check the device: GPU or CPU. 
# Attention: do not give list brackets "[]" here, because tensor.to() doesn't recieve it.
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print("device is", device)

class model_run():
    def __init__(self):
        # Prepare data
        chars = ""
        with open('trading_strategies.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        #print(chars)
        #print(len(chars))
        # encode and decode the characters
        string_to_int = {ch:i for i, ch in enumerate(chars)}
        int_to_string = {i:ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [string_to_int[c] for c in s]
        self.decode = lambda l: ''.join([int_to_string[i] for i in l])
        #print(encode('hello world')) # show the encode
        #encoded_str = encode('hello world')
        #print(decode(encoded_str)) # show the decode

        # load model
        print("Loading model...")
        model = pickle.load(open('model_01.pkl', 'rb'))
        print("Model loading complete")
        model.eval()
        self.m = model.to(device)


    def train_test(self,
                   inputs,
                   mode = 'test',
                   batch_size = 4,
                   block_size = 8, # considering GPU abilities
                   n_embd = 384,
                   n_layer = 4,
                   n_head = 4,
                   max_iters = 10000,
                   eval_iters = 250,
                   dropout = 0.2,
                   lr = 3e-4):
        
        if mode == 'train':
            # build data
            data = torch.tensor(self.encode(self.text), dtype=torch.long)
            #print(data[:100])
            separate_ind = int(0.8*len(data))
            train_data = data[:separate_ind]
            val_data = data[separate_ind:]

            '''
            #  Understand block size
            x = train_data[:block_size]
            y = train_data[1:block_size]

            # show the block size meaning
            for t in range(block_size):
                context = x[:t+1]
                target = y[t]
                print("When input is ", context, "target is", target)
            '''
            # setup data
            def get_batch(split):
                data = train_data if split == 'train' else val_data
                ix = torch.randint(len(data) - block_size, (batch_size, ))
                #print(ix)
                x = torch.stack([data[i:i+block_size] for i in ix])
                y = torch.stack([data[i+1:i+block_size+1] for i in ix])
                x, y = x.to(device), y.to(device)
                return x, y

            #x, y = get_batch('train') # debug the get_batch()
            #print('inputs: ', x)
            #print('targets: ', y)

            # Create model
            model = largelanguagemodel(vocab_size=self.vocab_size, 
                                       n_embd=n_embd, 
                                       n_layer=n_layer, 
                                       n_head=n_head, 
                                       block_size=block_size, 
                                       dropout=dropout)
            self.m = model.to(device)

            '''
            # outputs for debug
            print("[Before training]:")
            prompt = inputs
            context = torch.tensor(self.encode(prompt), dtype=torch.long, device=device)
            generate_chars = self.decode(self.m.generate(context.unsqueeze(0), max_new_tokens=200, block_size=block_size)[0].tolist())
            print(generate_chars)
            '''

            # Loss for evaluations
            @torch.no_grad()
            def estimate_loss():
                out = {}
                model.eval()
                for split in ["train", "val"]:
                    losses = torch.zeros(eval_iters)
                    for k in range(eval_iters):
                        X, Y = get_batch(split)
                        logits, loss = model(X,Y)
                        losses[k] = loss.item()
                    out[split] = losses.mean()
                model.train()
                return out

            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            print("Start training: ")
            # training process
            for iter in range(max_iters):
                # Evaluation loss
                if iter % eval_iters == 0:
                    losses = estimate_loss()
                    print(f'step: {iter}, train loss: {losses["train"]}, val loss: {losses["val"]}')
                # sample a batch of data
                xb, yb = get_batch('train')
                # evaluate the loss
                logits, loss = model.forward(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            print(loss.item()) # display the loss

            # Save the model
            with open('model_02.pkl', 'wb') as f:
                pickle.dump(model, f)
            print('model saved')
            
            # generate results
            return self._generation(inputs, block_size)
        else:
            # generate results
            return self._generation(inputs, block_size)


    def _generation(self, inputs, block_size):
        # outputs
        prompt = inputs
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=device)
        generate_chars = self.decode(self.m.generate(context.unsqueeze(0), max_new_tokens=200, block_size=block_size)[0].tolist())
        return generate_chars

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='network parameters.')
    parser.add_argument('--mode', default='test', help='Choose train or test')
    parser.add_argument('--block_size', type=int, default=8, help='Data input length')
    parser.add_argument('--n_embd', type=int, default=384, help='Data embedding dim')
    parser.add_argument('--n_head', type=int, default=4, help='Multi-heads attention')
    parser.add_argument('--n_layer', type=int, default=4, help='Decoder block number')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='The learning rate')
    parser.add_argument('--max_iters', type=int, default=10000, help='Max training iterations')
    parser.add_argument('--eval_iters', type=int, default=250, help='Validation iterations')

    args = parser.parse_args()
    print(args)
    
    # run the model
    modelrun = model_run()
    gen_chars = modelrun.train_test(inputs='Range tr',
                                    mode = args.mode,
                                    block_size = args.block_size, # considering GPU abilities
                                    n_embd = args.n_embd,
                                    n_layer = args.n_layer,
                                    n_head = args.n_head,
                                    max_iters = args.max_iters,
                                    eval_iters = args.eval_iters,
                                    dropout = args.dropout,
                                    batch_size = args.batch_size,
                                    lr = args.lr)
    
    print(gen_chars)
    







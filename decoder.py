import torch
import torch.nn as nn

#Hierarchical Decoder as explained 3.2
class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim, output_size, conductor_hidden_size=1024, conductor_output_dim=512, bottom_decoder_hidden_size=1024):
        super(HierarchicalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.conductor_hidden_size = conductor_hidden_size
        self.bottom_decoder_hidden_size=bottom_decoder_hidden_size
        
        # Fully connected layer for latent vector z
        self.fc_z = nn.Linear(latent_dim,conductor_hidden_size)



        # Conductor RNN: two-layer unidirectional LSTM
        self.conductor_rnn = nn.LSTM(batch_first=True,input_size=latent_dim, hidden_size=conductor_hidden_size, num_layers=2)
        self.fc_1 = nn.Linear(conductor_hidden_size, conductor_output_dim) # to make the output embedding has the size of 512


        # Shared fully-connected layer for conductor embeddings
        self.conductor_fc = nn.Linear(conductor_output_dim, conductor_output_dim)
        
        # Bottom-layer Decoder RNN: two-layer LSTM, assuming unidirectional since the paper only mention 2-LSTM
        self.bottom_decoder_rnn = nn.LSTM(batch_first=True,input_size=(conductor_output_dim + output_size), hidden_size=bottom_decoder_hidden_size, num_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(bottom_decoder_hidden_size, output_size)

        #Tanh activation
        self.tanh = nn.Tanh()
        
    def forward(self, z, input_sequence):
        # input_sequence : [BATCH_SIZE, seq_length, class][256, 64, 27]
        # z: [BATCH_SIZE,LATENT_DIM]

        batch_size = z.size(0)
       
        num_subsequences = input_sequence.size(1)
        seq_length = input_sequence.size(2)

        # Pass the latent vector through a fully-connected layer followed by a tanh activation
        # Initial state of conductor RNN
        z = self.fc_z(z)
        z = self.tanh(z)
        z = z.unsqueeze(1)
        z = z.repeat(1,2,1) # [BATCH_SIZE, num_layers, conductor_hidden_size]
        

        z = z.permute(1,0,2) # initial state expected to have shape of [num_layers, BATCH_SIZE, conductor_hidden_size] 
    
        # get embeddings from conductor
        conductor_input = torch.zeros(size=(batch_size, 1, self.latent_dim))
        embeddings = torch.empty(batch_size,16,self.latent_dim)
        state = (z,z)

        outputs = []
        previous = torch.zeros((batch_size, self.output_size))

        for i in range(16): # U=16
            conductor_out, state = self.conductor_rnn(conductor_input,state)

            conductor_out = self.fc_1(conductor_out)
           
            embeddings[:,i,:] = conductor_out[:,0]

            conductor_input = conductor_out

            output_decoder = []

            init = torch.zeros(size=(2, batch_size, self.bottom_decoder_hidden_size))
            
            state2 = (init,init)
            for _ in range(4): # T / U, loop through each subsequence length
                emb = self.conductor_fc(embeddings[:,i,:])
                emb = self.tanh(emb)
            
                l2_in = torch.cat((emb, previous), dim=1) # the current conductor embedding cu is concatenated with the previous output token to be used as the input
                l2_in = l2_in.unsqueeze(1)

         
                h2,state2 = self.bottom_decoder_rnn(l2_in,state2)

                previous = self.output_layer(h2)
             
                previous = previous.squeeze()
                output_decoder.append(previous)
            outputs.extend(output_decoder)
            previous = output_decoder[-1]

        output_tensor = torch.stack(outputs, dim=1)
        
        output = torch.sigmoid(output_tensor)



        return output
if __name__ == '__main__':
    e = HierarchicalDecoder(512,27)
    seq_length = 64
    batch_size = 3
    z = torch.rand(batch_size,512)
    input = torch.rand(batch_size,seq_length,27)

    output = e(z,input)

    
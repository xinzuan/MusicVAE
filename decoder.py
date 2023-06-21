import torch
import torch.nn as nn

#Hierarchical Decoder as explained 3.2
class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim, output_size, conductor_hidden_size=1024, conductor_output_dim=512, bottom_decoder_hidden_size=1024):
        super(HierarchicalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.conductor_hidden_size = conductor_hidden_size
        
        # Fully connected layer for latent vector z
        self.fc_z = nn.Linear(latent_dim,conductor_hidden_size)



        # Conductor RNN: two-layer unidirectional LSTM
        self.conductor_rnn = nn.LSTM(batch_first=True,input_size=conductor_hidden_size, hidden_size=conductor_hidden_size, num_layers=2)
        
        # Shared fully-connected layer for conductor embeddings
        self.conductor_fc = nn.Linear(conductor_hidden_size, conductor_output_dim)
        
        # Bottom-layer Decoder RNN: two-layer LSTM, assuming unidirectional since the paper only mention 2-LSTM
        self.bottom_decoder_rnn = nn.LSTM(batch_first=True,input_size=(conductor_output_dim + output_size), hidden_size=bottom_decoder_hidden_size, num_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(bottom_decoder_hidden_size, output_size)

        #Tanh activation
        self.tanh = nn.Tanh()
        
    def forward(self, z, input_sequence):
        # input_sequence : [BATCH_SIZE, SUBSEQUENCE, SEQ_LENGTH][256, 64, 27]
        # z: [BATCH_SIZE, SUBSEQUENCE, LATENT_DIM]

        batch_size = z.size(0)
        num_subsequences = input_sequence.size(1)
        seq_length = input_sequence.size(2)

        # Pass the latent vector through a fully-connected layer followed by a tanh activation
        z = self.fc_z(z)
        z = self.tanh(z)
 
        
        conductor_hidden = torch.zeros(batch_size, 2,self.bottom_decoder_rnn.hidden_size)
      


        # Initialize the output tensor
        output = torch.zeros(batch_size,num_subsequences,seq_length)


        # Autoregressively generate output tokens for each subsequence
        # Subsequence U is 16
        for i in range(4):

            # Pass the conductor input through the Conductor RNN
            embedding, _ = self.conductor_rnn(z[:,i*16,:], conductor_hidden) 
      
            #Each cu is individually passed through a shared fully-connected layer followed by a tanh activation to produce initial states for a final bottom-layer decoder RNN
            embedding = self.conductor_fc(embedding)
            embedding = self.tanh(embedding).unsqueeze(1)

            decoder_hidden = (torch.randn(2, batch_size, self.bottom_decoder_rnn.hidden_size),
                  torch.randn(2, batch_size, self.bottom_decoder_rnn.hidden_size))


            
            last_dim = embedding.shape[-1]
            embedding = embedding.expand(batch_size, 16, last_dim)

            #concatenated with the previous output token to be used as the input of bottom RNN
            e = torch.cat([embedding,input_sequence[:,range(i*16,i*16+16),:]],dim=-1)

           
            
            bottom_output, decoder_hidden = self.bottom_decoder_rnn(e, decoder_hidden)
            
            temp = self.output_layer(bottom_output)
            temp = torch.softmax(temp, dim=2)
                
            #generates 16 output per batch at a time

            output[:,range(i*16,i*16+16),:]=temp
        
        return output

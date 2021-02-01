
import torch
import torch.nn as nn


class vae(nn.Module):

    def __init__(self, enc_dims, latent_dim, dec_dims, ReLU_slope=0):

        super(vae, self).__init__()

        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.z_dim = latent_dim


        #Encoder
        encoder_layers = []
        for i in range(len(self.enc_dims)-1):
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.enc_dims[i], self.enc_dims[i+1]),
                    nn.LeakyReLU(negative_slope=ReLU_slope)
                )
            )

        self.encoder = nn.Sequential(*encoder_layers)


        # Bottleneck
        self.fc_mu = nn.Linear(self.enc_dims[-1], self.z_dim)
        self.fc_var = nn.Linear(self.enc_dims[-1], self.z_dim)


        #Decoder
        decoder_layers = []
        decoder_layers.append(
            nn.Sequential(self.z_dim, self.dec_dims[0]),
            nn.LeakyReLU(negative_slope=ReLU_slope)
        )

        for i in range(len(self.dec_dims)-1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.dec_dims[i], self.dec_dims[i+1]),
                    nn.LeakyReLU(negative_slope=ReLU_slope)
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)

    #Include for clarity
    def encode(self, data):
        result = self.encoder(data)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    #Include for clarity
    def decode(self, z):

        output = self.decoder(z)

        return output

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        e = torch.randn(*mu.size())
        z = mu + std * e

        return z

    def forward(self, data):

        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)

        return output, mu, logvar

    def sample(self, num=1):
        z = torch.randn(num, self.z_dim)
        output = self.decode(z)
        return output



if __name__ == '__main__':

    model = vae(enc_dims=[20,15,10,5], latent_dim=2, dec_dims=[5,10,15,20])
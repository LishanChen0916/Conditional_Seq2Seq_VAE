from __future__ import unicode_literals, print_function, division
import math
import random
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataloader import *

'''Hyper parameters'''
# 26 alphabet plus "SOS" and "EOS"
token_size = 28
condition_size = 4
condition_hidden_size = 8
# Token + condition
hidden_size = 256
latent_size = 32
learning_rate = 0.05
'''Hyper parameters'''

# Compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

# Compute Gaussian score
def Gaussian_score(words):
    words_list = []
    score = 0
    path = 'Data/train.txt'
    with open(path,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def KL_weight_Monotonic(KL_weight):
    slope = 0.001
    KL_weight += slope

    return min(KL_weight, 1.0)

class VAE(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size+8, hidden_size)
    
    def forward(self, x, condition):
        mu = self.mean(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, condition), dim=2)
        z = self.latent2hidden(z)
        
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(-1, 1, self.hidden_size)
        outputs, hidden = self.lstm(embedded, hidden)
        
        return outputs, hidden

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size - 8, device=device)
        c0 = torch.zeros(1, 1, self.hidden_size - 8, device=device)
        return (h0, c0)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(1, 1, self.hidden_size)
        embedded = F.relu(embedded)
        output, hidden = self.lstm(embedded, hidden)
        
        pred = self.out(output).view(-1, token_size)
        
        return pred, hidden

class Seq2SeqCVAE(nn.Module):
    def __init__(self, encoder, decoder, vae):
        super(Seq2SeqCVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

        self.embedding = nn.Embedding(condition_size, condition_hidden_size)

    def forward(self, inputs, targets, input_condition, target_condition, teacher_forcing_ratio=0.8):
        encoder_hidden_init = self.encoder.initHidden()
        input_condition_embedded = self.embedding(input_condition).view(1, 1, -1)
        target_condition_embedded = self.embedding(target_condition).view(1, 1, -1)

        hidden_ = torch.cat((encoder_hidden_init[0], input_condition_embedded), dim=2)
        cell_ = torch.cat((encoder_hidden_init[1], input_condition_embedded), dim=2)
        
        # -------------------------Encoder part-------------------------------#
        _, (hidden_, cell_) = self.encoder(inputs, (hidden_, cell_))
        hidden_z, hidden_mean, hidden_logvar = self.vae(hidden_, target_condition_embedded)
        cell_z, cell_mean, cell_logvar = self.vae(cell_, target_condition_embedded)

        mean = torch.cat((hidden_mean, cell_mean), dim=2)
        logvar = torch.cat((hidden_logvar, cell_logvar), dim=2)

        # -------------------------Decoder part-------------------------------#
        target_len = targets.size(0)
        outputs = torch.zeros(target_len, 1, token_size).to(device)
        
        # Decoder's first input: 'SOS' token
        input_ = torch.unsqueeze(inputs[0], 0)
        prediction = []
        
        for t in range(1, target_len):
            
            output, (hidden_z, cell_z) = self.decoder(input_, (hidden_z, cell_z))

            # Place predictions in a tensor for each token
            outputs[t] = output

            # Decide whether we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            pred = output.argmax(1)
            
            input_ = torch.unsqueeze(targets[t], 0) if teacher_force else pred
            
            if pred != 27 and t < targets.size(0) - 1:
                prediction.append(pred)
            else:
                break

        return outputs, prediction, mean, logvar
        
def loss_fn(recon_x, x, mean, logvar, criterion):
    CE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return CE, KLD

def train(train_loader, test_loader, model, epochs=150, optimizer=optim.SGD):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    CELoss = np.zeros(epochs)
    KLDLoss = np.zeros(epochs)
    bleu_score = []
    bleu_score_ = np.zeros(epochs)
    KL_Weight = np.zeros(epochs)
    KLW = 0.0
    
    for epoch in range(epochs):
        CELoss_sum = 0
        KLDLoss_sum = 0
        loss_sum = 0
        KLW = KL_weight_Monotonic(KLW)
        if epoch % 20 == 0:
            KLW = 0.0

        model.train()
        for train_data, condition in train_loader:
            # Remove all the dimensions of input of size 1
            train_data = torch.squeeze(train_data).to(device)
            condition = torch.squeeze(condition).to(device)
            train_label = train_data
            input_condition = condition
            output_condition = condition

            optimizer.zero_grad()
            
            output, prediction, mean, logvar = model(train_data, train_label, input_condition, output_condition)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = train_data[1:].view(-1)
            
            CE, KLD = loss_fn(output, target, mean, logvar, criterion=nn.CrossEntropyLoss())
            loss = CE + KLW * KLD
            CELoss_sum += CE
            KLDLoss_sum += KLW * KLD
            loss_sum = CELoss_sum + KLDLoss_sum
            CELoss[epoch] = CELoss_sum
            KLDLoss[epoch] = KLDLoss_sum
            KL_Weight[epoch] = KLW

            loss.backward()
            optimizer.step()

        print("Epochs[%3d/%3d] \nCrossEntropyLoss : %f KLDLoss : %f" % (epoch, epochs, CELoss_sum, KLDLoss_sum))

        model.eval()
        with torch.no_grad():
            for test_data, test_label, input_condition, target_condition in test_loader: 
                test_data = torch.squeeze(test_data).to(device)
                test_label = torch.squeeze(test_label).to(device)
                input_condition = torch.squeeze(input_condition).to(device)
                target_condition = torch.squeeze(target_condition).to(device)

                output, prediction, _, _ = model(test_data, test_label, input_condition, target_condition)
                
                targetString = c.LongtensorToString(test_label, show_token=False, check_end=True)
                outputString = c.LongtensorToString(prediction, show_token=False, check_end=False)
                
                bleu_score.append(compute_bleu(outputString, targetString))
                bleu_score_[epoch] = sum(bleu_score) / len(bleu_score)

            print('BlEU-4 Score: {score}'.format(score=sum(bleu_score) / len(bleu_score)))

    torch.save(model.state_dict(), 'Result/model/Seq2SeqCVAE_KLDSlope=0.001_tfr=0.8_Cyclical.pkl')
    plotCurve('Training: Loss&Score_KLDSlope=0.001_tfr=0.8_Cyclical', CELoss, KLDLoss, bleu_score_)

def evaluate(test_loader, train_dataset, model, path):
    bleu_score = []
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        for test_data, test_label, input_condition, target_condition in test_loader: 
            test_data = torch.squeeze(test_data).to(device)
            test_label = torch.squeeze(test_label).to(device)
            input_condition = torch.squeeze(input_condition).to(device)
            target_condition = torch.squeeze(target_condition).to(device)

            _, prediction, _, _ = model(test_data, test_label, input_condition, target_condition)

            print('=========================')
            print('Input:  ', c.LongtensorToString(test_data, show_token=False, check_end=True))
            print('Target: ', c.LongtensorToString(test_label, show_token=False, check_end=True))
            print('Pred:   ', c.LongtensorToString(prediction, show_token=False, check_end=False))
            print('=========================')

            targetString = c.LongtensorToString(test_label, show_token=False, check_end=True)
            outputString = c.LongtensorToString(prediction, show_token=False, check_end=False)
            bleu_score.append(compute_bleu(outputString, targetString))

        print('BlEU-4 Score: {score}\n'.format(score=sum(bleu_score) / len(bleu_score)))

        predictions = []
        for i in range(100):
            pred = []
            index = random.randint(0, 4907)
            for tense in range(4):
                _, prediction, _, _ = model(
                    train_dataset.__getitem__(index)[0].to(device), 
                    train_dataset.__getitem__(index-(index%4)+tense)[0].to(device), 
                    torch.LongTensor([index%4]).to(device), 
                    torch.LongTensor([tense]).to(device)
                )
                pred.append(c.LongtensorToString(prediction, show_token=False, check_end=False))
            print(pred)
            predictions.append(pred)
        print('Gaussian-score: {score}'.format(score=Gaussian_score(predictions)))

def plotCurve(title, CrossEntropyLoss, KLLoss, score):
    plt.figure(figsize=(8,4.5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(CrossEntropyLoss, label='CrossEntropyLoss', c='C1')
    plt.plot(KLLoss, label='KLLoss', c='C2')
    h1, l1 = plt.gca().get_legend_handles_labels()

    ax = plt.gca().twinx()
    plt.ylabel('Score')
    ax.plot(score, label='Score', c='C3')
    h2, l2 = ax.get_legend_handles_labels()
    

    plt.legend(loc='best')
    plt.title(title)

    ax.legend(h1+h2, l1+l2, loc='best')

    #plt.show()
    filename = title
    plt.savefig('Result/Line_Chart/' + filename + ".png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TextDataloader('Data', 'train')
    test_dataset = TextDataloader('Data', 'test')

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    c = CharDict()
    encoder = EncoderRNN(input_size=token_size, hidden_size=hidden_size).to(device)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size=token_size).to(device)
    vae = VAE(hidden_size=hidden_size, latent_size=latent_size).to(device)

    model = Seq2SeqCVAE(encoder, decoder, vae).to(device)

    # Train
    train(train_loader, test_loader, model)

    # Evaluate
    path = 'Result/model/Seq2SeqCVAE_KLDSlope=0.001_tfr=0.8_Cyclical.pkl'
    evaluate(test_loader, train_dataset, model, path)
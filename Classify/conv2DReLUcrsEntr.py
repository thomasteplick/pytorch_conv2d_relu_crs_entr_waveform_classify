'''
Created on Nov 14, 2026
Artificial Neural Network (ANN) using 2D Convolution, ReLU activation functions, and Cross Entropy loss function.
Otherwise known as Convolution Neural Network (CNN) using back-propagation of errors to train the weights and biases.
The task is to classify four waveforms, each consisting of several chirp signals: linear, quadratic, logarithmic, hyperbolic.
Display the learning curve and the distribution of the properties for each class.  Test the trained ANN with a separate
set of samples and tabulate the results.  The STFT is used to convert the time signal to an image with frequency and time axes.

The PyTorch framework was used to build the ANN.

@author: Tom Teplick
'''
import torch
from torch import nn
from scipy import signal    
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os.path
import random
import math
from datetime import datetime
from scipy.io.wavfile import write

nwaveforms = 4
# number of chirps in each waveform
nchirps = 5

# number of chirp parameters: chirp type, f1, f2, amplitude
nchirpParams = 4

# Print input tensor shapes in Sequential layers
class DebugPrint(nn.Module):
    def __init__(self, expected):
        super(DebugPrint, self).__init__()
        self.expected = expected

    def forward(self, x):
        print(f"wanted layer {self.expected},  have {x.shape}")
        return x

# Define ANN model, choose hidden layers and features in each layer
class NeuralNetwork(nn.Module):
    def __init__(self, epochs, conv2dlayers):
        super().__init__()
        self.conv2D_relu_crsEntr_stack = None
        match conv2dlayers:
            case 1:
                self.conv2D_relu_crsEntr_stack = nn.Sequential(
                    #DebugPrint("1, Size([1,32,32])"),
                    nn.Conv2d(1, 4, 5),
                    #DebugPrint("2, Size([[4,28,28])"),
                    nn.ReLU(),
                    #DebugPrint("3, Size([4,28,28])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    #DebugPrint("4, Size([4,14,14])"),
                    nn.Flatten(start_dim=0),
                    #DebugPrint("5, Size([784])"),
                    # [((32-5)+1)/2]^2 * 4 = 784
                    nn.Linear(784, nwaveforms))
                    #DebugPrint("6, Size([4])"))
            case 2:           
                self.conv2D_relu_crsEntr_stack = nn.Sequential(
                    #DebugPrint("1, Size([1,32,32])"),
                    nn.Conv2d(1, 4, 5),
                    #DebugPrint("2, Size([[4,28,28])"),
                    nn.ReLU(),
                    #DebugPrint("3, Size([4,28,28])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    #DebugPrint("4, Size([4,14,14])"),
                    nn.Conv2d(4, 8, 3),
                    #DebugPrint("5, Size([8,12,12])"),
                    nn.ReLU(),
                    #DebugPrint("6, Size([8,12,12])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    #DebugPrint("7, Size([8,6,6])"),
                    nn.Flatten(start_dim=0),
                    #DebugPrint("8, Size([288])"),
                    # [((14-3)+1)/2]^2 * 8 = 288
                    nn.Linear(288, nwaveforms))
                    #DebugPrint("9, Size([4])"))
            case 3:
                self.conv2D_relu_crsEntr_stack = nn.Sequential(
                    #DebugPrint("1, Size([1,32,32])"),
                    nn.Conv2d(1, 4, 5),
                    #DebugPrint("2, Size([[4,28,28])"),
                    nn.ReLU(),
                    #DebugPrint("3, Size([4,28,28])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    #DebugPrint("4, Size([4,14,14])"),
                    nn.Conv2d(4, 8, 3),
                    #DebugPrint("5, Size([8,12,12])"),
                    nn.ReLU(),
                    #DebugPrint("6, Size([8,12,12])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    #DebugPrint("7, Size([8,6,6])"),
                    nn.Conv2d(8, 16, 3),
                    #DebugPrint("8, Size([16,4,4])"),
                    nn.ReLU(),
                    #DebugPrint("9, Size([16,4,4])"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # [((6-3)+1)/2]^2 * 16 = 64
                    #DebugPrint("10, Size([16,2,2])"),
                    nn.Flatten(start_dim=0),
                    #DebugPrint("11, Size([64])"),
                    nn.Linear(64, nwaveforms))
                    #DebugPrint("12, Size([4])"))
            case _:
                print(f"{conv2dlayers} is not supported: choose (1,2,3)")
        self.loss = torch.zeros(epochs, dtype=torch.float32)
        self.count = torch.zeros(nwaveforms, dtype=torch.int32)
        self.correct = torch.zeros(nwaveforms, dtype=torch.int32)
        
        # Frequency-swept cosine generator types.  quadratic can be vertex_zero = True or False
        self.chirptype = ['linear', 'quadratic', 'logarithmic', 'hyperbolic']
                        
    def forward(self, x):
        logits = self.conv2D_relu_crsEntr_stack(x)
        return logits
    
    def createSamples(self, nsamples, imgdim, input_data, target, waveformData, snr):
        # Create the input data and the target
        # block size = 64, 32 frames, 2 sample hop, no padding, 126/64 sec  
        
        # block of waveform samples, Length of the FFT used
        block = 64
        # Time increment in signal samples for sliding window.
        hopTime = 2
        # sample rate = 64 samples/sec
        sr = 64.0
        # sample period in seconds
        ts = 1.0/sr       # sample period in seconds
        
        # calculate gaussian variance from snr: snr=10*log10(sigpwr/noisepwr)
        div = math.pow(10.0, float(snr)/10.0)
                
        # window for the STFT to reduce spectral leakage
        win = signal.windows.hamming(block)
                
        # waveform length in samples
        nwvfmSamples = block + (block//2 - 1)*hopTime
        t = np.arange(nwvfmSamples) * ts
        tend = nwvfmSamples*ts
        for samp in range(nsamples):
            wvform = torch.randint(0, nwaveforms, (1,), dtype=torch.uint8).item()
            target[samp] = wvform
            sumwvfrm = np.zeros(nwvfmSamples, dtype=np.float32)  
            # signal power
            sigpwr = 0.0
                # loop over the chirps
            for chrp in range(nchirps):
                # chirp param number
                ichirpParam = 0
                
                # get chirp type
                chirp_t = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                
                # get f1 and f2, start and stop frequences of chirp
                freq1 = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                freq2 = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                
                # get amplitude of chirp
                ampl = waveformData[wvform][chrp][ichirpParam]
                sigpwr += ampl
                
                # select the chirp types and sum them, add noise to each chirp
                match chirp_t:
                    case 0:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[0])
                    case 1:
                        if (freq1+freq2)%2 == 0:
                            sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=True)
                        else:
                            sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=False)                         
                    case 2:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[2])
                    case 3:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[3])
                    case _:
                        print(f"undefined chirp type in createSamples: {chirp_t}")
             
            # calculate standard deviation (sigma) of additive white gaussian noise (AWGN)
            sd = math.sqrt((sigpwr/2.0)/div)
            # create the random number generator
            rng = np.random.default_rng()
            # generate the gaussian variate
            gauss = rng.normal(loc=0.0, scale=sd, size=nwvfmSamples)
                      
            # Perform the STFT on this waveform + AWGN and assign it to input_data
            SFT = signal.ShortTimeFFT(win, hop=hopTime, fs=sr, mfft=block, scale_to='magnitude')
            aStft = abs(SFT.stft(sumwvfrm + gauss))
            input_data[samp] = torch.from_numpy(aStft[0:imgdim,31:31+imgdim])
                
                
    def train_(self, dataloader, loss_fn, optimizer, device, epoch):
        self.train()
        # one-hot vector target 
        target = torch.zeros(nwaveforms, dtype=torch.float32)
        losssum = 0.0
        count = 0
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            # target waveform has probability 1, all others have  probability 0
            target[y.item()] = 1.0
            # Compute prediction error
            pred = self.forward(X)
            #loss = loss_fn(pred, y.squeeze(0))
            loss = loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # clear the waveform having probabilitiy 1
            target[y.item()] = 0
            losssum += loss.item()
            count += 1
        # save the loss for learning curve display
        self.loss[epoch] = losssum/count
                    
    def test_(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                # loop over y tensor and determine if pred[i].argmax() is equal,
                # count the number of samples for each class
                for _,target in enumerate(y):
                    self.count[target.item()] += 1
                    if pred.argmax() == target:
                        self.correct[target.item()] += 1
                        
    def displayTestResults(self):
               
        # show classes, count, correct, %correct, totals
        # heading
        ttlCorrect = 0
        ttlCount = 0
        sep = 8
        horzBorder = '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|'
        print(horzBorder)
        print("|{0:8s}|{1:8s}|{2:8s}|{3:8s}|".format("Class", "Count", "Correct", "%Correct"))
        print(horzBorder)
        for i, val in enumerate(self.count):
            print("|{0:<8d}|{1:<8d}|{2:<8d}|{3:<8d}|".format(i+1, val, self.correct[i], self.correct[i]*100//val))
            print(horzBorder)
            ttlCount += val
            ttlCorrect += self.correct[i]
        print("|{0:<8s}|{1:<8d}|{2:<8d}|{3:<8d}|".format("Totals", ttlCount, ttlCorrect, round((ttlCorrect/ttlCount*100).item())))
        print(horzBorder)
        
    def  plotWaveformSTFT(self, waveformData):
        # plot the images of the STFT of the waveforms using matplotlib subplots(2,2)
        # show colormaps of the images
        # no gaussian noise
        # create AM waveform usimng 1000 Hz center frequency and save to WAV file
        
        block = 64
        # Time increment in signal samples for sliding window.
        hopTime = 2
        # sample rate = 64 samples/sec
        sr = 64
        # sample period in seconds = 1/sample rate
        ts = 1.0/sr       # sample period in seconds
        
        # window for the STFT to reduce spectral leakage
        win = signal.windows.hamming(block)
        
        nwvfmSamples = block + (block//2 - 1)*hopTime
        t = np.arange(nwvfmSamples) * ts
        tend = nwvfmSamples*ts                        
        fig, axs = plt.subplots(nrows=2, ncols=2, layout="constrained", figsize=(8.0, 6.0))
        axs = axs.flatten() 
        for wvform in range(nwaveforms):
            sumwvfrm = np.zeros(nwvfmSamples, dtype=np.float32)  
            # loop over the chirps
            for chrp in range(nchirps):
                # chirp param number
                ichirpParam = 0
                
                # get chirp type
                chirp_t = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                
                # get f1 and f2, start and stop frequences of chirp
                freq1 = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                freq2 = waveformData[wvform][chrp][ichirpParam]
                ichirpParam += 1
                
                # get amplitude of chirp
                ampl = waveformData[wvform][chrp][ichirpParam]
                
                # select the chirp types and sum them
                match chirp_t:
                    case 0:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[0])
                    case 1:
                        if (freq1+freq2)%2 == 0:
                            sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=True)
                        else:
                            sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=False)                         
                    case 2:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[2])
                    case 3:
                        sumwvfrm += ampl*signal.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[3])
                    case _:
                        print(f"undefined chirp type in createSamples: {chirp_t}")
                        
            # Perform the STFT on this waveform and add the image to the plot
            SFT = signal.ShortTimeFFT(win, hop=hopTime, fs=sr, mfft=block, scale_to='magnitude')
            aStft = abs(SFT.stft(sumwvfrm))
            axs[wvform].set_title(rf"Waveform {wvform+1} STFT ({SFT.m_num*SFT.T:g}$\,s$ Hamming window)")
            axs[wvform].set_xlabel(f"Time $t$ in seconds ({SFT.p_num(nwvfmSamples)} slices, " +
                           rf"$\Delta t = {SFT.delta_t:g}\,$s)")
            axs[wvform].set_ylabel(f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                           rf"$\Delta f = {SFT.delta_f:g}\,$Hz)")
            im = axs[wvform].imshow(aStft, origin='lower', aspect='auto',
                             extent=SFT.extent(nwvfmSamples), cmap='viridis')
            if wvform == 1 or wvform == 3:
                fig.colorbar(im, ax=axs[wvform], label="Magnitude $|S_x(t, f)|$")
            # save waveform to WAV file for listening on Media Player (low frequencies not audible)
            write(filename=f"data\\waveform{wvform}.wav", rate=sr, data=sumwvfrm/np.max(sumwvfrm))
        plt.show(block=True)  
                
    def plotLearningCurve(self):
        # show cross entropy loss per epoch from training the MLP
        _, ax = plt.subplots(1,1)
        x = np.arange(self.loss.size(0), dtype=np.uint32)
        y = self.loss.numpy()
        # line properties: width, style
        lnwid = 3
        lnsty = "solid"
        ax.plot(x, y, linewidth=lnwid, linestyle=lnsty)        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title('Learning Curve')
        plt.show(block=True)          
                                          
def runConv2DReluCrossEntropy():
    
    parser = argparse.ArgumentParser(prog='conv2DReLUcrsEntr', description="Convolution Neural Network (ReLU, Cross Entropy Loss) for waveform classification using PyTorch")
    parser.add_argument('epochs', type=int, choices=[100, 200, 500, 1000, 2000, 5000, 10000], help="the number of epochs for training and testing")
    parser.add_argument('batch_size', type=int, choices=[1, 2, 10, 16, 32, 64], help="the batch size for training and testing")
    parser.add_argument('learning_rate', type=float, choices=[.001, .002, .01, .02], help="learning rate hyperparameter for training")
    parser.add_argument('momentum', type=float, choices=[.5, .6, .7, .8, .9], help="momentum hyperparameter for training")
    parser.add_argument('convolution2Dlayers', type=int, choices=[1,2,3], help="number of convolution 2D layers")
    parser.add_argument('snr', type=int, choices=[5, 10, 15, 20], help="signal-to-noise ratio using gaussian noise")
    args = parser.parse_args()
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # seed the random number generators
    torch.manual_seed(int(time.time()))
    np.random.default_rng()
    
    # each epoch consists of batch_size samples which consists of (input,target)
    epochs = args.epochs  #args.epochs
    # number of samples consisting of (input,target)
    batch_size = args.batch_size #args.batch_size
    # learning rate to scale the backpropagation error
    learning_rate = args.learning_rate #args.learning_rate
    # momentum
    momentum = args.momentum    
    # Convolutional 2D layers
    convolution2D_layers = args.convolution2Dlayers 
    
    # signal-to-noise ration (SNR)
    snr = args.snr  
     
    # initialize random number generator
    random.seed()

    nsamples = epochs*batch_size
    
    # image dimension
    imgdim = 32
    
    conv2DReLUcrossEntropyModel = "data\\conv2DReLUcrossEntropy.pth"
    waveformDataFile = "data\\waveformData.txt"
           
    waveformData = np.zeros((nwaveforms, nchirps, nchirpParams), dtype=np.uint8)
    
    # Set waveform chirp frequencies if data file doesn't exist, otherwise read in the file
    if os.path.exists(waveformDataFile):
        with open(waveformDataFile, mode='r', encoding="utf-8") as fin:
            wvform = 0
            for line in fin:
                items = line.split(' ')
                for chrp in range(nchirps):
                    k = chrp*nchirpParams
                    for param in range(nchirpParams):
                        waveformData[wvform][chrp][param] = int(items[k + param])
                wvform += 1
    else:
        # Each waveform consists of nchirps chirps, starting at t1=0 sec and ending at t2=sr*N sec
        # Each chirp starts at a different f1 and ends at a different f2
        # Each chirp has a different amplitude
        freqlo = 5
        freqhi = 25
        freqmid = (freqlo+freqhi)//2
        
        # chirp amplitudes
        namps = 10
                
        chirptype_size = 4
        
        for wvform in range(nwaveforms):
            # permutation of the chirp types
            permchirp = torch.randperm(chirptype_size)
            # permutation of the amplitudes
            permamp = torch.randperm(namps) + 5
                                          
            # loop over the chirps
            for chrp in range(nchirps):
                # select chirp type
                # select f1 and f2, endpoint frequencies of chirp
                # select amplitude of chirp
                
                # chirp param number
                ichirpParam = 0
                                
                waveformData[wvform][chrp][ichirpParam] = permchirp[chrp%len(permchirp)].item()
                ichirpParam += 1
                         
                # select frequency start and stop          
                f1 =  torch.randint(freqlo, freqhi, (1,), dtype = torch.uint8)
                waveformData[wvform][chrp][ichirpParam] = f1.item()
                ichirpParam += 1
                if f1 < freqmid:
                    waveformData[wvform][chrp][ichirpParam] = torch.randint(freqmid, freqhi, (1,), dtype = torch.uint8)
                else:
                    waveformData[wvform][chrp][ichirpParam] = torch.randint(freqlo, freqmid, (1,), dtype = torch.uint8)
                ichirpParam += 1
                
                # select amplitude of chirp   
                waveformData[wvform][chrp][ichirpParam] = permamp[chrp%len(permamp)].item()
                ichirpParam += 1
                        
    # Input data and the target tensors comprise the samples used to train the ANN
    input_data = torch.empty(nsamples, imgdim, imgdim, dtype=torch.float32)
    target = torch.empty(nsamples, 1, dtype=torch.uint8)
    
    # create the ANN
    model = NeuralNetwork(epochs, convolution2D_layers).to(device)
    print(f"Model: {model}")    
    
    # Train the model
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, imgdim, input_data, target, waveformData, snr)
        
    # find max in the input_data
    maxData = torch.max(torch.abs(input_data)).item()
    # Create dataset from sample tensors, normalize the input_data to (0,1]
    dataset = data.TensorDataset(input_data/maxData, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Display the waveform's STFT (spectrogram) using matplotlib
    print("Display the waveform's STFT (spectrogram), no additive white Gaussian noise")
    model.plotWaveformSTFT(waveformData)
       
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    trainbegin = datetime.now()   
    print(f"Begin training the model: {trainbegin}")
    for t in range(epochs):
        print("*", end='', flush=True)
        model.train_(dataloader, loss_fn, optimizer, device, t)
        if (t+1) % 50 == 0:
            print()
    trainend = datetime.now()
    elapsedtime = trainend-trainbegin
    print(f"\nDone training: {trainend}")
    print(f"Elapsed time: {elapsedtime}")

    # Display Cross Entropy Loss using matplotlib
    print("Display the Cross Entropy Loss")
    model.plotLearningCurve()
    
    torch.save(model.state_dict(), conv2DReLUcrossEntropyModel)
    print("Saved PyTorch Model State to %s" % conv2DReLUcrossEntropyModel)
    
    # Save the waveform data, waveform, f0, f1, amp1, 
    with open(waveformDataFile, mode='w', encoding="utf-8") as fout:
        # loop over the waveforms
        for wvform in range(nwaveforms):
            # loop over the chirps
            for chrp in range(nchirps):
                # loop over the chirp parameters
                for param in range(nchirpParams):
                    print(f"{waveformData[wvform][chrp][param]} ", file=fout, end='')
            print(file=fout)
            
    # Test the model using different samples
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, imgdim, input_data, target, waveformData, snr)
    
    # find max in the input_data
    maxData = torch.max(torch.abs(input_data)).item()         
    # Create dataset from sample tensors, normalize the input_data to (0,1]
    dataset = data.TensorDataset(input_data/maxData, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    
    testbegin = datetime.now()   
    print(f"Begin testing the model: {testbegin}")
    model.test_(dataloader, device)
    testend = datetime.now()
    elapsedtime = testend-testbegin
    print(f"\nDone testing.  {testend}")
    print(f"Elapsed time: {elapsedtime} Test Results:")
        
    # Tabulate and display the test results: class, count, correct, %correct, totals
    model.displayTestResults()
    
if __name__ == '__main__':
    runConv2DReluCrossEntropy()
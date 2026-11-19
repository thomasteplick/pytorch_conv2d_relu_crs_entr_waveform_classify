'''
Created on Nov 14, 2026
Artificial Neural Network (ANN) using 2D Convolution, ReLU activation functions, and Cross Entropy loss function.
Otherwise known as Convolution Neural Network (CNN) using back-propagation of errors to train the weights and biases.
The task is to classify four waveforms, each consisting of several chirp signals: linear, quadratic, logarithmic, hyperbolic.
Display the learning curve and the distribution of the properties for each class.  Test the trained ANN with a separate
set of samples and tabulate the results.

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
from datetime import datetime
from scipy.io.wavfile import write

nwaveforms = 4
# number of chirps in each waveform
nchirps = 5

# number of chirp parameters: chirp type, f1, f2, amplitude
nchirpParams = 4

# Define ANN model, choose hidden layers and features in each layer
class NeuralNetwork(nn.Module):
    def __init__(self, epochs, conv2dlayers):
        super().__init__()
        self.conv2D_relu_crsEntr_stack = None
        match conv2dlayers:
            case 1:
                self.Conv2D_relu_crsEntr_stack = nn.Sequential(
                    nn.Conv2d(1, 4, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.flatten(),
                    # [((32-5)+1)/2]^2 * 4 = 784
                    nn.Linear(784, nwaveforms))
            case 2:           
                self.conv2D_relu_crsEntr_stack = nn.Sequential(
                    nn.Conv2d(1, 4, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(4, 8, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.flatten(),
                    # [((14-3)+1)/2]^2 * 8 = 288
                    nn.Linear(288, nwaveforms))
            case 3:
                self.conv2D_relu_crsEntr_stack = nn.Sequential(
                    nn.Conv2d(1, 4, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(4, 8, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(8, 16, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # [((6-3)+1)/2]^2 * 16 = 64
                    nn.flatten(),
                    nn.Linear(64, nwaveforms))
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
    
    def createSamples(self, nsamples, imgdim, input_data, target, waveformData):
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
        
        # waveform length in samples
        nwvfmSamples = block + (block//2 - 1)*hopTime
        t = np.arange(nwvfmSamples) * ts
        tend = nwvfmSamples*ts
        for samp in range(nsamples):
            wvform = torch.randint(0, nwaveforms, (1,), dtype=torch.uint8).item()
            target[samp] = wvform
            sumwvfrm = np.zeros((nwvfmSamples,1), dtype=np.float32)        
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
                
                # select the chirp types and sum them, add noise to each chirp
                match chirp_t:
                    case 0:
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[0])
                    case 1:
                        if (freq1+freq2)%2 == 0:
                            sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=True)
                        else:
                            sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=False)                            
                    case 2:
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[2])
                    case 3:
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[3])
                    case _:
                        print(f"undefined chirp type in createSamples: {chirp_t}")
                        
                # Perform the STFT on this waveform and assign it to input_data
                win = signal.windows.hamming(block)
                SFT = signal.ShortTimeFFT(win, hop=hopTime, fs=sr, mfft=block, scale_to='magnitude')
                aStft = abs(SFT.stft(sumwvfrm))
                input_data[samp] = torch.from_numpy(aStft[0:imgdim,0:imgdim])
                
                
    def train_(self, dataloader, loss_fn, optimizer, device):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = self.forward(X)
            loss = loss_fn(pred, y.squeeze(1))
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # save the loss for learning curve display
            self.loss[batch] = loss.item()
            
    def test_(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                # loop over y tensor and determine if pred[i].argmax() is equal,
                # count the number of samples for each class
                for i,target in enumerate(y):
                    self.count[target.item()] += 1
                    if pred[i].argmax() == target:
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
        
        block = 64
        # Time increment in signal samples for sliding window.
        hopTime = 2
        # sample rate = 64 samples/sec
        sr = 64.0
        # sample period in seconds = 1/sample rate
        ts = 1.0/sr       # sample period in seconds
        nwvfmSamples = block + (block//2 - 1)*hopTime
        t = np.arange(nwvfmSamples) * ts
        tend = nwvfmSamples*ts                        
        fig, axs = plt.subplots(nrows=2, ncols=2, layout="constrained")
        axs = axs.flatten() 
        for wvform in range(nwaveforms):
            sumwvfrm = np.zeros((nwvfmSamples,1), dtype=np.float32)        
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
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[0])
                    case 1:
                        vtx0 = True 
                        if (freq1+freq2)%2 == 0:
                            vtx0 = False 
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[1], vertex_zero=vtx0)
                    case 2:
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[2])
                    case 3:
                        sumwvfrm += ampl*torch.chirp(t, f0=freq1, f1=freq2, t1=tend, method=self.chirptype[3])
                    case _:
                        print(f"undefined chirp type in createSamples: {chirp_t}")                        
            # Perform the STFT on this waveform and add the image to the plot
            win = signal.windows.hamming(block)
            SFT = signal.ShortTimeFFT(win, hop=hopTime, fs=sr, mfft=block, scale_to='magnitude')
            aStft = abs(SFT.stft(sumwvfrm))
            axs[wvform].set_title(rf"Waveform {wvform+1} STFT ({SFT.m_num*SFT.T:g}$\,s$ Hamming window")
            axs[wvform].set_xlabel(f"Time $t$ in seconds ({SFT.p_num(nwvfmSamples)} slices, " +
                           rf"$\Delta t = {SFT.delta_t:g}\,$s)")
            axs[wvform].set_ylabel(f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                           rf"$\Delta f = {SFT.delta_f:g}\,$Hz)")
            im = axs[wvform].imshow(aStft, origin='lower', aspect='auto',
                             extent=SFT.extent(nwvfmSamples), cmap='viridis')
            if wvform == 1 or wvform == 3:
                fig.colorbar(im, ax=axs[wvform], label="Magnitude $|S_x(t, f)|$")
            # save waveform to WAV file for listening on Media Player (low frequencies not audible)
            write(file=f"data\\waveform{wvform}.wav", rate=sr, data=sumwvfrm/np.max(sumwvfrm))
        fig.suptitle('Waveform Short-Time Fourier Transform', y=0.95)
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
    args = parser.parse_args()
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # seed the random number generator
    torch.manual_seed(int(time.time()))
    
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

    nsamples = epochs*batch_size
    
    # image dimension
    imgdim = 32
    
    conv2DReLUcrossEntropyModel = "data\\conv2DReLUcrossEntropy.pth"
    waveformDataFile = "data\\waveformData.txt"
           
    waveformData = torch.empty((nwaveforms, nchirps, nchirpParams), dtype=torch.uint8)
    
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
                    waveformData[wvform][chrp][ichirpParam] = torch.randint(freqmid, freqhi, (1,), dtype = torch.unit8)
                else:
                    waveformData[wvform][chrp][ichirpParam] = torch.randint(freqlo, freqmid, (1,), dtype = torch.unit8)
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
    model.createSamples(nsamples, imgdim, input_data, target, waveformData)
        
    # find max in the input_data
    maxData = torch.max(torch.abs(input_data)).item()
    print(f"maxData={maxData}")         
    # Create dataset from sample tensors, normalize the input_data to (0,1]
    dataset = data.TensorDataset(input_data/maxData, target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Display the waveform's STFT (spectrogram) using matplotlib
    print("Display the waveform's STFT (spectrogram)")
    model.plotWaveformSTFT(waveformData)
       
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    trainbegin = datetime.now()   
    print(f"Begin training the model: {trainbegin}")
    for t in range(epochs):
        print("*", end='', flush=True)
        model.train_(dataloader, loss_fn, optimizer, device)
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
    
    # Save the waveform data, waveform, f0, f1, amp1, ... amp5
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
    model.createSamples(nsamples, imgdim, input_data, target, waveformData)
    
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
%{
N为帧长度
M为后一帧相对前一帧的位移
%}
% 每次计算一个音频的MFCC
function [finalMFCCs] = MFCC(audioFilePath, N, M, fs, numCoefficients)
    frames = frame_block(audioFilePath, N, M);
    [windowedFrames, periodograms] = applyWindowing(frames, N);
    figure(1)
    plotSpectrogram('E:\EEC201\Project\Final Project\StudentAudioRecording\Zero-Training\Zero_train3.wav', N, M, fs);
    finalMFCCs = computeMFCCs(periodograms, fs, numCoefficients, N, M);
end

function frames = frame_block(audioFilePath, N, M)
    [file, ~] = audioread(audioFilePath);
    file = downsample(file,6);
    startIndex = 1;
    endIndex = N;
    numSamples = length(file);
    numFrames = floor((numSamples - N) / M) + 1;

    % 帧矩阵，每一列表示一帧
    frames = zeros(N, numFrames);
    frameIndex = 1;
    % 最后一帧不完整就舍弃，确保帧矩阵维度正确
    while endIndex <= numSamples
        currentFrame = file(startIndex:endIndex);
        frames(:, frameIndex) = currentFrame;  
        startIndex = startIndex + M;
        endIndex = startIndex + N - 1;
        frameIndex = frameIndex + 1;
    end
end

function [windowedFrames, periodograms] = applyWindowing(frames, N)
    numFrames = size(frames, 2);
    windowedFrames = zeros(size(frames));
    periodograms = zeros(N, numFrames);
    
    hammingWindow = 0.54 - 0.46 * cos(2 * pi * (0:N-1).' / (N-1));
    
    % 对每一帧加汉明窗并计算功率谱密度
    for k = 1:numFrames
        windowedFrame = frames(:, k) .* hammingWindow;
        windowedFrames(:, k) = windowedFrame;
        
        periodogram = abs(fft(windowedFrame, N)).^2;
        periodograms(:, k) = periodogram;
    end

end

function plotSpectrogram(audioFilePath, N, M, fs)
frames = frame_block(audioFilePath, N, M);
[windowedFrames, periodograms] = applyWindowing(frames, N);
N = size(periodograms,1);
numFrames = size(periodograms,2);
frameTime = (((1:numFrames)-1) * M + N/2)/fs * 1000;
W2 = N/2 + 1;
n2 = 1:W2;
freq = (n2-1)*fs/N;

imagesc(frameTime,freq,periodograms(n2,:));
axis xy;xlabel('msec');ylabel('freqency/Hz');title('Spectrum before Mel-frequency Wrapping');

end


function mfccs = computeMFCCs(periodograms, fs, numCoefficients, N, M)
    % 计算MFCC
    numMelFilters = 20;  
    numFFTPoints = size(periodograms, 1);
    
    % 定义梅尔滤波器组
    melFilterBank1 = melFilterBank(numMelFilters, numFFTPoints, fs);

    figure(2)
    plotFilterBank(melFilterBank1, fs);

    % 计算梅尔频谱
    melSpectra = melFilterBank1 * periodograms;
    
    figure(3)
    plotMelSpectrogram(melSpectra, N, M, fs);

    
    % 对数转换和离散余弦变换
    logMelSpectra = log(melSpectra);
    mfccs = dct(logMelSpectra);

    % 保留前 numCoefficients 个系数
    mfccs = mfccs(1:numCoefficients, :);
end

function melFilterBank = melFilterBank(numMelFilters, numFFTPoints, fs)
    % 计算梅尔滤波器组
    function mel = hz2mel(hz)
        mel = 2595 * log10(1 + hz / 700);
    end
    
    function hz = mel2hz(mel)
        hz = 700 * (10.^(mel / 2595) - 1);
    end
    
    melMin = hz2mel(0);
    melMax = hz2mel(fs/2);
    
    % mel域频率采样点，并转换为hz域频率
    melPoints = linspace(melMin, melMax, numMelFilters + 2);
    hzPoints = mel2hz(melPoints);
    
    bin = floor((numFFTPoints + 1) * hzPoints / (fs/2));
    
    % mel滤波器组矩阵，每一行代表一个滤波器
    melFilterBank = zeros(numMelFilters, numFFTPoints);
    for i = 1:numMelFilters
        if bin(i) < 1
            bin(i) = 1;
        end

        if bin(i+1) > numFFTPoints
            bin(i+1) = numFFTPoints;
        end
        melFilterBank(i, bin(i):bin(i+1)) = (0:(bin(i+1)-bin(i))) / (bin(i+1)-bin(i));
        melFilterBank(i, bin(i+1):bin(i+2)) = (bin(i+2)-(bin(i+1):bin(i+2))) / (bin(i+2)-bin(i+1));
    end
end

function plotFilterBank(melFilterBank, fs)
numMelFilters = size(melFilterBank,1);
numFFTPoints = size(melFilterBank,2);
for i = 1:numMelFilters
    freq = ((1:numFFTPoints)-1) * (fs/2) / numFFTPoints;
    plot(freq,melFilterBank(i,:)),hold on;
end
hold off;xlabel('freqency/Hz');ylabel('amplitude');title('Mel-spaced Filterbank');
end

function plotMelSpectrogram(melSpectra, N, M, fs)
numMelFilters = size(melSpectra,1);
numFrames = size(melSpectra,2);
frameTime = (((1:numFrames)-1) * M + N/2)/fs * 1000;
melIndex=1:numMelFilters;
imagesc(frameTime,melIndex,melSpectra);
axis xy;xlabel('msec');ylabel('Melspectrum Coefficients');title('Spectrum after Mel-frequency Wrapping');

end
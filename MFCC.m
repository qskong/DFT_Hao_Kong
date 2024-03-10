
function [finalMFCCs] = MFCC(audioFilePath, N, M, fs, numCoefficients)

    frames = frame_block(audioFilePath, N, M);

    threshold = 0.01; 
    [frames, ~] = removeSilence(frames, threshold);

    [windowedFrames, periodograms] = applyWindowing(frames, N);
    finalMFCCs = computeMFCCs(periodograms, fs, numCoefficients);

end



function frames = frame_block(audioFilePath, N, M)
   
    [file, fs] = audioread(audioFilePath);

    startIndex = 1;
    endIndex = N;
    numSamples = length(file);

    numFrames = floor((numSamples - N) / (N - M)) + 1;

    frames = zeros(N, numFrames);
    frameIndex = 1;
    while endIndex <= numSamples
        currentFrame = file(startIndex:endIndex);
        frames(:, frameIndex) = currentFrame;  
        startIndex = startIndex + (N - M);
        endIndex = startIndex + N - 1;
        frameIndex = frameIndex + 1;
    end
end


function [windowedFrames, periodograms] = applyWindowing(frames, N)

    numFrames = size(frames, 2);

    windowedFrames = zeros(size(frames));
    periodograms = zeros(N, numFrames);
    
    hammingWindow = 0.54 - 0.46 * cos(2 * pi * (0:N-1).' / (N-1));
    
    for k = 1:numFrames
        windowedFrame = frames(:, k) .* hammingWindow;
        windowedFrames(:, k) = windowedFrame;
        
        periodogram = abs(fft(windowedFrame, N)).^2;
        periodograms(:, k) = periodogram;
    end
end

function [nonSilentFrames, energy] = removeSilence(frames, threshold)
  
    
    numFrames = size(frames, 2);
    energy = zeros(1, numFrames);
    
    for i = 1:numFrames
        energy(i) = sum(frames(:, i).^2);
    end
    
    nonSilentIdx = energy > threshold;
    nonSilentFrames = frames(:, nonSilentIdx);
end






function mfccs = computeMFCCs(periodograms, fs, numCoefficients)
   
    numMelFilters = 20;  
    numFFTPoints = size(periodograms, 1);
 
    function melFilterBank = melFilterBank(numMelFilters, numFFTPoints, fs)

        function mel = hz2mel(hz)
            mel = 2595 * log10(1 + hz / 700);
        end
        
        function hz = mel2hz(mel)
            hz = 700 * (10.^(mel / 2595) - 1);
        end
        
        melMin = hz2mel(0);
        melMax = hz2mel(fs/2);
        
        melPoints = linspace(melMin, melMax, numMelFilters + 2);
        
        hzPoints = mel2hz(melPoints);
        
        bin = floor((numFFTPoints + 1) * hzPoints / fs);
        
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

    melFilterBank1 = melFilterBank(numMelFilters, numFFTPoints, fs);

    melSpectra = melFilterBank1 * periodograms;
    
    logMelSpectra = log(melSpectra);
    mfccs = dct(logMelSpectra);

    mfccs = mfccs(1:numCoefficients, :);
end






clear;clc;

audioDir = 'E:\EEC201\Project\Final Project\StudentAudioRecording\Twelve-Training\';
speakers = {'Twelve_train1', 'Twelve_train2', 'Twelve_train3','Twelve_train4','Twelve_train6','Twelve_train7','Twelve_train8','Twelve_train9','Twelve_train10', 'Twelve_train11','Twelve_train12','Twelve_train13','Twelve_train14','Twelve_train15','Twelve_train16','Twelve_train17','Twelve_train18','Twelve_train19'}; % 假设你有11个说话者
N = 256;
M = 100;
fs = 8000;
numCoefficients = 20;
Q = 30; 

codebooks = cell(1, numel(speakers));

for i = 1:numel(speakers)
    audioFilePath = fullfile(audioDir, sprintf('%s.wav', speakers{i}));
    [mfccs] = MFCC(audioFilePath, N, M, fs, numCoefficients); 
    [~, C] = kmeans(mfccs', Q); 
    codebooks{i} = C; 

end



testAudioDir = 'E:\EEC201\Project\Final Project\StudentAudioRecording\Twelve-Testing\';
testFiles = {'Twelve_test1.wav', 'Twelve_test2.wav', 'Twelve_test3.wav','Twelve_test4.wav','Twelve_test6.wav','Twelve_test7.wav','Twelve_test8.wav','Twelve_test9.wav','Twelve_test10.wav', 'Twelve_test11.wav','Twelve_test12.wav','Twelve_test13.wav','Twelve_test14.wav','Twelve_test15.wav','Twelve_test16.wav','Twelve_test17.wav','Twelve_test18.wav','Twelve_test19.wav'}; % 更新为你的测试文件列表


testResults = strings(1, numel(testFiles));


for i = 1:numel(testFiles)
    testAudioFilePath = fullfile(testAudioDir, testFiles{i});
    testMFCCs = MFCC(testAudioFilePath, N, M, fs, numCoefficients);
    minDistortion = inf;
    minSpeaker = '';
    
    for j = 1:numel(speakers)
        C = codebooks{j}; 
        distortions = pdist2(testMFCCs', C); 
        meanDistortion = mean(min(distortions, [], 2)); 
        
        if meanDistortion < minDistortion
            minDistortion = meanDistortion;
            minSpeaker = speakers{j};
        end
    end
    
    testResults(i) = minSpeaker;
    fprintf('Test audio %s is closest to speaker: %s\n', testFiles{i}, minSpeaker);
end


audioFilePath1='E:\EEC201\Project\Final Project\StudentAudioRecording\Twelve-Training\Twelve_train2.wav';
audioFilePath2='E:\EEC201\Project\Final Project\StudentAudioRecording\Twelve-Training\Twelve_train10.wav';
plotMFCC(audioFilePath1,audioFilePath2, N, M, fs, numCoefficients);


function plotMFCC(audioFilePath1,audioFilePath2, N, M, fs, numCoefficients)
[mfccs1] = MFCC(audioFilePath1, N, M, fs, numCoefficients);
[mfccs2] = MFCC(audioFilePath2, N, M, fs, numCoefficients);
figure;
scatter(mfccs1(6,:),mfccs1(7,:),Marker="x");hold on;
scatter(mfccs2(6,:),mfccs2(7,:),Marker="o",MarkerEdgeColor='r');
xlabel('mfcc-6');ylabel('mfcc-7');legend('Speaker 2','Speaker 10');title('mfcc space');
end
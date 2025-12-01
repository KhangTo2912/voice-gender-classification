%Part 1: data preparation folder, file, dataset
folder = ["C:\Users\khang\OneDrive\Documents\MATLAB\PROJECT\Data\Male","C:\Users\khang\OneDrive\Documents\MATLAB\PROJECT\Data\Female"];
folderlabel= ["male", "female"];

%part2: set all record as same frequency speed 
fsSet = 20000; % all record need to be consistency
labels =[]; % empty label vector for male and female
features = []; % empty features vector

for k = 1 : length(folder)
    fol = folder(k);
    label = folderlabel(k);
files = dir(fullfile(fol, '*.wav'));
% Load audio files from the folder

for i = 1:length(files)
    fullpath = fullfile(fol, files(i).name); %build full file path to linking record in folder
    [y,Fs]= audioread(fullpath); %audio read file

    %Trim Silence in the record 
  
minSamples = fsSet * 1;
if length(y) < minSamples
        y = [y; zeros(minSamples-length(y), 1)];
end
    % Monaunal - Mono audio has only 1 chanel 
    % convert mono for simpler processing, and consistent dataset for all
    % features. 
    y = mean(y,2); %mono function
    y = y / max(abs(y)); % Normalize the audio signal
    % normalize - finds the loudest point in the audio 
    % same reason of mono, consistency dataset, improve data integrity for feature extraction
    y=resample(y,fsSet, Fs); % Set same sample rate
    % change the sample rate to value above
    % consistency and easier and faster processing to correct feature


%pitch features - fundamental frequency F0
%Pitch is the perceptual property of sounds. A higher frequency of vocal fold vibration results in a higher perceived pitch. 
% Male 85 to 155 Hz, female 165 to 255 Hz
   p = pitch(y, fsSet);
 % calculates fundamental frequency of record

    pMean=median(p);% The average of pitch of record
    % Male 85 to 155 Hz, female 165 to 255 Hz
    pIQR=iqr(p);
    %The Interquartile Range (IQR) is a statistical measure of the variable in mid 50% of a set of pitch data
% we use IQR to measures the variable the pitch is across the recording

%MFCC features: MFFCC is a timbre of the voice 
% Mel - Frequency Ceptral coefficients
numCoeffs = 8;
    mf = mfcc(y, fsSet,'NumCoeffs',numCoeffs); % function to calculate MFCC
    %  vector contains the values for a small frame of the audio
    mfMean=median(mf,1); 
    % same as pitch, we want a mean of MFCC across all frame as
    % characteristics of voice in the record
    mfSTD=std(mf,0,1);
    % show the standard or variablity of voice over time

%spectral features include 2 feature
% we want to know the center of mass of the spectrum - centroid 
% Bandwidth as the the range of frequencies 
centroid = spectralCentroid(y,fsSet);
bandwidth =powerbw(y, fsSet);
% computes the spectral centroid and bandwidth and the output are vector,
% one value per frame
centroid = mean(centroid);
bandwidth = mean(bandwidth);
rolloff = spectralRolloffPoint(y, fsSet);
rolloff = mean(rolloff);
% basically rolloff is feature measure the frequency below which a specific percentage of the total energy is contained
% all we looking for the average, from the machine can learning to classify

feat = [pMean, pIQR, mfMean, mfSTD, centroid, bandwidth, rolloff ];
features=[features; feat];
labels = [labels; label];

end
end

labels= categorical(labels);
train = cvpartition(labels, 'Holdout', 0.2) % 80percent train and 0.2 as 20% test
% Prepare training and testing datasets
XtrainData = features(training(train), :);
YtrainLabels = labels(training(train));
XtestData = features(test(train), :);
YtestLabels = labels(test(train));

%train classifier
model = fitcsvm(XtrainData, YtrainLabels,'KernelFunction', 'rbf');

%predict
pred = predict(model, XtestData);

% Accuracy
accur = sum(pred == YtestLabels)/numel(YtestLabels);
disp('Accuracy: ');
disp(accur)

% plot the confusion matrix
conf = confusionmat(YtestLabels, pred);
disp(conf);
figure;
confusionchart(conf,["male", "female"]);
title('Gender Classification');

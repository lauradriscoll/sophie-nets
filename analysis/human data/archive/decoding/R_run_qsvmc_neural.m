% Writing out infoabout parameters for quick reference:
% Allcellresponses (trials x frames x cells)
% Experience.reward etc (trials x frames)
% Trialparameters (trials x conditions)

% load('/Users/lauradriscoll/Desktop/yang_net_data/7.mat')
% R.taskDetails = taskDetails;
% R = continuous;

% targets = 409*[1 0; 0 1; -1 0; 0 -1];
% state = 20;
% stateID = single(cat(1,R.taskDetails.states.id));
% stateName = cat(1,{R.taskDetails.states.name});
% stateName{stateID==state}

%% pull out relevant data
cursorPos = R.effectorCursorPos(:,1:2);
contextLogic = R.state==20;
itiLogic = R.state==18;
successLogic = R.state==5;
failLogic = R.state==6;
numTargets = 4;


trial.start = find(diff([ 1; contextLogic])==1);
trial.contextOff = find(diff([ones(trial.start(1),1); contextLogic(trial.start(1):end)])==-1);
trial.end = find(diff([ones(trial.start(1),1); contextLogic(trial.start(1):end)])==-1);
% trial.end = find(diff([zeros(trial.start(1),1); itiLogic(trial.start(1):end)])==1);
numTrials = size(trial.start,1);
minTrialLength = min(trial.end - trial.start);
minContextLength = min(trial.contextOff - trial.start);
trial.stim = nan(numTrials,1);
trial.task = nan(numTrials,1);
trial.target = nan(numTrials,1);
trial.success = nan(numTrials,1);

Allcursor = nan(numTrials,minTrialLength,size(cursorPos,2));
Allcellresponses = nan(numTrials,minContextLength,size(R.meanSubtractSpikes,2));
Trialparameters = nan(numTrials,2);
%%
for nt = 1:numTrials
    trial.stim(nt) = mode(R.stimConds(trial.start(nt):trial.end(nt),2));
    trial.task(nt) = mode(R.stimConds(trial.start(nt):trial.end(nt),1));
    trial.success(nt) = sum(R.state(trial.start(nt):trial.end(nt))==5)>0;
    if trial.task(nt)<4
        trial.target(nt) = trial.stim(nt);
    else
        trial.target(nt) = trial.stim(nt)+numTargets/2;
        if trial.target(nt)>numTargets
            trial.target(nt) = trial.target(nt)-numTargets;
        end
    end
    
    Allcursor(nt,:,:) = cursorPos((trial.end(nt)-minTrialLength+1):trial.end(nt),:);
    Allcellresponses(nt,:,:) = R.meanSubtractSpikes((trial.contextOff(nt)-minContextLength+1):trial.contextOff(nt),:);
    Trialparameters(nt,:) = [trial.task(nt) trial.target(nt)];
end

%% useful logic
proLogic = trial.task<4;
taskRule = [trial.task==1 | trial.task==4; trial.task==2 | trial.task==5;...
    trial.task | trial.task ];

%% Running svm for all cells combined
targ = 3;
task_set = [2 5]; %rt [1 4]; %delay [3 6]; %memdelay

use_trials = trial.success==1;
trial_labels = trial.task>3; % trial labels
%%
trialsA = find(trial_labels==1 & use_trials); %separate trials into left and right groups
trialsB = find(trial_labels==0 & use_trials); %other group
frames = 1:size(Allcursor,2);

% Plotting all cells for both groups - doesn't seem to be a lot of
% selective cells
figure;
hold on
plot(Allcursor(trialsA,frames,1),Allcursor(trialsA,frames,2),'.b')
plot(Allcursor(trialsB,frames,1),Allcursor(trialsB,frames,2),'.r')
%%
% data must be a two dimensional array, use one cell at a time or a set of
% frames
frames = 1:size(Allcellresponses,2); % only look during delay
cels = 1:size(Allcellresponses,3); %include all cells

%average across frames of interest and reshape matrix into propper dimensions (predictors x examples)
data = squeeze(nanmean(Allcellresponses(:,frames,:),2))';

num_its = 10;
test_group = nan(num_its,1);
train_group = nan(num_its,1);
for it = 1:num_its
    [svm_struct, accuracy_test, accuracy_train] = ...
        qsvmc(data,trial_labels,cels,use_trials,.4);
    test_group(it) = accuracy_test;
    train_group(it) = accuracy_train;
end
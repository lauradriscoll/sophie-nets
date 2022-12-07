% Writing out infoabout parameters for quick reference:
% Allcellresponses (trials x frames x cells)
% Experience.reward etc (trials x frames)
% Trialparameters (trials x conditions)

% load('/Users/lauradriscoll/Desktop/yang_net_data/7.mat')
% stream{1}.taskDetails = taskDetails;
% stream{1}.continuous = continuous;

% targets = 409*[1 0; 0 1; -1 0; 0 -1];
% state = 20;
% stateID = single(cat(1,stream{1}.taskDetails.states.id));
% stateName = cat(1,{stream{1}.taskDetails.states.name});
% stateName{stateID==state}




%% Running svm for all cells combined
% targ = 3;
% task_set = [2 5]; %rt [1 4]; %delay [3 6]; %memdelay

use_trials = (trial.task==2 | trial.task==5) & trial.success==1;
trial_labels = trial.task>3; % trial labels
%% plot some cursor data

% trialsA = find(trial_labels==1 & use_trials); %separate trials into left and right groups
% trialsB = find(trial_labels==0 & use_trials); %other group
% frames = (minContextLength-500):minContextLength;
% 
% figure;
% hold on
% plot(Allcursor(trialsA,frames,1),Allcursor(trialsA,frames,2),'.b')
% plot(Allcursor(trialsB,frames,1),Allcursor(trialsB,frames,2),'.r')
%%

% data must be a two dimensional array, use one cell at a time or a set of
% frames
frames = 1:minContextLength; % only look during delay
cels = 1:size(Allcellresponses,3); %include all cells

%average across frames of interest and reshape matrix into propper dimensions (predictors x examples)
data = squeeze(nanmean(Allcellresponses(:,frames,:),2))';

num_its = 10;
test_group = nan(num_its,1);
train_group = nan(num_its,1);
for it = 1:num_its
    [svm_struct, accuracy_test, accuracy_train] = ...
        qsvmc(data,trial_labels,cels,use_trials,.1);
    test_group(it) = accuracy_test;
    train_group(it) = accuracy_train;
end
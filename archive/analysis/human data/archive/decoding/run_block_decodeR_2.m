% %% DECODE MOVEMENT
data_dur = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/20180730/');
blocks = [4:8 10:18];%3:12;
epochs = 1:3;
% 
figName = 'useRT_decodeTarget_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==2 | trial.task==5);';
trial_label_rule = 'trial_labels = (trial.target==2 | trial.target==3);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useDelay_decodeTarget_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==1 | trial.task==4);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useMemDelay_decodeTarget_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==3 | trial.task==6);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

%% PRO v ANTI
figName = 'useRT_decodeAnti_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==2 | trial.task==5);';
trial_label_rule = 'trial_labels = (trial.task>3);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useDelay_decodeAnti_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==1 | trial.task==4);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useMemDelay_decodeAnti_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==3 | trial.task==6);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)


%% COMPARE TASKS
figName = 'useDelay_decodeRTvDelay_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==1 | trial.task==2);';
trial_label_rule = 'trial_labels = (trial.task==2);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useMemDelay_decodeRTvMemDelay_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==3 | trial.task==2);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

figName = 'useDelayMemDelay_decodeDelayvMemDelay_epoch';
use_trial_rule = 'use_trials = trial.success==1 & trial.end>trial.go & (trial.task==1 | trial.task==3);';
trial_label_rule = 'trial_labels = (trial.task==1);';
block_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

%% process data
data_dur = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/20180718/');
blocks = 1;
data_process(data_dur,blocks)
%%
data_dur = fullfile('~/data/human/yangnet/20180730/');
data_dur = fullfile('~/data/');
blocks = [5];
data_process_20180730(data_dur,blocks)


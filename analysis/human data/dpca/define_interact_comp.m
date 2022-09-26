addpath('/Users/lauradriscoll/Documents/code/nptl_code/analysis/Frank/dPCA/dPCA-master/matlab/');
YYMMDD = '180815';%'180730';%
YYYYMMDD = ['20' YYMMDD];
%%
    spikes =  [];
    align_ids_go = [];
    align_ids_stim = [];
    conds = [];
    behave = [];
    
    
for block = [2:4 8:10 14:16]%[4:8 10:11]%%[5:7 11:13]%3:12%;%8]%
    
[R, ~] = load_ydata(YYMMDD,block);
data_dir = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/',['20' YYMMDD]);
% load(fullfile(data_dir,'process','zscore',['R_process_block' num2str(block)]),'trial') %%'allCursorR','allCellResponsesR',
load(fullfile(data_dir,'process','zscore','alignAll',['alignAll_' YYYYMMDD '_block' num2str(block) '.mat']),'trial') %%'allCursorR','allCellResponsesR',
% load(fullfile(data_dir,'process','zscore','catBlocks',['catBlocks_' YYYYMMDD '.mat']),'trial')

stim_events = round(trial.stimOn/20)+size(spikes,1);
go_events = round(trial.go/20)+size(spikes,1);
align_ids_go = cat(1,align_ids_go,go_events(~isnan(go_events)));
align_ids_stim = cat(1,align_ids_stim,stim_events(~isnan(stim_events)));
spikes =  cat(1,spikes,R.zScoreSpikes);
conds = cat(1,conds,R.stimConds);
behave = cat(1,behave,R.rigidBodyPosXYZ);
    
%     clear R trial
end
%%

features = spikes;
eventIdx = align_ids_stim; %floor(trial.contextOff/20);
task = conds(eventIdx,1);
target = conds(eventIdx,2);

eventIdx = eventIdx((task == 3 | task==6) & (target==2 | target==4));%eventIdx(task == 2);% eventIdx(task == 3 | task == 2);%eventIdx(task~=2 & task~=5); %
task = conds(eventIdx,1);
stim = conds(eventIdx,2);
target = mod(stim + 2*(task>3),4);
target(target==0) = 4;

trialCodes = cat(2,stim,target); %target;%cat(2,task == 3,target); %cat(2,task>3,mod(task,3));%

timeWindow = [-50 50];
timeStep = 0.02;
margNames=[];
maxDim = 20;
%%
out = apply_dPCA_simple( features, eventIdx, trialCodes, timeWindow, timeStep, margNames, maxDim );
%%
cond_indep_comp = out.W(:,1);
stim_comp = out.W(:,3);
targ_comp = out.W(:,2);
%%
%features: N x M matrix of M features and N time steps
%eventIdx: T x 1 vector of time step indices where each trial begins
%timeWindow: 1 x 2 vector defining the time window (e.g. [-50 100] for
%   1 second before and 2 seconds after the event index (assuming 0.02
%   second time step)
%timeStep: scalar defining the length of time step (0.02 seconds in our case)
%trialCodes: T x 1 (single factor) or T x 2 (two factor) matrix of
%   codes defining what condition each trial belongs to
%margNames: 2 x 1 (single factor) or 4 x 1 (two factor) cell vector
%   defining the names of each marginalization

%%
figure;
hold on
for t = 1:4
	plot(behave(eventIdx(target==t)+29,1),behave(eventIdx(target==t)+29,2),'o')
end
%%
figure; 
for feat = 1:4
    hold on;
    plot(squeeze(out.featureAverages(:,1,feat,:))'*cond_indep_comp,'-')
    plot(squeeze(out.featureAverages(:,2,feat,:))'*cond_indep_comp,':')
end

%%
figure; 
for feat = 1:4
    hold on;
    plot(squeeze(out.featureAverages(:,1,feat,:))'*stim_comp,'-')
    plot(squeeze(out.featureAverages(:,2,feat,:))'*stim_comp,':')
end
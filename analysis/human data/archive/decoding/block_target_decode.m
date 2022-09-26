data_dur = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/20180718/');
block = 10;%use_blocks
load(fullfile(data_dur,['formatted' num2str(block) '.mat']));

eval(['stream = block' num2str(block) 'stream;'])
eval(['R = block' num2str(block) 'binnedR;'])

clear(['block' num2str(block) 'binnedR'])
clear(['block' num2str(block) 'stream'])
spikeRasterCombined = cat(2,stream{1}.spikeRaster,stream{1}.spikeRaster2);
%%
[allCursor, allCellResponses, trial] = combine_trial_data(stream{1}.continuous.state, stream{1}.continuous.stimConds, ...
    stream{1}.continuous.windowsMousePosition, spikeRasterCombined);
%%
trial_labels = trial.target==4 | trial.target==3;
use_trials = trial.success==1 & trial.end>trial.go & (trial.task==1 | trial.task==3);
epoch = 3;
useInd = sum(squeeze(~isnan(allCellResponses{epoch}(:,:,1))))>0;
step_size = 500;
steps = 1:step_size:3*step_size;%sum(useInd)
cmap = spring(size(steps,2));
hmap = summer(size(steps,2));
%%
num_its = 10;
neuron_test_group = nan(size(steps,2),num_its);
neuron_train_group = nan(size(steps,2),num_its);
behave_test_group = nan(size(steps,2),num_its);
behave_train_group = nan(size(steps,2),num_its);

figure
for step = steps
    
    frames = step:(step+step_size);
    ndata = squeeze(nanmean(allCellResponses{epoch}(:,frames,:),2))';
    bdata = squeeze(nanmean(allCursor{epoch}(:,frames,:),2))';
    trial_t = trial_labels & use_trials;
    trial_f = trial_labels==0 & use_trials;
    hold on
    plot(bdata(1,trial_t),bdata(2,trial_t),'.','color',cmap(step==steps,:),'MarkerSize',10)
    plot(bdata(1,trial_f),bdata(2,trial_f),'.','color',hmap(step==steps,:),'MarkerSize',10)
    
    for it = 1:num_its
        
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.01);
        neuron_test_group(step==steps,it) = accuracy_test;
        neuron_train_group(step==steps,it) = accuracy_train;
        
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(bdata,trial_labels,1:size(bdata,1),use_trials,.1);
        behave_test_group(step==steps,it) = accuracy_test;
        behave_train_group(step==steps,it) = accuracy_train;
        
    end
end
%%
figure;
hold on
plot(nanmean(behave_test_group,2),':r','lineWidth',2)
plot(nanmean(behave_train_group,2),'-r','lineWidth',2)
plot(nanmean(neuron_test_group,2),':b','lineWidth',2)
plot(nanmean(neuron_train_group,2),'-b','lineWidth',2)
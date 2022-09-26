
% function block_target_decodeR(data_dur,blocks,use_trial_rule,trial_label_rule)
data_dur = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/20180730/');
epoch = 3;
figName = ['useRT_decodeTarget_epoch' num2str(epoch)];
figure;
for block = [4:8 10:11 13:18]%4:18%use_blocks
load(fullfile(data_dur,['formatted' num2str(block) '.mat']));

eval(['stream = block' num2str(block) 'stream;'])
eval(['R = block' num2str(block) 'binnedR;'])

clear(['block' num2str(block) 'binnedR'])
clear(['block' num2str(block) 'stream'])
spikeRasterCombined = cat(2,stream{1}.spikeRaster,stream{1}.spikeRaster2);
%%
[allCursor, allCellResponses, allCursorR, allCellResponsesR, trial] = combine_trial_data(stream{1}.continuous.state, stream{1}.continuous.stimConds, ...
    stream{1}.continuous.windowsMousePosition, stream{1}.continuous.rigidBodyPosXYZ_speed, spikeRasterCombined, R.zScoreSpikes);
%%
trial_labels = (trial.target==2 | trial.target==3); %trial.target==3 | trial.target==1;
use_trials = trial.success==1 & trial.end>trial.go & (trial.task==2 | trial.task==5);
useInd = sum(squeeze(~isnan(allCellResponsesR{epoch}(:,:,1))))>0;
step_size = 1;
steps = 1:step_size:30*step_size;%sum(useInd)
cmap = spring(2*size(steps,2));
hmap = summer(2*size(steps,2));
%%
num_its = 10;
neuron_test_group = nan(size(steps,2),num_its);
neuron_train_group = nan(size(steps,2),num_its);
behave_test_group = nan(size(steps,2),num_its);
behave_train_group = nan(size(steps,2),num_its);

for step = steps
    frames = step:(step+step_size);
    ndata = squeeze(nanmean(allCellResponsesR{epoch}(:,frames,:),2))';
    bdata = squeeze(nanmean(allCursorR{epoch}(:,frames,:),2))';
    trial_t = trial_labels & use_trials;
    trial_f = trial_labels==0 & use_trials;
    
    subplot(1,2,1)
    hold on
    plot(bdata(1,trial_t),bdata(2,trial_t),'.','color',cmap(step==steps,:),'MarkerSize',10)
    plot(bdata(1,trial_f),bdata(2,trial_f),'.','color',hmap(step==steps,:),'MarkerSize',10)
    
    for it = 1:num_its
        
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.0001);
        neuron_test_group(step==steps,it) = accuracy_test;
        neuron_train_group(step==steps,it) = accuracy_train;
        
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(bdata,trial_labels,1:size(bdata,1),use_trials,.1);
        behave_test_group(step==steps,it) = accuracy_test;
        behave_train_group(step==steps,it) = accuracy_train;
        
    end
end
%%
subplot(1,2,2)
hold on
plot(nanmean(behave_test_group,2),':r','lineWidth',2)
plot(nanmean(behave_train_group,2),'-r','lineWidth',2)
plot(nanmean(neuron_test_group,2),':b','lineWidth',2)
plot(nanmean(neuron_train_group,2),'-b','lineWidth',2)
end
%%
subplot(1,2,2)
title('Behavior and Neural Decoding')
axis square
legend('Behavior Test','Behavior Train','Neural Test','Neural Train','location','southeast')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
xticks(0:10:30)
xticklabels(20*(0:10:30))
ylabel('Fraction Correct')
ylabel('Time (ms)')

subplot(1,2,1)
title(['Epoch ' num2str(epoch) ' Cursor Tracking'])
axis square
legend('Down and Left','Up and Right','location','southeast')
xlabel('X Position on Screen')
ylabel('Y Position on Screen')
xlim([-.5 .5])
ylim([-.5 .5])
set(findall(gcf,'-property','FontSize'),'FontSize',20)

% save_path = fullfile(data_dur,'analysis','block_decode');
% if ~exist(save_path,'dir')
%     mkdir(save_path)
% end
% savefig(fullfile(save_path,figName))
% end

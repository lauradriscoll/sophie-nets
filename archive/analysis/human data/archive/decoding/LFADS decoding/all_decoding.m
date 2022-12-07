% %%
% 
% arm_head_20bins_0in_e1
% % movement_startAlign_20bins
% pm = rc.runs(1).loadPosteriorMeans();
figure;

task_set = 1:3;
num_its = 10;
step_size = 5;
inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
Xn = permute(pm.rates(:,:,inds),[3 2 1]);
sts = 1:step_size:(size(Xn,2)-step_size-1);
cmap = cool(2*size(task_set,2));

trial_labels = task < 4;
for t = 1:3
use_trials = (task== t | task== t+3) & data.success==1 & inds'<round(size(pm.rates,3)/2);

neuron_test_group = nan(size(sts,2),num_its);
neuron_train_group = nan(size(sts,2),num_its);

for st = sts
    frames = st:(st+step_size);
    ndata = squeeze(nanmean(Xn(:,frames,:),2))';
    
    for it = 1:num_its
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.0001);
        neuron_test_group(st==sts,it) = accuracy_test;
        neuron_train_group(st==sts,it) = accuracy_train;
    end
end

hold on
plot(nanmean(neuron_test_group,2),':','color',cmap(2*t,:),'lineWidth',2)
plot(nanmean(neuron_train_group,2),'-','color',cmap(2*t,:),'lineWidth',2)
end


xts = get(gca,'xtick');
set(gca,'xticklabel',xts*step_size);
xlabel('Time (ms)')
ylim([.3 1])
xlim([0  size(neuron_test_group,1)+1])
plot([0 xts(end)],[.5 .5],'-k')
text(1,.52,'Chance')
ylabel('Fraction Correct')

legend('Delay Test','Delay Train','RT Test','RT Train','Memory Test','Memory Train')
title('Pro vs. Anti (Context Period)')
save_dir = fullfile('figures','yangnet', datestr(now,'yyyymmdd'));
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
savefig(fullfile(save_dir,['svm_pro_v_anti_context_' num2str(step_size) '_' dataset_name(1:end-4)]))

%%
figure;
cmap = spring(2*size(task_set,2));

trial_labels = task == 3 | task == 6;
for t = [1 2]
use_trials = (task~= t & task~= t+3) & data.success==1 & inds'<round(size(pm.rates,3)/2);

neuron_test_group = nan(size(sts,2),num_its);
neuron_train_group = nan(size(sts,2),num_its);

for st = sts
    frames = st:(st+step_size);
    ndata = squeeze(nanmean(Xn(:,frames,:),2))';
    
    for it = 1:num_its
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.0001);
        neuron_test_group(st==sts,it) = accuracy_test;
        neuron_train_group(st==sts,it) = accuracy_train;
    end
end

hold on
plot(nanmean(neuron_test_group,2),':','color',cmap(2*t,:),'lineWidth',2)
plot(nanmean(neuron_train_group,2),'-','color',cmap(2*t,:),'lineWidth',2)
end


xts = get(gca,'xtick');
set(gca,'xticklabel',xts*step_size);
xlabel('Time (ms)')
ylim([.3 1])
xlim([0  size(neuron_test_group,1)+1])
plot([0 xts(end)],[.5 .5],'-k')
text(1,.52,'Chance')
ylabel('Fraction Correct')

legend('Memory v RT Test','Memory v RT Train','Memory v Delay Test','Memory v Delay Train')
title('Memory v. Other Task (Context Period)')
savefig(fullfile(save_dir,['svm_memory_v_other_context_' num2str(step_size) '_' dataset_name(1:end-4)]))

%%
arm_head_20bins_0in_e3

pm = rc.runs(1).loadPosteriorMeans();

figure;

task_set = 1:3;
step_size = 1;
inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
target = ceil(data.conditionId(inds)/6);
Xn = permute(pm.rates(:,:,inds),[3 2 1]);
sts = 1:step_size:(size(Xn,2)-step_size-1);
cmap = cool(2*size(task_set,2));

trial_labels = target < 3;
for t = task_set
use_trials = task== t | task== t+3 & data.success==1;

neuron_test_group = nan(size(sts,2),num_its);
neuron_train_group = nan(size(sts,2),num_its);

for st = sts
    frames = st:(st+step_size);
    ndata = squeeze(nanmean(Xn(:,frames,:),2))';
    
    for it = 1:num_its
        [~, accuracy_test, accuracy_train] = ...
            qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.0001);
        neuron_test_group(st==sts,it) = accuracy_test;
        neuron_train_group(st==sts,it) = accuracy_train;
    end
end

hold on
plot(nanmean(neuron_test_group,2),':','color',cmap(2*t,:),'lineWidth',2)
plot(nanmean(neuron_train_group,2),'-','color',cmap(2*t,:),'lineWidth',2)
end


xts = get(gca,'xtick');
set(gca,'xticklabel',xts*step_size);
xlabel('Time (ms)')
ylim([.3 1])
xlim([0  size(neuron_test_group,1)+1])
plot([0 xts(end)],[.5 .5],'-k')
text(1,.52,'Chance')
ylabel('Fraction Correct')

legend('Delay Test','Delay Train','RT Test','RT Train','Memory Test','Memory Train')
title('Target Direction (Movement Period)')
savefig(fullfile(save_dir,'svm_target_movement'))

%%
% movement_startAlign
% pm = rc.runs(1).loadPosteriorMeans();
%%

nr = 2;
nc = 2;
% 
% %%
X = [];
for t = 1:6
X = cat(3,X,nanmean(pm.factors(:,:,task==t),3));
end


X_shape = reshape(X,40,[]);
X_musub = X_shape - mean(X_shape,2);
% 
pcs = pca(X_musub');
num_pcs = 40;

pmfac_shape = reshape(pm.factors,40,[]);
pmfac_musub = pmfac_shape ;


pc_project = pcs(:,1:num_pcs)'*pmfac_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(pm.factors,3)]);

    figure
for target_id = 1
for trial_id = find(target==target_id,1,'first')'        
    hold on
    plot3(pmfac_musub_reshape(1,:,trial_id),pmfac_musub_reshape(2,:,trial_id),pmfac_musub_reshape(3,:,trial_id),'color',cmap(task(trial_id),:))
end
end
%%
figure;

task_set = [1:3];
num_its = 10;
step_size = 20;
inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
target = ceil(data.conditionId(inds)/6);
Xn = permute(pm.rates(:,:,inds),[3 2 1]);%allCellResponsesR{epoch};%
sts = 1:step_size:(size(Xn,2)-step_size-1);%1:step_size:size(pm.rates,2)-step_size;%sum(useInd)
cmap = spring(2*size(task_set,2));

trial_labels = task==2 | task==5;%data.conditionId(inds)>12;%trial.target==3 | trial.target==1;%(trial.task==2 | trial.task==5); %

% for t = task_set
use_trials = task~=1 & task~=4; %(task==t);% | task==t+3);%trial.success==1 & trial.end>trial.go & 

neuron_test_group = nan(size(sts,2),num_its);
neuron_train_group = nan(size(sts,2),num_its);

for st = sts
    frames = st:(st+step_size);
    ndata = squeeze(nanmean(Xn(:,frames,:),2))'; %ceil(frames/10) %frames
    trial_t = trial_labels(inds) & use_trials(inds);%use_trials;
    trial_f = trial_labels(inds)==0 & use_trials(inds);%use_trials;
    
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
% end






















































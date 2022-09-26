% 
% %%
% % arm_head_20bins_0in_e1_180813
% arm_head_20bins_0in_e1
% % movement_startAlign_20bins
% pm = rc.runs(1).loadPosteriorMeans();
% %%
section = [];
figname = [section '_half_decode_task'];
inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
target = ceil(data.conditionId(inds)/6);
nr = 2;
nc = 2;
% 
% %%
X = [];
for t = 1:6
X = cat(3,X,nanmean(pm.factors(:,:,task==t & inds' > 0),3));%round(size(pm.rates,3)/2)),3));%
end


X_shape = reshape(X,40,[]);
X_musub = X_shape - mean(X_shape,2);
% 
pcs = pca(X_musub');
num_pcs = 40;

pmfac_shape = reshape(pm.factors,40,[]);
pmfac_musub = pmfac_shape ;


pc_project = pcs(:,1:num_pcs)'*X_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(X,3)]);

task_set = 1:3;
cmap = cool(2*size(task_set,2));

task = [1:3 1:3];
anti = [0 0 0 1 1 1];
pc_set = [1 2 3];
%%
    figure   
    hold on
for task_id = task_set
for trial_id = find(task==task_id)      
    if anti(trial_id)==0
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),'color',cmap(2*task_id,:),'lineWidth',5)
    else
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),':','color',cmap(2*task_id,:),'lineWidth',5)
    end
end
end

for task_id = task_set
for trial_id = find(task==task_id)     
    plot3(pmfac_musub_reshape(pc_set(1),1,trial_id),pmfac_musub_reshape(pc_set(2),1,trial_id),pmfac_musub_reshape(pc_set(3),1,trial_id),'o','color',cmap(2*task_id,:),'MarkerSize',10,'lineWidth',2)
    plot3(pmfac_musub_reshape(pc_set(1),end,trial_id),pmfac_musub_reshape(pc_set(2),end,trial_id),pmfac_musub_reshape(pc_set(3),end,trial_id),'o','color',cmap(2*task_id,:),'MarkerSize',5,'lineWidth',5)
end
end
%%
view(-82, 56)
xlabel('LFADS Factor PC 1')
ylabel('LFADS Factor PC 2')
zlabel('LFADS Factor PC 3')

legend('Delay Pro','Delay Anti','RT Pro','RT Anti','Memory Pro','Memory Anti','Trial Start','Trial End')

save_dir = fullfile('figures','yangnet', datestr(now,'yyyymmdd'));
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
savefig(fullfile(save_dir,figname))


pc_project = pcs(:,1:num_pcs)'*pmfac_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(pm.factors,3)]);

task_set = 1:3;
cmap = cool(2*size(task_set,2));

task = mod(data.conditionId(inds)-1,6)+1;
anti = [0 0 0 1 1 1];

figure
hold on
nt = 100;

for task_id = task_set
for trial_id = find(task==task_id)'%,nt,section)'% | task==task_id*2,3,'first')'     
    if task(trial_id)==task_id
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),'color',cmap(2*task_id,:))
    else
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),':','color',cmap(2*task_id,:))
    end
end
end


for task_id = task_set
for trial_id = find(task==task_id)'%,nt,section)'% | task==task_id*2,3,'first')'       
    plot3(pmfac_musub_reshape(pc_set(1),1,trial_id),pmfac_musub_reshape(pc_set(2),1,trial_id),pmfac_musub_reshape(pc_set(3),1,trial_id),'o','color',cmap(2*task_id,:),'MarkerSize',10,'lineWidth',2)
    plot3(pmfac_musub_reshape(pc_set(1),end,trial_id),pmfac_musub_reshape(pc_set(2),end,trial_id),pmfac_musub_reshape(pc_set(3),end,trial_id),'o','color',cmap(2*task_id,:),'MarkerSize',5,'lineWidth',5)
end
end

view(-82, 56)
xlabel('LFADS Factor PC 1')
ylabel('LFADS Factor PC 2')
zlabel('LFADS Factor PC 3')
legend('Delay Pro','Delay Anti','RT Pro','RT Anti','Memory Pro','Memory Anti','Trial Start','Trial End')

savefig(fullfile(save_dir,[figname '_w_single']))

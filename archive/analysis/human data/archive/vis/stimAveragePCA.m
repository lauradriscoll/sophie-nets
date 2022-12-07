% 
% %%
% movement_startAlign_20bins
% pm = rc.runs(1).loadPosteriorMeans();
% %%
section = [];
figname = [section '_half_decode_target'];
inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
target = ceil(data.conditionId(inds)/6);

stim = target;
stim(task>3) = target(task>3) + 2;
stim(stim>4) = stim(stim>4)-4;

nr = 2;
nc = 2;
% 
% %%
X = [];
for t = 1:4
X = cat(3,X,nanmean(pm.factors(:,:,stim==t & inds'<round(size(pm.rates,3)/2)),3));
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

target_set = 1:4;
cmap = parula(size(target_set,2));

pc_set = [1 2 3];
%%
    figure   
    hold on
for target_id = target_set   
    plot3(pmfac_musub_reshape(pc_set(1),:,target_id),pmfac_musub_reshape(pc_set(2),:,target_id),pmfac_musub_reshape(pc_set(3),:,target_id),'color',cmap(target_id,:),'lineWidth',5)
end

for target_id = target_set
    plot3(pmfac_musub_reshape(pc_set(1),1,target_id),pmfac_musub_reshape(pc_set(2),1,target_id),pmfac_musub_reshape(pc_set(3),1,target_id),'o','color',cmap(target_id,:),'MarkerSize',10,'lineWidth',2)
    plot3(pmfac_musub_reshape(pc_set(1),end,target_id),pmfac_musub_reshape(pc_set(2),end,target_id),pmfac_musub_reshape(pc_set(3),end,target_id),'o','color',cmap(target_id,:),'MarkerSize',5,'lineWidth',5)
end
%%
view(-82, 56)
xlabel('LFADS Factor PC 1')
ylabel('LFADS Factor PC 2')
zlabel('LFADS Factor PC 3')

legend('Target 1','Target 2','Target 3','Target 4')
title('Movement Period')

save_dir = fullfile('figures','yangnet', datestr(now,'yyyymmdd'));
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
savefig(fullfile(save_dir,figname))


pc_project = pcs(:,1:num_pcs)'*pmfac_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(pm.factors,3)]);

figure
hold on
nt = [];

for target_id = target_set
for trial_id = find(target==target_id)'%,nt,section)'% | task==task_id*2,3,'first')'     
    if task(trial_id)<4
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),'color',cmap(target_id,:))
    else
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),':','color',cmap(target_id,:))
    end
end
end


for target_id = target_set
for trial_id = find(target==target_id)'%,nt,section)'% | task==task_id*2,3,'first')'       
    plot3(pmfac_musub_reshape(pc_set(1),1,trial_id),pmfac_musub_reshape(pc_set(2),1,trial_id),pmfac_musub_reshape(pc_set(3),1,trial_id),'o','color',cmap(target_id,:),'MarkerSize',10,'lineWidth',2)
    plot3(pmfac_musub_reshape(pc_set(1),end,trial_id),pmfac_musub_reshape(pc_set(2),end,trial_id),pmfac_musub_reshape(pc_set(3),end,trial_id),'o','color',cmap(target_id,:),'MarkerSize',5,'lineWidth',5)
end
end

view(-82, 56)
xlabel('LFADS Factor PC 1')
ylabel('LFADS Factor PC 2')
zlabel('LFADS Factor PC 3')
title('Movement Period')

savefig(fullfile(save_dir,[figname '_w_single']))

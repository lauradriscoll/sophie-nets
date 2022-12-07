% 
% %%
% arm_head_20bins_0in_e3
% movement_startAlign_20bins
% pm_all = rc.runs(1).loadPosteriorMeans();

d = 3;
pm = pm_all(d);
date = date_set{d};
blocks = block_set{d};

[ out ] = getLNDPaths( );
LFADS_data_dir = fullfile(out.dataPath,'human','yangnet',date,'process/zscore','alignMove');
filename = ['LFADS_' date '_block'];

data = [];
for block = blocks
data = data_process_LFADS_join(data,LFADS_data_dir,filename,block,blocks,epoch);
end

inds = 1:size(pm.rates,3);
task = mod(data.conditionId(inds)-1,6)+1;
target = ceil(data.conditionId(inds)/6);
nr = 2;
nc = 2;
% 
%
categors = unique(floor(data.trialAddress/(10^3)))';
X = [];
for x = categors
    if sum(data.trialAddress<(10^3*x))>10
        X = cat(3,X,nanmean(pm.factors(:,:,data.trialAddress<(10^3*x)),3));
    end
end

X_shape = reshape(X,40,[]);
X_musub = X_shape - mean(X_shape,2);
pcs = pca(X_musub');
num_pcs = 40;

pmfac_shape = reshape(pm.factors,40,[]);
pmfac_musub = pmfac_shape ;


pc_project = pcs(:,1:num_pcs)'*X_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(X,3)]);
cmap = jet(max(categors));

% task = [1:3 1:3];
pc_set = [1 2 4];

    figure
for trial_id = 1:size(pmfac_musub_reshape,3)     
    hold on
    plot3(pmfac_musub_reshape(pc_set(1),:,trial_id),pmfac_musub_reshape(pc_set(2),:,trial_id),pmfac_musub_reshape(pc_set(3),:,trial_id),'color',cmap(trial_id,:),'lineWidth',5)
end
%%


pc_project = pcs(:,1:num_pcs)'*pmfac_musub;
pmfac_musub_reshape = reshape(pc_project, [num_pcs, size(pm.factors,2), size(pm.factors,3)]);

task_set = 1:3;

for x = categors
    t = find(floor(data.trialAddress/(10^3))==x,3,'last')';
    if ~isempty(t)
    plot3(squeeze(pmfac_musub_reshape(pc_set(1),:,t)),squeeze(pmfac_musub_reshape(pc_set(2),:,t)),squeeze(pmfac_musub_reshape(pc_set(3),:,t)),'color',cmap(x,:))
    end
end

% figure;imagesc(X(:,:,1))
% data_dir = '/Users/lauradriscoll/Documents/data/human/yangnet/20180730';
% data_process_alignAll_noRT(data_dir,[4:8 10:11 13:18])
%%
allCR = [];
tar_pos = [];
epoch = 3;
for b = 4:8
block_temp = load(strcat('/Users/lauradriscoll/Documents/data/human/yangnet/20180730/process/zscore/alignAll/alignAll_noRT_20180730_block', string(b), '.mat'));
allCR = cat(1,allCR,block_temp.allCellResponsesR{epoch});
tar_pos = cat(1,tar_pos,block_temp.trial.target == 1:4);
end
col = tar_pos*[1:4]';
%%
use_trials = ~isnan(allCR(:,1,1));
F = cat(2,tar_pos(use_trials,4)-tar_pos(use_trials,2),tar_pos(use_trials,1)-tar_pos(use_trials,3),ones(sum(use_trials),1))';
F_titles = {'right';'up';'ones'}; %check this
regression_mat = (F*F')\F;

%%
remove_channels = [3 4 185:188];
cels = 1:192;
cels(remove_channels) = [];
%%

hidden = allCR(use_trials,:,cels);
inds = 10:size(hidden,2)-10;
all_data = squeeze(nanmean(hidden(:,inds,:),2));

%%
Beta = nan(size(all_data,2),size(F,1));
didx = 1:size(hidden,2);
for neur = 1:size(all_data,2)
    Beta(neur,:) = regression_mat*all_data(:,neur);
end
    
%% 6.6. Principal component analysis
N_pca = 10;
[pc_struct.coeff, pc_struct.score, pc_struct.latent, pc_struct.tsquared, ...
    pc_struct.explained, pc_struct.mu] = pca(all_data);

D = pc_struct.coeff(:,1:N_pca)*pc_struct.coeff(:,1:N_pca)';
%%
X_pca = D*all_data';

    %% 6.7. Regression subspace
B_vt = Beta;% reshape(Beta,size(hidden,3),[]);                                 %reshape to units x (conditions x time)
B_pca = D * B_vt;                                                       %denoise
% B_pca_vt = reshape(B_pca,size(hidden,2),size(F,1),size(hidden,2));     %reshape to units x conditions x time
B_vmax = B_pca;%squeeze(max(B_pca_vt,[],3));                             %find peak of all units for each condition
[Q,R] = qr(B_vmax);                                                     %orthogonalize condition dimensions
B_vortho = Q(:,1:size(F,1)-1);                                          %remove ones dimension

%%
figure;
hold on
cmap = colormap(lines(6));
for t = 1:size(allCR,1)
    use = squeeze(allCR(t,:,cels));
    plot(use*B_vortho(:,1),use*B_vortho(:,2),'color', cmap(col(t)+2,:),'linewidth',2)
end

%%
figure;
hold on
cmap = colormap(lines(6));
for t = 1:size(allCR,1)
    use = squeeze(allCR(t,:,cels));
    plot(use*pc_struct.coeff(:,1),use*pc_struct.coeff(:,2),'-o','color', cmap(col(t)+2,:),'linewidth',2)
end
%%
rad_stim = repmat(atan2(F(2,:),F(1,:)),size(inds,2),1);
[x_reg, y_reg, B, coeff] = pca_regress(all_data',rad_stim(1,:));
coeffB = coeff*B;
%%
figure;
hold on

cmap = colormap(lines(6));
for t = 1:size(allCR,1)
    use = squeeze(allCR(t,:,cels));
    plot(use*coeffB(:,1),use*coeffB(:,2),'-o','color', cmap(col(t)+2,:),'linewidth',2)
end

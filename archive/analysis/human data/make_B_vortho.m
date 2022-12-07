function B_vortho = make_B_vortho(hidden,F,inds)

all_data = squeeze(nanmean(hidden(:,inds,:),2));
regression_mat = (F*F')\F;

Beta = nan(size(all_data,2),size(F,1));
for neur = 1:size(all_data,2)
    Beta(neur,:) = regression_mat*all_data(:,neur);
end
    
%% 6.6. Principal component analysis
N_pca = 10;
[pc_struct.coeff, pc_struct.score, pc_struct.latent, pc_struct.tsquared, ...
    pc_struct.explained, pc_struct.mu] = pca(all_data);
D = pc_struct.coeff(:,1:N_pca)*pc_struct.coeff(:,1:N_pca)';

%% 6.7. Regression subspace
B_vt = Beta;                                %reshape to units x (conditions x time)
B_pca = D * B_vt;                          %find peak of all units for each condition
% [Q,~] = qr(B_pca);                                                     %orthogonalize condition dimensions
Q = B_pca;
B_vortho = Q(:,1:size(F,1)-1);                                          %remove ones dimension
end
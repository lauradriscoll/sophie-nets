function B_vortho = tdr(data,F)

regression_mat = (F*F')\F;

Beta = nan(size(data,2),size(F,1));
for neur = 1:size(data,2)
    Beta(neur,:) = regression_mat*data(:,neur);
end
    
%% 6.6. Principal component analysis
N_pca = min(10;
[pc_struct.coeff, pc_struct.score, pc_struct.latent, pc_struct.tsquared, ...
    pc_struct.explained, pc_struct.mu] = pca(data);

D = pc_struct.coeff(:,1:N_pca)*pc_struct.coeff(:,1:N_pca)';

%% 6.7. Regression subspace
B_vt = Beta;% reshape(Beta,size(hidden,3),[]);                                 %reshape to units x (conditions x time)
B_pca = D * B_vt;                                                       %denoise
% B_pca_vt = reshape(B_pca,size(hidden,2),size(F,1),size(hidden,2));     %reshape to units x conditions x time
B_vmax = B_pca;%squeeze(max(B_pca_vt,[],3));                             %find peak of all units for each condition
[Q,~] = qr(B_vmax);                                                     %orthogonalize condition dimensions
B_vortho = Q(:,1:size(F,1)-1);  

end
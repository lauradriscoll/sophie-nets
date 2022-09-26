addpath('/Users/lauradriscoll/Documents/code/yangnet/analysis/human data');
addpath('/Users/lauradriscoll/Documents/code/multitask-nets/utils/');

% data_dir = '/Users/lauradriscoll/Documents/data/human/yangnet/20180730';
% data_process_catBlocks(data_dir,[4:8 10:11 13:16 18])%


% data_dir = '/Users/lauradriscoll/Documents/data/human/yangnet/20180813';
% % data_process_alignAll(data_dir,0:18)%
% data_process_catBlocks(data_dir,[6:18])%

data_dir = '/Users/lauradriscoll/Documents/data/human/yangnet/20180730';
% data_process_alignAll(data_dir,0:18)%
% data_process_catBlocks(data_dir,[2:4 8:10 14:16])%arm blocks
%
save_path = fullfile(data_dir,'process','zscore','catBlocks');
load(fullfile(save_path,['catBlocks_' data_dir(end-7:end) '.mat']));

% remove channels

remove_channels = [3 4 185:188];
cels = 1:192;
cels(remove_channels) = [];

%%
inds = [];
for t = 1:size(trial.start,1)
inds = cat(2,inds,ceil(trial.start(t)/20):ceil((trial.go(t))/20));
end
all_data = R_all(inds,cels);

% 6.6. Principal component analysis
N_pca = 10;
[pc_struct.coeff, pc_struct.score, pc_struct.latent, pc_struct.tsquared, ...
    pc_struct.explained, pc_struct.mu] = pca(all_data);

D = cat(2,pc_struct.coeff(:,1),pc_struct.coeff(:,2),pc_struct.coeff(:,3));
% D = cat(2,B_vortho_move(:,1),B_vortho_move(:,2));
[Q,~] = qr(D); 

all_data = R_all(:,cels);

%%
figure;
memory_trials = trial.task(1:500)==1;% | trial.task(1:100)==6;
for targ = 2:3
%     subplot (2,2,targ)
    hold on
    cmap = colormap(lines(6));
    for t = find(memory_trials)'
        if trial.target(t)==targ

            c = cmap(trial.target(t)+2,:);
%             c = cmap(trial.task(t),:);

            inds = ceil(trial.stimOn(t)/20):ceil(trial.go(t)/20);
            if sum(isnan(inds))==0
                use = all_data(inds,:);

                p1 = plot3(use*Q(:,1),use*Q(:,2),use*Q(:,3),'-','color', c,'linewidth',2);
                p1.Color(4) = 0.15;
            end
     
            inds_go = ceil(trial.go(t)/20);
            if ~isnan(inds)
                use = all_data(inds_go,:);
                p2 = plot3(use*Q(:,1),use*Q(:,2),use*Q(:,3),'o','color', c,'MarkerFaceColor', c);
            end
% 
%             inds = ceil(trial.moveOn(t)/20);
%             if ~isnan(inds)
%                 use = all_data(inds,:);
%                 p3 = plot3(use*Q(:,1),use*Q(:,2),use*Q(:,3),'o','color', c);
%             end
% 
%             inds = ceil(trial.start(t)/20):10:ceil(trial.end(t)/20);
%             if ~isnan(inds)
%                 use = all_data(inds,:);
%                 p3 = plot3(use*Q(:,1),use*Q(:,2),use*Q(:,3),'.','color', c);
%             end
        end
    end
end

%%

% task_names = {'DelayGo','ReactGo','MemoryGo','DelayAnti','ReactAnti','MemoryAnti'};
% align_name = {'Stim','DelayOn' 'Go' 'MoveOn'};
% align_var = {trial.stimOn, trial.delayOn, trial.go, trial.moveOn};
% %%
% for task_id = [3:4 6]
%     for align_i = 1:size(align_name,2)
%         savedir = fullfile(data_dir,'analysis','example cells',task_names{task_id},align_name{align_i});
%         if ~exist(savedir,'dir')
%             mkdir(savedir)
%         end
%         close all
%         figure('position',[0 0 1000 1000]);
%         fig_num = 1;
%         width = 20;
%         u_trials = trial.task==task_id;% | trial.task(1:100)==6;
% %         c1 = round(align_var{align_i}(trial.target==2)/20);
% %         c2 = round(align_var{align_i}(trial.target==3)/20);
%         %     [B, look_cels] = sort(abs(nanmean(all_data(c1(~isnan(c1)),:)) - nanmean(all_data(c2(~isnan(c2)),:))));
%         
%         cel_count = 0;
%         for cel = 1:size(all_data,2)
%             if cel_count==15
%                 
%                 xlabel([align_name{align_i} ' Aligned Time (ms)'])
%                 ylabel('Firing Rate (z scored)')
%                 legend('4 targets')
%                 set(gcf,'color','w');
%                 
%                 savefig(fullfile(savedir, [ num2str(fig_num) '.fig']))
%                 set(findall(gcf,'-property','FontSize'),'FontSize',12)
%                 print(fullfile(savedir, [ num2str(fig_num) '.pdf']),'-dpdf','-fillpage')
%                 close all
%                 
%                 figure('position',[0 0 1400 800]);
%                 fig_num = fig_num + 1;
%                 cel_count=0;
%             end
%             cel_count = cel_count+1;
%             subplot(5,3,cel_count)
%             hold on
%             
%             cmap = colormap(lines(6));
%             for t = find(u_trials)'
%                 
%                 c = cmap(trial.target(t)+2,:);
%                 
%                 inds = ceil(align_var{align_i}(t)/20)-width:ceil(align_var{align_i}(t)/20)+width;
%                 if sum(isnan(inds))==0
%                     p1 = plot(all_data(inds,cel),'-','color', c,'linewidth',2);
%                     p1.Color(4) = 0.15;
%                 end
%                 
%             end
%             
%             xticks(0:10:2*width)
%             xticklabels({num2str(20*[-width:10:width]')})
%             title(['cell ' string(cel)])
%             set(findall(gcf,'-property','FontSize'),'FontSize',16)
%             
%             
%         end
%         
%         
%     end
% end


%%
figure
contexts = {'fdgo', 'delaygo', 'fdanti', 'delayanti'}; %'reactgo', 'reactanti', 
rts = cell(size(contexts));
rts_mean = zeros(size(contexts));
rts_std = zeros(size(contexts));
for context = 1:size(contexts,2)
    
    plot_trials = trial.task==context;
    tar_pos = trial.target == 1:4;
    F = cat(2,tar_pos(:,4)-tar_pos(:,2),tar_pos(:,1)-tar_pos(:,3),ones(size(tar_pos,1),1))';
    stim_data = nan(size(F,2),size(all_data,2));
    use_trials = zeros(size(F,2),1);
    
    for t = find(trial.task==context)'
        inds = ceil(trial.go(t)/20):ceil((trial.moveOn(t))/20);
        if sum(isnan(inds))==0
            stim_data(t,:) = nanmean(all_data(inds,:),1);
            use_trials(t) = ~isnan(sum(stim_data(t,:)));
        end
    end

    B_vortho_stim = tdr(stim_data(use_trials==1,:),F(:,use_trials==1));
    D = cat(2,B_vortho_stim(:,1),B_vortho_stim(:,2));
    [Q,~] = qr(D); 

    subplot(2,2,context)
    hold on
    cmap = colormap(lines(6));
    for t = find(plot_trials)'
    %     if trial.target(t)>2
            c = cmap(trial.target(t)+2,:);

            inds = ceil(trial.start(t)/20):ceil(trial.end(t)/20);
            use = all_data(inds,:);

            p1 = plot(use*Q(:,1),use*Q(:,2),'-','color', c,'linewidth',2);
            p1.Color(4) = 0.1;

            inds = ceil(trial.moveOn(t)/20);
            if ~isnan(inds)
                use = all_data(inds,:);
                p2 = plot(use*Q(:,1),use*Q(:,2),'o','color', c,'MarkerFaceColor', c);
            end

            inds = ceil(trial.go(t)/20);
            if ~isnan(inds)
                use = all_data(inds,:);
                p3 = plot(use*Q(:,1),use*Q(:,2),'d','color', c);%,'MarkerFaceColor', c);
            end

            inds = ceil(trial.stimOn(t)/20);
            if ~isnan(inds)
                use = all_data(inds,:);
                p3 = plot(use*Q(:,1),use*Q(:,2),'^','color', c,'MarkerFaceColor', c);
            end
    %     end
    end
    
    rts{context} = trial.moveOn(plot_trials) - trial.go(plot_trials);
    rts_mean(context) = nanmean(rts{context});
    rts_std(context) = nanstd(rts{context});

    title(contexts{context})
end


%% svm classifer

bin_size = 10;
inds_total = 50;
bins_total = 1;%2*ceil(inds_total/bin_size);

ut = ones(size(trial.go,1),1);
context = 1;
data_vec = nan(bins_total,size(ut,2),size(all_data,2));
center_ind = trial.moveOn;

g_vec = (trial.task==context) & ~isnan(center_ind) & ut;
for t = find(g_vec)'
    ii = ceil(center_ind(t)/20) - inds_total;
    for b = 1:bins_total
        inds = (ii+(b-1)*bin_size):(ii+(b*bin_size));
        data_vec(b,t,:) = nanmean(all_data(inds,:),1);
    end
end

%%
accuracy_test = nan(bins_total,1);
accuracy_train = nan(bins_total,1);
c1_vec = trial.target<3;%
u_vec = ones(1,size(data_vec,3));

for b = 1:bins_total
    dv = squeeze(data_vec(b,:,:))';
    [~, accuracy_test(b), accuracy_train(b)] = qsvmc(dv,c1_vec,u_vec,g_vec,.00001);
end

figure;
hold on
plot(accuracy_test,'lineWidth',2)
plot(accuracy_train,'lineWidth',2)
plot([0 bins_total],[.5 .5],':k')
ylim([0 1])

%%
[accuracy_test, accuracy_train] = qsvmc_across(permute(data_vec,[1 3 2]),c1_vec,u_vec,g_vec,.001);
figure;imagesc(accuracy_test,[0.5 1])
axis square


%% naive bayes classifer
figure('position',[0 0 900 180])
label_set = {'Delay Pro','React Pro','Memory Pro','Delay Anti','React Anti','Memory Anti'};
align_set = 2:5;
align = {'Prev End';'Stim';'Delay';'Go';'Move';'End'};
window_size = 80;
xtick_num = 20;

for context = [1 3 4 6]
    frac_corr = naive_bayes_decode(context,context,all_data,trial,trial.targets);
    

    for a = align_set
        subplot(1,size(align_set,2),a - min(align_set)+1)
        hold on
        plot(frac_corr{a},'linewidth',2,'DisplayName',label_set{context})
        
        p = plot([window_size window_size],[0 1],':k');
        set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        p = plot([0 2*window_size],[.25 .25],'-k');
        set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

        text(window_size*1.1,.28,'Chance')
        title(['Align to ' align{a}])
        ylabel('Fraction Correct')
        xlabel('Time (ms)')
        xticks(xtick_num:xtick_num:2*window_size)
        xticklabels({num2str(20*[xtick_num-window_size:xtick_num:window_size]')})
        ylim([0.1 1])
        xlim([0 3*window_size/2])
    end
end

set(findall(gcf,'-property','FontSize'),'FontSize',10)
legend()

%%
task_names = {'DelayGo','ReactGo','MemoryGo','DelayAnti','ReactAnti','MemoryAnti'};
fileName = ['targets_train_' task_names{context} '_test_' task_names{context}];
% fileName = 'targets_train_AllTasks_test_AllTasks';
savedir = fullfile(data_dir,'analysis','decode');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-bestfit')

savefig(gcf,fullfile(savedir, [ fileName '.png']),'format','png')

%% shift train and test
align = {'Prev End';'Stim';'Delay';'Go';'Move';'End'};
bin_size = 10;
window_size = 150;
ind_set = {[0; trial.end(1:end-1)],trial.stimOn,trial.delayOn,trial.go,trial.moveOn,trial.end};
frac_corr_train = nan(size(align));
frac_corr_test = cell(size(align));
context_train = 1;
context_test = 1;
align_set = 2:5;
inds_total = 50;
bins_total = 1;%2*ceil(inds_total/bin_size);

ut = ones(size(trial.go,1),1);
context = 1;
data_vec = nan(bins_total,size(ut,2),size(all_data,2));
c1_vec = trial.stim;%trial.target;
u_vec = ones(1,size(data_vec,3));

for a = 2:size(align,1)-1
    
    center_ind = ind_set{a};
    g_vec = (trial.task==context_train) & ~isnan(center_ind) & ut;
    data_vec_train = nan(size(ut,1),size(all_data,2));
    
    for t = find(g_vec)'
        ii = round(center_ind(t)/20);
        inds = (ii - round(bin_size/2)):(ii + round(bin_size/2));
        data_vec_train(t,:) = nanmean(all_data(inds,:),1);
    end
    
    allFeatures = data_vec_train(g_vec,:);
    trlCodes = c1_vec(g_vec);
    codeList = unique(trlCodes);
    
    obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','Prior',ones(size(codeList)));
    cvmodel = crossval(obj);
    L = kfoldLoss(cvmodel);
    predLabels = kfoldPredict(cvmodel);
    
    frac_corr_train(a) = sum((trlCodes - predLabels)==0)/size(trlCodes,1);
    frac_corr_test{a} = nan(window_size*2,1);
    
    for shift_ind = 1:window_size*2
        g_vec = (trial.task==context_test) & ~isnan(center_ind) & ut;
        data_vec_test = nan(size(ut,1),size(all_data,2));
        
        for t = find(g_vec)'
            
            ii = round(center_ind(t)/20) - window_size + shift_ind;
            
            if ii > ind_set{a-1}(t)/20 && (ii+bin_size) < ind_set{a+1}(t)/20
                inds = ii:(ii+bin_size);
                data_vec_test(t,:) = nanmean(all_data(inds,:),1);
            else
                g_vec(t) = 0;
            end
            
        end
        
        if sum(g_vec)>10
            allFeatures = data_vec_test(g_vec,:);
            trlCodes = c1_vec(g_vec);
            codeList = unique(trlCodes);
            
            predLabels_test = predict(obj,allFeatures);
            
            frac_corr_test{a}(shift_ind,1) = sum((trlCodes - predLabels_test)==0)/size(trlCodes,1);
        end
    end
end

figure('position',[0 0 1600 400])
for a = align_set
    subplot(1,size(align_set,2),a - min(align_set)+1)
    hold on
    plot(frac_corr_test{a},'linewidth',2)
    plot([window_size window_size],[0 1],':k')
    plot([0 2*window_size],[.25 .25],'-k')
    text(window_size*1.5,.28,'Chance')
    title(['Align to ' align{a}])
    ylabel('Fraction Correct')
    xlabel('Time (ms)')
    xticks(0:50:2*window_size)
    xticklabels({num2str(20*[-window_size:50:window_size]')})
    ylim([0.1 1.1])
end
set(findall(gcf,'-property','FontSize'),'FontSize',12)
%%
task_names = {'DelayGo','ReactGo','MemoryGo','DelayAnti','ReactAnti','MemoryAnti'};
fileName = ['targets_train_' task_names{context_train} '_test_' task_names{context_test}];
savedir = fullfile(data_dir,'analysis','decode');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-fillpage')

%%
C = confusionmat(trlCodes, predLabels);
for rowIdx=1:size(C,1)
    C(rowIdx,:) = C(rowIdx,:)/sum(C(rowIdx,:));
end

%% stim info
context1 = 1;
context2 = 4;
%%
frac_corr = naive_bayes_decode(context1,context2,all_data,trial,trial.target);

align = {'Prev End';'Stim';'Delay';'Go';'Move';'End'};
window_size = 80;
xtick_num = 40;

figure('position',[0 0 1600 200])
for a = 1:size(align,1)
    subplot(1,size(align,1),a)
    hold on
    plot(frac_corr{a},'linewidth',2)
    plot([window_size window_size],[0 1],':k')
    plot([0 2*window_size],[.25 .25],'-k')
    text(window_size*1.1,.28,'Chance')
    title(['Align to ' align{a}])
    ylabel('Fraction Correct')
    xlabel('Time (ms)')
    
    xticks(xtick_num:xtick_num:2*window_size)
    xticklabels({num2str(20*[xtick_num-window_size:xtick_num:window_size]')})
    ylim([0.1 1])
    xlim([0 2*window_size])
end
set(findall(gcf,'-property','FontSize'),'FontSize',12)
%%
task_names = {'DelayGo','ReactGo','MemoryGo','DelayAnti','ReactAnti','MemoryAnti'};
fileName = ['target_train_' task_names{context1} '_test_' task_names{context2}];
savedir = fullfile(data_dir,'analysis','decode');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-fillpage')


%% context info

align = {'Start';'Stim';'Delay';'Go';'Move';'End'};
bin_size = 10;
window_size = 80;
ind_set = {trial.start,trial.stimOn,trial.delayOn,trial.go,trial.moveOn,trial.end};
frac_corr = cell(size(align));
context1 = 2;
context2 = 5;

for a = 1
    
    center_ind = ind_set{a};
    frac_corr{a} = nan(window_size*2,1);
    
    for shift_ind = 1:window_size*2
        g_vec = (trial.task==context1 | trial.task==context2) & ~isnan(center_ind) & ut;
        data_vec = nan(size(ut,2),size(all_data,2));
        
        for t = find(g_vec)'
            
            ii = ceil(center_ind(t)/20) + shift_ind;
            
            if (ii+bin_size) < ind_set{a+1}(t)/20
                inds = ii:(ii+bin_size);
                data_vec(t,:) = nanmean(all_data(inds,:),1);
            else
                g_vec(t) = 0;
            end
            
        end
        
        if sum(g_vec)>40
            allFeatures = data_vec(g_vec,:);
            trlCodes = trial.task(g_vec);
            codeList = unique(trlCodes);
            
            obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','Prior',ones(size(codeList)));
            cvmodel = crossval(obj);
            L = kfoldLoss(cvmodel);
            predLabels = kfoldPredict(cvmodel);
            
            frac_corr{a}(shift_ind,1) = sum((trlCodes - predLabels)==0)/size(trlCodes,1);
        end
    end
end

%%
task_names = {'Delay Pro','React Pro','Memory Pro','Delay Anti','React Anti','Memory Anti'};
xtick_num = 40;

figure('position',[0 0 300 300])
for a = 1
    hold on
    plot(frac_corr{a},'linewidth',2)
    plot([0 0],[0 1],':k')
    plot([0 2*window_size],[1/size(codeList,1) 1/size(codeList,1)],'-k')
    text(window_size*1.6,1.1/size(codeList,1),'Chance')
    title([task_names{context1} ' vs ' task_names{context2}])
    ylabel('Fraction Correct')
    xlabel('Time (ms)')
    
    xticks(xtick_num:xtick_num:2*window_size)
    xticklabels({num2str(20*[xtick_num:xtick_num:2*window_size]')})
    ylim([0.1 1])
    xlim([0 2*window_size])
end
set(findall(gcf,'-property','FontSize'),'FontSize',12)


fileName = ['tasks_' task_names{context1} '_vs_' task_names{context2}];
savedir = fullfile(data_dir,'analysis','decode');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-bestfit')


%%
define_task_sep

%% plot in task_sep space

ut = ones(size(trial.go,1),1);
align = {'Start';'Stim';'Go';'Move';'End'};
window_size = 100;
ind_set = {trial.start,trial.stimOn,trial.go,trial.moveOn,trial.end};
data_vec = cell(size(align));

for a = 1:size(align,1)
    
    center_ind = ind_set{a};
    g_vec = ~isnan(center_ind) & ut;
    data_vec{a} = nan(2,sum(ut),2*window_size+1);
    
    for t = find(g_vec)'
        inds = (ceil(center_ind(t)/20)-window_size):(ceil(center_ind(t)/20)+window_size);
        data_vec{a}(1,t,:) = all_data(inds,:)*task_sep(cels);
        data_vec{a}(2,t,:) = all_data(inds,:)*cond_indep(cels);
    end
    
end
%%
figure('position',[0 0 1600 400])
window_set = [0,100,1,5,5;100,0,10,1,10];
for a = 1:size(align,1)
    inds = (window_size-window_set(1,a)):window_size+window_set(2,a);
    
    for task_num = 1:6
        
        subplot(2,size(align,1),a+size(align,1))
        hold on
        plot(squeeze(nanmean(data_vec{a}(2,trial.task==task_num,1+inds),2)),'linewidth',2)
        ylabel('Condition Independent Space')
        plot([window_set(1,a) window_set(1,a)],[-3 15],':k')
        ylim([-3 15])
        
        subplot(2,size(align,1),a)
        hold on
        plot(squeeze(nanmean(data_vec{a}(1,trial.task==task_num,1+inds),2)),'linewidth',2)
        ylabel('Task Separation Space')
        plot([window_set(1,a) window_set(1,a)],[-1 3],':k')
        ylim([-1 3])
        
        
    end
    subplot(2,size(align,1),a)
    title(['Align to ' align{a}])
    xticklabels([])
    
    subplot(2,size(align,1),size(align,1)+a)
    xt = xticks;
    xticklabels({num2str(20*[xt - window_set(1,a)]')})
    xlabel('Time (ms)')
end
set(findall(gcf,'-property','FontSize'),'FontSize',12)

%%
fileName = 'full trial_CIS_TSS';
savedir = fullfile(data_dir,'analysis','dim_reduc');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-fillpage')


%% plot in stim targ space

define_interact_comp
%%
ut = ones(size(trial.go,1),1);
align = {'Start';'Stim';'Delay';'Go';'Move';'End'};
window_size = 100;
ind_set = {trial.start,trial.stimOn,trial.delayOn,trial.go,trial.moveOn,trial.end};
data_vec = cell(size(align));

for a = 1:size(align,1)
    
    center_ind = ind_set{a};
    g_vec = ~isnan(center_ind) & ut;
    data_vec{a} = nan(3,sum(ut),2*window_size+1);
    
    for t = find(g_vec)'
        inds = (ceil(center_ind(t)/20)-window_size):(ceil(center_ind(t)/20)+window_size);
        data_vec{a}(2,t,:) = all_data(inds,:)*targ_comp(cels);
        data_vec{a}(3,t,:) = all_data(inds,:)*stim_comp(cels);
        data_vec{a}(1,t,:) = all_data(inds,:)*cond_indep_comp(cels);
    end
    
end
%%
figure('position',[0 0 1600 400])
cmap = lines(4);
line_set = {'-','-','-',':',':',':'};

window_set = [0,50,50,1,5,5;100,50,50,10,1,10];
for a = 1:size(align,1)
    inds = (window_size-window_set(1,a)):window_size+window_set(2,a);
    
    for task_num = [1 4]%:6
        for target_num = [1:4]
            t_set = trial.task==task_num & trial.target==target_num;
            subplot(3,size(align,1),a+size(align,1))
            hold on
            plot(squeeze(nanmean(data_vec{a}(3,t_set,1+inds),2)),line_set{task_num},'color',cmap(target_num,:),'linewidth',2)
            ylabel('Stim Space')
            plot([window_set(1,a) window_set(1,a)],[-3 3],':k')
            ylim([-3 3])

            subplot(3,size(align,1),a+2*size(align,1))
            hold on
            plot(squeeze(nanmean(data_vec{a}(2,t_set,1+inds),2)),line_set{task_num},'color',cmap(target_num,:),'linewidth',2)
            ylabel('Target Space')
            plot([window_set(1,a) window_set(1,a)],[-3 3],':k')
            ylim([-3 3])

            subplot(3,size(align,1),a)
            hold on
            plot(squeeze(nanmean(data_vec{a}(1,t_set,1+inds),2)),line_set{task_num},'color',cmap(target_num,:),'linewidth',2)
            ylabel('Condition Indep Space')
            plot([window_set(1,a) window_set(1,a)],[-3 3],':k')
            ylim([-3 3])
        
        end
    end
    subplot(3,size(align,1),a)
    title(['Align to ' align{a}])
    xticklabels([])
    
    subplot(3,size(align,1),size(align,1)+a)
    xt = xticks;
    xticklabels({num2str(20*[xt - window_set(1,a)]')})
    xlabel('Time (ms)')
end
set(findall(gcf,'-property','FontSize'),'FontSize',12)

%%
fileName = 'full trial_CIS_TSS';
savedir = fullfile(data_dir,'analysis','dim_reduc');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-fillpage')


%%
figure;
nr = 3;
nc = 3;

for t = 1:3
    use_t = (trial.task==t | trial.task == (3+t));
    subplot(nr,nc,1+((t-1)*nc))
    h1 = histogram(trial.contextOff(use_t)-trial.start(use_t),2000:100:4000,'Facecolor',[0.3010 0.7450 0.9330],'edgecolor','none');
    xticks([2000:1000:4000])
    title('Context Duration (ms)')

    subplot(nr,nc,2+((t-1)*nc))
    h2 = histogram(trial.delayOn(use_t)-trial.stimOn(use_t),0:250:3500,'Facecolor',[0.3010 0.7450 0.9330],'edgecolor','none');
    xticks(0:1000:3500)
    title('Stim Duration (ms)')

    subplot(nr,nc,3+((t-1)*nc))
    h3 = histogram(trial.go(use_t)-trial.delayOn(use_t),0:250:3500,'Facecolor',[0.3010 0.7450 0.9330],'edgecolor','none');
    xticks(0:1000:3500)
    title('Delay Duration (ms)')
end

set(findall(gcf,'-property','FontSize'),'FontSize',16)

fileName = 'epoch_length_histograms';
savedir = fullfile(data_dir,'analysis');
if ~exist(savedir,'dir')
    mkdir(savedir)
end

%%
savefig(fullfile(savedir, [ fileName '.fig']))
set(findall(gcf,'-property','FontSize'),'FontSize',12)
print(fullfile(savedir, [ fileName '.pdf']),'-dpdf','-bestfit')




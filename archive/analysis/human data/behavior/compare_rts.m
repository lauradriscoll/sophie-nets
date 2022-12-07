%% RT v delay Plot

[ out ] = getLNDPaths( );
data_dur = fullfile(out.data,'human/yangnet/20180730/');
process_dur = fullfile(data_dur,'process','zscore');

delayTime_all = cell(7,1);
reactTime_all = cell(7,1);
taskLabel_all = cell(7,1);

for block = [4:8 10:18]
load(fullfile(process_dur,['R_process_block' num2str(block)]),'trial')

delayTime = trial.go - trial.contextOff;
reactTime = trial.moveOn - trial.go;

delayTime_all{1} = cat(1,delayTime_all{1},delayTime((trial.task==1 | trial.task==4) & trial.success==1));
delayTime_all{2} = cat(1,delayTime_all{2},delayTime((trial.task==2 | trial.task==5) & trial.success==1));
delayTime_all{3} = cat(1,delayTime_all{3},delayTime((trial.task==3 | trial.task==6) & trial.success==1));

reactTime_all{1} = cat(1,reactTime_all{1},reactTime((trial.task==1 | trial.task==4) & trial.success==1));
reactTime_all{2} = cat(1,reactTime_all{2},reactTime((trial.task==2 | trial.task==5) & trial.success==1));
reactTime_all{3} = cat(1,reactTime_all{3},reactTime((trial.task==3 | trial.task==6) & trial.success==1));

taskLabel_all{1} = cat(1,taskLabel_all{1},trial.task((trial.task==1 | trial.task==4) & trial.success==1));
taskLabel_all{2} = cat(1,taskLabel_all{2},trial.task((trial.task==2 | trial.task==5) & trial.success==1));
taskLabel_all{3} = cat(1,taskLabel_all{3},trial.task((trial.task==3 | trial.task==6) & trial.success==1));
end

% load(fullfile(process_dur,['R_process_block' num2str(0)]),'trial')
% delayTime = trial.go - trial.contextOff;
% reactTime = trial.moveOn - trial.go;
% delayTime_all{4} = cat(1,delayTime_all{4},delayTime((trial.task==1 | trial.task==4) & trial.success==1));
% reactTime_all{4} = cat(1,reactTime_all{4},reactTime((trial.task==1 | trial.task==4) & trial.success==1));
% taskLabel_all{4} = cat(1,taskLabel_all{4},trial.task((trial.task==1 | trial.task==4) & trial.success==1));
% 
% load(fullfile(process_dur,['R_process_block' num2str(1)]),'trial')
% delayTime = trial.go - trial.contextOff;
% reactTime = trial.moveOn - trial.go;
% delayTime_all{5} = cat(1,delayTime_all{5},delayTime((trial.task==1 | trial.task==3) & trial.success==1));
% reactTime_all{5} = cat(1,reactTime_all{5},reactTime((trial.task==1 | trial.task==3) & trial.success==1));
% taskLabel_all{5} = cat(1,taskLabel_all{5},trial.task((trial.task==1 | trial.task==3) & trial.success==1));
% 
% load(fullfile(process_dur,['R_process_block' num2str(2)]),'trial')
% delayTime = trial.go - trial.contextOff;
% reactTime = trial.moveOn - trial.go;
% delayTime_all{6} = cat(1,delayTime_all{6},delayTime((trial.task==2 | trial.task==5) & trial.success==1));
% reactTime_all{6} = cat(1,reactTime_all{6},reactTime((trial.task==2 | trial.task==5) & trial.success==1));
% taskLabel_all{6} = cat(1,taskLabel_all{6},trial.task((trial.task==2 | trial.task==5) & trial.success==1));
% 
% load(fullfile(process_dur,['R_process_block' num2str(13)]),'trial')
% delayTime = trial.go - trial.contextOff;
% reactTime = trial.moveOn - trial.go;
% delayTime_all{6} = cat(1,delayTime_all{6},delayTime((trial.task==2 | trial.task==5) & trial.success==1));
% reactTime_all{6} = cat(1,reactTime_all{6},reactTime((trial.task==2 | trial.task==5) & trial.success==1));
% taskLabel_all{6} = cat(1,taskLabel_all{6},trial.task((trial.task==2 | trial.task==5) & trial.success==1));
% 
% load(fullfile(process_dur,['R_process_block' num2str(16)]),'trial')
% delayTime = trial.go - trial.contextOff;
% reactTime = trial.moveOn - trial.go;
% delayTime_all{7} = cat(1,delayTime_all{7},delayTime(trial.task==2 & trial.success==1));
% reactTime_all{7} = cat(1,reactTime_all{7},reactTime(trial.task==2 & trial.success==1));
% taskLabel_all{7} = cat(1,taskLabel_all{7},trial.task(trial.task==2 & trial.success==1));

figure;
hold on
rand_fac = 0;
cmap = {'ob','og','or'};
tasks = {'Delay','Memory Delay','RT'};
b = cell(3,1);
r = cell(3,1);
bint = cell(3,1);
rint = cell(3,1);

for x = [1 3 2]
    
plot(randn(size(delayTime_all{x},1),1)*rand_fac+delayTime_all{x},...
    reactTime_all{x},cmap{x},'MarkerSize',7)
end

plot([nanmean(delayTime_all{2})-200 nanmean(delayTime_all{2})+200],...
    [nanmean(reactTime_all{2}) nanmean(reactTime_all{2})],'color',cmap{2}(2),'lineWidth',4)
plot([nanmean(delayTime_all{2})-200 nanmean(delayTime_all{2})+200],...
    [nanmean(reactTime_all{2}) nanmean(reactTime_all{2})],'color','k','lineWidth',1)

for x = [1 3]
[b_temp, bint_temp, r_temp, rint_temp] = regress(reactTime_all{x},cat(2,delayTime_all{x},ones(size(delayTime_all{x}))));
b{x} = b_temp;
r{x} = r_temp;
bint{x} = bint_temp;
rint{x} = rint_temp;
inds = 750:2200;
plot(inds,b{x}(1)*inds+b{x}(2),'color',cmap{x}(2),'lineWidth',4)
plot(inds,b{x}(1)*inds+b{x}(2),'color','k','lineWidth',1)
end

xlabel('Stim and Delay (ms)')
ylabel('Movement Onset (ms)')
legend(tasks)
set(findall(gcf,'-property','FontSize'),'FontSize',20)
save_path_fig = fullfile(data_dur,'analysis','reactionTimeBehavior');

if ~exist(save_path_fig)
    mkdir(save_path_fig)
end
savefig(fullfile(save_path_fig,'scatter_mixed'))

%% compare mixed
figure;
hold on
for x = [1 3 2]
histogram(reactTime_all{x},5*x:15:600,'EdgeColor','none','Facecolor',cmap{x}(2),'FaceAlpha',.7)
end
ylabel('Trial Count')
xlabel('Reaction Time (ms)')
legend(tasks)
set(findall(gcf,'-property','FontSize'),'FontSize',20)

save_path_fig = fullfile(data_dur,'analysis','reactionTimeBehavior');
savefig(fullfile(save_path_fig,'hist_mixed'))

%% compare RTs
figure;
hold on
rtmap = parula(4);
gr = [7 6 2];
rand_fac = 2;
for x = gr
    inds = randperm(size(reactTime_all{x},1));
histogram(reactTime_all{x}(inds(1:44),1),x:30:600,'EdgeColor','none','Facecolor',rtmap(gr==x,:),'FaceAlpha',.7)

% plot(randn(42,1)*rand_fac+delayTime_all{x}(1:42),...
%     reactTime_all{x}(1:42),'.','color',rtmap(gr==x,:),'MarkerSize',10)
end
ylabel('Trial Count')
xlabel('Reaction Time (ms)')
legend({'Pro Only', 'Pro and Anti', 'Mixed'})
set(findall(gcf,'-property','FontSize'),'FontSize',20)

save_path_fig = fullfile(data_dur,'analysis','reactionTimeBehavior');
savefig(fullfile(save_path_fig,'hist_rts'))

%% compare RTs pro v anti
figure;
hold on
tasks = {'Reaction Time Task','Delay Task','Memory Task'};
x_set = [2 1 3];
for x = x_set
    subplot(1,3,find(x==x_set))
    hold on
rand_fac = .2;
inds_anti = taskLabel_all{x}>3;
inds_pro = taskLabel_all{x}<4;

shift_window = [min(delayTime_all{x}) max(delayTime_all{x})];
shift_ind = 2*(shift_window(2));

plot(-shift_ind+randn(sum(inds_anti),1)*rand_fac+delayTime_all{x}(inds_anti),reactTime_all{x}(inds_anti),...
    '.r','MarkerSize',10)

plot(randn(sum(inds_pro),1)*rand_fac+delayTime_all{x}(inds_pro),reactTime_all{x}(inds_pro),...
    '.g','MarkerSize',10)

plot(-shift_ind+[nanmean(delayTime_all{x}(inds_anti))-shift_ind/2 nanmean(delayTime_all{x}(inds_anti))+shift_ind/2],...
    [nanmean(reactTime_all{x}(inds_anti)) nanmean(reactTime_all{x}(inds_anti))],'color','r','lineWidth',4)
plot(-shift_ind+[nanmean(delayTime_all{x}(inds_anti))-shift_ind/2 nanmean(delayTime_all{x}(inds_anti))+shift_ind/2],...
    [nanmean(reactTime_all{x}(inds_anti)) nanmean(reactTime_all{x}(inds_anti))],'color','k','lineWidth',1)

plot([nanmean(delayTime_all{x}(inds_pro))-shift_ind/2 nanmean(delayTime_all{x}(inds_pro))+shift_ind/2],...
    [nanmean(reactTime_all{x}(inds_pro)) nanmean(reactTime_all{x}(inds_pro))],'color','g','lineWidth',4)
plot([nanmean(delayTime_all{x}(inds_pro))-shift_ind/2 nanmean(delayTime_all{x}(inds_pro))+shift_ind/2],...
    [nanmean(reactTime_all{x}(inds_pro)) nanmean(reactTime_all{x}(inds_pro))],'color','k','lineWidth',1)

ylabel('Reaction Time (ms)')
ylim([150 500])
set(gca,'xtick',[-.5*shift_ind,.5*shift_ind])
set(gca,'xticklabel',{'Anti', 'Pro'})
set(findall(gcf,'-property','FontSize'),'FontSize',20)
title(tasks{find(x==x_set)})

end
save_path_fig = fullfile(data_dur,'analysis','reactionTimeBehavior');
savefig(fullfile(save_path_fig,'hist_rts'))
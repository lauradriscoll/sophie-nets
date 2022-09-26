% addpath('/Users/lauradriscoll/Documents/code/yangnet/analysis/human data');
% 
% data_dir = '/Users/lauradriscoll/Documents/data/human/yangnet/20180730';
% data_process_alignAll(data_dir,4)%[4:8 10:11 13:18]
% %%

remove_channels = [3 4 185:188];
cels = 1:192;
cels(remove_channels) = [];


allCR = [];
allcursor = [];
tar_pos = [];
task = [];
epoch = 3;
for b = [4:8 10:11 13:18]
block_temp = load(strcat('/Users/lauradriscoll/Documents/data/human/yangnet/20180730/process/zscore/alignAll/alignAll_20180730_block', string(b), '.mat'));
allCR = cat(1,allCR,block_temp.allCellResponsesR{epoch}(:,:,cels));
allcursor = cat(1,allcursor,block_temp.allCursorR{epoch});
tar_pos = cat(1,tar_pos,block_temp.trial.target == 1:4);
task = cat(1,task,block_temp.trial.task);
end
col = tar_pos*[1:4]';

%%

%%
use_task = task == 3 | task == 6;
use_trials = ~isnan(allCR(:,1,1)) & use_task;
F = cat(2,tar_pos(use_trials,4)-tar_pos(use_trials,2),tar_pos(use_trials,1)-tar_pos(use_trials,3),ones(sum(use_trials),1))';
F_titles = {'right';'up';'ones'}; %check this

%%
hidden = allCR(use_trials,:,:);
stim_inds = 15:20;
move_inds = 26:31;
all_inds = [stim_inds move_inds];
B_vortho_stim = make_B_vortho(hidden,F,stim_inds);
B_vortho_move = make_B_vortho(hidden,F,move_inds);

%%
figure;
subplot(2,2,1)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_stim(:,1),B_vortho_stim(:,2),allCR(:,stim_inds,:),tinds,col)
ylabel('Memory Neural Movement Y')
xlabel('Memory Neural Movement X')
title('Memory Period')
axis square

subplot(2,2,2)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_move(:,1),B_vortho_move(:,2),allCR(:,stim_inds,:),tinds,col)
ylabel('Readout Neural Movement Y')
xlabel('Readout Neural Movement X')
title('Memory Period')
axis square

subplot(2,2,3)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_stim(:,1),B_vortho_stim(:,2),allCR(:,move_inds,:),tinds,col)
ylabel('Memory Neural Movement Y')
xlabel('Memory Neural Movement X')
title('Readout Period')
axis square

subplot(2,2,4)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_move(:,1),B_vortho_move(:,2),allCR(:,move_inds,:),tinds,col)
ylabel('Readout Neural Movement Y')
xlabel('Readout Neural Movement X')
title('Readout Period')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)


%%
subplot(1,2,1)
hold on
cmap = colormap(lines(6));
for t = find(use_task)'
    use = squeeze(allcursor(t,:,:));
    p1 = plot(use(stim_inds,1),use(stim_inds,2),'color', cmap(col(t)+2,:),'linewidth',2);
    p1.Color(4) = 0.25;
    p1 = plot(use(stim_inds(1),1),use(stim_inds(1),2),'^','color', cmap(col(t)+2,:),'linewidth',2);
%     p1 = plot(use(stim_inds(end),1),use(stim_inds(end),2),'o','color', cmap(col(t)+2,:),'linewidth',2);
end
ylabel('Cursor Movement Y')
xlabel('Cursor Movement X')
title('Memory Period')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)

subplot(1,2,2)
hold on
cmap = colormap(lines(6));
for t = find(use_task)'
    use = squeeze(allcursor(t,:,:));
    p2 = plot(use(move_inds,1),use(move_inds,2),'color', cmap(col(t)+2,:),'linewidth',2);
    p2.Color(4) = 0.25;
    p2 = plot(use(move_inds(1),1),use(move_inds(1),2),'^','color', cmap(col(t)+2,:),'linewidth',2);
%     p2 = plot(use(move_inds(end),1),use(move_inds(end),2),'o','color', cmap(col(t)+2,:),'linewidth',2);
end
ylabel('Cursor Movement Y')
xlabel('Cursor Movement X')
title('Readout Period')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)

%%
hidden = allCR(:,all_inds,:);

figure;

subplot(2,2,1)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_stim(:,1),B_vortho_stim(:,2),allCR(:,stim_inds,:),tinds,col)
ylabel('Memory (sin theta)')
xlabel('Memory (cos theta)')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)

subplot(2,2,2)
hold on
tinds = find(use_task)';
plot_mem_readout(B_vortho_move(:,1),B_vortho_move(:,2),allCR(:,move_inds,:),tinds,col)
ylabel('Readout (sin theta)')
xlabel('Readout (cos theta)')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)


subplot(2,2,3)
hold on
tinds = find(use_task & mod(col+1,2))';
plot_mem_readout(B_vortho_move(:,1),B_vortho_stim(:,1),allCR(:,all_inds,:),tinds,col)
ylabel('Memory (cos theta)')
xlabel('Readout (cos theta)')
title('Right/Left')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)

subplot(2,2,4)
hold on
tinds = find(use_task & mod(col,2))';
plot_mem_readout(B_vortho_move(:,2),B_vortho_stim(:,2),allCR(:,all_inds,:),tinds,col)
ylabel('Memory (sin theta)')
xlabel('Readout (sin theta)')
title('Up/ Down')
axis square
set(findall(gcf,'-property','FontSize'),'FontSize',16)
%%


D = cat(2,B_vortho_move(:,2),B_vortho_stim(:,2));
[Q,~] = qr(D); 
%%
% figure;
subplot(2,2,4)
hold on
cmap = colormap(lines(6));
for t = find(use_task & mod(col,2))'
    use = squeeze(allCR(t,:,:));
    p4 = plot(use*D(:,1),use*D(:,2),'color', cmap(col(t)+2,:),'linewidth',2);
    p4.Color(4) = 0.25;
    p4 = plot(use(1,:)*D(:,1),use(1,:)*D(:,2),'^','color', cmap(col(t)+2,:),'linewidth',2);
%     p4 = plot(use(end,:)*D(:,1),use(end,:)*D(:,2),'o','color', cmap(col(t)+2,:),'linewidth',2);
end
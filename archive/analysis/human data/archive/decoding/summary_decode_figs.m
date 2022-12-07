raw_fold = '/Users/lauradriscoll/Documents/data/human/yangnet/20180718/analysis/block_decode/zscore';
fileList = dir(fullfile(raw_fold,'*epoch1_blocks'));

for f = 1:size(fileList,1)
    
    figFile = fileList(f).name(1:end-14);
    epochNames = {'Context','Stim and Delay','Movement'};
    
    if strcmp(figFile(1:5),'useRT')
        epochs = [1 3];
        figure('position',[1089 405 588 543],'Name',figFile);
    else
        epochs = 1:3;
        figure('position',[640 320 1041 628],'Name',figFile);
    end
    
    for epoch = epochs
        
        epoch_folder = fullfile(raw_fold,[figFile '_epoch' num2str(epoch) '_blocks']);
        allBlockFiles = dir(fullfile(epoch_folder,'*.mat'));
        
        behave_test_group_all = [];
        behave_train_group_all = [];
        neuron_test_group_all = [];
        neuron_train_group_all = [];
        
        for block = 1:size(allBlockFiles,1)
            load(fullfile(epoch_folder,allBlockFiles(block).name))
            
            behave_test_group_all = cat(2,behave_test_group_all,behave_test_group);
            behave_train_group_all = cat(2,behave_train_group_all,behave_train_group);
            neuron_test_group_all = cat(2,neuron_test_group_all,neuron_test_group);
            neuron_train_group_all = cat(2,neuron_train_group_all,neuron_train_group);
        end
        
        subplot(2,size(epochs,2),find(epoch==epochs))
        hold on
        plot(nanmean(behave_test_group_all,2),':r','lineWidth',2)
        plot(nanmean(behave_train_group_all,2),'-r','lineWidth',2)
        plot(nanmean(neuron_test_group_all,2),':b','lineWidth',2)
        plot(nanmean(neuron_train_group_all,2),'-b','lineWidth',2)
        
        axis square
        title(epochNames{epoch})
        xticks(0:10:30)
        xticklabels(20*(0:10:30))
        ylabel('Fraction Correct')
        xlabel('Time (ms)')
        ylim([.4 1])
        
        subplot(2,size(epochs,2),size(epochs,2)+find(epoch==epochs))
        imagesc(corr_mat_temp(:,:,1));
        title('Decision Plane Stability')
        xticklabels(20*(0:10:30))
        yticklabels(20*(0:10:30))
        xlabel('Time (ms)')
        ylabel('Time (ms)')
        axis square
    end
    
    subplot(2,size(epochs,2),find(epoch==epochs))
    legend('Behavior Test','Behavior Train','Neural Test','Neural Train','location','southeast')
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    
    savefig(fullfile(epoch_folder,[figFile '_corr']))
end
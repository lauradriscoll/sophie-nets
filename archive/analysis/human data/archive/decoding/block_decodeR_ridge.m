function block_decodeR_ridge(data_dur,blocks,use_trial_rule,trial_label_rule,figName,epochs)

for epoch = epochs
    
    figure;
    fullFigName = [figName num2str(epoch)];
    step_size = 1;
    steps = 1:step_size:30*step_size;%sum(useInd)
    cmap = spring(2*size(steps,2));
    hmap = summer(2*size(steps,2));
    
    save_path_file = fullfile(data_dur,'analysis','block_decode','zscore',[fullFigName '_blocks']);
    save_path_fig = fullfile(data_dur,'analysis','block_decode','zscore');
    if ~exist(save_path_file,'dir')
        mkdir(save_path_file)
    end
    
    corr_mat = nan(10,size(steps,2),size(steps,2),2);
    for block = blocks%use_blocks
        
        load(fullfile(data_dur,'process','zscore',['R_process_block' num2str(block)]),...
            'allCursorR','allCellResponsesR','trial')
        fullFileName = [fullFigName '_block' num2str(block)];
        
        eval(use_trial_rule)
        eval(trial_label_rule)
        %%
        num_its = 10;
        neuron_test_group = nan(size(steps,2),num_its);
        neuron_train_group = nan(size(steps,2),num_its);
        behave_test_group = nan(size(steps,2),num_its);
        behave_train_group = nan(size(steps,2),num_its);
        dpn = nan(size(steps,2),size(allCellResponsesR{epoch},3),num_its);
        dpb = nan(size(steps,2),size(allCursorR{epoch},3),num_its);
        
        
        for step = steps
            frames = step:(step+step_size);
            ndata = squeeze(nanmean(allCellResponsesR{epoch}(:,frames,:),2))';
            bdata = squeeze(nanmean(allCursorR{epoch}(:,frames,:),2))';
            trial_t = trial_labels & use_trials;
            trial_f = trial_labels==0 & use_trials;
            
            subplot(1,3,1)
            hold on
            plot(bdata(1,trial_t),bdata(2,trial_t),'.','color',cmap(step==steps,:),'MarkerSize',10)
            plot(bdata(1,trial_f),bdata(2,trial_f),'.','color',hmap(step==steps,:),'MarkerSize',10)
            
            for it = 1:num_its
                
                [nsvm_struct, accuracy_test, accuracy_train] = ...
                    qsvmc(ndata,trial_labels,1:size(ndata,1),use_trials,.0001);
                
                b = ridge(trial_labels,X,k);
                neuron_test_group(step==steps,it) = accuracy_test;
                neuron_train_group(step==steps,it) = accuracy_train;
                dpn(step,:,it) = nsvm_struct.Beta/max(nsvm_struct.Beta);
                
                [bsvm_struct, accuracy_test, accuracy_train] = ...
                    qsvmc(bdata,trial_labels,1:size(bdata,1),use_trials,.1);
                behave_test_group(step==steps,it) = accuracy_test;
                behave_train_group(step==steps,it) = accuracy_train;
                dpb(step,:,it) = bsvm_struct.Beta/max(bsvm_struct.Beta);
                
            end
            
        end
        corr_mat_temp = cat(3,corrcoef(squeeze(nanmean(dpn,3))'),corrcoef(squeeze(nanmean(dpb,3))'));
        corr_mat(block,:,:,:) = corr_mat_temp;
        %%
        subplot(1,3,2)
        hold on
        plot(nanmean(behave_test_group,2),':r','lineWidth',2)
        plot(nanmean(behave_train_group,2),'-r','lineWidth',2)
        plot(nanmean(neuron_test_group,2),':b','lineWidth',2)
        plot(nanmean(neuron_train_group,2),'-b','lineWidth',2)
        
        save(fullfile(save_path_file,fullFileName),'behave_test_group','behave_train_group',...
            'neuron_test_group','neuron_train_group','corr_mat_temp')
    end
    
    subplot(1,3,3)
    title('Decision Plane Stability')
    imagesc(squeeze(nanmean(corr_mat(:,:,:,1),1)))
    xticklabels(20*(0:10:30))
    yticklabels(20*(0:10:30))
    xlabel('Time (ms)')
    ylabel('Time (ms)')
    axis square
    
    %%
    subplot(1,3,2)
    title(fullFigName)
    axis square
    legend('Behavior Test','Behavior Train','Neural Test','Neural Train','location','southeast')
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    xticks(0:10:30)
    xticklabels(20*(0:10:30))
    ylabel('Fraction Correct')
    xlabel('Time (ms)')
    
    subplot(1,3,1)
    title(['Epoch ' num2str(epoch) ' Cursor Tracking'])
    axis square
    legend('Down and Left','Up and Right','location','southeast')
    xlabel('X Position on Screen')
    ylabel('Y Position on Screen')
    xlim([-.5 .5])
    ylim([-.5 .5])
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    
    savefig(fullfile(save_path_fig,fullFigName))
end
end

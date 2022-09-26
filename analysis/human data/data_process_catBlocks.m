function data_process_catBlocks(data_dir,blocks)

YYYYMMDD = data_dir(end-7:end);
YYYY = YYYYMMDD(1:4);
MM = YYYYMMDD(5:6);
DD = YYYYMMDD(7:8);

R_all = [];
state_all = [];
stimConds_all = [];
cursor_all = [];

for block = blocks%use_blocks
    load(fullfile(data_dir,[YYYY '.' MM '.' DD '_block' num2str(block) '.mat']));
    stream = streams;
    R = binnedRstream;
    
    clear streams binnedRstream
    
    %%
    last_ind = find(stream{1}.continuous.state==18,1,'last');
    
    R_all = cat(1,R_all,R.zScoreSpikes(1:floor(last_ind/20),:));
    state_all = cat(1,state_all,stream{1}.continuous.state(1:last_ind,:));
    stimConds_all = cat(1,stimConds_all,stream{1}.continuous.stimConds(1:last_ind,:,:));
    cursor_all = cat(1,cursor_all,stream{1}.continuous.rigidBodyPosXYZ_speed(1:last_ind,:));

end

trial = combine_trial_data(state_all, stimConds_all, cursor_all);

save_path = fullfile(data_dir,'process','zscore','catBlocks');
if ~exist(save_path,'dir')
    mkdir(save_path)
end

save(fullfile(save_path,['catBlocks_' YYYYMMDD '.mat']), 'R_all','trial')
end
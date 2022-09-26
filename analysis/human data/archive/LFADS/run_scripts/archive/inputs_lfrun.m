%% YANG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set up data for lfads
date = '20180730';
blocks = [4:8 13:18]; %10:11

LFADS_data_dir = fullfile('/home/laura/data/yangnet',date,'process/LFADS');
filename = ['LFADS_' date '_block'];

data = [];
for block = blocks
data = data_process_LFADS_join(data,LFADS_data_dir,filename,block,blocks);
end
dc.name = 'yangnet';
dataset_name = [filename '_join_' num2str(blocks(1)) '_' num2str(blocks(end)) '.mat'];
dc = YangExperiment.DatasetCollection(fullfile(LFADS_data_dir,'join'));
ds = YangExperiment.Dataset(dc, dataset_name); % adds this dataset to the collection
dc.loadInfo(); % loads dataset metadata

%% Run a single model for each dataset, and one stitched run with all datasets
runRoot = fullfile('~/runs');
rc = YangExperiment.RunCollection(runRoot, 'inputs', dc);
rc.version = datenum('now','YYYYMMDD');

%% run files will live at ~/data/yangnet/first_attempt/

% Setup hyperparameters, 4 sets with number of factors swept through 2,4,6,8
par = YangExperiment.RunParams;
par.name = '2'; % completely optional
par.spikeBinMs = 2; % rebin the data at 2 ms
par.c_co_dim = 2; % no controller outputs --> no inputs to generator
par.c_batch_size = 150; % must be < 1/5 of the min trial count
par.c_gen_dim = 64; % number of units in generator RNN
par.c_ic_enc_dim = 64; % number of units in encoder RNN
par.c_ci_enc_dim = 64; % number of units in encoder RNN
par.c_con_dim = 64; % number of units in controller RNN
par.c_learning_rate_stop = 1e-3; % we can stop really early for the demo
par.c_max_ckpt_to_keep = 20; % number of checkpoints to keep;
par.c_max_ckpt_to_keep_lve = 20;
parSet = par.generateSweep('c_factors_dim', 40);
rc.addParams(parSet);

%% Setup which datasets are included in each run, here just the one
runName = dc.datasets(1).getSingleRunName(); % == 'single_dataset001'
rc.addRunSpec(YangExperiment.RunSpec(runName, dc, 1));
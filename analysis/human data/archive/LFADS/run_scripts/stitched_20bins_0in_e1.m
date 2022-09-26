%% YANG
% [ out ] = getLNDPaths( );
% data_dir = fullfile(out.dataPath,'human/yangnet/20180730/');
% blocks = [1:8 10:20];
% data_process(data_dir,blocks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set up data for lfads
date_set = {'20180730';'20180813';'20180815'};
blocks = {[4:8 10:11 13:18]; 3:18 ; 2:19}; %10:1110:12%
epoch = 1;

dataset_name_all = [];
for d = 1:size(date_set,1)
[ out ] = getLNDPaths( );
LFADS_data_dir = fullfile(out.dataPath,'human','yangnet',date_set{d},'process/zscore','alignMove');
filename = ['LFADS_' date_set{d} '_block'];
dataset_name = [filename '_join_' num2str(blocks{d}(1)) '_' num2str(blocks{d}(end)) '.mat'];
dc.name = 'yangnet';
dataset_name_all = cat(1,dataset_name_all,{dataset_name});
dc = YangExperiment.DatasetCollection(fullfile('~/data/human/yangnet','join'));
end
YangExperiment.Dataset(dc, dataset_name_all{1}); % adds this dataset to the collection
YangExperiment.Dataset(dc, dataset_name_all{2}); % adds this dataset to the collection
YangExperiment.Dataset(dc, dataset_name_all{3}); % adds this dataset to the collection
dc.loadInfo(); % loads dataset metadata

%% Run a single model for each dataset, and one stitched run with all datasets
runRoot = fullfile('~/runs');
rc = YangExperiment.RunCollection(runRoot, 'stitched_20bins_0in_e1', dc);
rc.version = datenum('now','YYYYMMDD');

%% run files will live at ~/data/yangnet/first_attempt/

% Setup hyperparameters, 4 sets with number of factors swept through 2,4,6,8
par = YangExperiment.RunParams;
par.name = 'inputs2_m'; % completely optional
par.spikeBinMs = 20; % rebin the data at 2 ms
par.c_co_dim = 0; % no controller outputs --> no inputs to generator
par.c_batch_size = 175; % must be < 1/5 of the min trial count
par.c_gen_dim = 64; % number of units in generator RNN
par.c_ic_enc_dim = 64; % number of units in encoder RNN
par.c_ci_enc_dim = 64; % number of units in encoder RNN
par.c_con_dim = 64; % number of units in controller RNN
par.c_learning_rate_stop = 1e-3; % we can stop really early for the demo
par.c_max_ckpt_to_keep = 7; % number of checkpoints to keep;
par.c_max_ckpt_to_keep_lve = 7;
parSet = par.generateSweep('c_factors_dim', 40);
rc.addParams(parSet);

%% Setup which datasets are included in each run, here just the one
% runName = dc.datasets(1).getSingleRunName(); % == 'single_dataset001'
rc.addRunSpec(YangExperiment.RunSpec('all', dc, 1:dc.nDatasets));

return;

%% Generate files needed for LFADS input on disk
rc.prepareForLFADS();
% % 
%% Write a python script that will train all of the LFADS runs using a
% load-balancer against the available CPUs and GPUs
rc.writeShellScriptRunQueue('display', 0, 'virtualenv', 'py27');


% cd
% data_all = [];
% for d = 1:size(date_set,1)
% load(fullfile('~/data/human/yangnet','join',dataset_name_all{d}))
% end
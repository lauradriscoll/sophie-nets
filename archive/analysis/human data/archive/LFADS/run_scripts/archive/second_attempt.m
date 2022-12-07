%% YANG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set up data for lfads
date = '20180730';
blocks = [4:8 10:11 13:18];

[ out ] = getLNDPaths( );
LFADS_data_dir = fullfile(out.dataPath,'human/yangnet',date,'process/LFADS','alignTextOffset');
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
rc = YangExperiment.RunCollection(runRoot, 'second_attempt', dc);
rc.version = 20180802;

%% run files will live at ~/data/yangnet/first_attempt/

% Setup hyperparameters, 4 sets with number of factors swept through 2,4,6,8
par = YangExperiment.RunParams;
par.name = 'second_attempt'; % completely optional
par.spikeBinMs = 2; % rebin the data at 2 ms
par.c_co_dim = 0; % no controller outputs --> no inputs to generator
par.c_batch_size = 150; % must be < 1/5 of the min trial count
par.c_gen_dim = 128; % number of units in generator RNN
par.c_ic_enc_dim = 128; % number of units in encoder RNN
par.c_learning_rate_stop = 1e-3; % we can stop really early for the demo
parSet = par.generateSweep('c_factors_dim', 40);
rc.addParams(parSet);

%% Setup which datasets are included in each run, here just the one
runName = dc.datasets(1).getSingleRunName(); % == 'single_dataset001'
rc.addRunSpec(LorenzExperiment.RunSpec(runName, dc, 1));

[ out ] = getLNDPaths( );
data_dir = fullfile(out.dataPath,'human/yangnet/20180730/');
blocks = [1:8 10:20];
data_process(data_dir,blocks)
data_process_20180730(data_dir,blocks)


movement_lfrun % LFADS on movement period
movement_lfrun2 % 

verify_outputs

reation_time_analysis
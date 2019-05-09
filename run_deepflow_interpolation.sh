END=0;
for i in $(seq 0 $END)
do 
python interpolate.py --working_dir ./ \
--output_dir runs/interpolation_5_1 \
--matlab_dir mrst/mrst-2018a/modules/optimization/examples/model2Dtest \
--mrst_dir mrst/mrst-2018a \
--checkpoints_dir checkpoints \
--reference_model reference/model_67_x.npy \
--use_prior_loss \
--optimize_flow \
--seed "$i"
done

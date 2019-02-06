END=100;
for i in $(seq 0 $END)
do 
python run_deepflow.py --working_dir ./ \
--output_dir runs/test_deepflow_both \
--matlab_dir mrst/mrst-2018a/modules/optimization/examples/model2Dtest \
--mrst_dir mrst/mrst-2018a \
--checkpoints_dir checkpoints \
--reference_model reference/model_67_x.npy \
--optimizer adam \
--beta1 0.9 \
--beta2 0.999 \
--lr 1e-1 \
--iterations 500 \
--weight_decay 0.1 \
--optimize_wells \
--optimize_flow \
--early_stopping \
--target_accuracy 0.9 \
--seed "$i"
done
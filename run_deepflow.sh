END=1;
for i in $(seq 0 $END)
do 
python run_deepflow.py --working_dir ./ \
--output_dir runs/test_kl_div \
--matlab_dir mrst/mrst-2018a/modules/optimization/examples/model2Dtest \
--mrst_dir mrst/mrst-2018a \
--checkpoints_dir checkpoints \
--reference_model reference/model_67_x.npy \
--optimizer adam \
--beta1 0.9 \
--beta2 0.999 \
--lr 3e-2 \
--iterations 500 \
--weight_decay 0.0 \
--use_prior_loss \
--optimize_flow \
--seed "$i"
done

for subset in `seq 0 20`
do
python -W ignore infinite_generator_3D.py --fold $subset --scale 16 --data /data2/brain_mri/genesis_adni_dataset --save /data2/brain_mri/genesis_adni_generated_cubes
done

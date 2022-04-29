for subset in `seq 0 99`
do
python -W ignore infinite_generator_3D.py --fold $subset --scale 32 --data /data2/brain_mri/genesis_dataset --save /data2/brain_mri/genesis_generated_cubes
done

import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = "/data2/brain_mri/genesis_generated_cubes"
    train_fold=[i for i in range(99)]
    valid_fold=[0]
    test_fold=[]
    hu_min = 0.0
    hu_max = 255.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 64
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 32
    optimizer = "adam"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 5
    patience = 50
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

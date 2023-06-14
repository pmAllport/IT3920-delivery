# Codebase for: 'Applications of fault-tolerant software architecture principles in detection of adversarial attacks'

This codebase consists of the code used to implement the fault-tolerant architectures from the thesis. The codebase is configured to be self-contained and quite 'plug and play', given some small initial setup. In essence, the main points of interest are:

* "detectors/recovery_block/recovery_block.ipynb". This notebook tests and demonstrates the recovery block architecture
* "detectors/nvp/nvp_detector.ipynb". This notebook tests and demonstrates the N-Version Programming architecture and the individual detectors.
* "detectors/nvp/nvp_detector_dualtransform.ipynb". This notebook tests and demonstrates the nvp "dualtransform" detectors.


Some other points of interest may be:

* "thesis_scrips/verification_customized_dataset.ipynb". This notebook tests the different iterations of the customized dataset discussed in section 4.4.
* "thesis_scrips/generate_fourier_report_imgs.ipynb". This notebook generates the images and plots used in the thesis to demonstrate the adversarial attacks.
* "scripts/models/*". This folder contains the scripts run on the IDUN server to generate the component detectors.
* "scripts/attacks/". This folder contains the scripts run on the IDUN server to generate the adversarial dataset.


## Requirements:
* Python 3.9 (Tested on python 3.9.13)
* [Jupyter notebook](https://jupyter.org/)
* [Virtualenv](https://pypi.org/project/virtualenv/)



## Install guide:
- Create virtualenv for the project. E.g. run the command "virtualenv thesis_venv"
- Activate the virtual environment. E.g. run the command "source thesis_venv/bin/activate"
- Install the requirements in the requirements.txt file. E.g. run the command "pip install -r requirements.txt"
- Run the command "python -m ipykernel install --user --name=thesis_venv". This installs the virtualenv to the jupyter notebook.
- Set the "PROJECT_ROOT" environment variable in the ".env"-file to point to the absolute path of this codebase. E.g. "/home/\<username\>/Documents/project". (No trailing slash)



## Notes for code under "Detector" folder:
* Performant CPU and GPU recommended, but not required. 
* Reduce variable "dataset_batch_size" if encountering issues with insufficient VRAM.
* Reduce variable "num_workers" down for function "gen_mixed_detection_tvt_dict", if enountering issues with slow speeds in retrieving data.
* Code tested locally with the following system specs:
    * CPU: AMD Ryzen 9 3900X 12-Core Processor
    * GPU: Nvidia GeForce RTX 2080 8GB
    * RAM: 32 GB DDR4 2666 MT/s

## Notes for code under "Scripts/models"
* Performant CPU and GPU recommended.
* If running the code, know that the trained weights are saved under a folder called "run_test" and does not overwrite the weights used by the code in the "Detectors" folder. E.g. running the file "scripts/models/nvp/train_nvp.py" saves the weights under "weights/NVP/combined/run_test/trained.pt" and not "weights/NVP/combined/trained.pt". This is to allow for verification that the code runs, without having the risk of overwriting the weights with an incomplete set of weights.
* The scrips need command-line arguments to run. 
    * The arguments are :
        * -e (epochs). The number of epochs to run the training for. Generally set to a high value, as the early stopping callback should stop the training and not the epochs themselves.
        * -lr (learning rate). The learning rate used to train the detector.
        * For Individual detectors: -t (transformation). Is used to select the transformation used in training the model.
            * 0 = jpg_minpool_jpg
            * 1 = maxpool
            * 2 = minpool
            * 3 = minpool_jpg
        * For duotransform detectors: -t (transformation). Is used to select the first transformation. Each run trains every combination of the first transformation selected and the other three transformations as their second. E.g. "-t 0" = jpg_minpool_jpg + maxpool, jpg_minpool_jpg + minpool, jpg_minpool_jpg + minpool_jpg
            * 0 = jpg_minpool_jpg
            * 1 = maxpool
            * 2 = minpool
            * 3 = minpool_jpg 
    * Examples of commands would be:
        *"python scripts/models/nvp/train_nvp.py -e 100 -lr 1e-05"
        *"python scripts/models/nvp/train_nvp_individual.py -e 100 -lr 1e-04 -t 0" . This trains the individual 'jpg_minpool_jpg' detector
        *"python scripts/models/nvp/train_nvp_dualtransform.py -e 100 -lr 1e-04 -t 0" . This trains the dualtransform detectors with the 'jpg_minpool_jpg' as its first transform and the other three transformations as its second transformation.  
* Detectors were trained on IDUN-nodes with one of following system specs:
    * Configuration 1:
        * CPU: Intel® Xeon® Gold 6248R Processor 
        * GPU: NVIDIA A100 40Gb / NVIDIA A100 80Gb
        * RAM: 32 GB
    * Configuration 2:
        * CPU: AMD EPYC 75F3
        * GPU: NVIDIA A100 80Gb
        * RAM: 32 GB


## Notes for code under "Scripts/attacks"
* Performant CPU and GPU recommended.
* The scrips need command-line arguments to run. 
    * The arguments are :
        * -f (folder location). This is needs to point to a folder for a specific dataset for a specific class. E.g. "datasets/final_datasets/benign/full_size/train/vehicle"
        * -t (target location). This is needs to point to a folder where the images should be written. E.g. "datasets/adversarial_test/train/vehicle". It is recommended to not point this at the existing adversarial dataset folder, as it will overwrite the existing adversarial images.
    * Examples of a command would be:
        *"python scripts/attacks/apgd/generate_indiv_apgd.py -f datasets/final_datasets/benign/full_size/train/vehicle -t datasets/adversarial_test/train/vehicle". This will take all the vehicle images from the training set and use APGD to attack them. 
* Adversarial images were generated on IDUN-nodes with one of following system specs:
    * Configuration 1:
        * CPU: Intel® Xeon® Gold 6248R Processor 
        * GPU: NVIDIA A100 40Gb / NVIDIA A100 80Gb
        * RAM: 32 GB
    * Configuration 2:
        * CPU: AMD EPYC 75F3
        * GPU: NVIDIA A100 80Gb
        * RAM: 32 GB  

## [OPTIONAL] Notes on the rust code under "src"-folder:
* As mentioned in section 4.6.2 in the thesis, a piece of rust code was written to calculate the optimal threshold for the recovery block detectors. The provided code already contains a compiled version of the "src" code, under the "target" folder. This means that no actions need to be taken to utilize this code.  
* However, if one wants to recompile it, rust's build system and package manager [Cargo](https://doc.rust-lang.org/book/ch01-03-hello-cargo.html), needs to be installed. Given that it is installed, the rust can be recompiled by running the terminal command  "cargo build --release".
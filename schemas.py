from pydantic import BaseModel
from typing import Optional
from fastapi import UploadFile



class InferenceInput(BaseModel):
    protein_path: str
    protein_sequence: Optional[str] = None
    ligand_description: str

class InferenceConfig(BaseModel):
    actual_steps: int = 19
    ckpt: str = 'best_ema_inference_epoch_model.pt'
    confidence_ckpt: str = 'best_model_epoch75.pt'
    confidence_model_dir: str = './workdir/v1.1/confidence_model'
    different_schedules: bool = False
    inf_sched_alpha: int = 1
    inf_sched_beta: int = 1
    inference_steps: int = 20
    initial_noise_std_proportion: float = 1.4601642460337794
    limit_failures: int = 5
    model_dir: str = './workdir/v1.1/score_model'
    no_final_step_noise: bool = True
    no_model: bool = False
    no_random: bool = False
    no_random_pocket: bool = False
    ode: bool = False
    old_confidence_model: bool = True
    old_score_model: bool = False
    resample_rdkit: bool = False
    samples_per_complex: int = 10
    sigma_schedule: str = 'expbeta'
    temp_psi_rot: float = 0.9022615585677628
    temp_psi_tor: float = 0.5946212391366862
    temp_psi_tr: float = 0.727287304570729
    temp_sampling_rot: float = 2.06391612594481
    temp_sampling_tor: float = 7.044261621607846
    temp_sampling_tr: float = 1.170050527854316
    temp_sigma_data_rot: float = 0.7464326999906034
    temp_sigma_data_tor: float = 0.6943254174849822
    temp_sigma_data_tr: float = 0.9299802531572672
    loglevel: str = 'WARNING'
    choose_residue: bool = False
    out_dir: str = 'results/user_inference'
    save_visualisation: bool = False
    batch_size: int = 10
    
class InferenceRequest(BaseModel):
    input: InferenceInput
    config: InferenceConfig
    
class ZipInputConfig(BaseModel):
    zip_file: UploadFile
    inference_config: InferenceConfig

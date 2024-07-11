#schemas.py
from pydantic import BaseModel, conint
from typing import Optional, List
from fastapi import UploadFile

class InferenceInput(BaseModel):
    protein_path: str
    protein_sequence: Optional[str] = None
    ligand_description: str

# class InferenceConfig(BaseModel):
#     actual_steps: Optional[int] = None
#     ckpt: Optional[str] = None
#     confidence_ckpt: Optional[str] = None
#     confidence_model_dir: Optional[str] = None
#     different_schedules: Optional[bool] = None
#     inf_sched_alpha: Optional[int] = None
#     inf_sched_beta: Optional[int] = None
#     inference_steps: conint(ge=1,le=50) = 20
#     initial_noise_std_proportion: Optional[float] = None
#     limit_failures: Optional[int] = None
#     model_dir: Optional[str] = None
#     no_final_step_noise: Optional[bool] = None
#     no_model: Optional[bool] = None
#     no_random: Optional[bool] = None
#     no_random_pocket: Optional[bool] = None
#     ode: Optional[bool] = None
#     old_confidence_model: Optional[bool] = None
#     old_score_model: Optional[bool] = None
#     resample_rdkit: Optional[bool] = None
#     samples_per_complex: conint(ge=1,le=50) = 10
#     sigma_schedule: Optional[str] = None
#     temp_psi_rot: Optional[float] = None
#     temp_psi_tor: Optional[float] = None
#     temp_psi_tr: Optional[float] = None
#     temp_sampling_rot: Optional[float] = None
#     temp_sampling_tor: Optional[float] = None
#     temp_sampling_tr: Optional[float] = None
#     temp_sigma_data_rot: Optional[float] = None
#     temp_sigma_data_tor: Optional[float] = None
#     temp_sigma_data_tr: Optional[float] = None
#     loglevel: Optional[str] = None
#     choose_residue: Optional[bool] = None
#     out_dir: Optional[str] = None
#     save_visualisation: Optional[bool] = None
#     batch_size: Optional[int] = None
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
    batch_size: int = 1

class InferenceConfig(BaseModel):
    smiles: List[str] 
    actual_steps: Optional[int] = None
    ckpt: Optional[str] = None
    confidence_ckpt: Optional[str] = None
    confidence_model_dir: Optional[str] = None
    different_schedules: Optional[bool] = None
    inf_sched_alpha: Optional[int] = None
    inf_sched_beta: Optional[int] = None
    inference_steps: conint(ge=1, le=50) = 20
    initial_noise_std_proportion: Optional[float] = None
    limit_failures: Optional[int] = None
    model_dir: Optional[str] = None
    no_final_step_noise: Optional[bool] = None
    no_model: Optional[bool] = None
    no_random: Optional[bool] = None
    no_random_pocket: Optional[bool] = None
    ode: Optional[bool] = None
    old_confidence_model: Optional[bool] = None
    old_score_model: Optional[bool] = None
    resample_rdkit: Optional[bool] = None
    samples_per_complex: conint(ge=1, le=50) = 10
    sigma_schedule: Optional[str] = None
    temp_psi_rot: Optional[float] = None
    temp_psi_tor: Optional[float] = None
    temp_psi_tr: Optional[float] = None
    temp_sampling_rot: Optional[float] = None
    temp_sampling_tor: Optional[float] = None
    temp_sampling_tr: Optional[float] = None
    temp_sigma_data_rot: Optional[float] = None
    temp_sigma_data_tor: Optional[float] = None
    temp_sigma_data_tr: Optional[float] = None
    loglevel: Optional[str] = None
    choose_residue: Optional[bool] = None
    out_dir: Optional[str] = None
    save_visualisation: Optional[bool] = None
    batch_size: Optional[int] = None

class InferenceRequest(BaseModel):
    input: InferenceInput
    config: InferenceConfig
    
class ZipInputConfig(BaseModel):
    zip_file: UploadFile
    inference_config: InferenceConfig

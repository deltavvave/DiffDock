import logging
import os
import yaml
import copy
from functools import partial
import warnings
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs
from tqdm import tqdm
from utils.logging_utils import configure_logger, get_logger
from utils.inference_utils import InferenceDataset, set_nones
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.download import download_and_extract
from utils.utils import get_model
from utils.visualise import PDBFile
from datasets.process_mols import write_mol_with_coords
from schemas import InferenceInput, InferenceConfig
import io
from zipfile import ZipFile
from argparse import Namespace
import pandas as pd 

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`")

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]

tasks = {}  # Ensure this is accessible

async def process_zip_and_run_inference(task_id: str, zip_path: str, config: InferenceConfig):
    # Update task status to 'running'
    tasks[task_id]['status'] = 'running'
    try:
        # Extract zip file
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(f"/tmp/{task_id}")

        input_dir = f"/tmp/{task_id}"
        
        # Find the PDB and SDF files
        pdb_file = None
        sdf_file = None
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.pdb'):
                pdb_file = file_name
            elif file_name.endswith('.sdf'):
                sdf_file = file_name

        # Check if both files are found
        if pdb_file is None or sdf_file is None:
            raise FileNotFoundError("Either PDB or SDF file is missing in the zip archive.")

        # Create InferenceInput object
        inference_input = InferenceInput(
            protein_path=os.path.join(input_dir, pdb_file),
            ligand_description=os.path.join(input_dir, sdf_file)
        )

        # Run inference task
        await run_inference_task(task_id, inference_input, config)
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        raise
def run_inference_task(task_id: str, config: InferenceConfig):
    configure_logger(config.loglevel)
    logger = get_logger()
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"DiffDock will run on {device}")

        if not os.path.exists(config.model_dir):
            logger.info(f"Models not found. Downloading")
            downloaded_successfully = False
            for remote_url in REMOTE_URLS:
                try:
                    logger.info(f"Attempting download from {remote_url}")
                    files_downloaded = download_and_extract(remote_url, os.path.dirname(config.model_dir))
                    if not files_downloaded:
                        logger.info(f"Download from {remote_url} failed.")
                        continue
                    logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                    downloaded_successfully = True
                    break
                except Exception as e:
                    logger.error(f"Failed to download from {remote_url}: {str(e)}")
            
            if not downloaded_successfully:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['error'] = "Models not found locally and failed to download"
                return
 
 
        os.makedirs(config.out_dir, exist_ok=True)

        with open(f'{config.model_dir}/model_parameters.yml') as f:
            score_model_args = Namespace(**yaml.full_load(f))
        if config.confidence_model_dir:
            with open(f'{config.confidence_model_dir}/model_parameters.yml') as f:
                confidence_args = Namespace(**yaml.full_load(f))

        # df = pd.read_csv(config.protein_ligand_csv)
        # complex_name_list = set_nones(df['complex_name'].tolist())
        # protein_path_list = set_nones(df['protein_path'].tolist())
        # protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        # ligand_description_list = set_nones(df['ligand_description'].tolist())

        df = pd.read_csv(config.protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist())
        protein_path_list = set_nones(df['protein_path'].tolist())
        protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())

        logging.info(df)

        for name in complex_name_list:
            write_dir = f'{config.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)

        test_dataset = InferenceDataset(
            out_dir=config.out_dir,
            complex_names=complex_name_list,
            protein_files=protein_path_list,
            ligand_descriptions=ligand_description_list,
            protein_sequences=protein_sequence_list,
            lm_embeddings=True,
            receptor_radius=score_model_args.receptor_radius,
            remove_hs=score_model_args.remove_hs,
            c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
            all_atoms=score_model_args.all_atoms,
            atom_radius=score_model_args.atom_radius,
            atom_max_neighbors=score_model_args.atom_max_neighbors,
            knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

        if config.confidence_model_dir and not confidence_args.use_original_model_cache:
            logger.info('Confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.')
            confidence_test_dataset = InferenceDataset(
                out_dir=config.out_dir,
                complex_names=complex_name_list,
                protein_files=protein_path_list,
                ligand_descriptions=ligand_description_list,
                protein_sequences=protein_sequence_list,
                lm_embeddings=True,
                receptor_radius=confidence_args.receptor_radius,
                remove_hs=confidence_args.remove_hs,
                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                all_atoms=confidence_args.all_atoms,
                atom_radius=confidence_args.atom_radius,
                atom_max_neighbors=confidence_args.atom_max_neighbors,
                precomputed_lm_embeddings=test_dataset.lm_embeddings,
                knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
            )
        else:
            confidence_test_dataset = None

        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=config.old_score_model)
        state_dict = torch.load(f'{config.model_dir}/{config.ckpt}', map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        if config.confidence_model_dir:
            confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True, old=config.old_confidence_model)
            state_dict = torch.load(f'{config.confidence_model_dir}/{config.confidence_ckpt}', map_location=device)
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None

        tr_schedule = get_t_schedule(inference_steps=config.inference_steps, sigma_schedule='expbeta')

        failures, skipped = 0, 0
        N = config.samples_per_complex
        test_ds_size = len(test_dataset)
        logger.info(f'Size of test dataset: {test_ds_size}')

        for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
            logging.info('-'*100)
            logging.info(f'Doing process for {orig_complex_graph["name"]}')
            logging.info('-'*100)
            if not orig_complex_graph.success[0]:
                skipped += 1
                logger.warning(f"The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
                continue
            try:
                if confidence_test_dataset:
                    confidence_complex_graph = confidence_test_dataset[idx]
                    if not confidence_complex_graph.success:
                        skipped += 1
                        logger.warning(f"The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                        continue
                    confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
                else:
                    confidence_data_list = None
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                                   initial_noise_std_proportion=config.initial_noise_std_proportion,
                                   choose_residue=config.choose_residue)

                lig = orig_complex_graph.mol[0]

                pdb = None
                if config.save_visualisation:
                    visualization_list = []
                    for graph in data_list:
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                data_list, confidence = sampling(data_list=data_list, model=model,
                                                 inference_steps=config.actual_steps if config.actual_steps is not None else config.inference_steps,
                                                 tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                                 device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                 visualization_list=visualization_list, confidence_model=confidence_model,
                                                 confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                                 batch_size=config.batch_size, no_final_step_noise=config.no_final_step_noise,
                                                 temp_sampling=[config.temp_sampling_tr, config.temp_sampling_rot, config.temp_sampling_tor],
                                                 temp_psi=[config.temp_psi_tr, config.temp_psi_rot, config.temp_psi_tor],
                                                 temp_sigma_data=[config.temp_sigma_data_tr, config.temp_sigma_data_rot, config.temp_sigma_data_tor])

                ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

                if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    confidence = confidence[re_order]
                    ligand_pos = ligand_pos[re_order]
                    write_dir = f'{config.out_dir}/{task_id}/{test_dataset.ligand_descriptions[idx]}'
                    os.makedirs(write_dir, exist_ok=True)
                    logger.debug(f"Created directory: {write_dir}")

                    try:
                        for rank, pos in enumerate(ligand_pos):
                            mol_pred = copy.deepcopy(lig)
                            if score_model_args.remove_hs:
                                mol_pred = RemoveAllHs(mol_pred)
                            output_file = os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf')
                            write_mol_with_coords(mol_pred, pos, output_file)
                            logger.debug(f"Wrote file: {output_file}")
                    except Exception as e:
                        logger.error(f"Error writing output files: {str(e)}")
                        tasks[task_id]['status'] = 'failed'
                        tasks[task_id]['error'] = f"Error writing output files: {str(e)}"
                        return

                if config.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                    else:
                        for rank, batch_idx in enumerate(ligand_pos):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

            except Exception as e:
                logger.warning(f"Failed on {orig_complex_graph['name']}: {str(e)}")
                failures += 1

        result_msg = f"""
        Failed for {failures} / {test_ds_size} complexes.
        Skipped {skipped} / {test_ds_size} complexes.
    """
        if failures or skipped:
            logger.warning(result_msg)
        else:
            logger.info(result_msg)
        
        tasks[task_id]['status'] = "completed"
        logger.info(f"Results saved in {config.out_dir}")

    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        raise

# def run_inference_task(task_id: str, input: InferenceInput, config: InferenceConfig):
#     configure_logger(config.loglevel)
#     logger = get_logger()
#     try:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"DiffDock will run on {device}")

#         if not os.path.exists(config.model_dir):
#             logger.info(f"Models not found. Downloading")
#             downloaded_successfully = False
#             for remote_url in REMOTE_URLS:
#                 try:
#                     logger.info(f"Attempting download from {remote_url}")
#                     files_downloaded = download_and_extract(remote_url, os.path.dirname(config.model_dir))
#                     if not files_downloaded:
#                         logger.info(f"Download from {remote_url} failed.")
#                         continue
#                     logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
#                     downloaded_successfully = True
#                     break
#                 except Exception as e:
#                     logger.error(f"Failed to download from {remote_url}: {str(e)}")
            
#             if not downloaded_successfully:
#                 tasks[task_id]['status'] = 'failed'
#                 tasks[task_id]['error'] = "Models not found locally and failed to download"
#                 return

#         os.makedirs(config.out_dir, exist_ok=True)

#         with open(f'{config.model_dir}/model_parameters.yml') as f:
#             score_model_args = Namespace(**yaml.full_load(f))
#         if config.confidence_model_dir:
#             with open(f'{config.confidence_model_dir}/model_parameters.yml') as f:
#                 confidence_args = Namespace(**yaml.full_load(f))

#         # Load the CSV file for multiple complexes
#         protein_path_list, ligand_description_list = [], []
#         with open(input.csv_path, mode='r') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 protein_path_list.append(row['protein_path'])
#                 ligand_description_list.append(row['ligand_description'])

#         complex_name_list = [f"complex_{i}" for i in range(len(protein_path_list))]
#         protein_sequence_list = [""] * len(protein_path_list)

#         for name in complex_name_list:
#             write_dir = f'{config.out_dir}/{name}'
#             os.makedirs(write_dir, exist_ok=True)

#         test_dataset = InferenceDataset(
#             out_dir=config.out_dir,
#             complex_names=complex_name_list,
#             protein_files=protein_path_list,
#             ligand_descriptions=ligand_description_list,
#             protein_sequences=protein_sequence_list,
#             lm_embeddings=True,
#             receptor_radius=score_model_args.receptor_radius,
#             remove_hs=score_model_args.remove_hs,
#             c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
#             all_atoms=score_model_args.all_atoms,
#             atom_radius=score_model_args.atom_radius,
#             atom_max_neighbors=score_model_args.atom_max_neighbors,
#             knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
#         )
#         test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

#         if config.confidence_model_dir and not confidence_args.use_original_model_cache:
#             logger.info('Confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.')
#             confidence_test_dataset = InferenceDataset(
#                 out_dir=config.out_dir,
#                 complex_names=complex_name_list,
#                 protein_files=protein_path_list,
#                 ligand_descriptions=ligand_description_list,
#                 protein_sequences=protein_sequence_list,
#                 lm_embeddings=True,
#                 receptor_radius=confidence_args.receptor_radius,
#                 remove_hs=confidence_args.remove_hs,
#                 c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
#                 all_atoms=confidence_args.all_atoms,
#                 atom_radius=confidence_args.atom_radius,
#                 atom_max_neighbors=confidence_args.atom_max_neighbors,
#                 precomputed_lm_embeddings=test_dataset.lm_embeddings,
#                 knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
#             )
#         else:
#             confidence_test_dataset = None

#         t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

#         model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=config.old_score_model)
#         state_dict = torch.load(f'{config.model_dir}/{config.ckpt}', map_location=device)
#         model.load_state_dict(state_dict, strict=True)
#         model = model.to(device)
#         model.eval()

#         if config.confidence_model_dir:
#             confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True, old=config.old_confidence_model)
#             state_dict = torch.load(f'{config.confidence_model_dir}/{config.confidence_ckpt}', map_location=device)
#             confidence_model.load_state_dict(state_dict, strict=True)
#             confidence_model = confidence_model.to(device)
#             confidence_model.eval()
#         else:
#             confidence_model = None
#             confidence_args = None

#         tr_schedule = get_t_schedule(inference_steps=config.inference_steps, sigma_schedule='expbeta')

#         failures, skipped = 0, 0
#         N = config.samples_per_complex
#         test_ds_size = len(test_dataset)
#         logger.info(f'Size of test dataset: {test_ds_size}')

#         for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
#             if not orig_complex_graph.success[0]:
#                 skipped += 1
#                 logger.warning(f"The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
#                 continue
#             try:
#                 if confidence_test_dataset:
#                     confidence_complex_graph = confidence_test_dataset[idx]
#                     if not confidence_complex_graph.success:
#                         skipped += 1
#                         logger.warning(f"The confidence dataset did not contain {orig_complex_graph['name']}. We are skipping this complex.")
#                         continue
#                     confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
#                 else:
#                     confidence_data_list = None

#                 data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
#                 randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
#                                    initial_noise_std_proportion=config.initial_noise_std_proportion,
#                                    choose_residue=config.choose_residue)

#                 lig = orig_complex_graph.mol[0]

#                 pdb = None
#                 if config.save_visualisation:
#                     visualization_list = []
#                     for graph in data_list:
#                         pdb = PDBFile(lig)
#                         pdb.add(lig, 0, 0)
#                         pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
#                         pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
#                         visualization_list.append(pdb)
#                 else:
#                     visualization_list = None

#                 data_list, confidence = sampling(data_list=data_list, model=model,
#                                                  inference_steps=config.actual_steps if config.actual_steps is not None else config.inference_steps,
#                                                  tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
#                                                  device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
#                                                  visualization_list=visualization_list, confidence_model=confidence_model,
#                                                  confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
#                                                  batch_size=config.batch_size, no_final_step_noise=config.no_final_step_noise,
#                                                  temp_sampling=[config.temp_sampling_tr, config.temp_sampling_rot, config.temp_sampling_tor],
#                                                  temp_psi=[config.temp_psi_tr, config.temp_psi_rot, config.temp_psi_tor],
#                                                  temp_sigma_data=[config.temp_sigma_data_tr, config.temp_sigma_data_rot, config.temp_sigma_data_tor])

#                 ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

#                 if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
#                     confidence = confidence[:, 0]
#                 if confidence is not None:
#                     confidence = confidence.cpu().numpy()
#                     re_order = np.argsort(confidence)[::-1]
#                     confidence = confidence[re_order]
#                     ligand_pos = ligand_pos[re_order]
#                     write_dir = f'{config.out_dir}/{task_id}'
#                     os.makedirs(write_dir, exist_ok=True)
#                     logger.debug(f"Created directory: {write_dir}")

#                     try:
#                         for rank, pos in enumerate(ligand_pos):
#                             mol_pred = copy.deepcopy(lig)
#                             if score_model_args.remove_hs:
#                                 mol_pred = RemoveAllHs(mol_pred)
#                             output_file = os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf')
#                             write_mol_with_coords(mol_pred, pos, output_file)
#                             logger.debug(f"Wrote file: {output_file}")
#                     except Exception as e:
#                         logger.error(f"Error writing output files: {str(e)}")
#                         tasks[task_id]['status'] = 'failed'
#                         tasks[task_id]['error'] = f"Error writing output files: {str(e)}"
#                         return

#                 if config.save_visualisation:
#                     if confidence is not None:
#                         for rank, batch_idx in enumerate(re_order):
#                             visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
#                     else:
#                         for rank, batch_idx in enumerate(ligand_pos):
#                             visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

#             except Exception as e:
#                 logger.warning(f"Failed on {orig_complex_graph['name']}: {str(e)}")
#                 failures += 1

#         result_msg = f"""
#         Failed for {failures} / {test_ds_size} complexes.
#         Skipped {skipped} / {test_ds_size} complexes.
#     """
#         if failures or skipped:
#             logger.warning(result_msg)
#         else:
#             logger.info(result_msg)
        
#         tasks[task_id]['status'] = "completed"
#         logger.info(f"Results saved in {config.out_dir}")

#     except Exception as e:
#         tasks[task_id]['status'] = 'failed'
#         tasks[task_id]['error'] = str(e)
#         raise

def zip_output_files(task_id: str, output_folder: str):
    zip_stream = io.BytesIO()
    with ZipFile(zip_stream, 'w') as zip_file:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(task_id, os.path.relpath(file_path, output_folder))
                zip_file.write(file_path, arcname)
    
    zip_stream.seek(0)
    return zip_stream

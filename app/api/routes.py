import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from ..services.mining_service import MiningService
from ..config.settings import Config

router = APIRouter()

@router.post("/mine")
def mine(
    graph_file: UploadFile = File(...), 
    job_id: str = Form(...),
    min_pattern_size: int = Form(...),
    max_pattern_size: int = Form(...),
    min_neighborhood_size: int = Form(...),
    max_neighborhood_size: int = Form(...),
    n_neighborhoods: int = Form(...),
    n_trials: int = Form(...),
    radius: int = Form(3), # Radius kept with default as removed from pipeline
    graph_type: str = Form(...),
    search_strategy: str = Form("greedy"),
    sample_method: str = Form("tree"),
    visualize_instances: bool = Form(...)
):
    # Validate file
    if not graph_file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the uploaded file
    filename = "{}.pkl".format(uuid.uuid4())
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(graph_file.file, buffer)
        
        # Prepare mining config
        mining_config = {
            'min_pattern_size': min_pattern_size,
            'max_pattern_size': max_pattern_size,
            'min_neighborhood_size': min_neighborhood_size,
            'max_neighborhood_size': max_neighborhood_size,
            'n_neighborhoods': n_neighborhoods,
            'n_trials': n_trials,
            'radius': radius,
            'graph_type': graph_type,
            'search_strategy': search_strategy,
            'sample_method': sample_method,
            'visualize_instances': visualize_instances
        }
            
        # Run miner with job_id and config
        result = MiningService.run_miner(
            filepath, 
            job_id=job_id,
            config=mining_config
        )

        # Construct response
        response = {
            'job_id': result['job_id'],
            'results_path': result['results_path'],
            'plots_path': result['plots_path'],
            'status': 'success'
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        print("Error: {}".format(str(e)), flush=True)
        return JSONResponse(status_code=500, content={'error': str(e)})
    finally:
        # Cleanup input file
        if os.path.exists(filepath):
            os.remove(filepath)

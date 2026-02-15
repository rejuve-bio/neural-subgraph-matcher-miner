import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from ..services.mining_service import MiningService
from ..config.settings import Config

router = APIRouter()

def _coerce_int(val, default, name="param"):
    """Coerce form value to int; use default if missing or invalid."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


@router.post("/mine")
def mine(
    graph_file: UploadFile = File(...),
    job_id: str = Form(...),
    min_pattern_size: str = Form("3"),
    max_pattern_size: str = Form("5"),
    min_neighborhood_size: str = Form("3"),
    max_neighborhood_size: str = Form("5"),
    n_neighborhoods: str = Form("500"),
    n_trials: str = Form("100"),
    radius: str = Form("3"),
    graph_type: str = Form("directed"),
    search_strategy: str = Form("greedy"),
    sample_method: str = Form("tree"),
    out_batch_size: str = Form("3"),
    visualize_instances: str = Form("false")
):
    # Validate file
    if not graph_file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Coerce all numeric form values (form often sends strings)
    min_ps = _coerce_int(min_pattern_size, 3, "min_pattern_size")
    max_ps = _coerce_int(max_pattern_size, 5, "max_pattern_size")
    min_ns = _coerce_int(min_neighborhood_size, 3, "min_neighborhood_size")
    max_ns = _coerce_int(max_neighborhood_size, 5, "max_neighborhood_size")
    n_neigh = _coerce_int(n_neighborhoods, 500, "n_neighborhoods")
    n_tr = _coerce_int(n_trials, 100, "n_trials")
    rad = _coerce_int(radius, 3, "radius")
    out_bs = _coerce_int(out_batch_size, 3, "out_batch_size")

    vi = str(visualize_instances).lower() in ("true", "1", "yes")

    # Debug: log received and coerced values so decoder gets user config
    print(
        "DEBUG miner /mine received: min_pattern_size={} max_pattern_size={} out_batch_size={} visualize_instances={}".format(
            min_ps, max_ps, out_bs, vi
        ),
        flush=True,
    )

    filename = "{}.pkl".format(uuid.uuid4())
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(graph_file.file, buffer)

        mining_config = {
            'min_pattern_size': min_ps,
            'max_pattern_size': max_ps,
            'min_neighborhood_size': min_ns,
            'max_neighborhood_size': max_ns,
            'n_neighborhoods': n_neigh,
            'n_trials': n_tr,
            'radius': rad,
            'graph_type': (graph_type or "directed").strip().lower(),
            'search_strategy': (search_strategy or "greedy").strip().lower(),
            'sample_method': (sample_method or "tree").strip().lower(),
            'out_batch_size': out_bs,
            'visualize_instances': vi
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

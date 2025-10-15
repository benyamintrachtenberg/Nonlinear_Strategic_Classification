#!/bin/bash


job_id="--j=$SLURM_JOB_ID"

### pass in budget as a comma separated sting (no spaces)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --times) times="--times=$2"; shift ;;
        --min) min_deg="--min=$2"; shift ;;
        --max) max_deg="--max=$2"; shift ;;
        --bud) budget="--bud=$2"; shift ;;
        --reg_thr) reg_thr="--reg_thr=$2"; shift ;;
        --strat_thr) strat_thr="--strat_thr=$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done


python3 expressivity_experiment.py $times "$budget" $min_deg $max_deg $reg_thr $strat_thr $job_id